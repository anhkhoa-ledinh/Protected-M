"""Make a file or folder readable by the current user only, on every OS.

POSIX: mode ``0o600`` for files, ``0o700`` for folders.
Windows: a protected DACL (no inherited entries) with exactly one allow entry
for the current user's SID, set and read through the Win32 security API with
``ctypes`` (standard library only). SIDs are compared, never account names,
because names are localized.

``is_private`` checks the same property in each system's own terms, so one
test can assert it everywhere. ``acl_report`` describes the permissions for
the message of a failing assertion.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
from ctypes import wintypes
from pathlib import Path


class PrivateFilesError(OSError):
    """The permissions could not be set or read."""


# ---- constants ------------------------------------------------------------------

SE_FILE_OBJECT = 1
DACL_SECURITY_INFORMATION = 0x4
PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
SE_DACL_PROTECTED = 0x1000
ACL_REVISION = 2
ACL_SIZE_INFORMATION = 2
TOKEN_QUERY = 0x8
TOKEN_USER = 1
ACCESS_ALLOWED_ACE_TYPE = 0
ACCESS_DENIED_ACE_TYPE = 1
OBJECT_INHERIT_ACE = 0x1
CONTAINER_INHERIT_ACE = 0x2
INHERITED_ACE = 0x10
FILE_ALL_ACCESS = 0x001F01FF

# An ACE is {BYTE type; BYTE flags; WORD size; DWORD mask; SID...}.
_ACE_SID_OFFSET = 8

# ---- Win32 functions: the one table every call goes through -----------------------

_P = ctypes.c_void_p
_PP = ctypes.POINTER(ctypes.c_void_p)

# name -> (dll, argtypes, restype)
_WIN_API = {
    "GetCurrentProcess": ("kernel32", [], wintypes.HANDLE),
    "CloseHandle": ("kernel32", [wintypes.HANDLE], wintypes.BOOL),
    "LocalFree": ("kernel32", [_P], _P),
    "OpenProcessToken": ("advapi32", [wintypes.HANDLE, wintypes.DWORD,
                                      ctypes.POINTER(wintypes.HANDLE)], wintypes.BOOL),
    "GetTokenInformation": ("advapi32", [wintypes.HANDLE, ctypes.c_int, _P, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
    "ConvertSidToStringSidW": ("advapi32", [_P, ctypes.POINTER(wintypes.LPWSTR)],
                               wintypes.BOOL),
    "ConvertStringSidToSidW": ("advapi32", [wintypes.LPCWSTR, _PP], wintypes.BOOL),
    "GetNamedSecurityInfoW": ("advapi32", [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD,
                                           _PP, _PP, _PP, _PP, _PP], wintypes.DWORD),
    "SetNamedSecurityInfoW": ("advapi32", [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
                                           _P, _P, _P, _P], wintypes.DWORD),
    "GetSecurityDescriptorControl": ("advapi32", [_P, ctypes.POINTER(wintypes.WORD),
                                                  ctypes.POINTER(wintypes.DWORD)],
                                     wintypes.BOOL),
    "GetAclInformation": ("advapi32", [_P, _P, wintypes.DWORD, ctypes.c_int], wintypes.BOOL),
    "GetAce": ("advapi32", [_P, wintypes.DWORD, _PP], wintypes.BOOL),
    "InitializeAcl": ("advapi32", [_P, wintypes.DWORD, wintypes.DWORD], wintypes.BOOL),
    "AddAccessAllowedAceEx": ("advapi32", [_P, wintypes.DWORD, wintypes.DWORD,
                                           wintypes.DWORD, _P], wintypes.BOOL),
    "GetLengthSid": ("advapi32", [_P], wintypes.DWORD),
}

_dlls: dict = {}
_funcs: dict = {}


def _api(name: str):
    """The Win32 function ``name`` with its types set. Loads the DLLs lazily."""
    func = _funcs.get(name)
    if func is None:
        dll_name, argtypes, restype = _WIN_API[name]
        dll = _dlls.get(dll_name)
        if dll is None:
            dll = _dlls[dll_name] = ctypes.WinDLL(dll_name, use_last_error=True)
        func = getattr(dll, name)
        func.argtypes = argtypes
        func.restype = restype
        _funcs[name] = func
    return func


def _ok(result, name: str) -> None:
    """Raise unless the BOOL ``result`` of ``name`` is true."""
    if not result:
        raise PrivateFilesError(f"{name} failed ({ctypes.get_last_error()})")


def _rc(code, name: str) -> None:
    """Raise unless ``code``, the return value of ``name``, is 0."""
    if code != 0:
        raise PrivateFilesError(f"{name} failed ({code})")


# ---- seams the tests replace --------------------------------------------------

def _is_windows() -> bool:
    return os.name == "nt"


def _sid_string(sid_ptr) -> str:
    text = wintypes.LPWSTR()
    _ok(_api("ConvertSidToStringSidW")(sid_ptr, ctypes.byref(text)),
        "ConvertSidToStringSidW")
    try:
        return text.value or ""
    finally:
        _api("LocalFree")(ctypes.cast(text, ctypes.c_void_p))


def _current_sid() -> str:
    """The string SID of the user of this process's token."""
    token = wintypes.HANDLE()
    _ok(_api("OpenProcessToken")(_api("GetCurrentProcess")(), TOKEN_QUERY,
                                 ctypes.byref(token)), "OpenProcessToken")
    try:
        need = wintypes.DWORD(0)
        get = _api("GetTokenInformation")
        get(token, TOKEN_USER, None, 0, ctypes.byref(need))   # size query: fails by design
        if need.value == 0:
            raise PrivateFilesError(f"GetTokenInformation failed ({ctypes.get_last_error()})")
        buf = ctypes.create_string_buffer(need.value)
        _ok(get(token, TOKEN_USER, buf, need.value, ctypes.byref(need)),
            "GetTokenInformation")
        sid_ptr = ctypes.c_void_p.from_buffer(buf).value      # TOKEN_USER.User.Sid, offset 0
        return _sid_string(sid_ptr)
    finally:
        _api("CloseHandle")(token)


def _read_dacl(path):
    """``(protected, [(type, flags, mask, sid), ...])`` or ``None`` for a NULL DACL."""
    dacl = _P()
    sd = _P()
    _rc(_api("GetNamedSecurityInfoW")(str(path), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
                                      None, None, ctypes.byref(dacl), None,
                                      ctypes.byref(sd)), "GetNamedSecurityInfoW")
    try:
        control = wintypes.WORD(0)
        revision = wintypes.DWORD(0)
        _ok(_api("GetSecurityDescriptorControl")(sd, ctypes.byref(control),
                                                 ctypes.byref(revision)),
            "GetSecurityDescriptorControl")
        if not dacl.value:
            return None
        protected = bool(control.value & SE_DACL_PROTECTED)

        info = (wintypes.DWORD * 3)()                  # AceCount, bytes in use, bytes free
        _ok(_api("GetAclInformation")(dacl, info, ctypes.sizeof(info), ACL_SIZE_INFORMATION),
            "GetAclInformation")
        aces = []
        for index in range(info[0]):
            ace = _P()
            _ok(_api("GetAce")(dacl, index, ctypes.byref(ace)), "GetAce")
            kind, flags, _size, mask = struct.unpack("<BBHI", ctypes.string_at(ace.value, 8))
            sid = ""
            if kind in (ACCESS_ALLOWED_ACE_TYPE, ACCESS_DENIED_ACE_TYPE):
                sid = _sid_string(ace.value + _ACE_SID_OFFSET)
            aces.append((kind, flags, mask, sid))
        return protected, aces
    finally:
        _api("LocalFree")(sd)


def _write_private_dacl(path, is_dir: bool) -> None:
    """Replace the DACL of ``path``: protected, one allow entry for the current user."""
    sid = _P()
    _ok(_api("ConvertStringSidToSidW")(_current_sid(), ctypes.byref(sid)),
        "ConvertStringSidToSidW")
    try:
        size = 8 + 8 + _api("GetLengthSid")(sid)       # ACL header + ACE header and mask + SID
        acl = ctypes.create_string_buffer(size)
        _ok(_api("InitializeAcl")(acl, size, ACL_REVISION), "InitializeAcl")
        flags = (OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE) if is_dir else 0
        _ok(_api("AddAccessAllowedAceEx")(acl, ACL_REVISION, flags, FILE_ALL_ACCESS, sid),
            "AddAccessAllowedAceEx")
        _rc(_api("SetNamedSecurityInfoW")(
            str(path), SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            None, None, acl, None), "SetNamedSecurityInfoW")
    finally:
        _api("LocalFree")(sid)


def _dacl_is_private(dacl, me_sid: str) -> bool:
    """True if ``dacl`` is protected and holds only plain allow entries for ``me_sid``."""
    if dacl is None:
        return False
    protected, aces = dacl
    if not protected or not aces:
        return False
    for kind, flags, _mask, sid in aces:
        if kind != ACCESS_ALLOWED_ACE_TYPE or flags & INHERITED_ACE or sid != me_sid:
            return False
    return True


# ---- public API ---------------------------------------------------------------

def make_private(path) -> None:
    path = Path(path)
    if _is_windows():
        _write_private_dacl(path, is_dir=path.is_dir())
        return
    os.chmod(path, 0o700 if path.is_dir() else 0o600)


def is_private(path) -> bool:
    path = Path(path)
    if _is_windows():
        return _dacl_is_private(_read_dacl(path), _current_sid())
    return (path.stat().st_mode & 0o077) == 0


def acl_report(path) -> str:
    """Describe the permissions of ``path`` for a failing assertion. Never raises."""
    try:
        path = Path(path)
        if not _is_windows():
            return f"{path}: mode {oct(path.stat().st_mode & 0o7777)}"
        parts = [f"{path}:"]
        try:                                   # an independent observer: not parsed
            done = subprocess.run(["icacls", str(path)], capture_output=True,
                                  text=True, timeout=30)
            parts.append(f"icacls rc={done.returncode}\n{done.stdout}{done.stderr}")
        except Exception as exc:
            parts.append(f"icacls could not run: {exc!r}")
        try:
            dacl = _read_dacl(path)
            if dacl is None:
                parts.append("_read_dacl: NULL DACL")
            else:
                protected, aces = dacl
                parts.append(f"_read_dacl: protected={protected}")
                parts.extend(f"  ace type={t} flags={f:#x} mask={m:#x} sid={s}"
                             for t, f, m, s in aces)
        except Exception as exc:
            parts.append(f"_read_dacl: error {exc!r}")
        try:
            parts.append(f"current sid: {_current_sid()}")
        except Exception as exc:
            parts.append(f"current sid: error {exc!r}")
        return "\n".join(parts)
    except Exception as exc:
        return f"acl_report failed: {exc!r}"
