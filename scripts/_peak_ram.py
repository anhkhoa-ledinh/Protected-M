r"""Peak working set của tiến trình hiện tại (MB) qua ``GetProcessMemoryInfo``."""
from __future__ import annotations

import ctypes


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_working_set_mb():
    """Đỉnh working set tính bằng MB; ``-1`` nếu gọi API hỏng."""
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(_PMC), ctypes.c_uint32]
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                    ctypes.byref(pmc), pmc.cb)
    return round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 1) if ok else -1.0
