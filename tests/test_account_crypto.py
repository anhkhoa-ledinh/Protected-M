"""
TASK_accounts_1119 Phase 1: ``src/account/crypto.py`` — hashing, username/
password rules, recovery codes, session tokens. Pure Python, no GL, no
file I/O — safe on CI.
"""

from __future__ import annotations

import unicodedata

import pytest

from src.account import crypto
from src.shared import account_rules


# ---- hashing ----------------------------------------------------------------

def test_round_trip_verifies():
    stored = crypto.hash_secret("correct horse battery staple")
    assert crypto.verify_secret("correct horse battery staple", stored)


def test_wrong_password_fails():
    stored = crypto.hash_secret("correct horse battery staple")
    assert not crypto.verify_secret("incorrect horse", stored)


def test_two_hashes_of_the_same_password_differ():
    a = crypto.hash_secret("same password 123")
    b = crypto.hash_secret("same password 123")
    assert a != b
    assert crypto.verify_secret("same password 123", a)
    assert crypto.verify_secret("same password 123", b)


@pytest.mark.parametrize("stored", [
    "garbage", "", "scrypt$abc$8$1$x$y", "scrypt$16384$8$1$not-base64!!$y",
    "scrypt$16384$8$1$" + "=" * 8, "bcrypt$10$salt$hash",
])
def test_tampered_or_malformed_stored_string_fails_without_crashing(stored):
    assert crypto.verify_secret("anything", stored) is False


def test_rehash_happens_when_parameters_change():
    stored = crypto.hash_secret("a password")
    assert crypto.needs_rehash(stored) is False
    # Simulate old params by hand-building a hash with a different N.
    old = crypto._scrypt("a password", b"0" * 16, 2 ** 10, 8, 1, 32)
    old_stored = f"scrypt$1024$8$1${crypto._b64(b'0'*16)}${crypto._b64(old)}"
    assert crypto.verify_secret("a password", old_stored)
    assert crypto.needs_rehash(old_stored) is True


def test_malformed_stored_string_needs_rehash():
    assert crypto.needs_rehash("not a real hash") is True


def test_nfc_and_nfd_forms_of_a_vietnamese_password_both_verify():
    nfc = unicodedata.normalize("NFC", "Việt Nam 2026!!")
    nfd = unicodedata.normalize("NFD", "Việt Nam 2026!!")
    assert nfc != nfd  # the two forms really are different byte sequences
    stored = crypto.hash_secret(nfc)
    assert crypto.verify_secret(nfd, stored)
    stored2 = crypto.hash_secret(nfd)
    assert crypto.verify_secret(nfc, stored2)


# ---- usernames ----------------------------------------------------------------

USERNAME_CASES = [
    ("anhkhoa", True), ("ab", False), ("abc", True), ("a" * 16, True),
    ("a" * 17, False), ("has space", False), ("guest", False),
    ("Guest", False), ("GUEST", False), ("anh.khoa", True),
    ("anh_khoa", True), ("anh-khoa", True), ("😀😀😀", False),
    ("anhkhôa", True), ("   anhkhoa   ", True),  # strip makes this valid
    ("", False), ("ab\tcd", False),
]


@pytest.mark.parametrize("raw,expect_valid", USERNAME_CASES)
def test_username_validity_table(raw, expect_valid):
    display, reason = account_rules.validate_username(raw)
    assert (display is not None) == expect_valid, reason


def test_username_key_is_case_insensitive():
    assert account_rules.username_key("AnhKhoa") == account_rules.username_key("anhkhoa")
    assert account_rules.username_key("AnhKhoa") == account_rules.username_key("ANHKHOA")


def test_username_display_form_preserved_exactly():
    display, _ = account_rules.validate_username("  AnhKhoa  ")
    assert display == "AnhKhoa"


def test_username_nfc_normalised():
    nfd = unicodedata.normalize("NFD", "Việt")
    display, reason = account_rules.validate_username(nfd)
    assert display == unicodedata.normalize("NFC", nfd)


# ---- passwords ----------------------------------------------------------------

@pytest.mark.parametrize("raw,expect_valid", [
    ("1234567", False), ("12345678", True), ("x" * 64, True),
    ("x" * 65, False), ("", False),
])
def test_password_length_table(raw, expect_valid):
    ok, reason = account_rules.validate_password(raw)
    assert ok == expect_valid, reason


# ---- recovery codes ----------------------------------------------------------

def test_recovery_code_shape():
    code = crypto.generate_recovery_code()
    groups = code.split("-")
    assert len(groups) == 3
    assert all(len(g) == 4 for g in groups)
    assert all(c in crypto._CROCKFORD_ALPHABET for g in groups for c in g)


def test_recovery_code_excludes_confusable_letters():
    alphabet = crypto._CROCKFORD_ALPHABET
    assert not set("ILOU") & set(alphabet)


def test_normalize_recovery_code_strips_separators_and_upcases():
    assert account_rules.normalize_recovery_code("abcd-efgh-jkmn") == "ABCDEFGHJKMN"
    assert account_rules.normalize_recovery_code("ABCD EFGH JKMN") == "ABCDEFGHJKMN"
    assert account_rules.normalize_recovery_code("ABCDEFGHJKMN") == "ABCDEFGHJKMN"


def test_recovery_codes_are_unique_in_practice():
    codes = {crypto.generate_recovery_code() for _ in range(200)}
    assert len(codes) == 200


# ---- session tokens -----------------------------------------------------------

def test_session_token_round_trip():
    token = crypto.generate_session_token()
    stored = crypto.hash_token(token)
    assert crypto.verify_token(token, stored)
    assert not crypto.verify_token("different-token", stored)


def test_session_tokens_are_unique():
    tokens = {crypto.generate_session_token() for _ in range(100)}
    assert len(tokens) == 100
