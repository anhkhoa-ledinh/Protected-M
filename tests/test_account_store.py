"""
TASK_accounts_1119 Phase 1: ``src/account/store.py`` — the atomic JSON
store (with ``.bak`` and corrupt-file recovery), accounts, sessions,
recovery codes, the failure throttle. Pure Python, no GL — real
filesystem via ``tmp_path``, safe on CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.account import crypto, store
from src.shared import account_rules

ROOT = Path(__file__).resolve().parent.parent


# ---- .gitignore --------------------------------------------------------------

def test_gitignore_lists_every_account_file():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("assets/config/accounts.json", "assets/config/accounts.json.bak",
                    "assets/config/accounts.corrupt-*.json", "assets/config/session.json"):
        assert pattern in text, f"{pattern!r} missing from .gitignore"


# ---- the atomic store itself --------------------------------------------------

def test_fresh_load_is_empty_and_not_corrupted(tmp_path):
    result = store.load(tmp_path)
    assert result.data == {"accounts": {}, "last_account": None}
    assert result.recovered is False
    assert result.reset is False


def test_save_then_load_round_trips(tmp_path):
    data = store.load(tmp_path).data
    key, code = store.create_account(data, "Alice", "password123")
    store.save(tmp_path, data)

    reloaded = store.load(tmp_path).data
    assert key in reloaded["accounts"]
    assert reloaded["accounts"][key]["display"] == "Alice"


def test_bak_is_written_on_the_second_save(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "Alice", "password123")
    store.save(tmp_path, data)
    bak = store._bak_path(store.accounts_path(tmp_path))
    assert not bak.exists()  # nothing to back up yet on the FIRST save

    data2 = store.load(tmp_path).data
    store.create_account(data2, "Bob", "password123")
    store.save(tmp_path, data2)
    assert bak.exists()
    bak_data = json.loads(bak.read_text(encoding="utf-8"))
    assert list(bak_data["accounts"].keys()) == ["alice"]


def test_corrupt_main_file_recovers_from_bak(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "Alice", "password123")
    store.save(tmp_path, data)
    data2 = store.load(tmp_path).data
    store.create_account(data2, "Bob", "password123")
    store.save(tmp_path, data2)

    store.accounts_path(tmp_path).write_text("{not valid json", encoding="utf-8")
    result = store.load(tmp_path)
    assert result.recovered is True
    assert result.reset is False
    assert list(result.data["accounts"].keys()) == ["alice"]


def test_both_corrupt_renames_bad_file_and_starts_empty(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "Alice", "password123")
    store.save(tmp_path, data)

    path = store.accounts_path(tmp_path)
    path.write_text("{not valid", encoding="utf-8")
    store._bak_path(path).write_text("{also not valid", encoding="utf-8")

    result = store.load(tmp_path)
    assert result.reset is True
    assert result.data == {"accounts": {}, "last_account": None}
    assert not path.exists()
    corrupt_files = list(path.parent.glob("accounts.corrupt-*.json"))
    assert len(corrupt_files) == 1


def test_missing_file_is_not_treated_as_corruption(tmp_path):
    # No accounts.json at all (first run) must NOT rename anything or
    # set reset=True — that's reserved for an UNREADABLE existing file.
    result = store.load(tmp_path)
    assert result.reset is False
    assert list((tmp_path / "assets" / "config").glob("accounts.corrupt-*.json")) == []


def test_simulated_crash_mid_write_leaves_the_original_intact(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "Carol", "password123")
    store.save(tmp_path, data)
    path = store.accounts_path(tmp_path)
    original = path.read_bytes()

    # Simulate a crash AFTER the temp file is written but BEFORE
    # os.replace() swaps it in — exactly where save() itself would be
    # interrupted; the real file must never be touched before that point.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(b"garbage from an interrupted write")

    assert path.read_bytes() == original


# ---- accounts: sign-up / sign-in -------------------------------------------------

def test_create_account_hashes_the_password_not_plaintext(tmp_path):
    data = store.load(tmp_path).data
    key, code = store.create_account(data, "Dave", "hunter2222")
    record = data["accounts"][key]
    assert "hunter2222" not in record["password_hash"]
    assert record["password_hash"].startswith("scrypt$")


def test_create_account_returns_a_recovery_code_only_its_hash_is_stored(tmp_path):
    data = store.load(tmp_path).data
    key, code = store.create_account(data, "Dave", "hunter2222")
    record = data["accounts"][key]
    assert code not in json.dumps(record)
    assert crypto.verify_secret(account_rules.normalize_recovery_code(code),
                                record["recovery_hash"])


def test_find_key_is_case_insensitive(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "AnhKhoa", "password123")
    assert store.find_key(data, "anhkhoa") == "anhkhoa"
    assert store.find_key(data, "ANHKHOA") == "anhkhoa"
    assert store.find_key(data, "nobody") is None


def test_is_available_false_for_taken_or_invalid_name(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "AnhKhoa", "password123")
    assert store.is_available(data, "AnhKhoa") is False
    assert store.is_available(data, "anhkhoa") is False  # case-insensitive
    assert store.is_available(data, "guest") is False
    assert store.is_available(data, "ab") is False  # too short
    assert store.is_available(data, "SomeoneNew") is True


def test_verify_credentials_three_outcomes(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "Eve", "correctpass")

    assert store.verify_credentials(data, "eve", "correctpass")[0] == store.RESULT_OK
    assert store.verify_credentials(data, "eve", "wrongpass11")[0] == store.RESULT_WRONG
    assert store.verify_credentials(data, "nosuchuser", "x" * 8)[0] == store.RESULT_UNKNOWN


def test_verify_credentials_rehashes_stale_params_in_place(tmp_path, monkeypatch):
    data = store.load(tmp_path).data
    key, _code = store.create_account(data, "Fay", "correctpass")
    before = data["accounts"][key]["password_hash"]
    monkeypatch.setattr(crypto, "needs_rehash", lambda stored: True)
    store.verify_credentials(data, "fay", "correctpass")
    after = data["accounts"][key]["password_hash"]
    assert after != before
    assert crypto.verify_secret("correctpass", after)


# ---- sessions -------------------------------------------------------------------

def test_session_created_when_requested(tmp_path):
    data = store.load(tmp_path).data
    key, _code = store.create_account(data, "Gail", "password123")
    store.start_session(tmp_path, data, key)
    assert data["accounts"][key]["session_token_hash"] is not None
    assert store.session_path(tmp_path).exists()


def test_resolve_session_succeeds_for_a_valid_token(tmp_path):
    data = store.load(tmp_path).data
    key, _code = store.create_account(data, "Gail", "password123")
    store.start_session(tmp_path, data, key)
    store.save(tmp_path, data)

    reloaded = store.load(tmp_path).data
    assert store.resolve_session(tmp_path, reloaded) == key


def test_resolve_session_fails_when_expired(tmp_path):
    data = store.load(tmp_path).data
    key, _code = store.create_account(data, "Gail", "password123")
    store.start_session(tmp_path, data, key, now=1000.0)
    far_future = 1000.0 + store.SESSION_LIFETIME_S + 1.0
    assert store.resolve_session(tmp_path, data, now=far_future) is None


def test_resolve_session_fails_when_tampered(tmp_path):
    data = store.load(tmp_path).data
    key, _code = store.create_account(data, "Gail", "password123")
    store.start_session(tmp_path, data, key)
    path = store.session_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["token"] = "not-the-real-token"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.resolve_session(tmp_path, data) is None


def test_switch_off_deletes_session(tmp_path):
    data = store.load(tmp_path).data
    key, _code = store.create_account(data, "Gail", "password123")
    store.start_session(tmp_path, data, key)
    assert store.session_path(tmp_path).exists()
    store.clear_session(tmp_path, data, key)
    assert not store.session_path(tmp_path).exists()
    assert data["accounts"][key]["session_token_hash"] is None


def test_sign_out_deletes_session(tmp_path):
    data = store.load(tmp_path).data
    key, _code = store.create_account(data, "Gail", "password123")
    store.start_session(tmp_path, data, key)
    store.clear_session(tmp_path, data, key)  # sign-out uses the same call
    assert store.resolve_session(tmp_path, data) is None


# ---- recovery ---------------------------------------------------------------

def test_correct_code_resets_password_and_issues_a_new_code(tmp_path):
    data = store.load(tmp_path).data
    key, old_code = store.create_account(data, "Hana", "oldpassword1")
    result, found_key = store.verify_recovery(data, "hana", old_code)
    assert result == store.RESULT_OK and found_key == key
    new_code = store.reset_with_recovery(data, key, "newpassword1")
    assert new_code != old_code
    assert store.verify_credentials(data, "hana", "newpassword1")[0] == store.RESULT_OK


def test_old_code_no_longer_works_after_reset(tmp_path):
    data = store.load(tmp_path).data
    key, old_code = store.create_account(data, "Hana", "oldpassword1")
    store.reset_with_recovery(data, key, "newpassword1")
    result, _ = store.verify_recovery(data, "hana", old_code)
    assert result == store.RESULT_WRONG


def test_wrong_recovery_code_result(tmp_path):
    data = store.load(tmp_path).data
    store.create_account(data, "Hana", "oldpassword1")
    result, key = store.verify_recovery(data, "hana", "ZZZZ-ZZZZ-ZZZZ")
    assert result == store.RESULT_WRONG
    assert key == "hana"


def test_recovery_for_unknown_account(tmp_path):
    data = store.load(tmp_path).data
    result, key = store.verify_recovery(data, "nosuchuser", "ZZZZ-ZZZZ-ZZZZ")
    assert result == store.RESULT_UNKNOWN
    assert key is None


# ---- the in-memory failure throttle ----------------------------------------------

def test_five_failures_start_the_cooldown_and_it_ends_after_30s():
    throttle = store.FailThrottle()
    now = 100.0
    for _ in range(store.MAX_FAILURES):
        throttle.record_failure("eve", now)
    assert throttle.cooldown_remaining("eve", now) == pytest.approx(store.COOLDOWN_S)
    assert throttle.cooldown_remaining("eve", now + store.COOLDOWN_S) == 0.0


def test_fewer_than_five_failures_no_cooldown():
    throttle = store.FailThrottle()
    now = 0.0
    for _ in range(store.MAX_FAILURES - 1):
        throttle.record_failure("eve", now)
    assert throttle.cooldown_remaining("eve", now) == 0.0


def test_successful_sign_in_resets_the_counter():
    throttle = store.FailThrottle()
    now = 0.0
    for _ in range(store.MAX_FAILURES - 1):
        throttle.record_failure("eve", now)
    throttle.reset("eve")
    for _ in range(store.MAX_FAILURES - 1):
        throttle.record_failure("eve", now)
    # Still under the threshold since reset() cleared the earlier count.
    assert throttle.cooldown_remaining("eve", now) == 0.0


def test_throttle_is_per_account(tmp_path):
    throttle = store.FailThrottle()
    now = 0.0
    for _ in range(store.MAX_FAILURES):
        throttle.record_failure("eve", now)
    assert throttle.cooldown_remaining("eve", now) > 0.0
    assert throttle.cooldown_remaining("adam", now) == 0.0


# ---- saved progress ---------------------------------------------------------------

def test_history_saved_for_one_account_does_not_leak_to_another(tmp_path):
    data = store.load(tmp_path).data
    key_a, _ = store.create_account(data, "Alice", "password123")
    key_b, _ = store.create_account(data, "Bob", "password123")
    store.save_progress(data, key_a, {"photo1.jpg": {"damage": 5}}, ["photo1.jpg"], 100.0)
    store.save_progress(data, key_b, {"photo2.jpg": {"damage": 9}}, ["photo2.jpg"], 50.0)

    loaded_a = store.load_progress(data, key_a)
    loaded_b = store.load_progress(data, key_b)
    assert loaded_a["history"] == {"photo1.jpg": {"damage": 5}}
    assert loaded_b["history"] == {"photo2.jpg": {"damage": 9}}
    assert loaded_a["best_score"] == 100.0
    assert loaded_b["best_score"] == 50.0


def test_guest_history_is_never_written(tmp_path):
    data = store.load(tmp_path).data
    store.save_progress(data, store.GUEST_KEY, {"photo1.jpg": {}}, ["photo1.jpg"], 10.0)
    assert store.GUEST_KEY not in data["accounts"]


def test_best_score_and_progress_survive_a_reload(tmp_path):
    data = store.load(tmp_path).data
    key, _ = store.create_account(data, "Ivy", "password123")
    store.save_progress(data, key, {"p.jpg": {"damage": 1}}, ["p.jpg"], 77.0)
    store.save(tmp_path, data)

    reloaded = store.load(tmp_path).data
    loaded = store.load_progress(reloaded, key)
    assert loaded["best_score"] == 77.0
    assert loaded["history"] == {"p.jpg": {"damage": 1}}
    assert loaded["recent"] == ["p.jpg"]


def test_save_progress_round_trips_path_entries_in_recent(tmp_path):
    """Regression: app.py puts ``Path`` objects into ``recent``; before the
    fix, ``save_progress`` stored them as-is and ``json.dumps`` (inside
    ``save``) raised ``TypeError: Object of type WindowsPath is not JSON
    serializable``."""
    data = store.load(tmp_path).data
    key, _ = store.create_account(data, "Jan", "password123")
    recent_in = [Path("a.jpg"), Path("sub") / "b.jpg"]
    store.save_progress(data, key, {}, recent_in, None)
    store.save(tmp_path, data)   # must not raise

    reloaded = store.load(tmp_path).data
    loaded = store.load_progress(reloaded, key)
    assert all(isinstance(p, str) for p in loaded["recent"])
    assert [Path(p) for p in loaded["recent"]] == recent_in
