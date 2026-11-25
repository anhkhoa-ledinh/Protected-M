"""Account economy fields: migration at load, spend, record_win."""

import json

from src.account import store

OLD = {"accounts": {"ann": {
    "display": "Ann", "password_hash": "scrypt$x$y", "recovery_hash": "scrypt$a$b",
    "created_at": 1.5, "session_token_hash": None, "session_expires_at": None,
    "history": {"p.jpg": {"date": "01/01/2026", "eaten": 4.0, "score": 77}},
    "recent": ["p.jpg"], "best_score": 77, "last_signed_in_at": 2.5}},
    "last_account": "ann"}


def _write(root, data):
    path = store.accounts_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_migration_adds_defaults_and_keeps_the_rest(tmp_path):
    _write(tmp_path, OLD)
    data = store.load(tmp_path).data
    record = data["accounts"]["ann"]
    assert (record["wallet"], record["earned_total"],
            record["levels_cleared"]) == (100, 0, [])
    for key, value in OLD["accounts"]["ann"].items():
        assert record[key] == value
    store.save(tmp_path, data)
    again = json.loads(store.accounts_path(tmp_path).read_text("utf-8"))
    assert again["accounts"]["ann"] == record


def test_existing_values_are_not_overwritten(tmp_path):
    old = json.loads(json.dumps(OLD))
    old["accounts"]["ann"].update(wallet=5, earned_total=9, levels_cleared=[2])
    _write(tmp_path, old)
    record = store.load(tmp_path).data["accounts"]["ann"]
    assert (record["wallet"], record["earned_total"],
            record["levels_cleared"]) == (5, 9, [2])


def test_new_account_starts_with_100():
    data = {"accounts": {}, "last_account": None}
    key, _ = store.create_account(data, "Bob", "password1", now=1.0)
    assert store.get_economy(data, key) == {
        "wallet": 100, "earned_total": 0, "levels_cleared": []}


def test_spend_and_win():
    data = store.migrate(json.loads(json.dumps(OLD)))
    assert store.spend(data, "ann", 30)
    assert store.get_economy(data, "ann")["wallet"] == 70
    assert not store.spend(data, "ann", 71)            # cannot afford
    assert store.get_economy(data, "ann")["wallet"] == 70
    store.record_win(data, "ann", 3, 120)
    store.record_win(data, "ann", 3, 10)
    eco = store.get_economy(data, "ann")
    assert eco == {"wallet": 200, "earned_total": 130, "levels_cleared": [3]}
    assert store.spend(data, "ann", 200)
    assert store.get_economy(data, "ann")["earned_total"] == 130   # never lowered


def test_guest_has_no_wallet():
    data = {"accounts": {}, "last_account": None}
    assert store.get_economy(data, store.GUEST_KEY) is None
    assert not store.spend(data, store.GUEST_KEY, 1)
    store.record_win(data, store.GUEST_KEY, 1, 5)       # no-op, no crash
