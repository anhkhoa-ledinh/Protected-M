"""Part of ``V3UI``, moved out of ``v3_ui.py`` unchanged (B5): see ``AccountMixin``."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from src.shared import account_rules
from src.ui_album import v3_flow as F
from src.ui_album import ids as ID

logger = logging.getLogger("src.ui_album.v3_ui")


class AccountMixin:
    # ---- TASK_accounts_1119: accounts — local store, no server -----------
    @property
    def account_busy(self) -> bool:
        """True while a sign-in/sign-up hash job is running on its own
        thread — the primary button must ignore repeat clicks during this
        window (scrypt takes tens to hundreds of ms; a second click while
        one is in flight would start a SECOND concurrent hash for no
        reason, and could race the first one's result application)."""
        return self._account_job is not None

    def pop_account_corrupt_toast(self) -> bool:
        """True exactly once, the first time it's read after a launch
        where ``accounts.json`` AND its ``.bak`` were both unreadable —
        ``app.py`` shows a toast for this, then it never fires again this
        process (the store already started empty and renamed the bad
        file aside; nothing more to say after the first toast)."""
        return self._accounts.take_corrupt_notice()

    def _start_account_job(self, kind: str, fn, *args) -> None:
        if self.account_busy:
            return
        holder: dict = {}
        self._account_job = (kind, holder)

        def worker() -> None:
            try:
                holder["result"] = fn(*args)
            except Exception as exc:                          # noqa: BLE001
                holder["error"] = exc
            finally:
                holder["done"] = True

        threading.Thread(target=worker, name=f"account-{kind}", daemon=True).start()

    def _poll_account_job(self) -> None:
        if self._account_job is None:
            return
        kind, holder = self._account_job
        if not holder.get("done"):
            return
        self._account_job = None
        if "error" in holder:
            logger.exception("TASK_accounts_1119: viec '%s' loi", kind,
                             exc_info=holder["error"])
            F.say(self.state, "Something went wrong. Try again.")
            return
        {
            "sign_in": self._apply_sign_in_result,
            "sign_up": self._apply_sign_up_result,
            "forgot": self._apply_forgot_result,
        }[kind](holder["result"])

    def sign_in_submit(self) -> None:
        """``ACT_SIGN_IN`` (``app.py``): the cooldown check is cheap and
        synchronous (in-memory, no hashing) and happens HERE, before even
        starting a thread — a name mid-cooldown must never spend a scrypt
        call just to be told to wait."""
        username, password = self.state.player, self.state.password
        now = time.perf_counter()
        remaining = self._accounts.throttle_remaining(username, now)
        if remaining > 0.0:
            self.state.sign_in_error = f"Too many tries. Wait {int(round(remaining))} s."
            self.state.go(F.SIGN_IN_ERR)
            return
        self._start_account_job("sign_in", self._verify_credentials_job,
                                username, password)

    def _verify_credentials_job(self, username: str, password: str):
        return self._accounts.verify_credentials(username, password)

    def _apply_sign_in_result(self, result) -> None:
        outcome, key = result
        now = time.perf_counter()
        if outcome == account_rules.RESULT_OFFLINE:
            self.state.sign_in_error = "Can't reach the server. Try again, or play as guest."
            self.state.go(F.SIGN_IN_ERR)
            return
        if outcome == account_rules.RESULT_OK:
            self._accounts.throttle_reset(key)
            self.state.account_key = key
            self.state.player = self._accounts.display_of(key)
            self.state.password = self.state.password2 = ""
            self.state.focus = None
            self._load_progress(key)
            # TASK_accounts_1119 Phase 4: the switch's CURRENT state wins
            # on every sign-in — on means create/refresh a session, off
            # means make sure no stale one from an EARLIER sign-in (with
            # the switch on, since changed) survives this one.
            self._accounts.remember(key, bool(self.state.remember))
            self.state.go(F.MENU)
            return
        # RESULT_UNKNOWN has no key (no such account) — throttle on the
        # NAME the user actually typed, same as RESULT_WRONG's real key
        # (an unknown name still burns cooldown attempts, matching the
        # task's own "5 failures in a row for one name", not "one
        # account").
        throttle_key = key or self.state.player
        self._accounts.throttle_failure(throttle_key, now)
        remaining = self._accounts.throttle_remaining(throttle_key, now)
        if remaining > 0.0:
            self.state.sign_in_error = f"Too many tries. Wait {int(round(remaining))} s."
        elif outcome == account_rules.RESULT_UNKNOWN:
            self.state.sign_in_error = f"No album named {self.state.player}. Create one below."
        else:
            self.state.sign_in_error = "Wrong password. Try again."
        self.state.go(F.SIGN_IN_ERR)

    def sign_up_submit(self) -> None:
        """``ACT_SIGN_UP`` — ``v3_routes`` already checked the FORMAT
        (name shape, password length/match) synchronously; this does the
        one check that needs the store (name actually free) plus the
        hashing, both off-thread."""
        self._start_account_job("sign_up", self._create_account_job,
                                self.state.player, self.state.password)

    def _create_account_job(self, display: str, password: str):
        return self._accounts.create_account(display, password)

    def _apply_sign_up_result(self, result) -> None:
        outcome, key, code = result
        if outcome == account_rules.RESULT_OFFLINE:
            self.state.touched.add(ID.FIELD_USER)
            self.state.hint_bad = True
            self.state.hint_text = "Can't reach the server"
            return
        if outcome == "taken":
            self.state.touched.add(ID.FIELD_USER)
            self.state.hint_bad = True
            self.state.hint_text = "Username already taken"
            return
        self.state.account_key = key
        self.state.player = self._accounts.display_of(key)
        self.state.password = self.state.password2 = ""
        self.state.focus = None
        self.sync_economy()
        self.state.go(F.MENU)
        # TASK_accounts_1119 Phase 6: shown exactly once, right now — the
        # recovery dialog overlays whatever screen is current (MENU,
        # just set above), same as the share-album confirm dialog does.
        self.state.recovery_code_shown = code

    def forgot_submit(self) -> None:
        """``ACT_FORGOT_SUBMIT`` (``app.py``): same cooldown-before-thread
        pattern as ``sign_in_submit`` — a wrong recovery code burns the
        SAME per-username throttle as a wrong password (task: "wrong
        codes use the same 5-try/30s cooldown")."""
        username = self.state.forgot_user
        now = time.perf_counter()
        remaining = self._accounts.throttle_remaining(username, now)
        if remaining > 0.0:
            self.state.forgot_error = f"Too many tries. Wait {int(round(remaining))} s."
            return
        self._start_account_job("forgot", self._verify_recovery_job, username,
                                self.state.forgot_code, self.state.forgot_new_pass)

    def _verify_recovery_job(self, username: str, code: str, new_password: str):
        return self._accounts.recover(username, code, new_password)

    def _apply_forgot_result(self, result) -> None:
        outcome, key, new_code = result
        now = time.perf_counter()
        if outcome == account_rules.RESULT_OFFLINE:
            self.state.forgot_error = "Can't reach the server. Try again."
            return
        if outcome == account_rules.RESULT_OK:
            self._accounts.throttle_reset(key)
            self.state.forgot_open = False
            self.state.forgot_focus = None
            self.state.forgot_user = self.state.forgot_code = ""
            self.state.forgot_new_pass = self.state.forgot_new_pass2 = ""
            self.state.forgot_error = ""
            # TASK_accounts_1119 Phase 6: the old code is used up the
            # moment it verifies — show the SAME recovery-code dialog
            # again with the new one, once.
            self.state.recovery_code_shown = new_code
            return
        # Same "throttle on the typed name" fallback as
        # ``_apply_sign_in_result`` — an unknown name still burns a
        # cooldown attempt.
        throttle_key = key or self.state.forgot_user
        self._accounts.throttle_failure(throttle_key, now)
        remaining = self._accounts.throttle_remaining(throttle_key, now)
        if remaining > 0.0:
            self.state.forgot_error = f"Too many tries. Wait {int(round(remaining))} s."
        elif outcome == account_rules.RESULT_UNKNOWN:
            self.state.forgot_error = f"No album named {self.state.forgot_user}. Create one below."
        else:
            self.state.forgot_error = "Wrong code. Try again."

    def copy_recovery_code(self) -> None:
        """``ACT_COPY_RECOVERY_CODE`` — clipboard copy, same mechanism as
        the PvP room-code "Copy" button (``app.py``)."""
        if self._window is not None:
            self._window.set_clipboard(self.state.recovery_code_shown)

    def save_progress(self) -> None:
        """TASK_accounts_1119 Phase 5: ``history``/``recent``/the best
        score (derived from ``history``), so the leaderboard "you" row
        and the menu photo caption/damage survive a restart. Called
        after every match result, on sign-out and on exit (``app.py``) —
        a no-op for guest (never saved) or no account signed in at all.
        """
        key = self.state.account_key
        if not key or key == account_rules.GUEST_KEY:
            return
        scores = [float(record.get("score", 0.0)) for record in self.state.history.values()
                 if isinstance(record, dict)]
        best_score = max(scores) if scores else None
        try:
            self._accounts.save_progress(key, self.state.history,
                                         self.state.recent, best_score)
        except Exception:
            logger.error("Khong luu duoc tien trinh tai khoan", exc_info=True)
            F.say(self.state, "Couldn't save your progress.")

    def dev_sign_in(self, key: str) -> None:
        """Evidence hooks only (``HCPD_ECONOMY_SIGNIN``, scripted runs on a
        temp accounts dir): act as signed in as ``key`` without typing the
        password, then land on the menu."""
        display = self._accounts.display_of(key)
        if display is None:
            logger.error("dev_sign_in: no account %r", key)
            return
        self.state.account_key = key
        self.state.player = display
        self._load_progress(key)
        self.state.go(F.MENU)

    def sync_economy(self) -> None:
        """Copy the signed-in account's points into the flow (guest or
        nobody signed in: no wallet)."""
        eco = self._accounts.economy(self.state.account_key)
        if eco is None:
            self.state.wallet = None
            self.state.earned_total = 0
            self.state.levels_cleared = ()
            return
        self.state.wallet = eco["wallet"]
        self.state.earned_total = eco["earned_total"]
        self.state.levels_cleared = tuple(eco["levels_cleared"])

    def charge_items(self, items, mode: str = "solo") -> bool:
        """Pay for ``items`` when a match starts. ``True`` = go on (paid,
        or a guest, or nothing to pay); ``False`` = cannot afford, nothing
        taken (the caller keeps the player on the loadout). On the central
        server a signed-in player always asks, even for no items: the server
        needs the ticket that the end of the match closes."""
        cost = self.state.price_total(items)
        key = self.state.account_key
        if self._accounts.economy(key) is None:
            return True
        if cost <= 0 and not self._accounts.wants_ticket:
            return True
        outcome = self._accounts.begin_match(key, self.state.level, list(items),
                                             cost, mode)
        if outcome != "ok":
            self.sync_economy()
            if outcome == "offline":
                F.say(self.state, "Can't reach the server - playing without items.")
            elif outcome == "locked_mode":
                F.say(self.state, "Win levels 1-5 to unlock PvP.")
            return False
        self._save_accounts()
        self.sync_economy()
        return True

    def _save_accounts(self) -> None:
        try:
            self._accounts.save()
        except Exception:
            logger.error("Khong luu duoc tai khoan", exc_info=True)
            F.say(self.state, "Couldn't save your points.")

    def note_match_finished(self, won: bool = False, level: int = 0,
                            score: int = 0, used: dict | None = None) -> int:
        """TASK_accounts_1119 Phase 5: call once right after a match
        result is written into ``state.history`` (``app.py``'s
        ``_tick_album_match``). A signed-in account saves immediately;
        a guest instead gets a ONE-TIME toast (never saved — this IS the
        explanation why)."""
        key = self.state.account_key
        if self._accounts.economy(key) is not None:
            score = self._accounts.end_match(key, level, won, dict(used or {}), score)
            self.sync_economy()
        if key == account_rules.GUEST_KEY:
            if not self._guest_toast_shown:
                self._guest_toast_shown = True
                F.say(self.state, "Playing as guest — create an album to keep your scores.")
            return score
        self.save_progress()
        return score

    def _load_progress(self, key: str) -> None:
        """The other half of ``save_progress`` — called right after a
        successful sign-in (``_apply_sign_in_result``) and on auto
        sign-in at launch, so ``flow.history``/``recent`` reflect what
        THIS account saved last time, not whatever a previous
        guest/account session left in memory."""
        progress = self._accounts.load_progress(key)
        if progress is None:
            return
        self.state.history = progress["history"]
        self.state.recent = [Path(p) for p in progress["recent"]]
        self.sync_economy()

    def sign_out(self) -> None:
        """TASK_accounts_1119 Phase 4: a deliberate sign-out must not
        leave a "remember me" session alive — same ``clear_session`` the
        switch's OFF path uses on a sign-in. Guest "sessions" never had
        one to begin with, but clearing ``session.json`` unconditionally
        is still correct (and cheap) either way. Phase 5: also the LAST
        chance to persist this session's progress before the account
        state about to be wiped by app.py's own "logout" handler."""
        self.save_progress()
        self._accounts.sign_out(self.state.account_key)
