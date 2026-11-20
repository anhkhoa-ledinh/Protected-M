r"""
TASK_arch_refactor_1120: compare a real automatch event log against the
R0 baseline for G3 ("event log identical to the baseline").

    python scripts/automatch_log_diff.py <baseline.txt> <current.txt>

Every line must match EXACTLY except the final ``Album match: ...``
line's score, which may differ by at most 1 point: the score formula
(``AlbumMatch.score``) subtracts elapsed WALL-CLOCK time, and the
autowin hook's kills land ~0.1 s apart in real time, not simulated
time -- measured directly (two same-commit baseline reruns, R2) to
occasionally round to a score 1 point lower depending on real
scheduling jitter between the match starting and the last kill
landing, with every other field (weapon, damage, bug id, hp_after,
kill flag, kill order, win/lose, bugs found/total) always identical.
That is the ONLY tolerance here; everything else is exact.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_SCORE_LINE_RE = re.compile(r"^(Album match: \w+ . cap \d+, \d+/\d+ bo, diem )(-?\d+)$")


def lines_match(baseline_line: str, current_line: str) -> tuple[bool, str]:
    if baseline_line == current_line:
        return True, "exact match"
    m_base = _SCORE_LINE_RE.match(baseline_line)
    m_cur = _SCORE_LINE_RE.match(current_line)
    if m_base and m_cur and m_base.group(1) == m_cur.group(1):
        delta = abs(int(m_base.group(2)) - int(m_cur.group(2)))
        if delta <= 1:
            return True, f"score within tolerance (delta={delta})"
        return False, f"score delta {delta} > 1 tolerance"
    return False, "differs"


def compare(baseline_lines: list[str], current_lines: list[str]) -> tuple[bool, list[str]]:
    details = []
    ok = len(baseline_lines) == len(current_lines)
    if not ok:
        details.append(f"line count differs: {len(baseline_lines)} vs {len(current_lines)}")
        return False, details
    for i, (b, c) in enumerate(zip(baseline_lines, current_lines)):
        matched, why = lines_match(b, c)
        details.append(f"line {i}: {'OK' if matched else 'DIFF'} -- {why}")
        ok = ok and matched
    return ok, details


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <baseline.txt> <current.txt>")
        return 2
    baseline = Path(argv[1]).read_text(encoding="utf-8").splitlines()
    current = Path(argv[2]).read_text(encoding="utf-8").splitlines()
    ok, details = compare(baseline, current)
    for line in details:
        print(line)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
