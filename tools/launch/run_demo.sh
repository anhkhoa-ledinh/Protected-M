#!/usr/bin/env bash
# Protected M - research demo (Linux), the twin of run_demo.bat.
# Only G, T, B, + = - 0, F10 (relief on/off), F6, drag-drop, F4, Esc work; reads assets/config/content.demo.json,
# writes no config file. Log (UTF-8): logs/demo_<date>_<time>.log.
# Use: bash tools/launch/run_demo.sh [extra main.py args]. Override the interpreter with $PM_PYTHON.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." || exit 1
PY="${PM_PYTHON:-$HOME/.venvs/protected-m/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python venv not found: $PY"
  echo "Create it first (see the Linux section of README.md), or set PM_PYTHON."
  exit 1
fi
mkdir -p logs
LOG="logs/demo_$(date +%Y%m%d_%H%M%S).log"
echo "HCPD demo - log: $LOG"
PYTHONIOENCODING=utf-8 "$PY" main.py --demo "$@" > "$LOG" 2>&1
code=$?
echo "Exited (code $code). Log: $LOG"
exit "$code"
