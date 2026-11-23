#!/usr/bin/env bash
# Protected M - run the game (Linux). Use: bash run_game.sh [extra main.py args]
# Log (UTF-8): logs/game_<date>_<time>.log. Override the interpreter with $PM_PYTHON.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
PY="${PM_PYTHON:-$HOME/.venvs/protected-m/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python venv not found: $PY"
  echo "Create it first (see the Linux section of README.md), or set PM_PYTHON."
  exit 1
fi
mkdir -p logs
LOG="logs/game_$(date +%Y%m%d_%H%M%S).log"
echo "Protected M - log: $LOG"
PYTHONIOENCODING=utf-8 "$PY" main.py --album "$@" > "$LOG" 2>&1
code=$?
echo "Exited (code $code). Log: $LOG"
exit "$code"
