#!/usr/bin/env bash
# Protected M - PvP LAN server (TCP 47821 + UDP discovery 47822). No webcam, no window.
# Use: bash run_server.sh [extra args]. Ctrl+C stops it. Override Python with $PM_PYTHON.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
PY="${PM_PYTHON:-$HOME/.venvs/protected-m/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python venv not found: $PY"
  echo "Create it first (see the Linux section of README.md), or set PM_PYTHON."
  exit 1
fi
echo "Protected M - PvP server. Open TCP 47821 and UDP 47822 if you run a firewall. Ctrl+C to stop."
PYTHONIOENCODING=utf-8 "$PY" -m src.net.server -v "$@"
code=$?
echo "Exited (code $code)."
exit "$code"
