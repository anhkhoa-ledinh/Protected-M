#!/usr/bin/env bash
# Run the ubuntu CI job (.github/workflows/tests.yml) in a clean container:
# no DISPLAY, no GPU, same apt packages, Python 3.13, CPU torch from the lock.
#   bash tools/ci_local.sh [extra pytest args]
# The repo is mounted read-only; the committed HEAD is cloned inside, like
# actions/checkout does (fetch-depth 0). Commit before you run it.
# Everything that installs runs inside the container, never on the host.
# Exit code = pytest's exit code.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${CI_LOCAL_IMAGE:-python:3.13-slim}"
CACHE_VOLUME="${CI_LOCAL_PIP_VOLUME:-hcpd-ci-pip-cache}"

command -v docker >/dev/null 2>&1 || { echo "docker is not installed" >&2; exit 3; }

inner='
set -euo pipefail
apt-get update -qq
apt-get install -y -qq --no-install-recommends git libgl1 libglib2.0-0 >/dev/null
git config --global --add safe.directory /src
git clone -q /src /work
cd /work
TV=$(grep -E "^torch==" requirements-lock.txt | cut -d= -f3)
VV=$(grep -E "^torchvision==" requirements-lock.txt | cut -d= -f3)
pip install -q "torch==$TV" "torchvision==$VV" --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements-lock.txt
set +e
HF_HUB_OFFLINE=1 python -m pytest -q -ra "$@" 2>&1 | tee /tmp/pytest.out | tail -n 400
rc=${PIPESTATUS[0]}
echo "--- ci_local: last line ---"
tail -n 1 /tmp/pytest.out
exit "$rc"
'

exec docker run --rm \
    -e HF_HUB_OFFLINE=1 -e PIP_CACHE_DIR=/pip-cache -e PIP_ROOT_USER_ACTION=ignore \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$ROOT:/src:ro" -v "$CACHE_VOLUME:/pip-cache" \
    "$IMAGE" bash -c "$inner" ci_local "$@"
