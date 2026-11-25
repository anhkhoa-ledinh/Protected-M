#!/usr/bin/env bash
# Copy cert.pem and key.pem from a directory into the Docker volume the server
# reads (default hcpd-certs). Pipes the files through stdin, so it needs no
# shared host paths (works with Docker Desktop too).
# Usage: bash deploy/import_certs.sh <dir with cert.pem and key.pem> [volume]
set -euo pipefail
dir="${1:?usage: bash deploy/import_certs.sh <dir> [volume]}"
vol="${2:-${HCPD_CERTS_VOLUME:-hcpd-certs}}"
image="${HCPD_IMAGE:-hcpd-server}"
for f in cert.pem key.pem; do
    [ -f "$dir/$f" ] || { echo "missing $dir/$f" >&2; exit 2; }
done
docker volume create "$vol" >/dev/null
for f in cert.pem key.pem; do
    mode=644; [ "$f" = key.pem ] && mode=600
    docker run --rm -i --user 0 --cap-drop ALL --cap-add CHOWN --cap-add FOWNER \
        --cap-add DAC_OVERRIDE -v "$vol:/certs" --entrypoint sh "$image" \
        -c "umask 077; cat > /certs/$f && chown 10001:10001 /certs/$f && chmod $mode /certs/$f" \
        < "$dir/$f"
done
echo "imported cert.pem and key.pem into volume $vol"
