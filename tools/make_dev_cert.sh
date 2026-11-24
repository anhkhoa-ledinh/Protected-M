#!/usr/bin/env bash
# Self-signed TLS pair for the central server (development / LAN).
# Usage: bash tools/make_dev_cert.sh <out_dir> [extra-ip-or-dns ...]
# Needs the system `openssl`. Never writes inside the repo (the key is secret).
set -euo pipefail

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is not installed; install it, then run this again" >&2
    exit 3
fi
if [ $# -lt 1 ]; then
    echo "usage: bash tools/make_dev_cert.sh <out_dir> [extra-ip-or-dns ...]" >&2
    exit 2
fi

out="$1"; shift
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
out="$(realpath -m "$out")"
case "$out/" in
    "$repo"/*) echo "refused: $out is inside the repo" >&2; exit 2 ;;
esac
mkdir -p "$out"
if [ -e "$out/cert.pem" ] || [ -e "$out/key.pem" ]; then
    echo "refused: cert.pem or key.pem already exists in $out" >&2
    exit 2
fi

san="DNS:localhost,IP:127.0.0.1"
for extra in "$@"; do
    if [[ "$extra" =~ ^[0-9.]+$ || "$extra" == *:* ]]; then
        san="$san,IP:$extra"
    else
        san="$san,DNS:$extra"
    fi
done

umask 077
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
    -days 825 -subj "/CN=hcpd-dev" -addext "subjectAltName=$san" \
    -keyout "$out/key.pem" -out "$out/cert.pem" 2>/dev/null
chmod 600 "$out/key.pem"
chmod 644 "$out/cert.pem"
echo "cert: $out/cert.pem"
echo "key:  $out/key.pem"
echo "san:  $san"
