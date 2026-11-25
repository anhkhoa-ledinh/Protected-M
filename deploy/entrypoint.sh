#!/bin/sh
# Starts the server inside the container. Everything comes from environment
# variables set by docker-compose.yml; the cert and key come from /certs.
set -eu
exec python -m src.server \
    --data-dir /data \
    --cert /certs/cert.pem --key /certs/key.pem \
    --host 0.0.0.0 --port "${HCPD_PORT}" \
    --max-conns "${HCPD_MAX_CONNS:-200}" \
    --max-conns-per-ip "${HCPD_MAX_CONNS_PER_IP:-10}" \
    --rate-per-s "${HCPD_RATE_PER_S:-20}" \
    --rate-burst "${HCPD_RATE_BURST:-40}" \
    --connects-per-min "${HCPD_CONNECTS_PER_MIN:-60}" \
    --backup-dir /backups \
    --backup-every "${HCPD_BACKUP_EVERY_S:-86400}" \
    --backup-keep "${HCPD_BACKUP_KEEP:-7}"
