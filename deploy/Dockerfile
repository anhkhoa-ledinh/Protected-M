# Central server image. Standard library only, so no pip step.
# Build from the repo root:  docker build -f deploy/Dockerfile -t hcpd-server .
FROM python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HCPD_PORT=47830

RUN groupadd --system --gid 10001 hcpd \
 && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin hcpd \
 && mkdir /data /certs /backups \
 && chown 10001:10001 /data /certs /backups

WORKDIR /app
COPY src/ /app/src/
COPY deploy/entrypoint.sh /app/entrypoint.sh

USER 10001:10001
VOLUME ["/data", "/certs", "/backups"]
EXPOSE 47830

HEALTHCHECK --interval=15s --timeout=6s --start-period=5s --retries=3 \
    CMD ["sh", "-c", "python -m src.server.healthcheck --port ${HCPD_PORT} --cert /certs/cert.pem"]

ENTRYPOINT ["sh", "/app/entrypoint.sh"]
