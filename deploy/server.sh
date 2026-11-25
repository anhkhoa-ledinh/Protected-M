#!/usr/bin/env bash
# One script to run the central server in Docker.
#   bash deploy/server.sh help
# Works from any directory. Needs docker (and docker compose) usable by your user.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEPLOY="$ROOT/deploy"
ENV_FILE="$DEPLOY/.env"
IMAGE="hcpd-server"

die() { echo "error: $*" >&2; exit 1; }
say() { echo "$*"; }

# Values from deploy/.env, unless the environment already sets them.
load_env() {
    [ -f "$ENV_FILE" ] || return 0
    local line key val
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key="${line%%=*}"; val="${line#*=}"
        case "$key" in HCPD_*) [ -n "${!key+x}" ] || export "$key=$val" ;; esac
    done < "$ENV_FILE"
}

need_docker() {
    command -v docker >/dev/null 2>&1 || die "docker is not installed"
    docker info >/dev/null 2>&1 || die "docker does not answer (is it running? can your user use it?)"
    docker compose version >/dev/null 2>&1 || die "docker compose is not available"
}

setup_names() {
    load_env
    CONTAINER="${HCPD_CONTAINER:-hcpd-server}"
    V_DATA="${HCPD_DATA_VOLUME:-hcpd-data}"
    V_CERTS="${HCPD_CERTS_VOLUME:-hcpd-certs}"
    V_BACKUPS="${HCPD_BACKUPS_VOLUME:-hcpd-backups}"
    PORT="${HCPD_PORT:-47830}"
    BIND="${HCPD_BIND:-127.0.0.1}"
    KEEP="${HCPD_BACKUP_KEEP:-7}"
    CERT_DIR="${HCPD_CERT_DIR:-$HOME/hcpd-certs}"
}

dc() {
    local envopt=()
    [ -f "$ENV_FILE" ] && envopt=(--env-file "$ENV_FILE")
    docker compose "${envopt[@]}" -f "$DEPLOY/docker-compose.yml" "$@"
}

# Run a command from the image in a throw-away container (non-root, no caps).
# Usage: oneoff <docker run options...> -- <command...>
oneoff() {
    local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done
    shift
    docker run --rm --cap-drop ALL --security-opt no-new-privileges:true \
        "${opts[@]}" --entrypoint "$1" "$IMAGE" "${@:2}"
}

is_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" = true ]
}

image_exists() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

# Volumes carry the labels compose expects, so compose does not warn about them.
make_volumes() {
    local proj="${COMPOSE_PROJECT_NAME:-hcpd}"
    docker volume create --label "com.docker.compose.project=$proj" --label com.docker.compose.volume=data "$V_DATA" >/dev/null
    docker volume create --label "com.docker.compose.project=$proj" --label com.docker.compose.volume=certs "$V_CERTS" >/dev/null
    docker volume create --label "com.docker.compose.project=$proj" --label com.docker.compose.volume=backups "$V_BACKUPS" >/dev/null
}

# Run a command, show its output only if it fails.
quiet() {
    local out
    out="$("$@" 2>&1)" || { echo "$out" >&2; return 1; }
}

wait_healthy() {
    local i st
    for i in $(seq 1 60); do
        st="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CONTAINER" 2>/dev/null || echo missing)"
        [ "$st" = healthy ] && return 0
        sleep 1
    done
    docker logs --tail 15 "$CONTAINER" >&2 || true
    die "the server did not become healthy within 60 s (health: $st)"
}

backup_names() {   # newest last
    oneoff -v "$V_BACKUPS:/backups:ro" -- sh -c 'ls -1 /backups 2>/dev/null | grep "^server-.*\.db$" | sort' || true
}

backup_now() {     # prints the new file name
    local out
    if is_running; then
        out="$(docker exec "$CONTAINER" python -m src.server.backup run \
            --data-dir /data --backup-dir /backups --keep "$KEEP" 2>&1)" || { echo "$out" >&2; die "backup failed"; }
    else
        out="$(oneoff -v "$V_DATA:/data" -v "$V_BACKUPS:/backups" -- python -m src.server.backup run \
            --data-dir /data --backup-dir /backups --keep "$KEEP" 2>&1)" || { echo "$out" >&2; die "backup failed"; }
    fi
    echo "$out" | sed -n 's/.*backup ok: \([^ ]*\) .*/\1/p' | tail -1
}

cmd_setup() {
    need_docker; setup_names
    if [ ! -f "$ENV_FILE" ]; then
        cp "$DEPLOY/.env.example" "$ENV_FILE"
        say "created deploy/.env from the example (edit it to change the address or limits)"
    fi
    setup_names
    if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
        say "keeping the existing certificate in $CERT_DIR"
    else
        bash "$ROOT/tools/make_dev_cert.sh" "$CERT_DIR" "$@" || die "could not make the certificate"
        say "made cert.pem and key.pem in $CERT_DIR (key.pem is secret: never send it)"
    fi
    say "building the image ..."
    quiet dc build || die "image build failed"
    make_volumes
    HCPD_IMAGE="$IMAGE" bash "$DEPLOY/import_certs.sh" "$CERT_DIR" "$V_CERTS" >/dev/null || die "could not load the certificate"
    say "certificate loaded into volume $V_CERTS"
    say "setup done. Next: bash deploy/server.sh start"
}

cmd_start() {
    need_docker; setup_names
    [ -f "$ENV_FILE" ] || die "no deploy/.env yet: run 'bash deploy/server.sh setup' first"
    make_volumes
    quiet dc up -d || die "could not start"
    wait_healthy
    say "server is healthy"
    say "address:  $BIND:$PORT"
    say "cert.pem: $CERT_DIR/cert.pem (give this file, not key.pem, to players)"
}

cmd_stop() {
    need_docker; setup_names
    quiet dc stop || die "could not stop"
    say "server stopped (data kept)"
}

cmd_status() {
    need_docker; setup_names
    if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
        say "container:  not created (run: bash deploy/server.sh start)"; return 0
    fi
    local running health started
    running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER")"
    health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CONTAINER")"
    started="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")"
    say "running:    $running"
    say "healthy:    $health"
    say "address:    $BIND:$PORT"
    if [ "$running" = true ]; then
        local s e
        s="$(date -d "$started" +%s 2>/dev/null || echo 0)"; e="$(date +%s)"
        say "uptime:     $(( (e - s) / 3600 )) h $(( (e - s) % 3600 / 60 )) min"
    fi
    local names count last
    names="$(backup_names)"
    count="$(printf '%s' "$names" | grep -c . || true)"
    last="$(printf '%s\n' "$names" | tail -1)"
    say "backups:    $count (last: ${last:-none})"
    if [ "$running" = true ]; then
        say "accounts:   $(docker exec "$CONTAINER" python -m src.server.admin --data-dir /data list-accounts 2>/dev/null | tail -1 | sed 's/ accounts\?$//')"
    else
        say "accounts:   (server is stopped)"
    fi
}

cmd_logs() {
    need_docker; setup_names
    docker logs --tail "${1:-50}" "$CONTAINER" 2>&1
}

cmd_backup() {
    need_docker; setup_names
    local f; f="$(backup_now)"
    [ -n "$f" ] || die "backup finished but no file name was reported"
    say "backup: $f (in volume $V_BACKUPS)"
}

cmd_backups() {
    need_docker; setup_names
    oneoff -v "$V_BACKUPS:/backups:ro" -- ls -l /backups
}

# Replace the data volume content with a backup that is already in the backups volume.
do_restore() {   # <name> <force 0|1>
    local name="$1" force="$2" has
    oneoff -v "$V_BACKUPS:/backups:ro" -- test -f "/backups/$name" || die "no backup named $name (see: bash deploy/server.sh backups)"
    make_volumes
    has=0
    oneoff -v "$V_DATA:/data:ro" -- test -f /data/server.db && has=1
    if [ "$has" = 1 ] && [ "$force" != 1 ]; then
        die "the data volume already has a database; restore would replace it. Add --force (a safety backup is made first)"
    fi
    if [ "$has" = 1 ]; then
        if is_running; then quiet dc stop; fi
        local safe
        safe="$(oneoff -v "$V_DATA:/data" -v "$V_BACKUPS:/backups" -- python -m src.server.backup run \
            --data-dir /data --backup-dir /backups --keep 1000 2>&1 | sed -n 's/.*backup ok: \([^ ]*\) .*/\1/p' | tail -1)"
        [ -n "$safe" ] || die "could not make the safety backup; nothing was changed"
        say "safety backup of the current data: $safe"
        oneoff -v "$V_DATA:/data" -- sh -c 'rm -f /data/server.db /data/server.db-wal /data/server.db-shm'
    else
        if is_running; then quiet dc stop; fi
    fi
    oneoff -v "$V_DATA:/data" -v "$V_BACKUPS:/backups:ro" -- python -m src.server.backup restore \
        --data-dir /data "/backups/$name" || die "restore failed"
    say "restored $name"
}

cmd_restore() {
    need_docker; setup_names
    local name="" force=0 a
    for a in "$@"; do
        if [ "$a" = --force ]; then force=1; else name="$a"; fi
    done
    [ -n "$name" ] || die "usage: bash deploy/server.sh restore <backup-file-name> [--force]"
    name="$(basename "$name")"
    do_restore "$name" "$force"
    cmd_start
}

cmd_update() {
    need_docker; setup_names
    [ -f "$ENV_FILE" ] || die "no deploy/.env yet: run 'bash deploy/server.sh setup' first"
    say "rebuilding the image from the current code ..."
    quiet dc build || die "image build failed"
    quiet dc up -d --force-recreate || die "could not recreate the container"
    wait_healthy
    say "updated; the container was recreated, data kept"
}

cmd_export() {
    need_docker; setup_names
    local dir="${1:-}"
    [ -n "$dir" ] || die "usage: bash deploy/server.sh export <dir>"
    image_exists || die "image $IMAGE not found: run setup first"
    mkdir -p "$dir"
    [ -z "$(ls -A "$dir")" ] || die "$dir is not empty"
    chmod 700 "$dir"
    say "saving the image ..."
    docker save "$IMAGE" -o "$dir/image.tar"
    local f; f="$(backup_now)"
    [ -n "$f" ] || die "could not make a fresh backup"
    oneoff -v "$V_BACKUPS:/backups:ro" -- cat "/backups/$f" > "$dir/$f"
    ( umask 077
      oneoff -v "$V_CERTS:/certs:ro" -- cat /certs/cert.pem > "$dir/cert.pem"
      oneoff -v "$V_CERTS:/certs:ro" -- cat /certs/key.pem > "$dir/key.pem" )
    [ -f "$ENV_FILE" ] && cp "$ENV_FILE" "$dir/env"
    say "exported to $dir: image.tar, $f, cert.pem, key.pem, env"
    say "WARNING: $dir/key.pem is the server's secret key. Move the folder privately and delete it afterwards."
}

cmd_import() {
    need_docker; setup_names
    local dir="${1:-}"
    [ -n "$dir" ] || die "usage: bash deploy/server.sh import <dir>"
    local f
    for f in image.tar cert.pem key.pem; do [ -f "$dir/$f" ] || die "missing $dir/$f"; done
    local backup_file; backup_file="$(ls -1 "$dir" | grep '^server-.*\.db$' | sort | tail -1 || true)"
    [ -n "$backup_file" ] || die "no server-*.db backup in $dir"
    if [ -f "$ENV_FILE" ] && ! cmp -s "$ENV_FILE" "$dir/env"; then
        mv "$ENV_FILE" "$ENV_FILE.before-import"
        say "kept your old deploy/.env as deploy/.env.before-import"
    fi
    [ -f "$dir/env" ] && cp "$dir/env" "$ENV_FILE"
    setup_names
    say "loading the image ..."
    docker load -i "$dir/image.tar" >/dev/null
    make_volumes
    HCPD_IMAGE="$IMAGE" bash "$DEPLOY/import_certs.sh" "$dir" "$V_CERTS" >/dev/null
    oneoff -i --user 10001 -v "$V_BACKUPS:/backups" -- sh -c "cat > /backups/$backup_file" < "$dir/$backup_file"
    if oneoff -v "$V_DATA:/data:ro" -- test -f /data/server.db; then
        die "this machine already has server data; import only into an empty setup (or use restore --force)"
    fi
    do_restore "$backup_file" 0
    cmd_start
}

cmd_help() {
    cat <<'EOF'
bash deploy/server.sh <command>

  setup [host-or-ip ...]  first time: .env, certificate (in ~/hcpd-certs), image, load the cert
  start                   start, wait until healthy, print the address and cert.pem
  stop                    stop (data kept)
  status                  running, healthy, address, uptime, last backup, accounts
  logs [n]                last n log lines (default 50)
  backup                  back up now and print the file
  backups                 list backups
  restore <file> [--force]  put a backup back (refuses to replace live data without --force)
  update                  rebuild the image from the current code and recreate the container
  export <dir>            image + fresh backup + cert/key + .env, for another machine
  import <dir>            on the new machine: load image, cert, .env, restore, start
  help                    this text
EOF
}

main() {
    local c="${1:-help}"
    [ $# -eq 0 ] || shift
    case "$c" in
        setup) cmd_setup "$@" ;;
        start) cmd_start ;;
        stop) cmd_stop ;;
        status) cmd_status ;;
        logs) cmd_logs "$@" ;;
        backup) cmd_backup ;;
        backups) cmd_backups ;;
        restore) cmd_restore "$@" ;;
        update) cmd_update ;;
        export) cmd_export "$@" ;;
        import) cmd_import "$@" ;;
        help|-h|--help) cmd_help ;;
        *) echo "unknown command: $c" >&2; cmd_help >&2; exit 2 ;;
    esac
}

main "$@"
