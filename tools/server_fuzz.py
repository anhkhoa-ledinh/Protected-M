"""Protocol fuzz for the central server (TASK_platform_1124 Phase C gate).

Starts ``python -m src.server`` with TLS in a temp dir, throws random,
oversized and malformed messages plus slow-loris connections at it, then
checks: the process is alive, it still answers a valid ``hello``, every bad
connection got closed, and its peak RSS (VmHWM) stayed bounded.

    python tools/server_fuzz.py [--messages 2000] [--out report.json]

With ``--container NAME --target-port P --cert CERT`` it attacks a server
already running in Docker (the image of deploy/) instead of starting one;
peak memory is read from the container's ``/proc/1/status`` and tracebacks
from ``docker logs``. Start that container with the per-IP and rate limits
raised (HCPD_MAX_CONNS_PER_IP etc.), as the local mode does."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RSS_LIMIT_MB = 150            # bounded: far below what an unbounded buffer needs


def vm_hwm_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def container_hwm_mb(name: str) -> float:
    out = subprocess.run(["docker", "exec", name, "grep", "VmHWM", "/proc/1/status"],
                         capture_output=True, text=True).stdout.split()
    return int(out[1]) / 1024.0 if len(out) >= 2 else -1.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tls_connect(port: int, cert: Path, timeout: float = 8.0):
    ctx = ssl.create_default_context(cafile=str(cert))
    raw = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    return ctx.wrap_socket(raw, server_hostname="localhost")


def drain_until_closed(sock, wait: float) -> bool:
    """True if the server closed (or reset) within ``wait`` seconds."""
    deadline = time.monotonic() + wait
    sock.settimeout(0.5)
    while time.monotonic() < deadline:
        try:
            if sock.recv(65536) == b"":
                return True
        except socket.timeout:
            continue
        except (OSError, ssl.SSLError):
            return True
    return False


def make_payload(rng: random.Random, kind: int) -> tuple[bytes, bool]:
    """(bytes to send, must the server close the connection afterwards)."""
    types = ["hello", "ping", "sign_in", "start_match", "admin", "", "x" * 300]
    if kind == 0:
        return bytes(rng.randrange(256) for _ in range(rng.randrange(1, 400))) + b"\n", True
    if kind == 1:
        return b"\xff\xfe\xfa" + b"\n", True
    if kind == 2:
        return (b"[" * 60000) + b"\n", True              # deep nesting
    if kind == 3:
        return b'{"t":"ping","pad":"' + b"a" * rng.randrange(70_000, 300_000), True
    if kind == 4:
        return json.dumps({"t": rng.choice(types), "id": rng.choice(
            [None, 1, "a", [1], {"x": 1}, True, "z" * 500])}).encode() + b"\n", False
    if kind == 5:
        return json.dumps({"t": "hello", "proto": rng.choice(
            [1, "1", None, True, 1.5, [1], 10**30]),
            "name": rng.choice(["ok", "n" * 99, 5, None])}).encode() + b"\n", False
    if kind == 6:
        return json.dumps([rng.random() for _ in range(5)]).encode() + b"\n", True
    return json.dumps({"t": rng.choice(["hello", "ping"]),
                       **{f"k{i}": "v" * rng.randrange(0, 2000)
                          for i in range(rng.randrange(0, 20))}}).encode() + b"\n", False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", type=int, default=2000)
    ap.add_argument("--loris", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1124)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--container", help="attack this running container instead")
    ap.add_argument("--target-port", type=int)
    ap.add_argument("--cert", type=Path, help="pinned cert of the --container server")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="hcpd_fuzz_"))
    if args.container:
        if not (args.target_port and args.cert):
            ap.error("--container needs --target-port and --cert")
        cert, port, proc = args.cert, args.target_port, None
        rss_start = container_hwm_mb(args.container)
    else:
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_dev_cert.py"),
                        str(work / "pki")], check=True, capture_output=True)
        cert, key = work / "pki" / "cert.pem", work / "pki" / "key.pem"
        port = free_port()
        log = open(work / "server.log", "wb")
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.server", "--data-dir", str(work / "data"),
             "--cert", str(cert), "--key", str(key), "--port", str(port),
             "--max-conns-per-ip", "500", "--rate-per-s", "100000",
             "--rate-burst", "100000", "--connects-per-min", "1000000"],
            cwd=ROOT, stdout=log, stderr=log)
        for _ in range(100):
            try:
                tls_connect(port, cert, 1).close()
                break
            except OSError:
                time.sleep(0.1)
        rss_start = vm_hwm_mb(proc.pid)

    rng = random.Random(args.seed)
    stats = {"sent": 0, "must_close": 0, "closed_ok": 0, "not_closed": 0,
             "tls_errors": 0}
    lock = threading.Lock()
    jobs = [(i, rng.randrange(8), random.Random(rng.random()))
            for i in range(args.messages)]

    # slow-loris first, so they overlap with everything else
    loris = []
    for _ in range(args.loris):
        s = tls_connect(port, cert)
        s.sendall(b'{"t":"pi')
        loris.append(s)

    def worker(chunk):
        for _, kind, r in chunk:
            payload, must_close = make_payload(r, kind)
            try:
                s = tls_connect(port, cert)
            except (OSError, ssl.SSLError):
                with lock:
                    stats["tls_errors"] += 1
                continue
            try:
                s.sendall(payload)
            except OSError:
                pass
            closed = None
            if must_close:
                closed = drain_until_closed(s, 8.0)
            with lock:
                stats["sent"] += 1
                if must_close:
                    stats["must_close"] += 1
                    stats["closed_ok" if closed else "not_closed"] += 1
            try:
                s.close()
            except OSError:
                pass

    threads = [threading.Thread(target=worker, args=(jobs[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # half-open TLS and plain-TCP junk
    for _ in range(50):
        raw = socket.create_connection(("127.0.0.1", port), timeout=3)
        raw.sendall(os.urandom(100))
        raw.close()

    time.sleep(0.2)
    loris_closed = sum(drain_until_closed(s, 14.0) for s in loris)

    if args.container:
        alive = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}",
                                args.container], capture_output=True,
                               text=True).stdout.strip() == "true"
    else:
        alive = proc.poll() is None
    answers = False
    if alive:
        s = tls_connect(port, cert)
        s.sendall(b'{"t":"hello","proto":1}\n')
        s.settimeout(5)
        answers = b'"ok"' in s.recv(4096)
        s.close()
    if args.container:
        rss_end = container_hwm_mb(args.container) if alive else -1.0
        logs = subprocess.run(["docker", "logs", args.container],
                              capture_output=True, text=True)
        text = logs.stdout + logs.stderr
        (work / "server.log").write_text(text)
    else:
        rss_end = vm_hwm_mb(proc.pid) if alive else -1.0
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        text = (work / "server.log").read_text(errors="replace")
    tracebacks = text.count("Traceback")

    report = {
        "messages": args.messages, **stats,
        "loris": args.loris, "loris_closed": loris_closed,
        "server_alive": alive, "answers_hello_after": answers,
        "rss_peak_start_mb": round(rss_start, 1), "rss_peak_end_mb": round(rss_end, 1),
        "rss_limit_mb": RSS_LIMIT_MB, "tracebacks_in_server_log": tracebacks,
        "server_log": str(work / "server.log"),
    }
    ok = (alive and answers and stats["not_closed"] == 0
          and loris_closed == args.loris and 0 < rss_end <= RSS_LIMIT_MB
          and tracebacks == 0)
    report["PASS"] = ok
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
