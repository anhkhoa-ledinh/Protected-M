#!/usr/bin/env python3
"""Self-signed TLS pair for the central server (development / LAN).

Usage: python tools/make_dev_cert.py <out_dir> [extra-ip-or-dns ...]

Calls the ``openssl`` executable (on PATH). Never writes inside the repo: the
key is secret. Refuses to overwrite an existing cert.pem / key.pem.
Exit codes: 0 ok, 2 refused or bad usage, 3 openssl missing or failed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shared.private_files import PrivateFilesError, make_private  # noqa: E402

USAGE = "usage: python tools/make_dev_cert.py <out_dir> [extra-ip-or-dns ...]"


def subject_alt_names(extra: list[str]) -> str:
    san = "DNS:localhost,IP:127.0.0.1"
    for name in extra:
        kind = "IP" if re.fullmatch(r"[0-9.]+", name) or ":" in name else "DNS"
        san += f",{kind}:{name}"
    return san


def main(argv: list[str]) -> int:
    if not argv:
        print(USAGE, file=sys.stderr)
        return 2
    out = Path(argv[0]).expanduser().resolve()
    if out == ROOT or ROOT in out.parents:
        print(f"refused: {out} is inside the repo", file=sys.stderr)
        return 2
    cert, key = out / "cert.pem", out / "key.pem"
    if cert.exists() or key.exists():
        print(f"refused: cert.pem or key.pem already exists in {out}", file=sys.stderr)
        return 2
    openssl = shutil.which("openssl")
    if not openssl:
        print("openssl is not installed or not on PATH; install it, then run this again",
              file=sys.stderr)
        return 3

    out.mkdir(parents=True, exist_ok=True)
    san = subject_alt_names(argv[1:])
    key.touch()                    # private before openssl writes the secret into it
    try:
        make_private(key)
        done = subprocess.run(
            [openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt",
             "ec_paramgen_curve:prime256v1", "-nodes", "-days", "825",
             "-subj", "/CN=hcpd-dev", "-addext", f"subjectAltName={san}",
             "-keyout", str(key), "-out", str(cert)],
            capture_output=True, text=True)
    except PrivateFilesError as exc:
        key.unlink(missing_ok=True)
        print(f"could not protect key.pem: {exc}", file=sys.stderr)
        return 3
    if done.returncode != 0:
        key.unlink(missing_ok=True)
        cert.unlink(missing_ok=True)
        print(f"openssl failed: {done.stderr.strip()}", file=sys.stderr)
        return 3
    print(f"cert: {cert}")
    print(f"key:  {key}")
    print(f"san:  {san}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
