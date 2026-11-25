"""Server settings and the checks that must hold before anything listens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PORT = 47830
PROTO_VERSION = 1

#: Repo root: the data dir must never live inside it.
_REPO_ROOT = Path(__file__).resolve().parents[2]

LOOPBACK = "127.0.0.1"

#: SHA-256 of the public test certificates in tests/data/pki (their keys are
#: published in the repo, so a real server must never use them).
TEST_CERT_SHA256 = frozenset({
    "dee22e5cc4039301679af22fb625c2f8355c8d53be0b39efaa2da65afc0763a0",
    "1618b3ac91d2637a7a8c1e56530a6cfc5f9a96f4c252fd8d5c1ea5900033d2ea",
})


def is_test_cert(path: Path) -> bool:
    """True when the PEM file holds one of the public test certificates."""
    import hashlib
    import ssl
    try:
        der = ssl.PEM_cert_to_DER_cert(Path(path).read_text(encoding="ascii"))
    except (OSError, ValueError):
        return False
    return hashlib.sha256(der).hexdigest() in TEST_CERT_SHA256


class ConfigError(ValueError):
    """The settings are unsafe or inconsistent; the server does not start."""


@dataclass(frozen=True)
class ServerConfig:
    data_dir: Path
    host: str = LOOPBACK
    port: int = DEFAULT_PORT
    cert: Path | None = None
    key: Path | None = None
    insecure_localhost: bool = False
    # Limits (docs/server.md).
    max_conns: int = 200
    max_conns_per_ip: int = 10
    max_message_bytes: int = 64 * 1024
    rate_per_s: float = 20.0       # sustained messages per second, per IP
    rate_burst: int = 40
    connects_per_min_per_ip: int = 60
    handshake_timeout_s: float = 5.0
    idle_timeout_s: float = 60.0
    line_timeout_s: float = 10.0   # a started line must finish within this
    write_timeout_s: float = 10.0
    workers: int = 4
    # Password hashing is slow: sign-up/in/recovery have their own, stricter
    # per-IP budget on top of the message rate.
    heavy_rate_per_s: float = 1.0
    heavy_burst: int = 5
    max_accounts: int = 100_000
    max_rooms: int = 50          # two players each
    photo_line_bytes: int = 6 * 1024 * 1024   # line cap while in a room (photo)

    def validate(self) -> None:
        if not (0 <= self.port < 65536):   # 0 = pick a free port (tests)
            raise ConfigError("port out of range")
        data = Path(self.data_dir).expanduser().resolve()
        if data == _REPO_ROOT or _REPO_ROOT in data.parents:
            raise ConfigError("the data dir must be outside the repo")
        if self.insecure_localhost:
            if self.host != LOOPBACK:
                raise ConfigError("--insecure-localhost needs the bind address "
                                  f"{LOOPBACK}")
        else:
            if not (self.cert and self.key):
                raise ConfigError("TLS is required: give --cert and --key "
                                  "(or --insecure-localhost on 127.0.0.1)")
            for label, path in (("cert", self.cert), ("key", self.key)):
                if not Path(path).is_file():
                    raise ConfigError(f"{label} file not found: {path}")
        for name in ("max_conns", "max_conns_per_ip", "max_message_bytes",
                     "rate_burst", "connects_per_min_per_ip", "workers"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1")
        for name in ("rate_per_s", "handshake_timeout_s", "idle_timeout_s",
                     "line_timeout_s", "write_timeout_s"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")
