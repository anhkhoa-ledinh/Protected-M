# Central server (design, 24/11)

One Python process we own. Standard library only (`asyncio`, `ssl`, `sqlite3`, `secrets`,
`hmac`). It runs on a LAN machine today and on a cloud VM later (Oracle Always Free, Azure for
Students). Code: `src/server/`. It may import `src.game` (rules), `src.account.crypto`,
`src.shared` and the framing in `src.net.protocol`. It never imports the UI, camera, tracking,
content or the rest of `src.net` (checked by `tests/test_architecture.py`).

Run: `python -m src.server --data-dir <dir outside the repo> --cert c.pem --key k.pem`.
Dev certificate: `bash tools/make_dev_cert.sh <dir>` (needs the system `openssl`).
Admin: `python -m src.server.admin --data-dir <dir> ...` on the server machine only.

## Wire format

TLS 1.2+ over TCP (default port 47830). One JSON object per line, UTF-8, `"t"` = message
type (the framing of `src/net/protocol.py`). A request may carry `"id"`; the reply echoes it.

- Success: `{"t":"ok","re":"<type>","id":..., ...data}`
- Failure: `{"t":"error","re":"<type>","id":...,"code":"<code>","msg":"<text>"}`
- Codes: `bad_request`, `forbidden`, `rate_limited`, `unknown_type`, `hello_required`,
  `bad_credentials`, `locked`, `taken`, `no_funds`, `bad_ticket`, `busy`, `internal`.
- A protocol violation (bad JSON, too large, a type used before `hello`) closes the
  connection. A normal failure (wrong password) keeps it open.

## Roles and the permission table

Roles: `anonymous` (no sign-in), `player` (valid session token), `admin`.
**Admin is never reachable from the network**: the dispatcher refuses any table row whose role
is `admin`; admin actions exist only in the local CLI, which opens the SQLite file directly.

One table, `src/server/permissions.py`, maps every message type to the role it needs.
Default deny: a type that is not in the table gets `unknown_type`. A test lists every
registered handler and fails if one has no row (and if a row has no handler).

| Type | Role | Step |
|---|---|---|
| `hello`, `ping` | anonymous | C1 |
| `sign_up`, `sign_in`, `recover`, `resume` | anonymous | C2 |
| `sign_out`, `me` | player | C2 |
| `start_match`, `finish_match`, `leaderboard`, `buy` | player | C3 |
| `create_room`, `join_room`, `queue`, `photo`, `bugs`, `ready`, `hit`, `killed`, `used_item`, `eye`, `relocate`, `end`, `rematch`, `cancel`, `rejoin` | player | C5 |

## Data model (SQLite, `server.db` in the data dir)

`PRAGMA user_version` is the schema version; migrations run in order at start.
- `meta(key, value)` (C1)
- `accounts(key, display, pw_hash, rec_hash, wallet, earned_total, levels_cleared, failures, locked_until, created_at)` (C2)
- `sessions(token_hash, account_key, expires_at, revoked)`: only the SHA-256 of a token is stored (C2)
- `tickets(nonce, account_key, level, items, started_at, used)` (C3)
- `scores(id, account_key, level, score, seconds, items_used, at)` (C3)

The data dir is given on the command line and must be outside the repo (the server refuses a
path inside it). Created with mode 0700.

## Threats (from `docs/security.md` section 7, items 2, 3, 4) and how each is handled

| Threat | Handling |
|---|---|
| 3. Server exposure: any host, unlimited sockets, no auth, no TLS (N1, N3, N7) | TLS only; plain TCP refused unless `--insecure-localhost` **and** bind address 127.0.0.1 (peer must be loopback). Max connections (global and per IP), per-IP message rate limit, handshake timeout, idle timeout, per-message deadline (slow-loris), write timeout (slow reader). Role table with default deny. Sessions are random 32-byte tokens, stored hashed, with expiry. |
| 3. Oversized or malformed input (N2) | Line cap 64 KiB for every connection (the API needs less); the cap is checked on the raw buffer before a newline arrives. Any `ProtocolError` closes the connection. Handlers validate every field. Nothing is logged from a message body. |
| 2. PvP photo relay unchecked (P7, N5) | C5: a `photo` needs a signed-in player in the room; the server raises the line cap for that one type only, checks base64, byte size and the pixel/size limits of the photo gate (header only, never a full decode), and drops the room on a violation. The receiving client still runs the photo gate. |
| 4. Remote data trusted (P5) | The server never fetches a URL and never writes a client-chosen file name; the shared album stays a client-side matter (already behind the photo gate). |
| Cheating client (N4) | The server never believes a score or a wallet value from a client (C3). |

## What the server cannot verify

It cannot know that a match was really won: `finish_match(won=true)` is the client's word. The
server controls only what it can: the match must have been started by that account (ticket),
the ticket works once and expires, the time part of the score comes from the **server clock**,
the items were charged at `start_match`, and the wallet moves in SQLite transactions. A modified
client can still claim a win; the damage is bounded by the minimum play time per level and by
what a win pays. Anti-cheat beyond that (replaying hits server-side) is out of scope.
