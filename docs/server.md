# Central server (design, 24/11)

One Python process we own. Standard library only (`asyncio`, `ssl`, `sqlite3`, `secrets`,
`hmac`). It runs on a LAN machine today and on a cloud VM later (Oracle Always Free, Azure for
Students). Code: `src/server/`. It may import `src.game` (rules), `src.account.crypto`,
`src.shared` and the framing in `src.net.protocol`. It never imports the UI, camera, tracking,
content or the rest of `src.net` (checked by `tests/test_architecture.py`).

Run: `python -m src.server --data-dir <dir outside the repo> --cert c.pem --key k.pem`.
Dev certificate: `python tools/make_dev_cert.py <dir>` (needs `openssl` on PATH).
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
| `logout`, `me` | player | C2 |
| `start_match`, `begin_play`, `finish_match`, `leaderboard` | player | C3 |
| `create_room`, `join_room`, `leave_room`, `photo` | player | C5 |

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
| Guessing passwords or tokens (C2) | 5 wrong tries lock one name for 30 s (stored in SQLite, survives a restart; same numbers as the local game). Unknown name and wrong password give the same reply and cost the same time (dummy hash). Sign-up, sign-in and recovery also share a stricter per-IP budget (1/s, burst 5) because scrypt is slow, and the account count is capped. Tokens are 32 random bytes; only their SHA-256 is stored; a token lasts 30 days, `logout`, recovery and the admin CLI revoke it, and a revoked token stops working on live connections at the next request. |
| Secrets in logs (C2) | Handlers never log a message body, a password, a token or a recovery code; `tests/test_server_accounts.py` runs the real server, signs up/in/recovers and greps its log. |
| Cheating client (N4) | The server never believes a score or a wallet value from a client (C3). |

## The economy (C3)

- `start_match {level, items, mode?}`: `mode` is `solo` (default), `Competitive` or `Hunt & Hide`.
  A PvP mode needs levels 1-5 cleared. Items must be distinct, known, and fit the level's slots.
  The price (`weight x 10`, the game's own table) is taken from the wallet in one transaction
  (`UPDATE ... WHERE wallet >= cost` and `CHECK (wallet >= 0)`); not enough points means no
  charge and no ticket. The reply carries `ticket` (random, only its hash is stored), the
  server's `started_at` and `expires_at`. Starting a new match voids the account's older unused
  ticket (its items stay paid).
- `begin_play {ticket}`: optional, once. Moves the ticket's start to the server's "now", so the
  seconds spent loading a photo do not count against the player. Same account only.
- `finish_match {ticket, won, items_used}`: the ticket is spent first (so it works once, even
  when two requests race), must belong to the caller and be unexpired (level time + level-5
  overtime + 120 s). The time part comes from `server now - started_at`; the formula is
  `src.game.scoring.match_score`, the same function the game uses. A solo win credits wallet and
  `earned_total` by the score and adds the level to `levels_cleared`. A PvP ticket pays nothing.
  `items_used` may only name items the ticket paid for, each 0-50.
- Two refusals that turn a claimed win into a loss: faster than 1 s per bug (`too_fast`), and
  later than the level time + 10 s (`time`; level 5 also gets its 30 s overtime).
- `leaderboard`: top 20 by `earned_total`, ties share a rank, then name; plus the caller's rank
  (`null` while they have won nothing). PvP boards stay "No matches yet".

## PvP rooms (C5)

Every room message is a `player` row: no token, no room. `create_room {mode}` / `join_room
{room}` need levels 1-5 cleared (the same lock as the game); a room has two seats, an account is
in one room at a time, the number of rooms is capped (`max_rooms`, default 50, so at most 100
players), and a room ends when its host leaves or either player's connection drops (the other
gets `{"t":"peer_left"}`). The server pushes to a peer through a bounded queue; a peer that never
reads is disconnected.

`photo {jpeg_b64, sha1}` is relayed only to the other player in the room, after: base64 and sha1
check, at most 4 MiB, JPEG/PNG/WebP magic, and the width/height read from the header (no
decoding on the server): longest side 16384, at most 64,000,000 pixels (equal to
`media_gate.MAX_PIXELS`, kept equal by a test). Any refusal ends the room. The line cap is 64 KiB
for everyone; a player inside a room may send one photo-sized line (6 MiB), and a big line that
is not a `photo` closes the connection. The receiving game still runs its media gate.

**Not ported yet:** the in-match messages of the LAN protocol (`ready`, `start`, `bugs`, `hit`,
`killed`, `used_item`, `eye`, `relocate`, `tick`, `end`, `rematch`, matchmaking `queue`). The game's
PvP client still plays over the LAN server in `src/net`; rooms on the central server today give
the signed-in access control, the room and player caps and the photo relay, not a full match.

## What the server cannot verify

It cannot know that a match was really won: `finish_match(won=true)` is the client's word. The
server controls only what it can: the match must have been started by that account (ticket),
the ticket works once and expires, the time part of the score comes from the **server clock**,
the items were charged at `start_match`, and the wallet moves in SQLite transactions. A modified
client can still claim a win; the damage is bounded by the minimum play time per level and by
what a win pays. Anti-cheat beyond that (replaying hits server-side) is out of scope.

## Items (D1)

Only `Flare` and `Thermal scan` work (`src/game/pvp.py: ITEM_EFFECT_S`, seconds of effect);
the other six are "Coming soon": the loadout shows them dimmed, cannot equip them, and
`start_match` refuses them (`bad_request`), so they are never charged. A used item counts once
per match in `used`, which the score formula and `finish_match(items_used)` both read.
The server only checks that a used item was bought for that ticket and the count is sane; it
cannot know the item was really *used* in the game.

## Open risks (24/11, end of TASK_platform_1124)

1. **PvP in-match play is not on the central server.** C5 put sign-in, rooms, the photo relay and
   its limits on the server, but the match itself (hit, killed, tick, end, queue, item relay)
   still runs on the old LAN server (`src/net/server.py`) that the game's PvP client uses. So
   PvP scores, the PvP tabs of the leaderboard and PvP wallet effects are not server-authoritative.
2. **A win is the client's word** (see above). The server rejects a win faster than
   1 s per bug (`too_fast`) or later than the level time (`time`); a client that lies within
   those bounds is accepted.
3. **The result screen trusts the client first.** When the server refuses a win (`too_fast`),
   the screen still says "Memory saved" and lists the time bonus, while SCORE shows the server's
   number (0). The server is right; the screen is misleading. Not changed in this task.
4. **Progress and play history stay on the player's computer** (`remote_progress.json`); only
   wallet, earned points, levels cleared and scores are on the server.
5. **Certificate pinning is a file the user copies by hand.** No rotation or revocation; a new
   certificate means every player must get the new file. Self-signed dev certs are for LAN use.
6. **Leaderboard**: only the Solo tab is served; the screen subtitle still reads "Accounts on this
   computer" in remote mode.
7. **No log of requests on the server** (by design: nothing from a message body is logged), so a
   run is matched to the server only by the database (`scores`, `tickets`) and connection lines.
8. **Offline resume**: with a remembered session and the server down, the game opens on the
   sign-in screen (name filled in, no message); the explicit "Can't reach the server" appears
   only after pressing Sign in.
9. **Account copies**: importing `accounts.json` is an admin CLI command; hashes move as hashes,
   the scrypt parameters must match.
10. **One process, one SQLite file, no backup job.** Fine for a LAN; a cloud VM needs a backup
    and a firewall rule (only the server port).
