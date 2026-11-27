# Protected M

![tests](https://github.com/anhkhoa-ledinh/Protected-M/actions/workflows/tests.yml/badge.svg)

**A glasses-free 3D bug hunt for an ordinary laptop screen.**

> ⚠️ **Demo build.** The game is playable end to end, but it still has bugs
> and some features are simulated. Read [Known issues](#known-issues) before
> you report a problem.

It is the year 3902. A meteor shower has carried bugs to Earth, and they hide
inside photos. You load any photo, the game splits it into depth layers, and
the bugs hide behind those layers. A webcam tracks your head, so the screen
behaves like a window: lean left or right and you can look *around* the near
layers to find what is behind them. Then shoot.

There is no VR headset, no glasses and no stereo. The 3D effect is motion
parallax only, driven by where your eyes are.

Protected M is the demo application of **HCPD (Head-Coupled Perspective
Display)**. HCPD is the engine underneath, and you can still run it on its own
(see [HCPD sandbox mode](#hcpd-sandbox-mode)).

---

## How it works

1. **Head tracking.** MediaPipe Face Landmarker finds your face in the webcam
   image. `solvePnP` and a filter turn that into a 3D eye position in
   millimetres.
2. **Off-axis projection.** The virtual camera sits at your eye position. The
   view frustum is built from the real screen rectangle with Kooima's
   generalized perspective projection, so the monitor acts like a window.
3. **Depth layers from one photo.**
   - Depth Anything (V1 Small) estimates a depth map.
   - K-means splits that map into 2–4 layers, with a vote over image regions
     so a layer boundary does not cut through one object.
   - The holes revealed behind each layer are filled with LaMa (ONNX). If the
     LaMa model is missing, Telea inpainting is used instead.
4. **Baking runs in a child process.** The result is cached in
   `assets/cache/maps/<sha1>/`. Opening the same photo with the same settings
   a second time is instant.
5. **Bugs hide in the scene, not on top of it.** Bugs are drawn inside the 3D
   scene with depth testing, so a nearer layer really covers them. A shot
   counts only if it hits the bug's actual sprite silhouette and no nearer
   layer blocks the line from your eye.

Everything runs on the CPU. A dedicated GPU is not required.

---

## Requirements

- Windows 10/11 or Linux (developed on CachyOS + Hyprland). Launch scripts:
  `run_game.bat` / `run_server.bat` on Windows, `bash run_game.sh` /
  `bash run_server.sh` on Linux. Tested with Python 3.13.
- A webcam, with your face lit from the front.
- A few GB of free disk space for PyTorch and the models downloaded on first
  run.
- Optional: Docker, only on the machine that hosts the
  [central server](#servers).

---

## Linux

The game also runs on Linux (developed on CachyOS + Hyprland).

1. Create a Python 3.13 venv at `~/.venvs/protected-m` and install
   `requirements-lock.txt` into it. Use another interpreter by setting
   `PM_PYTHON=/path/to/python`.
2. For the "open photo" dialog install the system package for Tk
   (`tk` on Arch, `python3-tk` on Debian/Ubuntu).
3. Run `bash run_game.sh` (log in `logs/`) and `bash run_server.sh` for the
   PvP server. The scripts need no exec bit, so they work on any drive.
4. **Hyprland (tiling): float the game window.** The number of bugs that
   fit on screen depends on the window size, so a tiled window changes the
   game. `hyprctl clients -j` shows the window as class and title
   `HCPD — Virtual Room` (it runs under XWayland). Rules tried on
   Hyprland 0.57 (Lua config), loaded with `hyprctl eval`. `float`, `size`
   (tested with 1091x613) and `move` (tested with 107,100) all took effect
   on the game window; 1280x720 below is the same syntax, not re-measured:

   ```lua
   hl.window_rule({ name = "pm-float", match = { class = "^HCPD.*" }, float = true })
   hl.window_rule({ name = "pm-size",  match = { class = "^HCPD.*" }, size = {1280, 720} })
   ```

   Add `move = {107, 100}` to place it. The window opens at the size the
   game asks for and the rule resizes it right after, so the game sees a
   resize event, as when you drag an edge.

---

## Install

```bat
git clone https://github.com/anhkhoa-ledinh/Protected-M.git
cd Protected-M

py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins exact versions (CI installs from it too, so the
test suite runs against the same versions locally and in Actions). To pick
up newer dependency versions instead, use `requirements.txt` (unpinned
`>=`) and regenerate the lock with `pip freeze > requirements-lock.txt`
once the suite is green again.

Then check these assets:

| Asset | In git? | What to do |
|---|---|---|
| `assets/models/face_landmarker.task` | yes | Nothing. Without it, the head is simulated instead of tracked. |
| Depth Anything model (`LiheYoung/depth-anything-small-hf`) | no | Downloaded automatically by `transformers` on the first photo. You need internet the first time. |
| `assets/models/lama_fp32.onnx` (208 MB) | no | Optional. Download `lama_fp32.onnx` from [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) and put it in `assets/models/`. Without it, holes are filled with Telea, which looks blurrier. |
| `assets/ui_v3/` (game UI) | yes | Nothing. |
| `assets/ui_album/` (bug sprites + unused v2 UI leftovers) | yes | Nothing. The v3 UI only reads `sprites/` and `manifest.json` from here. |

### Screen size, for correct parallax

Measure the **visible display area** of your screen with a ruler. Write it in
metres in `assets/config/screen.json`:

```json
{
  "screen_width": 0.344,
  "screen_height": 0.194,
  "camera_offset": [0.0, 0.103, 0.005]
}
```

- `camera_offset` is the webcam position relative to the centre of the
  screen.
- The axes are X right, Y up, Z toward you.

### Optional: face-scale calibration

```bat
python main.py --calibrate
```

This fixes absolute distance, which a single camera cannot measure on its own.

1. Measure the distance from your eyes to the screen with a tape and enter it.
2. Hold still and press `C` to capture.
3. Press `N`, then `S` to save.

The result goes to `assets/config/user.json`. That file is personal and is
never committed.

---

## Play

```bat
run_game.bat
```

On Linux: `bash run_game.sh`. Both run `python main.py --album` and write a
log to `logs/game_<date>_<time>.log`. If the Python environment does not
exist yet, the script tells you how to create it.

A round goes like this:

1. **Sign in.** Accounts are stored on this computer
   (`assets/config/accounts.json`) by default, or on a
   [central server](#servers) if one is configured. Passwords
   are scrypt-hashed; **Remember me** keeps you signed in next launch, and a
   recovery code gets you back in if you forget your password. Playing as
   guest skips sign-in but saves nothing.
2. **Choose a level.**
3. **Choose a photo.** You can:
   - pick one from your computer;
   - press `R` to get a random photo from the shared album;
   - press `U` to share yours to the shared album.

   Then click **Split layers & continue** and wait for the layers to bake.
4. **Loadout.** Pick one of three guns and put items in your slots. Items
   cost points (see [Points](#points)).
5. **Head-tilt check.** Tilt your head left and right. **Enter match** unlocks
   anyway after 15 s, so a weak webcam cannot block you.
6. **Hunt.** Move your head to look behind the layers, aim with the mouse and
   left-click to shoot. Win by clearing every bug before time runs out.

### Levels

Higher levels are harder in ways the photo cannot cap: bugs hide where you
must lean further, blend in more, are smaller, take more hits, and on levels
4–5 move to a new hiding spot once you have seen them for a while.

| Level | Bugs asked | Time | Item slots | Lean to reveal | Camouflage | Bug size | Moves after being seen | Bug HP |
|---|---|---|---|---|---|---|---|---|
| 1 · Easy | 6 | 180 s | 2 | 40–70 mm | 0.40 | ×1.0 | — | 100 |
| 2 · Normal | 8 | 150 s | 3 | 50–85 mm | 0.55 | ×1.0 | — | 200 |
| 3 · Hard | 10 | 150 s | 3 | 60–100 mm | 0.70 | ×1.0 | — | 300 |
| 4 · Infestation | 12 | 120 s | 4 | 75–110 mm | 0.85 | ×0.875 | 4 s | 400 |
| 5 · Asian Mode | 12 | 120 s | 4 | 90–120 mm | 1.00 | ×0.75 | 2.5 s | 500 |
On level 5 a hit that does not kill has a 50% chance to send the bug to a new hiding spot; it keeps its HP and cannot flee again for 1 s.

On levels 4–5 the game keeps one hiding spot free, so a photo with few spots gets one bug fewer; a dead bug's spot is freed for the others.

A photo can only hide as many bugs as it has hiding spots. When it holds
fewer than the level asks for, the loading screen and the HUD say so (e.g.
"photo holds 4 of 12"). Bug positions are random each match; the game tries
eight layouts and keeps the fullest. Every number here lives in one table,
`src/game/levels.py`.

### Guns

| Gun | Magazine | Time between shots | Reload | Damage per hit | Armour pierce |
|---|---|---|---|---|---|
| Sniper rifle | 12 | 0.50 s | 1.1 s | 150 (×2 on a headshot) | 1.00 |
| Shotgun | 8 | 0.38 s | 0.9 s | 5 pellets × 37.5 (no headshot bonus) | 0.85 |
| Heavy-mag gun | 100 | 0.18 s | 5.0 s | 51 (×1.5 on a headshot) | 0.70 |

Damage to a bug = base damage × armour pierce × headshot bonus. It does not
depend on the level; a bug has 100 × level health.
Press `R` to reload to a full magazine at any time. Hold right mouse to scope
in with the Sniper rifle. Hold left click with the Heavy-mag gun for
full-auto; the other two guns fire once per click.

In PvP every hit still kills in one shot, and the Heavy-mag gun keeps its PvP
magazine (50 rounds, 2.5 s reload).

### Items

| Item | Effect | Weight | Price |
|---|---|---|---|
| Flare | 8 s: every bug loses its camouflage and gets an outline | 3 | 30 |
| Thermal scan | 6 s: bugs show as heat marks, even behind the photo layers | 8 | 80 |
| Antidote, Acid shield, Filter lens, Compass, Energy drink, Smoke bomb | Coming soon (shown, not sold) | | |

Press `1`, `2` or `3` in a match to use the item in that slot, once per match.
Levels 4 and 5 have a fourth slot, but it has no key yet.

### Points

Points are both your score and your currency.

- **Match score** = seconds left × level. Using an item costs no score.
  A lost match scores 0. On level 5 the clock does not stop at zero: you get
  30 more seconds at −5 points per second, then the match is lost.
- **Wallet:** a new account starts with 100 points. A win adds its score to
  your wallet and to your **lifetime total**, which ranks the leaderboard;
  spending never lowers your rank.
- **Items cost points:** price = item weight × 10, paid when the match starts.
- **PvP unlocks** once you have won every level 1–5 at least once.
- **Leaderboard:** real players, ranked by lifetime total — the accounts on
  this computer, or everyone on the central server.

### Controls

| Input | Action |
|---|---|
| Mouse | Aim, and click buttons |
| Left click | Shoot (during a match); hold for full-auto with the Heavy-mag gun |
| Right mouse (hold) | Scope in with the Sniper rifle |
| `1` `2` `3` | Use the item in that slot (in a match) · select a slot (loadout) |
| `H` | Hide/show the HUD (in a match only) |
| `C` | Recentre to where you are sitting now |
| `R` | Reload to a full magazine (in a match) · play again (result screen) · random photo from the shared album (photo screen) |
| `U` | Share the chosen photo to the shared album (photo screen) |
| `Enter` | Main button of the current screen |
| `↑` / `↓` | Move focus on the main menu |
| Click a text field | Place the caret |
| `←` `→` `Home` `End` `Backspace` `Delete` | Edit and move the caret inside a text field |
| `Tab` | Next text field |
| Eye icon (sign-in) | Show/hide the typed password |
| Remember me (sign-in) | Keep you signed in on the next launch |
| `F11` | Switch between fullscreen and window mode (remembered in `assets/config/user_prefs.json`) |
| `F12` | Screenshot to `results/ui_v3_1107/live/` |
| `Esc` | Back one screen |

In window mode the parallax still follows the window's real position on the
screen, so you can drag the window around.

Each finished match appends one row to `results/prop_game.csv`.

---

## Shared photo album (optional)

Players can share photos to a common album on
[Cloudinary](https://cloudinary.com/) and draw a random photo from it. This
uses unsigned upload only: no API key or secret is stored in the app.

1. In your Cloudinary account, create an **unsigned upload preset** that
   adds a tag, for example `protected-m`.
2. Under **Settings → Security**, allow **Resource list**. Otherwise the app
   reports *"Album list is blocked in the Cloudinary account"*.
3. Copy `assets/config/shared_album.example.json` to
   `assets/config/shared_album.json` and fill in the three fields:

   ```json
   { "cloud_name": "your-cloud", "upload_preset": "your-unsigned-preset", "tag": "protected-m" }
   ```

`shared_album.json` is ignored by git. Without it, the shared-album buttons
report *"Shared album not configured"* and the rest of the game works
normally.

Anyone who has your unsigned preset can upload to your account. The app
therefore treats everything it downloads as untrusted. Every photo — shared,
picked from your computer, or received in PvP — goes through one import gate
(`src/content/media_gate.py`) before anything decodes it:

- the file type is decided by its content, not its name (PNG, JPEG, WebP);
- size and pixel limits are checked before decoding (no decompression bombs);
- multi-frame files are refused; extra bytes after the image are dropped;
- the photo is decoded once and re-encoded to a clean PNG with no metadata,
  and only that copy goes further;
- no file path is ever built from a remote name.

---

## Servers

Most of the game needs no server. Pick what you need:

| I want to… | Run | Ports |
|---|---|---|
| Play alone | Nothing | – |
| Play PvP on the same Wi-Fi/LAN | The PvP server, `run_server` | TCP **47821**, UDP **47822** |
| Share accounts, wallets and the leaderboard across machines | The central server (Docker, TLS) | TCP **47830** |

A PvP match itself always runs on the LAN server, even when you also use the
central server (see [Known issues](#known-issues)).

### PvP server

Competitive and Hunt & Hide are real matches over your LAN — no internet, no
cloud account. One machine runs the server; both games find it automatically.

1. **Start it** on one machine, on the same Wi-Fi/LAN as the players. Leave
   the window open for the whole session; `Ctrl+C` stops it.

   ```bat
   run_server.bat
   ```

   ```bash
   bash run_server.sh
   ```

2. **Open the firewall.**
   - Windows: the first time you run `run_server.bat` (or `run_game.bat`),
     Windows asks to allow Python on private networks. Click **Allow
     access**. If you miss the prompt, other games can't reach the server or
     discover it.
   - Linux: only if you run a firewall such as ufw or firewalld, open TCP
     47821 and UDP 47822:

     ```bash
     # ufw
     sudo ufw allow 47821/tcp
     sudo ufw allow 47822/udp
     # firewalld
     sudo firewall-cmd --permanent --add-port=47821/tcp --add-port=47822/udp
     sudo firewall-cmd --reload
     ```

3. **Find a match.** On each player's machine start the game, open the
   **PvP** tab, and either:
   - **Find match** — joins the queue for the chosen mode and level; the
     server pairs the first two players waiting; or
   - **Invite a friend** — creates a 4-digit room code the other player types
     into **Join room**.

   PvP unlocks after you win levels 1–5.

4. **How games find the server.** Automatically, over UDP discovery; there is
   nothing to type. If discovery fails (different subnet, a router or VPN
   blocks UDP broadcast), copy `assets/config/pvp.example.json` to
   `assets/config/pvp.json` and put the server's IP and port in it:

   ```json
   { "server": "192.168.1.42:47821" }
   ```

   `pvp.json` is ignored by git.

#### Try PvP on one machine

Only one process can open the webcam, so run the game with `--no-webcam`
(a simulated head pose). `scripts/pvp_bot.py` is a headless client (no GL, no
webcam) that joins the queue or a room code, so you can play against it. Use
three terminals.

Linux:

```bash
bash run_server.sh
bash run_game.sh --no-webcam
~/.venvs/protected-m/bin/python scripts/pvp_bot.py --mode Competitive --level 1 --host 127.0.0.1
```

Windows:

```bat
run_server.bat
run_game.bat --no-webcam
.venv\Scripts\python.exe scripts\pvp_bot.py --mode Competitive --level 1 --host 127.0.0.1
```

Bot flags: `--mode` (`Competitive` or `Hunt & Hide`), `--level`, `--room`
(a 4-digit code; default is the queue), `--kill-every`, `--matches`, `--name`,
`--host` (skip discovery), `--port` (default 47821), `-v`.

### Central server

The central server keeps accounts, wallets and the leaderboard in one place.
It is the referee: it charges items and computes each score from its own
clock with one-time match tickets, so a modified game cannot report a bigger
score. The code is `src/server/` (Python standard library only); it runs as
one Docker container, driven by `deploy/server.sh`.

You need Docker (Engine or Desktop), `python3` and `openssl` on the machine
that hosts it.

1. **First run:**

   ```bash
   bash deploy/server.sh setup      # deploy/.env, certificate, image
   bash deploy/server.sh start      # start and wait until healthy
   bash deploy/server.sh status     # running? last backup? how many accounts?
   ```

2. **Point the game at it** (on every player's machine):
   - copy `cert.pem` (`setup` made it in `~/hcpd-certs`) to
     `assets/config/server_cert.pem`;
   - copy `assets/config/central.example.json` to
     `assets/config/central.json` and set `"mode": "remote"` (the example
     says `local`), `"host"` to `127.0.0.1` on the same machine, and `"cert"`
     to `server_cert.pem` (the path is relative to that file). An optional
     `"name"` shows on the leaderboard:

     ```json
     {"mode": "remote", "host": "127.0.0.1", "port": 47830, "cert": "server_cert.pem", "name": "Home server"}
     ```

   - delete `central.json` to go back to local accounts.

3. **Everyday commands** (`bash deploy/server.sh <command>`):

   | Command | What it does |
   |---|---|
   | `setup [host-or-ip ...]` | first time: `.env`, certificate (in `~/hcpd-certs`), image, load the cert |
   | `start` | start, wait until healthy, print the address and `cert.pem` |
   | `stop` | stop (data kept) |
   | `status` | running, healthy, address, uptime, last backup, accounts |
   | `logs [n]` | last n log lines (default 50) |
   | `backup` | back up now and print the file |
   | `backups` | list backups |
   | `restore <file> [--force]` | put a backup back (refuses to replace live data without `--force`) |
   | `update` | rebuild the image from the current code and recreate the container |
   | `export <dir>` | image + fresh backup + cert/key + `.env`, for another machine |
   | `import <dir>` | on the new machine: load image, cert, `.env`, restore, start |
   | `help` | print this list |

   It backs itself up at start and every 24 h, keeping the last 7.

4. **Let friends join.** The server listens on `127.0.0.1` (this machine
   only) until you set `HCPD_BIND` in `deploy/.env`, for example to a
   [Tailscale](https://tailscale.com/) `100.x` address, so friends can join
   without opening your router. The certificate must name that address: make
   it again with `bash deploy/server.sh setup 100.x.y.z` (after removing
   `~/hcpd-certs`), then `start`. Give players `cert.pem`, **never
   `key.pem`**, and have them use that address as `"host"`.

Full guide (Vietnamese): [docs/deploy.md](docs/deploy.md). Design, roles and
known limits: [docs/server.md](docs/server.md); threat review:
[docs/security.md](docs/security.md).

---

## Known issues

This is a demo build. Known problems so far:

- **The player health bar never changes.** There is no way to damage the
  player yet — a level is lost only by running out of time.
- **The fourth item slot (levels 4–5) has no key.** Only `1`–`3` are bound.
- **Only two items work.** Flare and Thermal scan have real effects; the other
  six are shown as "Coming soon" and cannot be bought.
- **In-match PvP still runs on the LAN server.** The central server handles
  sign-in, rooms and photo checks for PvP, but the match itself (hits,
  timers, results) still goes through `run_server` on the local network, and
  PvP results are not on the leaderboard yet.
- **A win the server refuses** shows "Not saved" and a score of 0, but the win
  art (the "PROTECTED" stamp and the "Next level" button) is still drawn.
- **Saving a result to the central server blocks the screen** for up to 8 s on
  a slow network.
- **Hunt & Hide bug placement is automatic, not click-to-place.**
- **The first load of a new photo is slow.** Depth estimation, layer splitting
  and LaMa all run on the CPU. On a 2-core laptop this can take a minute or
  more. The same photo loads instantly the second time, from
  `assets/cache/maps/`; delete that folder to force a re-bake.
- **Head tracking needs light from the front.** Sit 40–50 cm from the screen.
  Close other apps that hold the webcam (Zalo, Meet, Teams, browser tabs)
  before starting.
- **On Hyprland, a tiled window changes the game** (fewer hiding spots fit).
  Float it — see [Linux](#linux).
- **`cv2` may fail to import.** `requirements.txt` lists both
  `opencv-python` and `opencv-contrib-python`. If `import cv2` fails, uninstall
  both and reinstall only `opencv-contrib-python`.

If the game misbehaves, the log is in `logs/`. Please attach it when you
report a bug.

---

## Roadmap

- In-match PvP on the central server, with PvP results on the leaderboard.
- Bugs that fight back, player health and the six remaining items.
- Click-to-place bug hiding for Hunt & Hide's hider.
- Sound.

---

## HCPD sandbox mode

To run the engine on its own, without the game UI:

```bat
tools\launch\run.bat        :: sandbox: all research keys (main.py, no flags)
tools\launch\run_demo.bat   :: presentation mode: only G, T, B, + = - 0, F10, F4, F6, Esc (main.py --demo)
```

On Linux the demo launcher is `bash tools/launch/run_demo.sh` (same flags; log in
`logs/demo_<date>_<time>.log`; set `PM_PYTHON` to use another interpreter).

`--album` is the game (`run_game.bat` in the repo root); `--demo` is this
research presentation mode. They are independent — the game no longer needs
`--demo`, and running both together behaves exactly like `--album` alone.

### Virtual room (`--demo` only)

The demo shows the photo inside a room: a box of grid walls behind the screen,
the screen being its open front. At the default size the box is straight; zoomed
in (`s` > 1) the back wall grows by `s`, so the walls flare and no part of the photo
is ever cut. The ceiling grid takes the colour of the
photo's sky (top 8 % of rows) and the floor grid the colour of its ground
(bottom 8 %). The room is on by default. The game (`--album`) has no room.

| Key | Action (`--demo`) |
|---|---|
| `B` | Room on / off |
| `+` (or `=`), `-` | Photo larger / smaller inside the room, x1.1 per press (0.6 to the largest size at which the photo has its old size at the centre). Above 1 the room flares wider toward the back, so zooming never cuts the photo |
| `0` | Photo scale back to 1 |
| `F10` | Per-layer relief on / off (flat "discrete" vs relief "hybrid" layers, thesis §3.5.6 and §4.5.2). The demo starts with relief on. Works with the room on or off. Not available in the game |

The other sandbox A/B keys (`F1`, `F2`, `F3`, `F7`, `F8`, `F9`) stay blocked in the demo.


| Key | Action (sandbox) |
|---|---|
| `F1` | Debug overlay: FPS, latency, pose, layer stats |
| `F2` | Cycle the pose filter (one_euro → kalman → ema → none) |
| `F3` | Motion prediction on/off |
| `F4` | Webcam preview |
| `F6` | Open an image or video (or drag and drop it onto the window) |
| `F7`–`F10` | A/B toggles: mist, atmospheric grading, hole filling, intra-layer relief |
| `G` | Hidden-object game (the predecessor of Protected M) |
| `T` | Head tracking on/off |
| `+` / `−` | Virtual viewing distance |
| `Esc` | Quit |

`F1`'s debug overlay uses `imgui[glfw]`, not in `requirements.txt` (no
wheel for Python 3.12+); without it, `F1` shows a plainer textured-quad
HUD instead.

---

## Tests

```bat
.venv\Scripts\python.exe -m pytest -q
```

On Linux: `~/.venvs/protected-m/bin/python -m pytest -q`.

Tests that need a real OpenGL context skip automatically, with a clear
reason, on a machine without a real OpenGL 3.3 driver — this is what
happens on CI's runners. The suite fails if any test changes the real files
in `assets/config/`.

---

## Configuration

| File | Purpose | In git |
|---|---|---|
| `assets/config/default.json` | Camera, filters, tracking, renderer | yes |
| `assets/config/screen.json` | Physical screen size, in metres | yes |
| `assets/config/content.game.json` | Layer pipeline settings used by the game | yes |
| `assets/config/content.demo.json` | Layer pipeline settings for `--demo` without the game | yes |
| `assets/config/content.json` | Layer pipeline settings for the sandbox | yes |
| `assets/config/user.json` | Face-scale calibration (written by `--calibrate`) | no |
| `assets/config/user_prefs.json` | Display mode and window position | no |
| `assets/config/shared_album.json` | Cloudinary settings | no |
| `assets/config/pvp.json` | PvP server `IP:port` override (skips discovery) | no |
| `assets/config/accounts.json` | Local accounts: display name, scrypt-hashed password, recovery-code hash, play history, wallet, lifetime points, levels cleared | no |
| `assets/config/accounts.json.bak` | Previous good copy of `accounts.json`, used to recover from an interrupted write | no |
| `assets/config/session.json` | "Remember me" session token, for auto sign-in on the next launch | no |
| `assets/config/central.json` | Use a central server: `mode`, `host`, `port`, pinned `cert`, display `name` | no |
| `deploy/.env` | Central server settings (bind address, port, limits, backups); copy of `deploy/.env.example` | no |

---

## Architecture

Dependencies only point down, and data flows one way:

```
input → routes (action name) → V3UI.dispatch → services / game → MatchView → painter

app (thin shell: window, loop, wiring)
  -> ui, game, net, services   ui never imports account, game internals, camera or tracking
  -> renderer, tracking, content
server (separate process / container) -> game rules, shared account rules, account crypto
```

- `game/` is plain Python — match rules, weapons, firing, hit resolution,
  scoring — with no import of the engine, renderer, UI, fx, net, OpenGL, glfw
  or PIL.
- The UI talks to accounts only through the `AccountService` interface (a
  local and a remote implementation), reads the match only through a frozen
  `MatchView`, and writes no files.
- `server/` imports no UI, camera, tracking or content code; camera frames
  never reach the network.
- `devtools/` (bench/evidence/automatch hooks) uses only the public
  `DevHooks` facade and is imported only when an `HCPD_*` flag is set.

`tests/test_architecture.py` parses every import with `ast` and fails if any
of this is broken.

---

## Project layout

```
main.py                 Entry point (--album = game, --demo = research demo)
run_game.bat / .sh      Launch the game (Windows / Linux)
run_server.bat / .sh    Launch the PvP LAN server
deploy/                 Central server: Dockerfile, docker-compose.yml,
                        server.sh (the one script to run it), .env.example
tools/launch/           Research launchers: run.bat/run.ps1 (sandbox),
                        run_demo.bat / run_demo.sh (presentation mode)
src/
  engine/               Main loop, worker threads, app wiring (app.py)
  game/                 Match and weapon rules, plain Python — no engine/
                        renderer/ui_album/fx/net/OpenGL/glfw/PIL import
  ui_album/             Game UI (v3): screens and flow, bugs, text fields,
                        hit-testing/hotspots, the painter
  devtools/             Bench/evidence/shots/automatch test hooks; loaded
                        only when an `HCPD_*` env flag is set
  common/               Small helpers shared by renderer/fx/ui_album
                        (stage timing, texture bleed) — depend on neither
  content/              Depth, layer splitting, hole filling, bake cache, shared album
  renderer/             OpenGL window, shaders, content plane, Kooima camera
  tracking/ pose/       MediaPipe landmarks, solvePnP, eye position, calibration
  filters/              One Euro, Kalman, EMA, motion prediction
  math/                 Projection, matrices, quaternions
  camera/ scene/ gui/   Webcam capture, scene graph (meshes), calibration wizard
  net/                  PvP wire protocol, LAN server, UDP discovery, client,
                        central-server client (depends on game, not on ui_album)
  server/               Central server: TLS, roles, accounts, wallet, scores,
                        rooms, backups (standard library only)
  shared/               Rules shared by the game and the server
                        (usernames/passwords)
  account/              AccountService (local and remote), store, crypto
assets/
  ui_v3/                Game UI pack exported from Penpot (in git)
  ui_album/             Bug sprites (sprites/) + manifest.json (sprite
                        registry); the rest is unused v2 UI art (in git)
  models/               face_landmarker.task (+ optional lama_fp32.onnx)
  config/ shaders/
scripts/                Measurement and report tools used for the thesis
                        (pvp_bot.py — headless PvP test client)
tests/                  pytest suite
tools/                  pin_window.py (fixed window for measurements),
                        make_dev_cert.py, research launchers
docs/                   Technical documentation: security.md, server.md,
                        deploy.md (Vietnamese), KY_THUAT_HE_THONG.md (Vietnamese)
```

---

## Credits

- **Head tracking:** [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Apache-2.0).
- **Depth:** [Depth Anything](https://huggingface.co/LiheYoung/depth-anything-small-hf) V1 Small (Apache-2.0).
- **Hole filling:** [LaMa](https://github.com/advimman/lama) (Suvorov et al., 2022), ONNX export by [Carve](https://huggingface.co/Carve/LaMa-ONNX) (Apache-2.0).
- **Projection:** R. Kooima, *Generalized Perspective Projection* (2009).
- **Bug sprites:** [Top-down Alien Bug Sprites](https://whiteknightstudios.itch.io/top-down-bug-sprites) by W_K_Studio (CC0).
- **Hit/UI effects:** rendered from [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx) by Justin Finkelman (MIT), recoloured to the ui_v3 palette.
- **Fonts** in the UI packs are under the SIL Open Font License.

Built as an undergraduate thesis project (HCPD, KLCN164).

---

## License

- **Source code:** MIT. See [LICENSE](LICENSE).
- **Game art and design:** CC BY-NC 4.0. This covers the UI packs in
  `assets/ui_v3/` and `assets/ui_album/` and the Protected M story; you may use
  them for non-commercial purposes with credit. See
  [LICENSE-ASSETS.md](LICENSE-ASSETS.md).
- **Third-party files:** the model, fonts and bug sprites keep their own
  licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
