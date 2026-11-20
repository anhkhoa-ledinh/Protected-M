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

- Windows 10 or 11. The launch scripts are `.bat` files. Tested with Python
  3.13.
- A webcam, with your face lit from the front.
- A few GB of free disk space for PyTorch and the models downloaded on first
  run.

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

This runs `python main.py --album` with the `.venv` in this folder and
writes a log to `logs\game_<date>_<time>.log`. If `.venv` does not exist yet,
the script tells you how to create it.

A round goes like this:

1. **Sign in.** Accounts are real and local — stored only on this computer
   (`assets/config/accounts.json`), with a scrypt-hashed password, **Remember
   me** (keeps you signed in next launch) and a recovery code if you forget
   your password. Playing as guest skips sign-in but does not save your
   progress.
2. **Choose a level.**
3. **Choose a photo.** You can:
   - pick one from your computer;
   - press `R` to get a random photo from the shared album;
   - press `U` to share yours to the shared album.

   Then click **Split layers & continue** and wait for the layers to bake.
4. **Loadout.** Pick one of three guns and put items in your slots.
5. **Head-tilt check.** Tilt your head left and right. **Enter match** unlocks
   anyway after 15 s, so a weak webcam cannot block you.
6. **Hunt.** Move your head to look behind the layers, aim with the mouse and
   left-click to shoot. Win by clearing every bug before time runs out.

### Levels

| Level | Bugs | Time | Item slots | Bugs fight back | Bug HP |
|---|---|---|---|---|---|
| 1 · Easy | 6 | 180 s | 2 | no | 100 |
| 2 · Normal | 8 | 150 s | 3 | no | 100 |
| 3 · Hard | 10 | 150 s | 3 | yes | 150 |
| 4 · Infestation | 12 | 120 s | 4 | yes | 200 |
| 5 · Asian Mode | 12 | 120 s | 4 | yes; stamina and overtime count | 200 |

### Guns

| Gun | Magazine | Time between shots | Reload | Damage |
|---|---|---|---|---|
| Sniper rifle | 12 | 0.50 s | 1.1 s | 100 per hit (×2 on a headshot) |
| Shotgun | 8 | 0.38 s | 0.9 s | 5 pellets × 25 per shot (no headshot bonus) |
| Heavy-mag gun | 50 | 0.18 s | 2.5 s | 34 per hit (×1.5 on a headshot) |

Hold right mouse to scope in with the Sniper rifle. Hold left click with the
Heavy-mag gun for full-auto; the other two guns fire once per click.

### Controls

| Input | Action |
|---|---|
| Mouse | Aim, and click buttons |
| Left click | Shoot (during a match); hold for full-auto with the Heavy-mag gun |
| Right mouse (hold) | Scope in with the Sniper rifle |
| `1` `2` `3` | Select an item slot |
| `H` | Hide/show the HUD (in a match only) |
| `C` | Recentre to where you are sitting now |
| `R` | Play again (result screen) · random photo from the shared album (photo screen) |
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
therefore treats everything it downloads as untrusted:

- it checks that the file decodes as an image;
- it rejects anything over 4096 px on the long side;
- it never builds a file path from a remote name.

---

## PvP on a local network

Competitive and Hunt & Hide are real matches over your LAN — no internet,
no cloud account. One machine runs the server; both games find it
automatically.

1. **Start the server.** On one machine, on the same Wi-Fi/LAN as the
   players:

   ```bat
   run_server.bat
   ```

   Leave that window open for the whole session (`Ctrl+C` to stop it).

2. **Allow Python through Windows Firewall.** The first time you run
   `run_server.bat` (or `run_game.bat`), Windows will ask to allow Python
   through the firewall for private networks — click **Allow access**. If
   you miss the prompt, other players' games won't be able to reach the
   server or discover it.

3. **Find a match.** On each player's machine, run `run_game.bat`, open the
   **PvP** tab, and either:
   - **Find match** — joins the matchmaking queue for the chosen mode and
     level; the server pairs the first two players waiting; or
   - **Invite a friend** — creates a 4-digit room code the other player
     types into **Join room**.

   Players discover the server automatically over UDP; nothing to type in
   by default.

4. **`pvp.json` override (optional).** If discovery doesn't find the
   server (different subnet, UDP broadcast blocked by a router/VPN), copy
   `assets/config/pvp.example.json` to `assets/config/pvp.json` and fill in
   the server's IP and port:

   ```json
   { "server": "192.168.1.42:47821" }
   ```

   `pvp.json` is ignored by git — it's specific to your machine/session.

### Testing PvP on one machine

- `--no-webcam` runs the game with a simulated head pose instead of
  opening the real webcam, so you can run two game instances on the same
  laptop (only one process can open the webcam at a time):

  ```bat
  run_game.bat --no-webcam
  ```

- `scripts\pvp_bot.py` is a headless client (no GL, no webcam) that joins
  a queue or a room code and plays either role — useful to test the server
  and the other, real game instance without needing two players:

  ```bat
  .venv\Scripts\python.exe scripts\pvp_bot.py --mode Competitive --level 1
  ```

---

## Known issues

This is a demo build. Known problems so far:

- **The player health bar never changes.** There is no way to damage the
  player yet, in solo play or PvP — a level is lost only by running out of
  time, never by taking a hit.
- **Loadout items have no effect yet.** Picking one into a slot and using it
  (PvP only — solo play has no way to use an item at all) only costs points
  via its weight and shows up in the match result; none of the 8 items
  (Flare, Antidote, Acid shield, Filter lens, Compass, Thermal scan, Energy
  drink, Smoke bomb) heals, reveals or helps in any other way.
- **The leaderboard still shows sample entries.** Accounts and match history
  are real and local now (see [Play](#play) above); only the leaderboard
  itself is not wired to real player data yet — your own row is computed
  from your real history, the rest are fixed sample rows.
- **Hunt & Hide bug placement is automatic, not click-to-place.** The
  hider's bugs are placed by the same algorithm solo matches use (hidden
  from the design rest eye), not by clicking the photo. Relocating one bug
  during the hunt isn't implemented either.
- **The first load of a new photo is slow.** Depth estimation, layer splitting
  and LaMa all run on the CPU. On a 2-core laptop this can take a minute or
  more. The same photo loads instantly the second time, from
  `assets/cache/maps/`; delete that folder to force a re-bake.
- **Head tracking needs light from the front.** Sit 40–50 cm from the screen.
  Close other apps that hold the webcam (Zalo, Meet, Teams, browser tabs)
  before starting.
- **`cv2` may fail to import.** `requirements.txt` lists both
  `opencv-python` and `opencv-contrib-python`. If `import cv2` fails, uninstall
  both and reinstall only `opencv-contrib-python`.

If the game misbehaves, the log is in `logs\`. Please attach it when you
report a bug.

---

## Roadmap

- Click-to-place bug hiding for Hunt & Hide's hider, and bug relocation
  during the hunt.
- A real leaderboard instead of sample data.
- Sound.

---

## HCPD sandbox mode

To run the engine on its own, without the game UI:

```bat
tools\launch\run.bat        :: sandbox: all research keys (main.py, no flags)
tools\launch\run_demo.bat   :: presentation mode: only G, T, F4, F6, Esc (main.py --demo)
```

`--album` is the game (`run_game.bat` in the repo root); `--demo` is this
research presentation mode. They are independent — the game no longer needs
`--demo`, and running both together behaves exactly like `--album` alone.

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

Tests that need a real OpenGL context skip automatically, with a clear
reason, on a machine without a real OpenGL 3.3 driver — this is what
happens on CI's runners.

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
| `assets/config/accounts.json` | Local accounts: display name, scrypt-hashed password, recovery-code hash, play history and best score | no |
| `assets/config/accounts.json.bak` | Previous good copy of `accounts.json`, used to recover from an interrupted write | no |
| `assets/config/session.json` | "Remember me" session token, for auto sign-in on the next launch | no |

---

## Architecture

Dependencies only point down:

```
app (thin shell: window, loop, wiring)
  -> ui, game, net        ui and net read game; game imports none of them
  -> renderer, tracking, content
```

`game/` is plain Python — match rules, weapons, firing, hit resolution,
scoring — with no import of `src.engine`, `src.renderer`, `src.ui_album`,
`src.fx`, `src.net`, OpenGL, glfw or PIL. `net/` depends on `game` for match
constants, not on the UI. `devtools/` (bench/evidence/automatch test hooks)
plugs into `app` through one entry point and is only imported when an
`HCPD_*` flag is set. `tests/test_architecture.py` parses every import with
`ast` and fails if any of this is broken.

---

## Project layout

```
main.py                 Entry point (--album = game, --demo = research demo)
run_game.bat            Launch the game
run_server.bat          Launch the PvP LAN server
tools/launch/           Research launchers: run.bat/run.ps1 (sandbox),
                        run_demo.bat (presentation mode)
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
  net/                  PvP wire protocol, LAN server, UDP discovery, client
                        (depends on game, not on ui_album)
assets/
  ui_v3/                Game UI pack exported from Penpot (in git)
  ui_album/             Bug sprites (sprites/) + manifest.json (sprite
                        registry); the rest is unused v2 UI art (in git)
  models/               face_landmarker.task (+ optional lama_fp32.onnx)
  config/ shaders/
scripts/                Measurement and report tools used for the thesis
                        (pvp_bot.py — headless PvP test client)
tests/                  pytest suite
docs/                   Technical documentation (docs/KY_THUAT_HE_THONG.md is in Vietnamese)
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
