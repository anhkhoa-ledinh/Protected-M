# Security review (24/11)

Scope: every place where data from outside the process reaches the app, what checks it
today, what is missing, and who may call what. Read-only review: no code was changed in this
step. Line numbers are for the commit this file was added in.

Rank: **high** = a remote or crafted input can crash the app, exhaust memory or run
untrusted decoder code; **medium** = needs a local/LAN foothold or a user mistake;
**low** = shipped or self-written data, or a missing explicit flag with a safe default.

## 1. Photos

All photo paths end in the same decoders: `ImageSource._load` (`src/content/image_source.py:65`)
-> `imread_bgr` (`src/content/cv_io.py:15`, `cv2.imdecode`) on the render thread or in the bake
subprocess (`src/content/bake.py:239`). Camouflage colour re-decodes the file
(`src/engine/app.py:959`). The upload path decodes with Pillow (`src/content/shared_album.py:84`).
Videos (`mp4 mov avi mkv`) go to `cv2.VideoCapture`/ffmpeg (`src/content/cv_io.py:41`).

| # | Entry point | Code path | Checks today | Gap | Rank |
|---|---|---|---|---|---|
| P1 | File dialog (album "pick file") | `src/gui/file_dialog.py:11` -> `src/engine/app.py:1204` -> `_try_load_media` `src/engine/app.py:440` -> `ContentPlaneRenderer.load_file` `src/renderer/content_plane.py:835` -> `_swap_source` `:1496` | Exists; extension in `SUPPORTED_EXTENSIONS` (`src/content/formats.py:7`); long side capped to 2048 **after** decode (`image_source.py:23`) | Extension only, no magic bytes; no size or pixel limit before decode (OpenCV's own cap is ~1 Gpx); symlinks and devices followed; the "All files" filter lets any file be chosen; file is decoded by `cv2` and again by `cv2` for camo | high |
| P2 | F6 dialog (non-album) | `src/engine/app.py:2559` -> `_try_load_media` | same as P1 | same as P1 | high |
| P3 | Drag and drop | `window.pop_dropped_path()` loop `src/engine/app.py:2637` -> `_try_load_media` | same as P1 | same as P1; a dropped path can be any file or folder | high |
| P4 | Command line `--config` / `content*.json` `content_path` | `main.py:38` -> `src/config/loader.py:14` -> `src/config/schema.py:443` `resolved_media_path` -> `ContentPlaneRenderer.__init__` `src/renderer/content_plane.py:332` | JSON parsed; falls back to a procedural plane when load fails | Same decoders, no gate; `content_path` may point anywhere on disk (`..`, absolute, symlink) | medium |
| P5 | Shared album download (Cloudinary) | `src/engine/app.py:571` -> `shared_album.pick_random_and_download` `src/content/shared_album.py:267` (`download` `:216`) -> `start_pipeline` -> `_try_load_media` | https only, `_SSL_CTX` default context; 15 s timeout; body capped while streaming at 20 MiB (`_read_capped` `:136`); format whitelist (`ALLOWED_FORMATS` `:49`); file named from sha1 of `public_id`, path checked to stay in `dest_dir`; `verify_photo` (`:267`): `Image.open`+`verify()`, long side <= 4096 | Format is taken from the remote JSON, not from the bytes; `verify()` does not decode pixels; Pillow's bomb guard is the default (warn at ~89 Mpx); a bad file is written into `assets/cache/shared/` before it is verified and is reused on the next run (`if dest.exists(): return dest`, `:236`) | high |
| P6 | Shared album upload | `src/engine/app.py:564` `_start_share_upload` -> `shared_album.upload` `:148` -> `_reencode` `:84` | Decodes with Pillow, EXIF transposed then dropped, long side <= 2048, JPEG q90, result <= `MAX_BYTES` 10 MiB | Decodes an unchecked local file (same gap as P1); a polyglot or a huge header is decoded before any check | medium |
| P7 | PvP photo from the other player | `src/net/server.py:580` relays `jpeg_b64` -> `src/engine/app.py:1564` `_pvp_handle_photo` -> `assets/cache/pvp/<sha1>.jpg` -> `_try_load_media` | base64 `validate=True`; sha1 recomputed locally; file name never built from remote data | The sha1 only proves the sender did not corrupt it, not that it is a photo; size limit is only the 16 MiB line cap (a 16 MiB JPEG can be hundreds of Mpx); no format check; decoded by `cv2` | high |
| P8 | Anything dropped into `assets/cache/` | `shared/`, `pvp/` read by P5/P7 (`dest.exists()`); `maps/<sha1>/` read by `bake.load_from_cache` (see C1) | None: a file with the right name is trusted | Local write access is enough to plant a file that is loaded as a photo or as a baked layer | medium |
| P9 | Videos | `src/content/cv_io.py:41` (`cv2.VideoCapture(..., CAP_FFMPEG)`), `src/renderer/video_player.py:27` | Extension whitelist only | ffmpeg demuxers parse untrusted containers in-process; not covered by the photo gate | medium |

## 2. Caches read back

| # | Entry point | Code path | Checks today | Gap | Rank |
|---|---|---|---|---|---|
| C1 | Bake cache `assets/cache/maps/<sha1>/` (`depth.npy`, `layer*.png`, `mask*.png`, `meta.json`) | `src/content/bake.py:171` `load_from_cache` (`np.load` `:180`, `cv2.imread` `:184`) | `meta.json` `format` version; shape cross-check layer vs depth; whole body inside `try/except` (a bad cache is re-baked) | `np.load` relies on numpy's default `allow_pickle=False` (safe, but not explicit); layer PNGs decoded by `cv2` with no gate; directory name is a hash of the inputs, not of the content | medium |
| C2 | LaMa fill cache | `src/content/lama_fill.py:192` (`cv2.imread` of `cached`) | Cache file name is a content hash | Same as C1 | low |
| C3 | FX library (`assets/fx/**`, sheets and `fx.json`) | `src/fx/spec.py:63` (json), `:98` and `src/fx/library.py:353` (`Image.open`), `src/fx/player.py:65` | Shipped with the app; failures are logged | A swapped pack is trusted; `fx.json` values are not range-checked beyond use | low |
| C4 | UI atlas, `manifest.json`, `ui_v3` art | `src/ui_album/manifest.py:61`, `src/ui_album/v3.py:83`, `src/ui_album/textures.py:165`, `src/ui_album/board.py:114,133`, `src/ui_album/v3_hover.py:124`, `src/ui_album/hud_cards.py:100` | Shipped with the app; `textures.py` caps size from `image.info` before decode | Same trust as C3 | low |
| C5 | ML models (`*.onnx`, Hugging Face weights) | `src/content/depth_estimator.py:153,201,206`, `src/content/lama_fill.py:80` | `HF_HUB_OFFLINE` in CI; paths fixed in code | Model files are loaded by torch/ORT; a swapped `.bin`/pickle model would run code. No hash check | medium |

## 3. Config and account JSON

| # | Entry point | Code path | Checks today | Gap | Rank |
|---|---|---|---|---|---|
| J1 | App config (`default.json`, `content*.json`, `screen.json`, `ui.json`, calibration) | `src/config/loader.py:14`, `src/content/persistence.py:42`, `src/engine/app.py:327,367,407`, `src/pose/calibration.py:169,399,529`, `src/pose/head_pose.py:121` | `json.loads`; schema coercion in `AppConfig.from_dict`; YAML path uses `yaml.safe_load` (`loader.py:107`) | No file size limit; values mostly trusted; `content_path` unrestricted (see P4) | low |
| J2 | `user_prefs.json` | `src/config/prefs.py:51` (load), `:85` (save) | try/except; every value coerced; unknown display mode falls back | None significant | low |
| J3 | `accounts.json`, `session.json` | `src/account/store.py:68` `_try_read_json`, shape check, `.bak` and `accounts.corrupt-*` recovery; atomic `save` `:140` | scrypt hashes, atomic replace, corrupt file renamed, lockout after `MAX_FAILURES` | No size limit on read; file is plain local data (anyone with disk access can edit the wallet; fine for a local game, must move to the server later) | low |
| J4 | `shared_album.json`, `pvp.json` | `src/content/shared_album.py:60`, `src/net/client.py:40` | Type checks, port range | `shared_album.json` holds the Cloudinary name and the upload preset in clear text on disk; no signing | low |

## 4. PvP LAN protocol

TCP 47821 (JSON lines) and UDP 47822 (discovery beacon). No encryption, no authentication,
no per-peer rate limit. Meant for a trusted LAN only.

| # | Entry point | Code path | Checks today | Gap | Rank |
|---|---|---|---|---|---|
| N1 | Server bind | `src/net/server.py:221` `bind((host, port))`, default host `0.0.0.0`; thread per connection `:277` | Backlog 16 | Listens on every interface; no cap on open connections or threads; a host that opens many sockets exhausts threads | high |
| N2 | Framing | `src/net/protocol.py:115` `LineReader`; `decode` | 16 MiB cap on the raw buffer before a newline, UTF-8 and JSON errors become `ProtocolError`, connection closed | 16 MiB per line is far above the real need (only `photo` is large); no rate limit per second | medium |
| N3 | Handshake | `src/net/server.py:321` `_do_handshake` | `hello` must come first; `name`/`version` must be strings; version must match | No auth, any LAN peer can join a room; name length capped later (`MAX_NAME_LEN` 24) | medium |
| N4 | Message bodies | `src/net/server.py` handlers (`_on_bugs` `:591`, `hit`, `used_item`, ...) | `_valid_bug` type/range checks, `MAX_BUGS`, `MAX_ITEM_LEN`; `ProtocolViolation` closes the peer | Server trusts the reporting client for hits (a cheating client wins); fine for casual LAN, not for the central server | medium |
| N5 | Photo relay | `src/net/server.py:580` `_on_photo` | `jpeg_b64`/`sha1` types, sha1 format | Forwarded without any size or format check; see P7 | high |
| N6 | Discovery beacon | `src/net/discovery.py:27` (limits), `:118` `recvfrom` | `MAX_BEACON_BYTES` 1024, name/version trimmed, port range | Any LAN host can advertise a fake server (spoofing); UDP source is not verified | low |
| N7 | Client | `src/net/client.py:143` `_run` (`create_connection`, timeout), `LineReader` | Same framing as the server; the client only talks to the chosen host | No TLS: a LAN attacker can read or alter traffic | medium |
| N8 | Shared album HTTPS | `src/content/shared_album.py:189,216,148` | `ssl.create_default_context()`, 15 s timeout, capped reads, quoted URL parts | Remote JSON drives the URL and the filename extension (see P5) | medium |

## 5. Camera frames

| Step | Code path | Result |
|---|---|---|
| Capture | `src/camera/capture.py:108` `cv2.VideoCapture(device_id)`; thread `src/camera/async_capture.py`, `src/engine/camera_worker.py:9` | Frames stay in memory |
| Tracking | `src/engine/tracking_worker.py` (`src/tracking`, `src/pose`) -> eye position only | Only numbers (position, pose) leave this stage |
| Preview | `src/renderer/webcam_preview.py:5`, `src/renderer/renderer.py:27` (`FramePacket`), `V3UI.set_camera_frame` `src/ui_album/v3_ui.py:1012` | Uploaded to a GL texture, drawn locally |
| Disk | Searched `src/` for `imwrite`, `.save(`, `VideoWriter`, `write_bytes`, `np.save`: none writes a camera frame. The only writers are the baker/caches (`bake.py:155-160`, `lama_fill.py:208`), shots (`ui_album/shots.py:27`, F12/devtools), accounts/prefs, the sha-named photo caches and `z_diag.csv` (`src/pose/head_pose.py:157`, numbers only, off unless `debug.z_diagnostics`) | Frames never reach disk |
| Network | `src/net`, `src/account`, `src/content/shared_album` and `src/ui_album` do not import `src.camera`, `src.tracking`, `src.pose` today (checked with the import graph). The PvP messages carry bug positions, hits and a photo the user chose, never a camera frame | Frames never reach the network |

Risk: **low today**, but nothing enforces it. Step 5 of this task adds an architecture test.

## 6. Who may call what (target, and the state today)

| Capability | Allowed modules (target) | Today |
|---|---|---|
| Read the camera | `src/camera`, `src/engine/camera_worker.py`, `src/gui/calibration_wizard.py` | As target; `src/engine/app.py:218` only builds a `CaptureConfig` |
| See camera frames / landmarks | `src/engine`, `src/tracking`, `src/pose`, `src/renderer/webcam_preview.py` | `src/renderer/renderer.py:27` imports `FramePacket` for the preview (fine) |
| Open sockets | `src/net/{server,client,discovery}.py`, `src/content/shared_album.py` (+ `shared_album_worker.py`) | As target |
| Write files | caches (`bake.py`, `lama_fill.py`, `shared_album.download`, `engine/app.py:1585`), `src/account/store.py`, `src/config/prefs.py`, `src/content/persistence.py`, shots, devtools | `engine/app.py:1585` writes the PvP photo directly (should go through the photo gate, Step 4) |
| Decode a user photo | **only** the new media gate (Step 4) | Five places decode user bytes: `cv_io.py:35`, `shared_album.py:87,270`, `lama_fill.py:192`, `bake.py:184` |
| Parse remote data | `src/net/protocol.py`, `src/content/shared_album.py` | As target |
| Import `ui_album` | `src/engine` only | `net` and `account` do not import it (checked by `tests/test_architecture.py`) |

## 7. Top risks (in the order to fix)

1. **No import gate for photos (P1-P7).** Any file with a photo extension is handed straight to
   `cv2.imdecode`; no magic-byte check, no pixel limit before decode, symlinks followed. A crafted
   file can exhaust memory or hit a decoder bug. -> Step 4.
2. **PvP photo is relayed and decoded unchecked (P7, N5).** The other player can send a 16 MiB
   "JPEG" that expands to hundreds of megapixels. -> Step 4 covers the decode side; the server
   should also cap `jpeg_b64` (central server TASK).
3. **PvP server exposure (N1).** Binds `0.0.0.0`, thread per connection, no connection cap, no
   auth, no TLS. Fine at home, not on a shared network. -> central server TASK.
4. **Shared album download trusts the remote format and keeps bad files (P5).** Extension comes
   from remote JSON, the file is written before it is verified and is reused later. -> Step 4
   (the gate verifies by content) and drop unverified files.
5. **Caches and model files are trusted by name (C1, C5, P8).** A planted `layer*.png`, `.npy`
   or model file is loaded without any integrity check. Needs a hash check or a signed manifest;
   not in this task.

Smaller items: videos go to ffmpeg unchecked (P9); `allow_pickle=False` is not explicit in
`bake.py:180`; account and wallet data is a plain local file (J3).

## 8. Status after this task

| Id | Now |
|---|---|
| P1-P3, P5, P6, P7 | Every path ends in `media_gate.import_photo`: regular file, size, magic bytes, extra bytes after the image dropped with a warning, header pixels before decode, one frame, clean PNG in `assets/cache/photos`. A refused file shows "This file is not a photo we can open" on the album screen. |
| P4 | Start-up photo from `content*.json` goes through the same gate (`ContentPlaneRenderer.load_file`). `content_path` itself is still unrestricted (a user-edited file). |
| P5 | `verify_photo` runs the gate with the 4096 px rule and `allowed_root` = the shared folder; a refused download is deleted. The format still comes from the remote JSON, but the gate decides by content. |
| P8, C1, C2, C5 | Open: caches and model files are still trusted by name. |
| P9 | Open: videos still go to ffmpeg unchecked. |
| N1-N4, N6-N8 | Open: central server TASK. |
| N5 | Receiving side is gated; the server relay still has no size cap. |
| Camera | Enforced by `tests/test_architecture.py`: net, account, shared album and album UI cannot reach `src.camera`/`tracking`/`pose`, and only four known modules write images. |

Known trade-offs of the gate: a JPEG with a `.png` name is accepted (the project's own
`t21.png` is one; `strict_extension=True` refuses it); a JPEG/PNG/WebP with data after its end
(a Samsung motion photo, image+ZIP) is accepted with the extra bytes dropped and a
warning logged; the clean copy is re-encoded from pixels and carries none of it.
