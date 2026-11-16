# CREDITS — bug sprite art

TASK_bug_art_0530 Phase 3. Mỗi dòng: file, tác giả, URL nguồn, giấy phép, ngày.

Các file nằm ở `assets/ui_album/sprites/` (đúng thư mục mà `AlbumManifest`
đọc mọi sprite của bộ bọ, bất kể giao diện album hay v3 — xem
`src/ui_album/manifest.py:19`). Thư mục đó không theo dõi bởi git
(`.gitignore`), nên các file PNG và `manifest.json` chỉ nằm trên đĩa máy
này; file CREDITS này được đặt ở `assets/ui_v3/sprites/` theo đúng yêu cầu
của kịch bản, là nơi DUY NHẤT trong việc này mà git còn theo dõi.

| File | Tác giả | Nguồn | Giấy phép | Ngày |
|---|---|---|---|---|
| `assets/ui_album/sprites/b1_wk_topdown.png` | W_K_Studio | https://whiteknightstudios.itch.io/top-down-bug-sprites | Creative Commons Zero v1.0 Universal (CC0), nêu rõ trên chính trang tác giả | 2026-11-11 |
| `assets/ui_album/sprites/b2_wk_topdown.png` | W_K_Studio | https://whiteknightstudios.itch.io/top-down-bug-sprites | Creative Commons Zero v1.0 Universal (CC0), nêu rõ trên chính trang tác giả | 2026-11-11 |
| `assets/ui_album/sprites/b3_wk_topdown.png` | W_K_Studio | https://whiteknightstudios.itch.io/top-down-bug-sprites | Creative Commons Zero v1.0 Universal (CC0), nêu rõ trên chính trang tác giả | 2026-11-11 |
| `assets/ui_album/sprites/b4_wk_topdown.png` | W_K_Studio | https://whiteknightstudios.itch.io/top-down-bug-sprites | Creative Commons Zero v1.0 Universal (CC0), nêu rõ trên chính trang tác giả | 2026-11-11 |

Ghi chú lấy file: trang itch.io của tác giả ghi đúng dòng giấy phép
"Creative Commons Zero v1.0 Universal" và "can be used Commercially". File
tải trực tiếp (đằng sau luồng "pay what you want") không lấy được qua công
cụ đang có; bốn sprite trên được cắt từ chính ảnh xem trước (preview) công
khai trên trang đó (`img.itch.zone/.../original/P605QY.png`, lưu nguyên bản
không chỉnh sửa tại `assets/ui_album/sprites/wk_bugs_topdown_source_preview.png`),
ảnh xem trước này đã có sẵn kênh alpha trong suốt thật (không phải nền
trắng) — bốn ô trong lưới 3×3 được cắt riêng ra, không vẽ lại, không xử lý
AI. Toàn bộ artwork trong ảnh xem trước là CÙNG một tác phẩm mà trang đã
nêu rõ giấy phép CC0.
