# ui_album — Protected M (hướng "Album bọc nhựa") cho ImGui

Bộ asset xuất từ thiết kế Penpot "Game UI" (các board `moi_*`). Toạ độ thiết kế 1280×720 (bảng thành phần 1280×960); ảnh render ở `image_scale` = 1.5 (1920×1080).

## Cách vẽ một màn (theo đúng thứ tự)
1. `base_<theme>.png` phủ kín màn (theme: `ngay` | `dem`). Đã ẩn: ảnh, lớp phủ trên ảnh, và mọi ô động.
2. Ô ảnh `photos[]`: `kind = "photo"` → vẽ ảnh của người chơi (cover-fit, cắt giữa) hoặc `sample_photo`; `kind = "viewport"` → vùng vẽ cảnh 3D HCPD (màn chơi).
3. `top_<theme>.png` (PNG trong suốt): góc dán ảnh, vệt loá, vết gặm, vụn, bọ trang trí, tem/ghi chú đè lên ảnh.
4. Ô động `dyn[]`: vẽ bằng ImGui theo `key` với đúng kiểu chữ/màu trong bản ghi (`ff`, `fw`, `fs`, `col` = tên token, `al`, `rot`), giá trị lấy từ trạng thái game. `children[]` mô tả từng phần con (vd. bảng điểm, nhãn hiệu ứng, sticker vật phẩm) để dựng lại.
5. Lớp chọn/hover: vẽ khung `primary` 3 px quanh `hotspots[]` đang chọn (các thẻ được xuất ở trạng thái trung tính, `sel: true` đánh dấu thẻ vốn được chọn trong bản thiết kế).

## manifest.json
- `tokens.ngay` / `tokens.dem`: tên token → mã hex (đúng token Penpot của người dùng).
- `screens.<tên>`: `size`, `base`, `top`, `photos[]`, `dyn[]`, `hotspots[]` (nút `nut_<nhãn>`, tab `tab_<nhãn>`, tab con `tab_con_<nhãn>`, thẻ cấp `cap_<n>_tui`, vũ khí `vu_khi_<i>_tui`, sticker `sticker_<tên>`, ô nhập `o_<NHÃN>`, chọn theme `o_Ngày`/`o_Đêm`, kỹ năng `ky_nang_<tên>`, thẻ chế độ `che_do_<tên>_tui`...), `texts[]` (mọi chữ tĩnh kèm vị trí, dùng cho link như "Chưa có album? Tạo album mới").
- `fonts.files`: họ chữ → độ đậm → file TTF; `fonts.dynamic_text_styles`: các cỡ cần nạp cho ô động; `fonts.glyph_ranges_hint`: dải ký tự tiếng Việt.
- `sprites`: 4 loại bọ theo cấp. `sample_photo`: ảnh mẫu.

Mọi `rect` là `[x, y, w, h]` trong toạ độ thiết kế (đã xoay thì là khung bao). Font: SIL OFL (kèm file OFL_*.txt).

## v2 (06/11 22:xx)
- Mỗi thanh có hai ô động: `<khoá>_track` (rãnh, rộng = 100 %) và `<khoá>_bar` (phần đầy). Vẽ phần đầy rộng = `track.w × tỉ lệ`; **không** dùng bề rộng mẫu của `_bar`. Khoá: `hp`, `st`, `kills`, `opp`, `exposure`, `pct`. Muốn ẩn hẳn một thanh (vd. stamina ngoài cấp Châu Á) thì không vẽ cả `_track` lẫn `_bar` và nhãn của nó.
- Nhãn chứa thanh trên màn chơi không còn xoay (dễ khớp); vệt loá trên màn chơi nhạt hơn (0,14).
