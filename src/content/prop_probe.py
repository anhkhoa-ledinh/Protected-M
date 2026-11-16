"""
Vật thử chẩn đoán cho PB-015 C2. Không phải đường chơi; bật bằng
``HCPD_PROP_PROBE``, mặc định tắt.

Giữ lại vì đây là thứ duy nhất phân biệt được "mô hình che sai" với
"renderer sai": báo cáo đầu tiên đổ lỗi cho renderer, và chính đường này
bác bỏ điều đó — quả đỏ sau tấm nền đục biến mất đúng như phải thế.
"""

from __future__ import annotations


#: Ba màu phân biệt cho vật thử, theo thứ tự #0, #1, #2.
PROBE_COLORS = ((0.95, 0.15, 0.15), (0.20, 0.95, 0.25), (0.25, 0.45, 1.00))


def probe_at(props):
    """
    Vật thử đặt ĐÚNG vị trí vật thật, chỉ đổi màu để phân biệt từng quả.

    Dùng khi mô hình che nói "bị che" mà mắt lại thấy: cần biết quả nào hiện
    thì mới khoanh được chỗ mô hình sai.
    """
    return [{**p, "color": PROBE_COLORS[i % len(PROBE_COLORS)]}
            for i, p in enumerate(props)]


def probe_props(planes):
    """
    Ba vật thử ở độ sâu BIẾT TRƯỚC, không qua ``place_props`` — chẩn đoán C2.

    Hình học nói ba vật của t22 phải bị che 92–100% ở mọi vị trí mắt đo được,
    nhưng trên máy chủ sở hữu cả ba đều hiện. Ba độ sâu này tách bạch nguyên
    nhân: sau tấm nền đục 100% thì KHÔNG thể thấy nếu depth test chạy đúng;
    trước kính thì LUÔN phải thấy, nên nó là đối chứng chứng minh cầu có vẽ.

    Mỗi quả một MÀU và một góc riêng: lần chạy trước không phân biệt được quả
    nào đã biến mất, mà chính điều đó quyết định chẩn đoán.
    """
    plate_mm = planes[0].z_m * 1000.0
    gap_mm = (planes[0].z_m + planes[1].z_m) * 500.0 if len(planes) > 1 else -200.0
    spec = ((-120.0, 60.0, plate_mm - 100.0),   # ĐỎ, sau tấm nền đục
            (0.0, 0.0, gap_mm),                 # LỤC, giữa hai lớp
            (120.0, -60.0, 50.0))               # LAM, trước kính
    return [{"pos_mm": d, "radius_mm": 0.0, "reveal_axis": "-",
             "reveal_450_mm": 0.0, "reveal_330_mm": None, "color": PROBE_COLORS[i]}
            for i, d in enumerate(spec)]
