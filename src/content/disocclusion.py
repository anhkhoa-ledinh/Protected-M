"""Disocclusion fill for layered depth images (task C1).

When layers are separated in Z with no welding, moving the eye sideways slides
a near layer across a far one and **reveals area that was hidden behind it**.
That area has no pixels in the source photograph, so without treatment the
viewer sees straight through to the void clear colour.

Following Shih et al., *3D Photography using Context-aware Layered Depth
Inpainting* (CVPR 2020), the fix is to synthesise background content behind
each layer's silhouette.  This is the lightweight variant the project calls for:
OpenCV Telea inpainting over a bounded band, no learned model, run once per
image load.

Two design points matter more than the choice of inpainting algorithm:

1. **The fill is band-limited.**  Only ``fill_px`` pixels can ever be revealed
   (see ``max_reveal_px``), so filling the entire foreground silhouette would be
   wasted work.

2. **The source is restricted to background pixels.**  ``extract_layer_rgba``
   keeps full RGB at every pixel — the foreground object's colours included —
   so a naive ``cv2.inpaint`` would happily smear the foreground into the gap.
   Masking out *everything* the nearer layers cover leaves only background as
   valid source.  This is the depth-aided refinement (task C3), applied up
   front rather than as a later fix, because without it C1 produces exactly the
   artefact C3 exists to prevent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from src.content.debug_marks import (fill_colour, paint_zone,
                                     plate_fill_colour)
from src.content.lama_fill import inpaint_lama

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DisocclusionParams:
    """Knobs for the background fill behind each layer silhouette."""

    enabled: bool = True

    fill_px: int = 64
    """How far behind a silhouette to synthesise background, in texture pixels."""

    inpaint_radius: int = 5
    """Telea neighbourhood radius (OpenCV ``inpaintRadius``)."""

    feather_px: int = 8
    """Alpha ramp at the outer rim of the band, so it does not end in a hard cut."""

    fill_method: str = "telea"
    """
    PB-074: thuật toán lấp lỗ. ``"telea"`` (mặc định) = hôm nay.
    ``"fsr"`` = ``cv2.xphoto`` Rapid Frequency Selective Reconstruction, dựng
    lại vùng thiếu bằng tổ hợp thưa các thành phần tần số nên **giữ được vân**
    thay vì lan màu từ mép vào như Telea. ``"lama"`` = big-lama chạy ONNX trên
    CPU (xem ``src/content/lama_fill.py``), một lượt suy luận mỗi lớp, có cache
    ra đĩa. Giá trị lạ, hoặc LaMa không nạp được, thì rơi về ``"telea"``.
    """

    plate_fill: str = "telea"
    """
    PB-075: cách lấp riêng cho **tấm nền** (lớp xa nhất, rank 0).

    Tấm nền khác hẳn các lớp giữa: nó lấp **cả lỗ** chứ không phải một dải
    64 px, nên vùng Telea trên nó chiếm tới 31 % khung hình và mịn hơn hẳn
    xung quanh (PB-074 Phase 0b). ``"telea"`` (mặc định) = y hôm nay ·
    ``"shiftmap"`` = `cv2.xphoto` INPAINT_SHIFTMAP, chép **mảng điểm** giống
    nhất từ chỗ đã biết nên giữ được vân · ``"lama"`` = big-lama ONNX.
    Lớp giữa **không** đụng tới: chúng vẫn theo ``fill_method``.
    """

    max_work_dim: int = 640
    """
    Inpaint at this longest side, then upscale — the fill is low-frequency.

    Measured on a 2912×1632 texture: 960 costs 1625 ms, 640 costs 878 ms, and
    the two differ by 4.35/255 mean inside the band. Not worth the extra second.
    """

    fringe_dilate_px: int = 0
    """
    L3 (TASK_game_1110) — nở ``nearer_mask`` ra ngần này px TRƯỚC khi tính
    band/lỗ, quy theo ``fringe_reference_width_px``. 0 (mặc định) = hệt hôm
    nay, nên các golden test MD5 ở ``test_fill_method.py``/``test_plate_fill.py``
    không đổi một byte.

    Vì sao cần: mask của lớp gần là mặt nạ CỨNG từ k-means, còn alpha THẬT vẽ
    ra có feather vài px tràn ra ngoài mặt nạ ấy. Lớp xa coi "lỗ" đúng bằng
    ``nearer_mask`` thì để lại một vành hẹp nơi RGB của chính nó (không phải
    RGB đã lấp) nằm NGAY DƯỚI vành mờ của lớp gần — khi lớp gần trượt đi lúc
    đầu dịch, vành đó lộ ra như một quầng màu tiền cảnh bám theo đúng đường
    bao vật, tức "ghost" người dùng thấy 07/11. Nở lỗ ra trước khi tính thì
    vùng dưới feather cũng được lấp, không còn gì sống sót.
    """

    fringe_reference_width_px: int = 1280
    """``fringe_dilate_px`` được đo ở độ rộng này; ảnh khác cỡ tự quy đổi."""


@dataclass(frozen=True)
class DisocclusionReport:
    """What the fill actually did to one layer (for logging / the F1 overlay)."""

    layer_index: int
    band_px: int
    source_px: int
    elapsed_ms: float


def max_reveal_px(
    *,
    eye_distance_m: float,
    plane_z_m: float,
    layer_z_near_m: float,
    layer_z_far_m: float,
    plane_width_m: float,
    texture_width_px: int,
    head_shift_mm: float,
) -> float:
    """
    Texture pixels of background revealed between the near and far layer.

    Projects both layer planes onto the physical screen, takes the differential
    parallax, and converts back into the far layer's own texture pixels. Used to
    choose ``fill_px`` from geometry instead of guesswork.
    """
    z_near = plane_z_m + layer_z_near_m
    z_far = plane_z_m + layer_z_far_m
    if eye_distance_m <= 0.0:
        raise ValueError("eye_distance_m must be positive")

    def projection(z_layer: float) -> float:
        denom = eye_distance_m - z_layer
        if abs(denom) < 1e-9:
            raise ValueError("layer coincides with the eye plane")
        return eye_distance_m / denom

    t_near = projection(z_near)
    t_far = projection(z_far)
    # Screen-space travel per unit eye travel is (1 - t); the gap is the difference.
    relative = abs((1.0 - t_far) - (1.0 - t_near))
    screen_shift_m = relative * head_shift_mm * 0.001
    far_plane_shift_m = screen_shift_m / t_far
    return far_plane_shift_m / plane_width_m * float(texture_width_px)


def disocclusion_band(
    own_mask: npt.NDArray[np.bool_],
    nearer_mask: npt.NDArray[np.bool_],
    fill_px: int,
) -> npt.NDArray[np.bool_]:
    """
    Strip of hidden area within ``fill_px`` of this layer's silhouette.

    Only this strip can ever be revealed, so only this strip is worth keeping.
    """
    if own_mask.shape != nearer_mask.shape:
        raise ValueError("own_mask and nearer_mask shapes differ")
    radius = max(1, int(fill_px))
    # Equivalent to dilating by a disc of `radius`, but O(pixels) instead of
    # O(pixels · radius²).  cv2.dilate does not decompose an ELLIPSE kernel, so
    # at fill_px=64 it convolves a 129×129 element over the whole image and
    # dominates the entire fill — measured at ~2 s on a 2912×1632 texture.
    #
    # maskSize=5, not 3: the 3×3 chamfer approximation drifts ~2 px at radius
    # 60, which would silently widen the band. The 5×5 mask is exact at this
    # radius for 58 ms on a 2912×1632 texture (DIST_MASK_PRECISE costs 191 ms
    # for the same answer).
    outside = np.where(own_mask, np.uint8(0), np.uint8(255))
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    grown = distance <= float(radius)
    return grown & nearer_mask & ~own_mask


#: Số pixel nở mặt nạ lỗ trước ``cv2.inpaint``, ở độ phân giải làm việc.
#: Chọn bằng đo (L1b, 2026-10-13) trên 6 ảnh: xem bảng ``fill_leak_dL`` với
#: 0 / 2 / 4 px trong báo cáo phase. Giữ số nhỏ hơn trừ khi 4 px hạ rõ rệt
#: ``dL`` của t23 — ở 0.751x thì 2 px máy ≈ 2.7 px ảnh gốc.
INPAINT_HOLE_DILATE_PX = 2


def inpaint_hole(rgb, hole, radius, method="telea"):
    """
    Lấp ``hole`` (255 = điểm phải dựng lại) bằng thuật toán ``method``.

    ``cv2.xphoto`` nhận mask **ngược** với ``cv2.inpaint``: khác 0 = điểm ĐÃ
    BIẾT. Đảo ở đây, ngay cạnh lời gọi, để chỗ gọi không phải nhớ.
    """
    name = str(method).lower().strip()
    if name == "shiftmap":
        # BẪY: ``cv2.xphoto`` nhận mask NGƯỢC với ``cv2.inpaint`` — khác 0 =
        # điểm ĐÃ BIẾT. Đưa nhầm chiều thì nó đi dựng lại cả ảnh TRỪ cái lỗ
        # (đo thử: 3303 ms, và điểm ngoài lỗ cũng bị đổi).
        #
        # Chạy trong Lab chứ không phải RGB: SHIFTMAP so mảng điểm theo khoảng
        # cách màu, mà Lab mới gần cảm nhận mắt. Đo thử trên t25 640×360:
        # Lab cho fill_tex_ratio 0,958 so với RGB 0,949.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        out = np.zeros_like(lab)
        cv2.xphoto.inpaint(lab, cv2.bitwise_not(hole), out,
                           cv2.xphoto.INPAINT_SHIFTMAP)
        return cv2.cvtColor(out, cv2.COLOR_Lab2RGB)
    if name == "lama":
        # Trả None khi thiếu model / ORT hỏng: rơi xuống Telea ngay dưới, chứ
        # một khoá hình ảnh không được quyền làm app chết.
        out = inpaint_lama(rgb, hole)
        if out is not None:
            return out
        logger.warning("LaMa khong dung duoc — dung telea cho lan lap nay")
        return cv2.inpaint(rgb, hole, radius, cv2.INPAINT_TELEA)
    if name == "fsr":
        known = cv2.bitwise_not(hole)
        out = np.zeros_like(rgb)
        cv2.xphoto.inpaint(rgb, known, out, cv2.xphoto.INPAINT_FSR_FAST)
        return out
    if name not in ("telea", ""):
        logger.warning("fill_method la gi the: %s — dung telea", method)
    return cv2.inpaint(rgb, hole, radius, cv2.INPAINT_TELEA)


def _band_alpha(
    band: npt.NDArray[np.bool_],
    own_mask: npt.NDArray[np.bool_],
    feather_px: int,
) -> npt.NDArray[np.uint8]:
    """Opaque across the band, ramping down over ``feather_px`` at its outer rim."""
    alpha = np.where(band, np.uint8(255), np.uint8(0))
    if feather_px <= 0:
        return np.ascontiguousarray(alpha)
    # Gieo distance transform trên ``band ∪ own_mask``, KHÔNG phải riêng dải.
    # Gieo riêng dải thì MỌI mép đều là biên, kể cả mép áp vào đường bao của
    # chính lớp — nơi alpha riêng cũng đang được feather. ``np.maximum`` ở khối
    # C1 khi đó để lại một vệt bán trong suốt chạy dọc đường bao, mắt đọc thành
    # viền tối. Đo 2026-10-13 trên 6 ảnh: alpha thấp nhất sát đường bao là
    # 23-62 trên 255. Gộp own vào mầm thì mép trong không còn là biên, chỉ mép
    # NGOÀI của dải mới dốc — đúng ý đồ ban đầu.
    #
    # maskSize=3 is fine here — the ramp is a few pixels wide, so the chamfer
    # approximation error stays sub-pixel.
    seed = np.where(band | own_mask, np.uint8(255), np.uint8(0))
    distance = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
    ramp = np.clip(distance / float(feather_px), 0.0, 1.0)
    return np.ascontiguousarray(np.where(band, (ramp * 255.0), 0.0).astype(np.uint8))


def fill_layer_disocclusion(
    rgb: npt.NDArray[np.uint8],
    own_mask: npt.NDArray[np.bool_],
    nearer_mask: npt.NDArray[np.bool_],
    params: DisocclusionParams,
    *,
    layer_index: int = 0,
    plate: bool = False,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], DisocclusionReport]:
    """
    Synthesise background behind one layer's silhouette.

    Returns ``(rgb_with_background, band_alpha, report)``.  ``band_alpha`` is
    non-zero only inside the band, so the caller can composite it over the
    layer's existing feathered alpha without disturbing the silhouette itself.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    if own_mask.shape != rgb.shape[:2]:
        raise ValueError("mask and image sizes differ")

    started = time.perf_counter()
    if params.fringe_dilate_px > 0:
        # L3 (TASK_game_1110): ``disocclusion_band`` luôn loại ``own_mask``
        # (``& ~own_mask``) — nên chỉ nở ``nearer_mask`` không đụng gì tới
        # phần lỗ/dải, vì nó vẫn bị cổng đó chặn lại. Phải CẮT một dải khỏi
        # ``own_mask`` ngay sát biên với lớp gần rồi nhập dải đó vào phần
        # "lớp gần" — đúng nghĩa "pixel của lớp xa trong vùng nở này được coi
        # là lỗ" của kịch bản. Dải ấy chính là nơi feather của lớp gần (vài
        # px rộng) tràn ra ngoài mặt nạ cứng của nó mà RGB thật của ảnh ở đó
        # vẫn mang màu pha trộn tiền cảnh — "foreground fringe" của kịch bản.
        width = rgb.shape[1]
        fringe_px = max(1, round(
            params.fringe_dilate_px * width
            / max(int(params.fringe_reference_width_px), 1)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * fringe_px + 1, 2 * fringe_px + 1))
        grown_nearer = cv2.dilate(
            np.where(nearer_mask, np.uint8(255), np.uint8(0)), kernel) > 0
        fringe_zone = grown_nearer & own_mask
        own_mask = own_mask & ~fringe_zone
        nearer_mask = nearer_mask | fringe_zone
    band = disocclusion_band(own_mask, nearer_mask, params.fill_px)
    # C3: mask out EVERYTHING the nearer layers cover, not just the band.  The
    # inpainter may then only read background pixels, so foreground colour
    # cannot bleed into the revealed gap.
    hole = nearer_mask & ~own_mask
    empty = np.zeros(rgb.shape[:2], dtype=np.uint8)
    # Chế độ plate lấp CẢ lỗ, nên cổng thoát sớm phải xét lỗ chứ không xét dải:
    # dải rỗng mà lỗ vẫn còn thì tấm nền vẫn phải được lấp.
    if not bool((hole if plate else band).any()):
        return (
            np.ascontiguousarray(rgb),
            empty,
            DisocclusionReport(layer_index, 0, 0, 0.0),
        )

    source_px = int((~hole).sum())
    if source_px == 0:
        return (
            np.ascontiguousarray(rgb),
            empty,
            DisocclusionReport(layer_index, int(band.sum()), 0, 0.0),
        )

    height, width = rgb.shape[:2]
    longest = max(height, width)
    scale = min(1.0, float(params.max_work_dim) / float(longest))
    if scale < 1.0:
        small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        small_rgb = cv2.resize(rgb, small_size, interpolation=cv2.INTER_AREA)
        small_hole = cv2.resize(
            np.where(hole, np.uint8(255), np.uint8(0)), small_size,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        small_rgb = np.ascontiguousarray(rgb)
        small_hole = np.where(hole, np.uint8(255), np.uint8(0))

    # Nở lỗ trước khi inpaint, tính ở ĐỘ PHÂN GIẢI LÀM VIỆC. ``INTER_AREA`` lấy
    # trung bình nên điểm sát biên trong ``small_rgb`` là hỗn hợp nền + vật,
    # còn mặt nạ đi qua ``INTER_NEAREST`` nên không co theo cùng kiểu: những
    # điểm lai đó nằm NGOÀI lỗ và Telea coi là nguồn hợp lệ, kéo màu vật vào
    # trong. Nở lỗ ra thì chúng được tổng hợp lại thay vì làm nguồn.
    #
    # CHỈ cho tấm nền. Lớp khác chỉ chép lại dải nên nở lỗ không giúp gì mà còn
    # làm Telea tổng hợp khác đi ngay sát mép dải: đo 2026-10-13 thấy ba lớp
    # không phải plate xấu đi (t11 L2 1.10→7.69, t24 L1 2.95→4.85,
    # t25 L1 2.59→4.57). Giữ chúng ở hành vi cũ.
    if plate and INPAINT_HOLE_DILATE_PX > 0:
        small_hole = cv2.dilate(
            small_hole,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=INPAINT_HOLE_DILATE_PX,
        )
    radius = max(1, int(params.inpaint_radius))
    # PB-075: tấm nền có khoá riêng. Lớp giữa chỉ lấp một dải 64 px nên Telea
    # ở đó gần như không thấy; tấm nền mới là chỗ vùng lấp chiếm tới 31 % khung.
    method = str(params.plate_fill if plate else params.fill_method)
    filled_small = inpaint_hole(small_rgb, small_hole, radius, method)
    filled = (
        cv2.resize(filled_small, (width, height), interpolation=cv2.INTER_LINEAR)
        if scale < 1.0
        else filled_small
    )

    # Lớp xa nhất là tấm nền: sau nó không còn gì, nên chép kết quả Telea vào
    # CẢ lỗ và để alpha đục trên cả lỗ. Dùng lại chính ``filled`` ở trên, không
    # inpaint lần thứ hai. Các lớp khác giữ dải như cũ — tấm nền phía sau chúng
    # giờ đã liền.
    # Tấm nền chép về CẢ VÀNH quanh lỗ, không chỉ đúng lỗ. Điểm ngay ngoài lỗ
    # là của chính tấm nền nhưng là hỗn hợp vật + nền (mép anti-alias, mép depth
    # lệch vài px): không được inpaint vì ngoài lỗ, cũng không bị thay vì chỉ
    # ``zone`` được chép — nên vật trượt đi rồi mà viền 1-3 px của nó vẫn nằm
    # đó. Vành này bị mép feather của lớp gần che ở vị trí mắt giữa, nên thay
    # bằng nền là đúng. Nở ở ĐỘ PHÂN GIẢI GỐC, quy đổi từ số px đã nở lúc nhỏ.
    if plate:
        grow = int(np.ceil(INPAINT_HOLE_DILATE_PX / max(scale, 1e-6)))
        zone = cv2.dilate(
            np.where(hole, np.uint8(255), np.uint8(0)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=max(0, grow),
        ) > 0
    else:
        zone = band
    out = np.ascontiguousarray(rgb, dtype=np.uint8).copy()
    out[zone] = filled[zone]
    # PB-074 (đo): tô vùng vừa lấp để đếm được nó trên khung hình. Không đặt
    # HCPD_MARK_FILL thì ``paint_zone`` trả nguyên ``out``.
    # PB-075b: tấm nền có màu riêng nếu đặt HCPD_MARK_PLATE, để tách được
    # "vùng lấp nhìn thấy" thành phần của tấm nền và phần của dải lớp giữa.
    paint_zone(out, zone, plate_fill_colour() if plate else fill_colour())
    alpha = (
        np.where(zone, np.uint8(255), np.uint8(0))
        if plate
        else _band_alpha(band, own_mask, params.feather_px)
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    report = DisocclusionReport(
        layer_index=layer_index,
        band_px=int(band.sum()),
        source_px=source_px,
        elapsed_ms=elapsed_ms,
    )
    logger.info(
        "Disocclusion fill L%d: band=%dpx source=%dpx work=%dx%d %.0fms",
        layer_index,
        report.band_px,
        source_px,
        small_rgb.shape[1],
        small_rgb.shape[0],
        elapsed_ms,
    )
    return out, alpha, report
