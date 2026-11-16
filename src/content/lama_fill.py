"""
PB-074 Phase 5 — lấp khe hở bằng **big-lama** (LaMa, Suvorov et al. 2022) chạy
trên `onnxruntime` CPU đã cài sẵn.

Model: `assets/models/lama_fp32.onnx` (bản xuất ONNX của Carve, Apache-2.0,
208 MB, sha256 `1faef530…`). Vào `image` [1,3,512,512] RGB 0–1 và `mask`
[1,1,512,512] (1 = điểm phải dựng lại); ra `[1,3,512,512]` **đã ở thang 0–255**.

Ô vào cố định 512×512, nên **mỗi lớp đúng MỘT lượt suy luận**: ảnh ở cỡ làm
việc (≤ 640 px cạnh dài) được thu vừa khung 512 giữ tỉ lệ, đệm bằng
`BORDER_REPLICATE` cho đủ ô, chạy một lần, rồi cắt và phóng trả về.

Thiếu model hay `onnxruntime` thì mọi hàm ở đây trả ``None`` và bên gọi tự
quay về Telea — không bao giờ làm app chết vì một khoá hình ảnh.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "assets" / "models" / "lama_fp32.onnx"
CACHE_DIR = ROOT / "artifacts" / "lama_cache"
TILE = 512
#: Theo kịch bản PB-074: 2 luồng trong một phép, 1 phép một lúc.
INTRA_THREADS, INTER_THREADS = 2, 1

_session = None
_session_failed = False
_preload_thread = None

#: PB-076: khoá cho việc dựng phiên. Thiếu nó thì luồng nền (preload) và luồng
#: GL (lúc cần lấp) cùng qua được cổng ``if _session is not None`` rồi **dựng
#: hai phiên**: đo ngày 01/11 trong `D:\hcpd_demo` thấy đúng hai dòng
#: "LaMa ONNX loaded" (80,0 s + 63,3 s) và lượt nạp ảnh kéo lên 118 s.
_session_lock = threading.Lock()

#: Windows: THREAD_PRIORITY_LOWEST.
_THREAD_LOWEST = -2


def _get_session():
    """
    Phiên ORT dùng chung, dựng **đúng một lần** dù có mấy luồng cùng hỏi.

    Kiểm hai lần quanh khoá: lần ngoài để đường nóng (đã có phiên) không phải
    lấy khoá, lần trong để hai luồng cùng vào thì chỉ một luồng dựng.
    """
    global _session, _session_failed
    if _session is not None or _session_failed:
        return _session
    with _session_lock:
        if _session is not None or _session_failed:
            return _session
        return _load_session_locked()


def _load_session_locked():
    """Dựng phiên; **chỉ gọi khi đang giữ** ``_session_lock``."""
    global _session, _session_failed
    try:
        import onnxruntime as ort

        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        options = ort.SessionOptions()
        options.intra_op_num_threads = INTRA_THREADS
        options.inter_op_num_threads = INTER_THREADS
        t0 = time.perf_counter()
        _session = ort.InferenceSession(str(MODEL_PATH), options,
                                        providers=["CPUExecutionProvider"])
        logger.info("LaMa ONNX loaded in %.0f ms (%s)",
                    (time.perf_counter() - t0) * 1000.0, MODEL_PATH.name)
    except Exception as err:                    # thiếu file, thiếu ORT, model hỏng
        logger.warning("LaMa khong nap duoc (%s) — quay ve telea", err)
        _session_failed = True
        _session = None
    return _session


def session_ready() -> bool:
    """``True`` khi phiên ORT đã nạp xong — dùng để log và để test."""
    return _session is not None


def preload_session_async() -> bool:
    """
    Nạp phiên ORT ở **luồng nền, ưu tiên thấp nhất**. Trả ``False`` nếu không
    cần (đã nạp, đã hỏng, hoặc đang nạp dở).

    Gọi **sau khi cảnh đầu đã hiện**, không bao giờ trước: ảnh đầu tiên không
    được chậm đi vì việc này. Ảnh mở sau đó (F6) nhờ vậy chỉ phải trả ~5,5 s
    mỗi lớp thay vì cộng thêm ~42 s nạp model.

    Ưu tiên thấp nhất vì máy chỉ có 2 nhân: luồng tracking phải thắng khi
    tranh nhau (cùng bài học của cổng SPIKE-12, PB-065).
    """
    global _preload_thread
    if _session is not None or _session_failed:
        return False
    if _preload_thread is not None and _preload_thread.is_alive():
        return False

    def _work():
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, _THREAD_LOWEST)
        except AttributeError:
            pass                                    # không phải Windows
        t0 = time.perf_counter()
        logger.info("LaMa preload: bat dau nap phien ORT o luong nen")
        _get_session()
        logger.info("LaMa preload: xong sau %.1f s (san sang=%s)",
                    time.perf_counter() - t0, session_ready())

    _preload_thread = threading.Thread(target=_work, name="lama-preload",
                                       daemon=True)
    _preload_thread.start()
    return True


def _fit_into_tile(rgb, hole):
    """
    Thu ảnh + mask vào ô 512×512 **giữ tỉ lệ**, đệm bằng điểm biên.

    Đệm bằng ``BORDER_REPLICATE`` chứ không phải màu đen: lớp FFC của LaMa đọc
    toàn ảnh làm ngữ cảnh, một viền đen rộng sẽ kéo cả vùng lấp tối đi. Mask ở
    phần đệm để 0 nên model không coi đó là chỗ phải dựng lại.
    """
    height, width = rgb.shape[:2]
    scale = min(TILE / float(width), TILE / float(height))
    new_w = max(1, min(TILE, int(round(width * scale))))
    new_h = max(1, min(TILE, int(round(height * scale))))
    small_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_hole = cv2.resize(hole, (new_w, new_h),
                            interpolation=cv2.INTER_NEAREST)
    pad_r, pad_b = TILE - new_w, TILE - new_h
    tile_rgb = cv2.copyMakeBorder(small_rgb, 0, pad_b, 0, pad_r,
                                  cv2.BORDER_REPLICATE)
    tile_hole = cv2.copyMakeBorder(small_hole, 0, pad_b, 0, pad_r,
                                   cv2.BORDER_CONSTANT, value=0)
    return tile_rgb, tile_hole, (new_w, new_h)


def _run(tile_rgb, tile_hole):
    """Một lượt suy luận; trả ảnh 512×512 uint8, hoặc ``None`` nếu hỏng."""
    session = _get_session()
    if session is None:
        return None
    mask = (tile_hole > 0).astype(np.float32)[None, None]
    image = tile_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    # Xoá sẵn phần lỗ: bản gốc cũng nhân (1 − mask) ở trong, làm ở đây thì
    # đúng với cả hai cách xuất model, và không bao giờ rò ảnh cũ vào chỗ lỗ.
    image = image * (1.0 - mask)
    try:
        out = session.run(["output"], {"image": image, "mask": mask})[0]
    except Exception as err:
        logger.warning("LaMa chay loi (%s) — quay ve telea", err)
        return None
    return np.clip(out[0].transpose(1, 2, 0), 0.0, 255.0).astype(np.uint8)


def _cache_key(rgb, hole) -> str:
    """Khoá băm theo đúng nội dung đưa vào, nên hai ảnh khác nhau không đụng."""
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(rgb).tobytes())
    digest.update(np.ascontiguousarray(hole > 0).tobytes())
    digest.update(f"{TILE}".encode())
    return digest.hexdigest()


def inpaint_lama(rgb, hole, use_cache: bool = True):
    """
    Lấp ``hole`` (≠ 0 = điểm phải dựng lại) bằng LaMa. ``None`` = không chạy được.

    Kết quả cache ra đĩa theo nội dung: dựng lại cùng một ảnh thì lần sau không
    tốn lượt suy luận nào (kịch bản PB-074 yêu cầu cache theo ảnh).
    """
    key = _cache_key(rgb, hole)
    cached = CACHE_DIR / f"{key}.png"
    if use_cache and cached.exists():
        out = cv2.imread(str(cached), cv2.IMREAD_COLOR)
        if out is not None and out.shape[:2] == rgb.shape[:2]:
            logger.info("LaMa cache hit: %s", cached.name)
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    tile_rgb, tile_hole, (new_w, new_h) = _fit_into_tile(rgb, hole)
    t0 = time.perf_counter()
    filled_tile = _run(tile_rgb, tile_hole)
    if filled_tile is None:
        return None
    height, width = rgb.shape[:2]
    out = cv2.resize(filled_tile[:new_h, :new_w], (width, height),
                     interpolation=cv2.INTER_LINEAR)
    logger.info("LaMa fill: %dx%d -> o %d, %.0f ms", width, height, TILE,
                (time.perf_counter() - t0) * 1000.0)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cached), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out
