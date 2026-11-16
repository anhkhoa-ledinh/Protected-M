r"""
TASK_game_1110 L1 — **nướng** bộ lớp trong một tiến trình con, có cache ra đĩa.

Vì sao phải là tiến trình con, không phải luồng: hai lượt chơi của người dùng
chết ngay giữa lúc dựng lớp (23:01 và 01:04 ngày 07/11), mã thoát
−805306369, không có traceback Python. Cả hai lần luồng GL đang chờ phiên
onnxruntime mà luồng nền vừa nạp xong. Bỏ LaMa đi thì hết lỗi, nhưng cũng mất
luôn cách lấp khe duy nhất không bị nhoè — và chính vết nhoè Telea là "khối
mèo xám" mà người dùng thấy.

Cách ở đây giữ được cả hai: **không luồng nào của app đụng onnxruntime nữa**.
Depth, cụm lớp và lấp khe chạy hết trong một tiến trình ``spawn`` riêng, ghi
kết quả ra ``assets/cache/maps/<sha1>/``, rồi chết. Luồng GL chỉ đọc PNG.

Khoá cache = sha1 của (nội dung file ảnh + bộ khoá cấu hình có ảnh hưởng tới
bước nướng). Mở lại cùng ảnh với cùng cấu hình là **không tính lại gì cả**.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: Phiên bản định dạng cache. Tăng số này là mọi thư mục cũ thành vô hiệu —
#: cần khi đổi cách ghi, chứ không phải khi đổi một tham số (tham số đã nằm
#: trong ``settings_hash``).
CACHE_FORMAT = 3

#: Cạnh vào của model depth. Depth Anything small chạy ở 518 (``size`` của
#: ``preprocessor_config.json``, giữ tỉ lệ, bội số 14). Ghi hẳn ra đây để số
#: trong báo cáo có chỗ bám, và để đổi được mà không phải sửa processor.
DEPTH_INPUT_SIZE = 518

#: Những khoá ``ContentSettings`` **thật sự** đổi kết quả nướng. Khoá nào chỉ
#: ảnh hưởng lúc vẽ (``layer_z_far_m``, overscan, relief, mist, màu void…)
#: KHÔNG nằm ở đây: đổi chúng thì cache vẫn dùng lại được, và đó chính là thứ
#: làm lượt quét z của L4 chạy nổi.
BAKE_KEYS = (
    "depth_model", "depth_backend",
    "layer_clustering_enabled", "layer_count", "layer_method",
    "adaptive_k_enabled", "adaptive_k_min", "adaptive_k_max", "k_policy",
    "gvf_threshold", "layer_seed",
    "layer_feather_dilate_px", "layer_feather_blur_px",
    "layer_rim_px", "rim_light_intensity", "rim_light_near_only",
    "edge_feather_px", "color_bleed_px", "edge_style",
    "edge_guided_refine", "edge_guided_radius_px", "edge_guided_feather_px",
    "atmospheric_enabled", "atmosphere_blend_max", "far_desaturation",
    "near_contrast_boost", "far_contrast_reduce", "far_fog_mode",
    "semantic_depth_enabled", "semantic_depth_bands",
    "semantic_depth_similarity", "semantic_depth_separation",
    "semantic_separation_min_horizontal_span",
    "semantic_max_segment_variance", "semantic_min_segment_pixels",
    "semantic_subdivide_pixels", "semantic_max_work_dim",
    "disocclusion_fill_enabled", "disocclusion_fill_px",
    "disocclusion_inpaint_radius", "disocclusion_feather_px",
    "disocclusion_max_work_dim", "disocclusion_fringe_dilate_px",
    "fill_method", "plate_fill",
)

#: Biến môi trường đánh dấu vùng (PB-074/075) đổi **pixel** của lớp, nên chúng
#: phải vào khoá cache; nếu không, lượt đo đánh dấu sẽ ăn cache của lượt thường.
MARK_ENVS = ("HCPD_MARK_PAD", "HCPD_MARK_FILL", "HCPD_MARK_TOP",
             "HCPD_MARK_PLATE")


@dataclass
class BakedLayer:
    """Một lớp đọc lại từ cache — đủ trường cho đường vẽ của ``_apply_layers``."""

    index: int
    mean_depth: float
    depth_lo: float
    depth_hi: float
    coverage: float
    depth_norm: float
    hole_pct: float
    rgba: npt.NDArray[np.uint8]
    mask: npt.NDArray[np.bool_]
    z_m: float = 0.0            # đặt lại lúc vẽ, không nằm trong cache
    grade_report: object = None  # LayerGradeReport, dựng lại từ meta.json


@dataclass
class BakeResult:
    """Kết quả một lượt nướng (dù vừa tính hay vừa đọc cache)."""

    layers: list[BakedLayer]
    depth: npt.NDArray[np.float32]
    k: int
    mist_color: tuple[float, float, float]
    bake_s: float
    cache_dir: Path
    from_cache: bool = False
    timing_ms: dict = field(default_factory=dict)


def settings_hash(content) -> str:
    """sha1 của đúng những khoá có ảnh hưởng tới bước nướng (+ cờ đánh dấu)."""
    payload = {key: _plain(getattr(content, key, None)) for key in BAKE_KEYS}
    payload["_format"] = CACHE_FORMAT
    payload["_depth_input"] = DEPTH_INPUT_SIZE
    for env in MARK_ENVS:
        value = os.environ.get(env)
        if value:
            payload[env] = value
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _plain(value):
    """Đưa giá trị về dạng JSON ghi được, giữ nguyên thứ tự."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def image_hash(path: Path) -> str:
    """sha1 **nội dung** file ảnh: đổi tên file không làm hỏng cache."""
    digest = hashlib.sha1()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_dir_for(root: Path, image_sha: str, cfg_sha: str) -> Path:
    """``assets/cache/maps/<sha1>/`` — sha1 gộp cả ảnh lẫn cấu hình."""
    joined = hashlib.sha1(f"{image_sha}:{cfg_sha}".encode()).hexdigest()
    return Path(root) / "assets" / "cache" / "maps" / joined


def _write_cache(cache_dir: Path, result, depth, meta) -> None:
    """Ghi lớp + depth + meta. Ghi vào thư mục tạm rồi đổi tên: lượt đọc song
    song không bao giờ thấy một thư mục dở dang."""
    tmp = cache_dir.with_name(cache_dir.name + ".tmp")
    if tmp.exists():
        for item in tmp.iterdir():
            item.unlink()
        tmp.rmdir()
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "depth.npy", np.asarray(depth, dtype=np.float32))
    for layer in result.layers:
        idx = int(layer.index)
        cv2.imwrite(str(tmp / f"layer{idx:02d}.png"),
                    cv2.cvtColor(layer.rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(tmp / f"mask{idx:02d}.png"),
                    np.where(layer.mask, np.uint8(255), np.uint8(0)))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1),
                                   encoding="utf-8")
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            item.unlink()
        cache_dir.rmdir()
    tmp.rename(cache_dir)


def load_from_cache(cache_dir: Path) -> BakeResult | None:
    """Đọc lại một lượt nướng. ``None`` nếu thiếu hoặc hỏng."""
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("format", -1)) != CACHE_FORMAT:
            return None
        depth = np.load(Path(cache_dir) / "depth.npy")
        layers = []
        for record in meta["layers"]:
            idx = int(record["index"])
            bgra = cv2.imread(str(Path(cache_dir) / f"layer{idx:02d}.png"),
                              cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(Path(cache_dir) / f"mask{idx:02d}.png"),
                              cv2.IMREAD_GRAYSCALE)
            if bgra is None or mask is None or bgra.shape[2] != 4:
                return None
            layers.append(BakedLayer(
                index=idx, mean_depth=float(record["mean_depth"]),
                depth_lo=float(record["depth_lo"]),
                depth_hi=float(record["depth_hi"]),
                coverage=float(record["coverage"]),
                depth_norm=float(record["depth_norm"]),
                hole_pct=float(record.get("hole_pct", 0.0)),
                rgba=np.ascontiguousarray(
                    cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)),
                mask=np.ascontiguousarray(mask > 127),
                grade_report=_grade_from(record.get("grade"))))
        if not layers or layers[0].rgba.shape[:2] != depth.shape:
            return None
        return BakeResult(
            layers=layers, depth=np.ascontiguousarray(depth, dtype=np.float32),
            k=int(meta["k"]), mist_color=tuple(meta["mist_color"]),
            bake_s=float(meta.get("bake_s", 0.0)), cache_dir=Path(cache_dir),
            from_cache=True, timing_ms=dict(meta.get("timing_ms", {})))
    except Exception:
        logger.warning("Cache lop hong o %s — nuong lai", cache_dir,
                       exc_info=True)
        return None


def bake(image_path: Path, content, root: Path, *,
         use_cache: bool = True) -> BakeResult:
    """
    Nướng (hoặc đọc cache) bộ lớp của một ảnh. **Không đụng GL.**

    Chạy được cả ở tiến trình con lẫn trong test; chính nó là toàn bộ phần
    nặng của lượt nạp ảnh.
    """
    from src.content.depth_estimator import DepthEstimator
    from src.content.image_source import ImageSource
    from src.content.layer_clustering import cluster_into_layers
    from src.content.adaptive_k import select_optimal_k

    image_path = Path(image_path)
    cfg_sha = settings_hash(content)
    img_sha = image_hash(image_path)
    cache_dir = cache_dir_for(root, img_sha, cfg_sha)
    if use_cache:
        hit = load_from_cache(cache_dir)
        if hit is not None:
            logger.info("Cache lop TRUNG: %s (%s)", image_path.name,
                        cache_dir.name[:12])
            return hit

    t0 = time.perf_counter()
    rgb = ImageSource(image_path).tick(0.0)
    if rgb is None:
        raise ValueError(f"khong doc duoc anh: {image_path}")
    estimator = DepthEstimator(content.depth_model, content.depth_backend)
    t_depth0 = time.perf_counter()
    depth = estimator.estimate(rgb)
    depth_ms = (time.perf_counter() - t_depth0) * 1000.0

    if bool(content.adaptive_k_enabled):
        n_clusters, _report = select_optimal_k(
            depth, k_min=int(content.adaptive_k_min),
            k_max=int(content.adaptive_k_max), seed=int(content.layer_seed),
            k_policy=str(content.k_policy),
            gvf_threshold=float(content.gvf_threshold),
            log_silhouette_comparison=bool(content.log_silhouette_comparison))
    else:
        n_clusters = max(1, int(content.layer_count))

    result = cluster_into_layers(
        rgb, depth, n_clusters=n_clusters,
        z_near_m=content.layer_z_near_m, z_far_m=content.layer_z_far_m,
        seed=int(content.layer_seed),
        feather_dilate_px=int(content.layer_feather_dilate_px),
        feather_blur_px=int(content.layer_feather_blur_px),
        rim_light_px=int(content.layer_rim_px),
        rim_intensity=float(content.rim_light_intensity),
        staging=_staging(content), semantic=_semantic(content),
        edge=_edge(content), disocclusion=_disocclusion(content),
        layer_method=str(content.layer_method))

    holes = hole_fractions(result.layers)
    bake_s = time.perf_counter() - t0
    meta = {
        "format": CACHE_FORMAT, "image": image_path.name,
        "image_sha1": img_sha, "settings_sha1": cfg_sha,
        "depth_input_size": DEPTH_INPUT_SIZE,
        "k": int(n_clusters), "bake_s": round(bake_s, 2),
        "mist_color": [float(v) for v in result.mist_color],
        "timing_ms": {"depth": round(depth_ms, 1),
                      **{k: round(float(v), 1)
                         for k, v in result.timing_ms.items()}},
        "layers": [{"index": int(layer.index),
                    "mean_depth": float(layer.mean_depth),
                    "depth_lo": float(layer.depth_lo),
                    "depth_hi": float(layer.depth_hi),
                    "coverage": float(layer.coverage),
                    "depth_norm": float(layer.depth_norm),
                    "hole_pct": round(holes[i], 3),
                    "grade": _grade_dict(layer.grade_report)}
                   for i, layer in enumerate(result.layers)],
    }
    _write_cache(cache_dir, result, depth, meta)
    logger.info("Nuong xong %s: K=%d, %.1f s -> %s", image_path.name,
                n_clusters, bake_s, cache_dir.name[:12])
    hit = load_from_cache(cache_dir)
    if hit is None:
        raise RuntimeError(f"ghi cache xong ma doc lai khong duoc: {cache_dir}")
    hit.from_cache = False
    return hit


def result_from_cluster(cluster_result, depth, k: int, *, bake_s: float = 0.0,
                        cache_dir: Path | None = None) -> BakeResult:
    """
    Gói kết quả ``cluster_into_layers`` thành ``BakeResult`` **không qua đĩa**.

    Dùng cho test đường vẽ: chúng cần một bộ lớp thật để đưa vào
    ``_apply_bake``, nhưng không có model depth và không nên đụng cache.
    """
    holes = hole_fractions(cluster_result.layers)
    layers = [BakedLayer(
        index=int(layer.index), mean_depth=float(layer.mean_depth),
        depth_lo=float(layer.depth_lo), depth_hi=float(layer.depth_hi),
        coverage=float(layer.coverage), depth_norm=float(layer.depth_norm),
        hole_pct=holes[i], rgba=layer.rgba, mask=layer.mask,
        z_m=float(layer.z_m), grade_report=layer.grade_report)
        for i, layer in enumerate(cluster_result.layers)]
    return BakeResult(layers=layers,
                      depth=np.ascontiguousarray(depth, dtype=np.float32),
                      k=int(k), mist_color=cluster_result.mist_color,
                      bake_s=float(bake_s), cache_dir=cache_dir or Path("."),
                      timing_ms=dict(cluster_result.timing_ms))


def _grade_dict(report):
    """``LayerGradeReport`` → dict JSON; ``None`` giữ nguyên ``None``."""
    if report is None:
        return None
    return {"index": int(report.index), "depth_norm": float(report.depth_norm),
            "atmosphere_blend": float(report.atmosphere_blend),
            "saturation_factor": float(report.saturation_factor),
            "contrast_factor": float(report.contrast_factor),
            "enabled": bool(report.enabled),
            "complement_t": float(report.complement_t)}


def _grade_from(record):
    """Dựng lại ``LayerGradeReport`` để lớp phủ F1 vẫn in đủ dòng như cũ."""
    if not record:
        return None
    from src.content.layer_clustering import LayerGradeReport

    return LayerGradeReport(**record)


def hole_fractions(layers) -> list[float]:
    """% lỗ của từng lớp: điểm bị lớp gần hơn che mà chính nó không có."""
    out = []
    for rank, layer in enumerate(layers):
        nearer = np.zeros_like(layer.mask)
        for other in layers[rank + 1:]:
            nearer |= other.mask
        out.append(100.0 * float((nearer & ~layer.mask).mean()))
    return out


def _staging(content):
    from src.content.layer_clustering import LayerStagingParams

    return LayerStagingParams(
        enabled=bool(content.atmospheric_enabled),
        atmosphere_blend_max=float(content.atmosphere_blend_max),
        far_desaturation=float(content.far_desaturation),
        near_contrast_boost=float(content.near_contrast_boost),
        far_contrast_reduce=float(content.far_contrast_reduce),
        far_fog_mode=str(content.far_fog_mode))


def _semantic(content):
    from src.content.semantic_depth import SemanticDepthParams

    return SemanticDepthParams(
        enabled=bool(content.semantic_depth_enabled),
        depth_band_count=int(content.semantic_depth_bands),
        similarity_threshold=float(content.semantic_depth_similarity),
        separation_offset=float(content.semantic_depth_separation),
        min_separation_horizontal_span=float(
            content.semantic_separation_min_horizontal_span),
        max_segment_variance=float(content.semantic_max_segment_variance),
        min_segment_pixels=int(content.semantic_min_segment_pixels),
        subdivide_above_pixels=int(content.semantic_subdivide_pixels),
        max_work_dim=int(content.semantic_max_work_dim))


def _edge(content):
    from src.content.layer_clustering import LayerEdgeParams

    return LayerEdgeParams(
        rim_light_intensity=float(content.rim_light_intensity),
        edge_feather_px=int(content.edge_feather_px),
        color_bleed_px=int(content.color_bleed_px),
        rim_light_near_only=bool(content.rim_light_near_only),
        edge_style=str(content.edge_style),
        guided_refine=bool(content.edge_guided_refine),
        guided_radius_px=int(content.edge_guided_radius_px),
        guided_feather_px=int(content.edge_guided_feather_px))


def _disocclusion(content):
    from src.content.disocclusion import DisocclusionParams

    return DisocclusionParams(
        enabled=bool(content.disocclusion_fill_enabled),
        fill_px=int(content.disocclusion_fill_px),
        inpaint_radius=int(content.disocclusion_inpaint_radius),
        feather_px=int(content.disocclusion_feather_px),
        max_work_dim=int(content.disocclusion_max_work_dim),
        fringe_dilate_px=int(content.disocclusion_fringe_dilate_px),
        fill_method=str(content.fill_method),
        plate_fill=str(content.plate_fill))


# ────────────────────────── tiến trình con ──────────────────────────────────

def _child_main(image_path: str, content_dict: dict, root: str,
                mark_env: dict, out: mp.Queue) -> None:
    """
    Thân của tiến trình con. **Hàm cấp module** — ``spawn`` phải pickle được nó.

    Trả về qua hàng đợi đúng một chuỗi: đường dẫn thư mục cache, hoặc lời báo
    lỗi. Mảng ảnh KHÔNG đi qua ống: chúng đã nằm trên đĩa, và một bộ lớp
    2048 px nặng ~50 MB mỗi lớp.
    """
    try:
        import logging as _logging

        _logging.basicConfig(level=_logging.INFO,
                             format="[bake] %(levelname)s %(message)s")
        os.environ.update({k: v for k, v in mark_env.items() if v})
        from src.config.schema import ContentSettings

        content = ContentSettings.from_dict(content_dict)
        result = bake(Path(image_path), content, Path(root))
        out.put(("ok", str(result.cache_dir), result.bake_s))
    except BaseException as err:                      # noqa: BLE001
        import traceback

        out.put(("err", f"{type(err).__name__}: {err}",
                 traceback.format_exc()))


class BakeWorker:
    """
    Nướng lớp ngoài tiến trình, một việc một lúc. Cùng API như ``DepthWorker``
    để ``ContentPlaneRenderer`` đổi sang mà không phải viết lại vòng nhịp.

    Mỗi việc một tiến trình **mới rồi chết**: nạp model mất ~10 s một lần, còn
    đổi lại là mọi thứ onnxruntime/torch cấp phát đều được trả về cho hệ điều
    hành. Trên máy 2 nhân, RAM commit cạn từng làm app chết giữa lúc dựng lớp.
    """

    #: Quá ngần này giây không xong thì coi như hỏng và giết tiến trình con.
    TIMEOUT_S = 900.0

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._ctx = mp.get_context("spawn")
        self._proc: mp.process.BaseProcess | None = None
        self._queue: mp.Queue | None = None
        self._job: tuple[str, Path, dict] | None = None
        self._started_s = 0.0
        self._ready: dict[str, BakeResult] = {}
        self._failed = False
        self._last_bake_s = 0.0

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def last_bake_s(self) -> float:
        return self._last_bake_s

    def start(self) -> None:
        """Có để gọi được như ``DepthWorker``; tiến trình mở theo từng việc."""

    def submit(self, key: str, path: Path, content) -> None:
        """Xin nướng ảnh ``path``. Cache trúng thì xong ngay, không mở gì."""
        if key in self._ready or (self._job is not None and self._job[0] == key):
            return
        from dataclasses import asdict, is_dataclass

        cache_dir = cache_dir_for(self._root, image_hash(Path(path)),
                                  settings_hash(content))
        hit = load_from_cache(cache_dir)
        if hit is not None:
            logger.info("Cache lop TRUNG (khong mo tien trinh): %s",
                        Path(path).name)
            self._ready[key] = hit
            self._last_bake_s = 0.0
            return
        payload = asdict(content) if is_dataclass(content) else dict(content)
        self._stop_proc()
        self._queue = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_child_main,
            args=(str(path), payload, str(self._root),
                  {k: os.environ.get(k, "") for k in MARK_ENVS}, self._queue),
            name="hcpd-bake", daemon=True)
        self._proc.start()
        self._job = (key, Path(path), payload)
        self._started_s = time.perf_counter()
        logger.info("Nuong lop o tien trinh con pid=%s: %s",
                    self._proc.pid, Path(path).name)

    def poll(self) -> None:
        """Nhịp mỗi khung. Không bao giờ chặn luồng GL."""
        if self._job is None or self._queue is None:
            return
        try:
            message = self._queue.get_nowait()
        except queue.Empty:
            if time.perf_counter() - self._started_s > self.TIMEOUT_S:
                logger.error("Nuong lop qua %.0f s — giet tien trinh con",
                             self.TIMEOUT_S)
                self._fail()
            elif self._proc is not None and not self._proc.is_alive():
                logger.error("Tien trinh nuong chet (ma %s) — khong co lop",
                             self._proc.exitcode)
                self._fail()
            return
        key, path, _payload = self._job
        self._job = None
        if message[0] != "ok":
            logger.error("Nuong lop hong: %s\n%s", message[1], message[2])
            self._fail()
            return
        result = load_from_cache(Path(message[1]))
        self._stop_proc()
        if result is None:
            logger.error("Tien trinh con bao xong ma cache doc khong duoc")
            self._failed = True
            return
        result.from_cache = False
        result.bake_s = float(message[2])
        self._last_bake_s = result.bake_s
        self._ready[key] = result
        logger.info("Lop cua %s san sang sau %.1f s (K=%d)", path.name,
                    result.bake_s, result.k)

    def pop_ready(self, key: str) -> BakeResult | None:
        return self._ready.pop(key, None)

    def has_ready(self, key: str) -> bool:
        return key in self._ready

    @property
    def busy(self) -> bool:
        return self._job is not None

    def stop(self) -> None:
        self._stop_proc()

    def _fail(self) -> None:
        self._job = None
        self._failed = True
        self._stop_proc()

    def _stop_proc(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=3.0)
            self._proc = None
        if self._queue is not None:
            self._queue.close()
            self._queue = None
