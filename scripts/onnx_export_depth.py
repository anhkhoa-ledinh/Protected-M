r"""
PB-060: xuat Depth Anything V1 Small ra ONNX roi luong hoa (quantize) INT8.

Mo hinh HF tra ve mot object co nhieu field; ONNX chi can ``predicted_depth``,
nen boc mo hinh trong mot ``nn.Module`` nho chi tra ve tensor do. Khong sua
``src/`` — script nay doc mo hinh tu cache HF (``local_files_only=True``).

    .\.venv\Scripts\python.exe scripts\onnx_export_depth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from onnx_head_patch import patch_head  # noqa: E402

MODEL_ID = "LiheYoung/depth-anything-small-hf"
OUT_DIR = ROOT / "artifacts" / "pb060_onnx" / "models"
FP32 = OUT_DIR / "da_v1_small_fp32.onnx"
INT8 = OUT_DIR / "da_v1_small_int8.onnx"


class DepthOnlyWrapper(nn.Module):
    """Boc mo hinh HF: vao ``pixel_values``, ra duy nhat ``predicted_depth``."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).predicted_depth


def load_wrapped():
    """
    Nap processor + mo hinh tu cache, va head, dat eval, boc lai bang wrapper.

    ``patch_head`` bo ``int()`` o buoc upsample cuoi (xem ``onnx_head_patch``);
    phai va truoc khi boc va truoc khi xuat, khong thi ONNX nuong cung 518x686.
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, local_files_only=True)
    patch_head(model)
    model.eval()
    return processor, DepthOnlyWrapper(model).eval()


def dummy_input(processor) -> torch.Tensor:
    """Tensor mau tu mot anh gia 552x736 — di qua dung processor cua app."""
    fake = np.zeros((552, 736, 3), dtype=np.uint8)
    inputs = processor(images=fake, return_tensors="pt")
    return inputs["pixel_values"]


def export_fp32(wrapped: nn.Module, sample: torch.Tensor) -> None:
    """
    ``torch.onnx.export`` opset 17, truc dong chi cho height/width; batch co
    dinh bang 1.

    Hai dieu kien de giu duoc kich thuoc symbolic:
    1. ``load_wrapped`` phai goi ``patch_head`` truoc (xem ``onnx_head_patch``).
       Day la dieu kien quyet dinh: con ``int()`` thi buoc upsample cuoi bi
       nuong thanh hang ``[518, 686]``, moi anh deu ra 518x686 bat ke input.
    2. ``dynamo=True``. Duong xuat TorchScript cu (``dynamo=False``) nuong cung
       luon ca input nen moi anh deu ra 518x686.

    Da va head roi thi ``Dim`` tu do cung xuat duoc (do ra ``14*(H//14)``), nen
    ``14 * Dim`` khong con bat buoc; van giu vi no dung voi thu processor tra ve
    (``ensure_multiple_of=14``) va cho shape goc gon hon: input
    ``[1, 3, 14*patch_h, 14*patch_w]``, output ``[1, 14*patch_h, 14*patch_w]``.
    Do kiem tren 8 anh app: output luon bang H x W cua tensor vao.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_h = torch.export.Dim("patch_h", min=8, max=104)
    patch_w = torch.export.Dim("patch_w", min=8, max=104)
    height, width = 14 * patch_h, 14 * patch_w
    torch.onnx.export(
        wrapped,
        (sample,),
        str(FP32),
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_shapes={"pixel_values": {2: height, 3: width}},
        opset_version=17,
        dynamo=True,
        # Mac dinh dynamo tach trong so ra file .onnx.data; gop vao mot file cho
        # de so sanh dung MB voi ban INT8 va de copy sang may khac.
        external_data=False,
    )


def check_and_quantize() -> None:
    """``onnx.checker`` tren file FP32 roi luong hoa dong trong so sang INT8."""
    model = onnx.load(str(FP32))
    onnx.checker.check_model(model)
    print(f"onnx.checker OK: {FP32.name} (ir_version={model.ir_version}, "
          f"opset={model.opset_import[0].version})")
    quantize_dynamic(
        model_input=str(FP32),
        model_output=str(INT8),
        weight_type=QuantType.QInt8,
    )
    quantized = onnx.load(str(INT8))
    onnx.checker.check_model(quantized)
    print(f"onnx.checker OK: {INT8.name}")


def main() -> int:
    # Trinh xuat dynamo in emoji; console cp1252 cua may nay se vo neu khong doi.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(2)
    processor, wrapped = load_wrapped()
    sample = dummy_input(processor)
    print(f"sample pixel_values: {tuple(sample.shape)}")
    with torch.no_grad():
        reference = wrapped(sample)
    print(f"predicted_depth torch: {tuple(reference.shape)}")
    export_fp32(wrapped, sample)
    check_and_quantize()
    for path in (FP32, INT8):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
