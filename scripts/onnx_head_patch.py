r"""
PB-060 Phase 1b: bo ``int()`` o buoc upsample cuoi cua depth head, chi tren
instance dang chay.

``DepthAnythingDepthEstimationHead.forward`` (``modeling_depth_anything.py:292``)
goi ``interpolate(..., (int(patch_height * self.patch_size),
int(patch_width * self.patch_size)), ...)``. Chay eager thi ``int()`` la vo
nghia vi ``patch_height`` da la int Python, nhung duoi ``torch.export`` chinh
no ep kich thuoc thanh hang ``[518, 686]``, nen ONNX xuat ra luon tra ve
518x686 bat ke anh vao co bao nhieu.

Ham ``patch_head`` thay ``forward`` bang ``types.MethodType`` **tren instance**,
copy y nguyen than ham, chi khac o cho khong goi ``int()``. Khong sua bat ky
file nao trong ``.venv``.
"""
from __future__ import annotations

import types

from torch import nn


def _forward_symbolic(self, hidden_states, patch_height, patch_width):
    """
    Ban copy cua ``DepthAnythingDepthEstimationHead.forward``, bo hai ``int()``.

    Moi dong, moi doi so va thu tu giu nguyen nhu ban goc; khac biet duy nhat la
    tuple kich thuoc truyen cho ``interpolate``.
    """
    hidden_states = hidden_states[self.head_in_index]

    predicted_depth = self.conv1(hidden_states)
    predicted_depth = nn.functional.interpolate(
        predicted_depth,
        (patch_height * self.patch_size, patch_width * self.patch_size),
        mode="bilinear",
        align_corners=True,
    )
    predicted_depth = self.conv2(predicted_depth)
    predicted_depth = self.activation1(predicted_depth)
    predicted_depth = self.conv3(predicted_depth)
    predicted_depth = self.activation2(predicted_depth) * self.max_depth
    predicted_depth = predicted_depth.squeeze(dim=1)  # shape (batch_size, height, width)

    return predicted_depth


def patch_head(model):
    """
    Gan ``_forward_symbolic`` vao ``model.head`` roi tra lai chinh ``model``.

    Bao loi ngay neu khong tim thay ``head`` — tot hon la xuat ra mot model van
    nuong cung kich thuoc ma khong ai biet.
    """
    head = getattr(model, "head", None)
    if head is None:
        raise AttributeError("Model khong co thuoc tinh 'head' de va")
    head.forward = types.MethodType(_forward_symbolic, head)
    return model
