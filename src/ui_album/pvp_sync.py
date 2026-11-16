"""
Bug placement <-> wire format for LAN PvP (TASK_pvp_lan_1116 P3/P4).

A bug's real position is mm in the *gap* between two consecutive layer
planes (``hidden_props_search.place_props`` always places it there), tied
to this machine's renderer geometry — not portable, same as the task's note
that raw mm are not portable across different ``screen.json`` files.

What travels instead:

- ``layer``: which gap, as an index into ``layer_planes(...)`` (far → near)
  — gap ``i`` is between ``planes[i]`` and ``planes[i + 1]``.
- ``x``, ``y``: the bug's fractional position within ``planes[i + 1]``'s own
  quad — the *nearer* plane bounding the gap, i.e. the one whose texture
  most directly determines what hides the bug. This is exactly the (u, v)
  ``hidden_props.alpha_at`` already uses to sample a layer's alpha mask, so
  it is anchored to the layer's own *content*, not to the physical screen:
  both players baked the identical photo through the identical pipeline,
  so planes[i + 1] holds the same image on both machines regardless of
  physical screen size.
- ``z``: the bug's fractional depth within the gap (``planes[i].z_max_m``
  .. ``planes[i + 1].z_min_m``), so occlusion reconstructs at (close to)
  the original relative depth even though each machine computed its own
  absolute relief bounds.
- ``size_k``, ``camo_rgb``: opaque floats already computed once
  (``assign_bug_sizes``/``assign_bug_camo``), carried through unchanged.

``radius_mm`` is never sent: both sides read their own (committed)
``content.game_prop_radius_mm``.
"""

from __future__ import annotations

#: Extra prop fields reconstructed bugs don't have real values for. Solo's
#: "reveal" bookkeeping (how a bug became exposed during placement search)
#: has no PvP equivalent — Hunt & Hide's exposure HUD uses
#: ``hidden_props.is_visible`` directly instead (see app.py), so these are
#: placeholders only so the prop dict shape matches what the renderer and
#: ``hit.pick_target`` expect.
_PLACEHOLDER_REVEAL = {"reveal_axis": "x", "reveal_450_mm": 0.0,
                        "reveal_330_mm": 0.0}


def _gap_index_for_z(planes, z_mm: float) -> int | None:
    """Which gap ``i`` (between ``planes[i]`` and ``planes[i + 1]``) a prop's
    z falls in, or the nearest gap if it's slightly outside every bound
    (bake rounding)."""
    if len(planes) < 2:
        return None
    best, best_d = None, None
    for i in range(len(planes) - 1):
        lo, hi = planes[i].z_max_m, planes[i + 1].z_min_m
        if lo <= z_mm / 1000.0 <= hi:
            return i
        d = min(abs(z_mm / 1000.0 - lo), abs(z_mm / 1000.0 - hi))
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def bug_to_wire(index: int, prop: dict, planes) -> dict | None:
    """One local prop -> one wire bug. ``None`` if it can't be anchored to a
    gap (fewer than 2 layer planes — should not happen for a prop
    ``place_props`` actually placed)."""
    x_mm, y_mm, z_mm = (float(v) for v in prop["pos_mm"])
    gap = _gap_index_for_z(planes, z_mm)
    if gap is None:
        return None
    lo, hi = planes[gap].z_max_m, planes[gap + 1].z_min_m
    z_frac = 0.5 if hi <= lo else max(0.0, min(1.0, (z_mm / 1000.0 - lo) / (hi - lo)))
    anchor = planes[gap + 1]
    w_mm, h_mm = anchor.quad_w_m * 1000.0, anchor.quad_h_m * 1000.0
    if w_mm <= 0.0 or h_mm <= 0.0:
        return None
    u = (x_mm - (anchor.centre_x_m * 1000.0 - w_mm / 2.0)) / w_mm
    v = ((anchor.centre_y_m * 1000.0 + h_mm / 2.0) - y_mm) / h_mm
    return {"id": int(index), "x": round(max(0.0, min(1.0, u)), 5),
            "y": round(max(0.0, min(1.0, v)), 5), "z": round(z_frac, 5),
            "layer": int(gap),
            "size_k": round(float(prop.get("size_k", 1.0)), 4),
            "camo_rgb": [round(float(c), 4)
                         for c in prop.get("camo_rgb", (0.5, 0.5, 0.5))]}


def bugs_to_wire(props, planes) -> list[dict]:
    """Every prop that can be anchored to a gap, in prop-list order — the
    list *index* doubles as the wire ``id`` and the hit-confirmation id."""
    out = []
    for index, prop in enumerate(props):
        bug = bug_to_wire(index, prop, planes)
        if bug is not None:
            out.append(bug)
    return out


def bug_from_wire(bug: dict, planes, radius_mm: float) -> dict | None:
    """One wire bug -> one local prop, using *this* machine's own
    ``planes`` for the gap named by ``bug["layer"]``. ``None`` if that gap
    doesn't exist locally (e.g. the two machines baked a different layer
    count for some reason — should not happen for the same photo + config,
    but a malformed/adversarial message could still claim any ``layer``)."""
    gap = int(bug["layer"])
    if not (0 <= gap < len(planes) - 1):
        return None
    lo, hi = planes[gap].z_max_m, planes[gap + 1].z_min_m
    z_mm = (lo + max(0.0, min(1.0, float(bug.get("z", 0.5)))) * (hi - lo)) * 1000.0
    anchor = planes[gap + 1]
    w_mm, h_mm = anchor.quad_w_m * 1000.0, anchor.quad_h_m * 1000.0
    x_mm = (anchor.centre_x_m * 1000.0 - w_mm / 2.0) + float(bug["x"]) * w_mm
    y_mm = (anchor.centre_y_m * 1000.0 + h_mm / 2.0) - float(bug["y"]) * h_mm
    prop = {"pos_mm": (x_mm, y_mm, z_mm), "radius_mm": float(radius_mm),
            "size_k": float(bug.get("size_k", 1.0)),
            "camo_rgb": tuple(float(c) for c in bug.get("camo_rgb", (0.5, 0.5, 0.5)))}
    prop.update(_PLACEHOLDER_REVEAL)
    return prop


def bugs_from_wire(bugs, planes, radius_mm: float) -> list[dict]:
    """Wire bugs -> a local prop list indexed exactly like the sender's —
    ``props[i]`` is the reconstruction of the bug whose wire ``id`` is ``i``.
    A bug a gap-mismatch rejects becomes ``None`` and is dropped (its id
    is simply absent — a ``hit``/``killed`` naming that id will not match
    anything locally, which is the same "ignore the unknown" posture as
    everywhere else untrusted data is handled)."""
    if not bugs:
        return []
    size = max(int(b["id"]) for b in bugs) + 1
    props: list[dict | None] = [None] * size
    for bug in bugs:
        props[int(bug["id"])] = bug_from_wire(bug, planes, radius_mm)
    return props
