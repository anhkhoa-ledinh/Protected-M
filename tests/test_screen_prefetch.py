"""TASK_perf_textures_1119 Fix 3: ``v3_routes.NEXT_SCREENS`` — the static
prefetch graph ``V3UI`` uses to queue a screen's likely neighbours in the
background before the player clicks there.

Two checks:
- every screen in the real manifest has an entry (even an empty one);
- every ``state.go(F.XXX)`` target this source file's own click handlers
  can reach is SOMEWHERE in the graph (a global reachability check, not a
  per-call-site one — attributing each call to its exact calling screen
  would need real control-flow analysis; this still catches a genuinely
  orphaned/typo'd target).
"""
from __future__ import annotations

import ast
import inspect

from src.ui_album import v3_routes as R


def _manifest_screen_names() -> set[str]:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "assets" / "ui_v3" / "manifest.json").read_text(encoding="utf-8"))
    return set(data["screens"])


def test_every_manifest_screen_has_a_next_screens_entry():
    missing = _manifest_screen_names() - set(R.NEXT_SCREENS)
    assert not missing, f"no NEXT_SCREENS entry for: {sorted(missing)}"


def test_next_screens_has_no_entries_outside_the_manifest():
    extra = set(R.NEXT_SCREENS) - _manifest_screen_names()
    assert not extra, f"NEXT_SCREENS entries not in the manifest: {sorted(extra)}"


def _go_targets_in_source(module) -> set[str]:
    """Every ``F.<NAME>`` passed to ``state.go(...)`` or found in a dict
    literal immediately passed to it (``v3_routes.py``'s
    ``state.go({"btn_Solo": F.LEVELS, ...}[name])`` pattern), by walking
    the module's own AST. Resolves ``F.<NAME>`` via the already-imported
    ``v3_flow`` module so the test fails loudly (``AttributeError``) if a
    name is ever renamed on one side and not the other."""
    import src.ui_album.v3_flow as F

    source = inspect.getsource(module)
    tree = ast.parse(source)
    targets: set[str] = set()

    def resolve(node) -> None:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "F":
            if hasattr(F, node.attr):
                targets.add(getattr(F, node.attr))

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "go" and node.args):
            arg = node.args[0]
            resolve(arg)
            if isinstance(arg, ast.Dict):
                for value in arg.values:
                    resolve(value)
    return targets


def test_every_state_go_target_in_v3_routes_is_reachable_somewhere():
    targets = _go_targets_in_source(R)
    assert targets, "AST walk found no state.go(...) targets — check the walker itself"
    reachable = set(R.NEXT_SCREENS) | {s for row in R.NEXT_SCREENS.values() for s in row}
    unreachable = targets - reachable
    assert not unreachable, f"state.go(...) targets missing from NEXT_SCREENS: {sorted(unreachable)}"
