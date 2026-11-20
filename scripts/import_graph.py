r"""
TASK_arch_refactor_1120 R0: package-level import graph + cycle report.

    python scripts/import_graph.py

Parses every ``import``/``from ... import ...`` under ``src/`` with
``ast`` (no execution, so it is safe to run with any interpreter that
has nothing installed beyond the stdlib). Groups modules by their
top-level package (``src.engine``, ``src.ui_album``, ...), builds a
directed package-level graph, and reports every cycle (2-node mutual
edges and longer cycles via Tarjan SCCs).

Reused by later TASK_arch_refactor_1120 steps (R1-R6) to check the
cycle list after each change, and by ``tests/test_architecture.py``
(R2) for the ongoing dependency-rule test.
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def iter_py_files(src: Path = SRC):
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def module_name(path: Path, root: Path = ROOT) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_of(mod: str) -> str:
    """``src.ui_album.v3_ui`` -> ``src.ui_album``; ``src`` -> ``src``."""
    parts = mod.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else parts[0]


def extract_imports(path: Path) -> list[str]:
    """Every ``src....`` module this file imports (absolute or
    relative), as fully-qualified dotted names. Best-effort: a
    ``from . import x`` relative import is resolved against the
    file's own package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    own_mod = module_name(path)
    own_pkg_parts = own_mod.split(".")
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.") or alias.name == "src":
                    out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative import: resolve against the importing file's package
                base_parts = own_pkg_parts[:-1]  # drop the module itself
                base_parts = base_parts[: len(base_parts) - (node.level - 1)] \
                    if node.level > 1 else base_parts
                base = ".".join(base_parts)
                full = f"{base}.{node.module}" if node.module else base
                if full.startswith("src"):
                    out.append(full)
            elif node.module and (node.module.startswith("src.") or node.module == "src"):
                out.append(node.module)
    return out


def build_package_graph(src: Path = SRC) -> dict[str, set[str]]:
    """``{package: {packages it imports from}}``, self-edges dropped."""
    graph: dict[str, set[str]] = defaultdict(set)
    for path in iter_py_files(src):
        own_pkg = package_of(module_name(path))
        for imp in extract_imports(path):
            other_pkg = package_of(imp)
            if other_pkg != own_pkg:
                graph[own_pkg].add(other_pkg)
    return dict(graph)


def find_sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC algorithm; returns components with >1 node (real
    cycles) -- a lone node is never reported even if the graph has a
    self-loop (packages importing their own submodules is normal)."""
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    nodes = set(graph.keys())
    for dests in graph.values():
        nodes.update(dests)

    sys.setrecursionlimit(10000)

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in graph.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                result.append(sorted(component))

    for v in sorted(nodes):
        if v not in index:
            strongconnect(v)
    return result


def find_mutual_pairs(graph: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Every unordered pair (A, B) where A imports B AND B imports A."""
    pairs = set()
    for a, dests in graph.items():
        for b in dests:
            if a != b and a in graph.get(b, ()):
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def main() -> None:
    graph = build_package_graph()
    print("Package-level edges (A -> B means A imports from B):")
    for pkg in sorted(graph):
        for dest in sorted(graph[pkg]):
            print(f"  {pkg} -> {dest}")

    print("\nMutual (2-node) cycles:")
    pairs = find_mutual_pairs(graph)
    if not pairs:
        print("  (none)")
    for a, b in pairs:
        print(f"  {a} <-> {b}")

    print("\nLarger cycles (SCCs with >1 package, Tarjan):")
    sccs = find_sccs(graph)
    bigger = [c for c in sccs if len(c) > 2]
    if not bigger:
        print("  (none beyond the 2-node pairs above)")
    for comp in bigger:
        print(f"  {comp}")


if __name__ == "__main__":
    main()
