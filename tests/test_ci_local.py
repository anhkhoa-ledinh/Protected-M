"""tools/ci_local.sh must run the same job as .github/workflows/tests.yml."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
SCRIPT = (ROOT / "tools" / "ci_local.sh").read_text(encoding="utf-8")


def _workflow_apt_packages():
    m = re.search(r"apt-get install [^\n]*?--no-install-recommends ([^\n]+)", WORKFLOW)
    assert m, "workflow has no apt-get install step"
    return set(m.group(1).split())


def test_apt_packages_match_workflow():
    wanted = _workflow_apt_packages()
    assert {"libgl1", "libglib2.0-0"} <= wanted
    m = re.search(r"apt-get install [^\n]*--no-install-recommends ([^>\n]+)>", SCRIPT)
    assert m, "ci_local.sh has no apt-get install line"
    have = set(m.group(1).split())
    # git is only needed to clone the repo inside the container.
    assert have - {"git"} == wanted


def test_python_version_matches_workflow():
    m = re.search(r'python-version:\s*"([\d.]+)"', WORKFLOW)
    assert m
    assert f"python:{m.group(1)}" in SCRIPT


def test_same_pytest_command_and_env():
    assert "python -m pytest -q -ra" in WORKFLOW
    assert "python -m pytest -q -ra" in SCRIPT
    assert 'HF_HUB_OFFLINE: "1"' in WORKFLOW
    assert "HF_HUB_OFFLINE=1" in SCRIPT


def test_torch_comes_from_the_lock_on_the_cpu_index():
    assert "whl/cpu" in WORKFLOW and "whl/cpu" in SCRIPT
    assert "requirements-lock.txt" in SCRIPT
