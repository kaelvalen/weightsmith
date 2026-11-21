"""trainscope ships on PyPI on its own: it must never depend on Weightsmith."""

import re

from conftest import ROOT

PACKAGE = ROOT / "trainscope" / "trainscope"
IMPORT = re.compile(r"^\s*(from|import)\s+weightsmith", re.MULTILINE)


def test_trainscope_does_not_import_weightsmith():
    offenders = [str(p.relative_to(ROOT)) for p in PACKAGE.rglob("*.py") if IMPORT.search(p.read_text())]
    assert offenders == []


def test_standalone_wheel_ships_the_shared_ui(tmp_path):
    """`pip install trainscope` carries the UI built from web/ (scope.html), and no Weightsmith dependency."""
    import shutil
    import subprocess
    import zipfile

    import pytest

    if not shutil.which("uv"):
        pytest.skip("uv not installed")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT / "trainscope")], check=True, capture_output=True)
    wheel = zipfile.ZipFile(next(tmp_path.glob("trainscope-*.whl")))
    names = wheel.namelist()
    assert "trainscope/ui/static/index.html" in names
    assert 'name="generator" content="weightsmith-web"' in wheel.read("trainscope/ui/static/index.html").decode()
    assert any(n.startswith("trainscope/ui/static/assets/") and n.endswith(".js") for n in names)
    meta = wheel.read(next(n for n in names if n.endswith("METADATA"))).decode()
    assert "weightsmith" not in meta.lower()
