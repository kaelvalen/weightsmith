"""trainscope ships on PyPI on its own: it must never depend on Weightsmith."""

import re

from conftest import ROOT

PACKAGE = ROOT / "trainscope" / "trainscope"
IMPORT = re.compile(r"^\s*(from|import)\s+weightsmith", re.MULTILINE)


def test_trainscope_does_not_import_weightsmith():
    offenders = [str(p.relative_to(ROOT)) for p in PACKAGE.rglob("*.py") if IMPORT.search(p.read_text())]
    assert offenders == []
