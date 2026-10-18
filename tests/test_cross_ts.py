"""The TS mirror is tested against fixtures the Python analyzer produced (web/src/ir/cross.test.ts).
This test keeps those fixtures in sync with Python."""

import importlib.util
import json

from conftest import ROOT


def test_fixtures_are_fresh():
    spec = importlib.util.spec_from_file_location("gen_fixtures", ROOT / "scripts" / "gen_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    catalog, fixtures = mod.build()
    out = ROOT / "web" / "src" / "ir"
    assert (out / "catalog.json").read_text() == catalog, "run: uv run python scripts/gen_fixtures.py"
    assert (out / "fixtures.json").read_text() == fixtures, "run: uv run python scripts/gen_fixtures.py"
    assert (out / "templates.json").read_text() == json.dumps(mod.templates.export(), indent=1) + "\n", "run: uv run python scripts/gen_fixtures.py"
