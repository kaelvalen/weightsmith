# Contributing to trainscope

trainscope is developed in the [Weightsmith](https://github.com/kaelvalen/weightsmith) repository, under `trainscope/`, and published to PyPI on its own.

## Development setup

```bash
git clone https://github.com/kaelvalen/weightsmith.git
cd weightsmith
uv sync
cd web && pnpm install
```

## Running tests

```bash
cd trainscope && uv run pytest -q     # trainscope only
scripts/check.sh                      # everything, from the repository root
```

## Linting and formatting

```bash
cd trainscope
uv run ruff check trainscope/ tests/ scripts/ examples/
uv run ruff format trainscope/ tests/
uv run mypy trainscope/ tests/
```

The project uses **ruff** for linting and formatting (line length 100) and **mypy** for type checking.

## UI

The browser UI is Weightsmith's web package (`web/src/scope/`): the same views are Weightsmith's Diagnose tab and the standalone `trainscope ui`. Build the standalone bundle into the package with:

```bash
cd web && pnpm run build:scope
```

## Rules

- `trainscope/trainscope/` must never import Weightsmith.
- The Python API, config surface, Arrow format and plugin surface are stable contracts (see the README's "Stability scope").

## Pull request workflow

1. Create a feature branch from `main`.
2. Make focused, minimal changes.
3. Add or update tests for new behavior.
4. Run `scripts/check.sh` locally.
