# Instructions for agents working on dionw

## Commands
```bash
uv venv && uv pip install -e . --group test --group dev && .venv/bin/pre-commit install
.venv/bin/pytest                 # skips `slow` (compile-heavy) and `bench` tests
.venv/bin/pytest -m slow         # compile-heavy tests
.venv/bin/pytest -m bench        # the README's performance claims (idle GPU)
.venv/bin/python -m bench.<name> # print a benchmark's table (repository root)
.venv/bin/pre-commit run --all-files   # hygiene, ruff check/format, ty
```
Set `TORCHINDUCTOR_CACHE_DIR` to a persistent directory to keep compiled kernels
between runs.

## Layout
- Public API: `dionw/__init__.py`; public modules `optimizer`, `param_groups`,
  `routing`, `report`, `newton_schulz`, `ema/`. Underscore modules are private.
- Keep files short (about 300 lines at most) and functions within the ruff
  limits in `pyproject.toml` (complexity 8, 30 statements).

## Design rules (hard requirements)
- No backward compatibility shims unless explicitly requested.
- No silent defaults or fallbacks inside functions; defaults are explicit in
  signatures. Fail fast with clear, actionable errors; never swallow exceptions.
- Strict typing (`int | None`, `list[str]`, `collections.abc`); avoid `Any` except
  for torch's param-group dicts.
- `match`/`case` with a raising default branch over `if`/`elif` chains on options.
- No `hasattr`, `getattr`, `setattr`.
- Enums for every selectable option; no magic strings.
- Short functions and classes with accurate docstrings (purpose, inputs,
  outputs, errors).
- Tests never rely on defaults: construct everything with explicit values.
- Never change a default to make a test pass.
- Update numerics only with a parity test against microsoft/dion
  (`tests/test_optimizer.py`) still passing at its tolerance.
- Every performance or precision figure in the README comes from a script in
  `bench/` and is checked by a test (`bench` marker for timings).

## Scope
One process, one CUDA device, float32 parameters. No distributed code.
