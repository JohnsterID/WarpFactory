# Contributing

See [CONTRIBUTING.md](https://github.com/JohnsterID/WarpFactory/blob/main/CONTRIBUTING.md)
in the repository for the full guide. The essentials:

## Workflow

1. Create an issue first to discuss major changes
2. Fork, branch, and write tests first (pytest, TDD)
3. Implement following the existing package structure
4. Run the checks locally before opening a pull request:

```bash
ruff check .            # lint (includes import sorting)
ruff format --check .   # formatting
pytest warpfactory/tests -q --no-cov
```

## Code style

- PEP 8, enforced by [Ruff](https://docs.astral.sh/ruff/)
  (configuration in `pyproject.toml`); `ruff check --fix .` and
  `ruff format .` apply fixes
- Type hints for all function parameters and returns (checked with
  mypy)
- Docstrings in NumPy format
- Comments only for non-obvious behavior

## Testing policy

- Tests exercise real code paths against analytic ground truth
  (Minkowski gives T = 0, Schwarzschild is vacuum, Alcubierre matches
  its closed-form energy density) -- no mocks
- Every bug fix includes a regression test
- Tests for optional backends (torch, Qt) skip when the extra is not
  installed
- Document any deviation from the original MATLAB implementation on
  the [MATLAB Parity](parity.md) page in the same change
