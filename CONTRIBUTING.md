# Contributing to cpz-quant

Thank you for considering a contribution. cpz-quant is maintained by [CPZ Lab](https://ai.cpz-lab.com) and welcomes issues and pull requests.

Parts of this library are developed with **Simons**, the AI research partner of the CPZAI operating system. Commits authored by Simons are AI-generated contributions, reviewed and released under CPZ Lab's maintainership — disclosed here because we believe AI contributions should be visible, not hidden.

## Development setup

```bash
git clone https://github.com/CPZ-Lab/cpz-quant.git
cd cpz-quant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Ground rules

- **Pure functions only.** No network calls, no file I/O, no global state in library code. Data in, results out.
- **Fail loudly.** Never add a silent fallback or a fabricated default. Missing optional dependencies must raise with install instructions.
- **Typed.** New code must pass `mypy cpz_quant` and `ruff check` (the configured profiles in `pyproject.toml`; full `--strict` adoption is on the roadmap).
- **Tested.** New features need tests; CI enforces 80%+ branch coverage on `cpz_quant`.
- **Cite your math.** Non-trivial algorithms should reference the paper or book they implement in the docstring.

## Pull requests

1. Fork, branch from `main`, keep the change focused.
2. `pytest && mypy cpz_quant && ruff check cpz_quant tests` locally.
3. Update `CHANGELOG.md` under an `Unreleased` heading.
4. Open the PR with a clear description of the motivation and the math, if any.

## Reporting bugs

Open a [GitHub issue](https://github.com/CPZ-Lab/cpz-quant/issues) with a minimal reproduction (inputs, expected vs actual output, versions). For security issues see [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0.
