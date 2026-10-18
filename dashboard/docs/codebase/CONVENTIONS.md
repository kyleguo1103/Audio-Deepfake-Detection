# Coding Conventions

## 1) Naming Rules

| Item | Rule | Example | Evidence |
|---|---|---|---|
| Files | lower-case `snake_case` | `run_dashboard.bat`, `requirements_dashboard.txt` | Root tree |
| Functions/methods | `snake_case`; private helpers start with `_` | `on_detect`, `_load_encoder`, `_regions` | `app.py`, `detector.py`, `plots.py` |
| Types/classes | `PascalCase` | `Engine`, `Probe`, `ModelScore`, `Analysis`, `ItwSampler` | `detector.py` |
| Constants/env vars | upper-case `SNAKE_CASE` | `CONDITIONS`, `DEFAULT_MODEL`, `HF_HOME` | `app.py`, `detector.py` |

## 2) Formatting and Linting

- Formatter: [TODO] no formatter or configuration file is present.
- Linter: [TODO] no linter or configuration file is present. Inline `noqa` comments suggest Flake8/Ruff-compatible intent, but neither is configured here.
- Observed style: four-space indentation, type hints on public data paths, short module/function docstrings, divider comments for major sections.
- Run commands: [TODO] add explicit format and lint commands after tools are selected.

## 3) Import and Module Conventions

- Imports are grouped as standard library, third-party packages, then local modules.
- Local dashboard modules use direct imports; no aliases or package-relative imports are configured.
- Pilot imports are delayed until `Engine` startup after its `src` folder is inserted into `sys.path`.
- Heavy/environment-sensitive imports are delayed where needed, while `HF_HOME` is set before importing Gradio/Hugging Face dependencies.

## 4) Error and Logging Conventions

- Setup failures use `SystemExit` with actionable messages for missing pilot code/embeddings.
- Invalid input uses `ValueError`; unavailable random sampling uses `RuntimeError`.
- UI handlers catch broad exceptions and return a user-visible error instead of crashing the server.
- Console logging uses `print` with the `[dashboard]` prefix and flushes immediately.
- Sensitive-data redaction: [TODO] no explicit policy exists. Errors can include local paths or decoder details, which is acceptable for a local research demo but should be reviewed before remote sharing.

## 5) Testing Conventions

- Data-free unit tests live in `tests/` and use the `test_*.py` naming pattern with Python `unittest`.
- Lightweight fake pilot/model/encoder classes isolate threshold, layer-selection and analysis behaviour without loading XLS-R or audio data.
- `python detector.py --check` remains the integration/reproducibility check, using a deterministic random seed and comparison with recorded pilot logits.
- Coverage expectation: [TODO] not defined.

## 6) Evidence

- `app.py`
- `detector.py`
- `plots.py`
- `tests/test_detector_core.py`
- `docs/codebase/.codebase-scan.txt`
