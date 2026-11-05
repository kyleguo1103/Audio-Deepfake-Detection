# Codebase Structure

## 1) Top-Level Map

| Path | Purpose | Evidence |
|---|---|---|
| `app.py` | Gradio UI, event handlers, result formatting, detector, explanation and research-evidence pages | Module docstring and `build_ui()` |
| `detector.py` | Loads the frozen pilot configuration, probes and encoder; decodes and scores audio; samples In-the-Wild; provides CLI self-check | Module docstring, `Engine`, `ItwSampler`, `main()` |
| `plots.py` | Creates waveform/spectrogram, six-model comparison and layer-profile figures | `waveform_figure()`, `comparison_figure()`, `layer_figure()` |
| `run_dashboard.bat` | Windows launcher that activates the `xlsr` Conda environment | File contents |
| `requirements_dashboard.txt` | Dashboard-only extra dependency; the pilot environment supplies the model stack | File comments and Gradio constraint |
| `README_zh.md` | User-facing setup, feature and limitation documentation | File contents |
| `docs/codebase/` | Maintainer-facing verified codebase documentation | This documentation set |

Generated `__pycache__/` content is not source and is excluded from the architecture description.

## 2) Entry Points

- Main runtime entry: `app.py` → `main()` → `Engine(...)` → `build_ui()` → `demo.launch(...)`.
- Secondary entry point: `detector.py` → `main()` for status output and `--check` reproducibility testing.
- Windows convenience entry: `run_dashboard.bat`, which passes its arguments to `app.py`.
- Entry selection is by the command the user runs; there is no installed console script.

## 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|---|---|---|
| UI (`app.py`) | Components, handlers, HTML/Markdown formatting, batch CSV assembly | XLS-R implementation or probe-training details |
| Detection engine (`detector.py`) | Configuration loading, model/probe lifecycle, decoding, preprocessing, scoring, sampling | Gradio component construction |
| Visualisation (`plots.py`) | Matplotlib figures from an `Analysis` object | Model loading, training or filesystem configuration |
| Reused pilot (`../pilot_la_to_itw/src/`) | Canonical preprocessing, augmentation, embedding and probe utilities | Dashboard presentation logic |
| Pilot/data artefacts outside this folder | Frozen config, manifests, embeddings, run results and restricted/raw audio | Source code for the dashboard |

## 4) Naming and Organization Rules

- File naming: lower-case `snake_case` Python modules; the Chinese README uses the conventional `README_zh.md` suffix.
- Directory organization: small layer-based layout at project root rather than a Python package.
- Imports: ordinary local imports (`import plots`, `from detector import ...`); pilot `src` is inserted into `sys.path` at runtime.
- Path resolution: dashboard location determines the default pilot directory; data paths come from the pilot configuration, CLI overrides or selected environment variables.

## 5) Evidence

- `docs/codebase/.codebase-scan.txt`
- `app.py`
- `detector.py`
- `plots.py`
- `run_dashboard.bat`
