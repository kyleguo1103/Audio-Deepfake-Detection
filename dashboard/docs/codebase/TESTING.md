# Testing Patterns

## 1) Test Stack and Commands

- Primary test framework: Python standard-library `unittest`.
- Assertion/mocking tools: `unittest` assertions plus small local fake classes; no external mocking package.
- Implemented integration self-check:

```bat
python -m unittest discover -s tests -v
set PILOT_DATA_ROOT=D:\path\to\pilot_data
set HF_HOME=%PILOT_DATA_ROOT%\hf_cache
set HF_HUB_OFFLINE=1
python detector.py --check
```

- E2E browser and coverage commands: [TODO] not implemented.

## 2) Test Layout

- Data-free tests are in `tests/test_detector_core.py`.
- `detector.py` contains the console self-check in its `main()` function.
- The check uses deterministic `random.Random(7841)` target sampling and defaults to five clips.

## 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|---|---|---|---|
| Unit | Partial | sigmoid midpoint, hidden-layer selection, thresholded analysis, empty-audio rejection | Four tests in `tests/test_detector_core.py` |
| Integration | Partial | config/manifests, probe cache, encoder, target archive and live scoring | `detector.py --check` |
| E2E browser | No | upload/microphone, tabs, batch CSV, public sharing | [TODO] No automated browser test |
| Research-result verification | External to this folder | pilot metrics and bootstrap outputs | Supplied by the separate reproducibility repository/pilot scripts |

## 4) Mocking and Isolation Strategy

- Tests use `FakeModel`, `FakePilotCommon` and `FakeEncoder` to avoid data, network and GPU dependencies.
- `Engine` accepts an optional `encoder` object, and the tests construct an uninitialised engine with only the required collaborators.
- The self-check uses real local model/data artefacts, so it is an integration test rather than an isolated unit test.
- Common failure mode: Hugging Face may attempt a network metadata request even with local weights; offline mode prevents this when the checkpoint is cached.

## 5) Coverage and Quality Signals

- Coverage tool + threshold: [TODO] none.
- Current reported coverage: [TODO] unavailable.
- Verified 2026-10-18 unit result: all four data-free tests passed in 0.002 seconds in the local `xlsr` environment.
- Verified 2026-10-18 result: offline self-check loaded run `run_20261011_113206`, config hash prefix `a26f7873`, verified manifests, used the RTX 3060 GPU, found 31,779 target clips, and reproduced all six recorded logits for five deterministic clips with maximum absolute difference `0.0000`; final output was `CHECK PASSED`.
- A preceding online-mode attempt loaded the probe cache but failed because the restricted environment blocked a Hugging Face network request. This was resolved by using the README's documented offline mode; it was not a model-score mismatch.
- Known gaps: unsupported/edge-case audio formats, empty/very long files, batch partial failures, CPU performance, UI callbacks, share mode and cache-corruption behavior are not covered automatically.

## 6) Evidence

- `detector.py`
- `README_zh.md`
- `requirements_dashboard.txt`
- `tests/test_detector_core.py`
- `docs/codebase/.codebase-scan.txt`
- `../pilot_la_to_itw/outputs/run_20261011_113206/predictions_target.csv`
