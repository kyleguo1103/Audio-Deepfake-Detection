# Codebase Concerns

## 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|---|---|---|---|---|
| High when remotely exposed | `--share`/`--host 0.0.0.0` has no application authentication, and Gradio is allowed to serve the pilot run directory plus the whole OS temp directory | `app.py` launch arguments; `README_zh.md` | Uploaded audio, results or local temporary files may be exposed beyond the intended audience | Use local mode by default; before remote use, narrow `allowed_paths`, add access control and use non-sensitive demonstration audio |
| High if cache provenance is lost | Probe cache is deserialized with `pickle.load` | `detector.py:174-200` | A tampered pickle can execute code | Treat cache as trusted local artefact; consider a safer model format or checksum/signature verification |
| Medium | The full environment is not locked in this folder | `requirements_dashboard.txt` only constrains Gradio | Dependency upgrades can change loading, UI or numerical behaviour | Export a complete Conda/lock file and document the known-good CUDA/driver combination |
| Medium | Strong coupling to local paths and external pilot layout | `run_dashboard.bat`, `detector.py`, pilot config | Moving files or publishing only this folder breaks the demo | Add a startup validation report and portable configuration/example paths |
| Medium | Core unit tests now exist, but no automated browser test or coverage measurement exists | `tests/test_detector_core.py`, `TESTING.md` | UI/callback regressions may escape the unit and five-clip checks | Add one upload/batch browser smoke test and coverage reporting |
| Medium | No explicit input size/duration limits; batch work is sequential | `on_batch()`, `analyse()`, full-waveform plotting | Very large or numerous files can consume memory and block the single queue | Enforce upload limits, cap plotted duration and document batch limits |

## 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|---|---|---|---|---|
| UI and orchestration combined | Fast research-prototype development | `app.py` | Harder to test handlers separately | Split formatting, handlers and results-page data into modules |
| Model/data lifecycle combined | One-file demo engine | `detector.py` | Cache, training, decoding and scoring changes interact | Separate loaders, probe repository, decoder and scorer |
| Runtime `sys.path` modification | Reuse pilot source without packaging it | `detector.py` | Import errors and name collisions after relocation | Package the shared pilot code or install it in editable mode |
| Hardcoded Windows launcher paths | Built for the current laptop | `run_dashboard.bat` | Low portability | Read paths from arguments/config and add a setup check |
| Broad exception catches | Keep the demo responsive | `app.py`, `detector.py` | Root cause may be hidden or local details shown remotely | Log structured details locally and show a short safe UI error |

## 3) Security Concerns

| Risk | OWASP category | Evidence | Current mitigation | Gap |
|---|---|---|---|---|
| Unauthenticated remote dashboard | A01 Broken Access Control | `--share`, `--host`; no auth parameter | Default bind is `127.0.0.1` | Remote modes need explicit access policy |
| Unsafe local pickle deserialization | A08 Software/Data Integrity Failures | `pickle.load` | Cache path is local and name includes run/config identifiers | No content checksum or safe deserializer |
| Local path/decoder details in UI errors | A05 Security Misconfiguration | broad UI exception messages | Local-only default | Sanitize messages before remote sharing |
| Broad file-serving paths | A01 Broken Access Control | `allowed_paths=[run_dir, tempfile.gettempdir()]` | Intended for generated artefacts | Scope to exact generated files/directories |

## 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---|---|---|---|---|
| One encoder and queue concurrency one | global `ENGINE`; `demo.queue(default_concurrency_limit=1)` | Requests run serially | Multiple users wait; public sharing is not scalable | Keep for laptop demo or add controlled worker/GPU scheduling |
| Sequential batch scoring | `on_batch()` loop | One XLS-R pass per file | Runtime grows linearly | Set a documented batch cap; consider safe batching later |
| Full original waveform retained and plotted | `Analysis.waveform`, `waveform_figure()` | Long files consume memory despite a 6-second model crop | Oversized files can slow plots or exhaust memory | Limit decode/plot duration and reject oversized uploads |
| 45 probes may be trained on first run | `_train_probes()` | First startup takes minutes without cache | Unfriendly setup; depends on embeddings | Ship a verifiable safe-format probe artefact or retain cache with integrity checks |

## 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|---|---|---|---|
| `detector.py` | Couples external pilot APIs, cache schema, model loading and scoring | No Git history is available; file is one of the two largest | Run offline self-check and compare live/recorded logits after every change |
| `app.py` | Four pages, callbacks, HTML and file serving in one module | No Git history is available; file is one of the two largest | Smoke-test Detect, random target, model switch, Batch and result gallery |
| Pilot output contract | Dashboard assumes exact JSON/CSV keys and folder names | External dependency rather than local churn | Pin run/config hashes and validate expected files/keys at startup |

## 6) `[ASK USER]` Questions

1. [ASK USER] Will the demo remain local/meeting-only, or do you intend to use `--share` or `--host 0.0.0.0` regularly? The answer determines whether authentication and file-serving hardening are required now.
2. [ASK USER] Do you want a complete locked environment and automated UI smoke test before the next assessed demonstration, or are the data-free unit tests plus the verified local integration check sufficient for the current milestone?

## 7) Evidence

- `docs/codebase/.codebase-scan.txt`
- `app.py`
- `detector.py`
- `plots.py`
- `run_dashboard.bat`
- `requirements_dashboard.txt`
- `tests/test_detector_core.py`
- `README_zh.md`
