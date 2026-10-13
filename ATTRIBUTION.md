# Attribution

## Authorship of this repository

All code, configuration, the pre-registration and the documentation in this repository were written
for this pilot study by Jingkai Guo (The University of Queensland). The experiments were run on the
author's laptop (NVIDIA RTX 3060 Laptop GPU) on 11 October 2026.

An AI assistant (Claude, Anthropic) helped draft the code and documentation; the author ran every
experiment, checked the results and is responsible for the content.

## Relationship to prior work in the same group

This study builds on the research direction of U. Thani, G. Singh and P. Singh, "Unsupervised domain
adaptation for audio deepfake detection with modular statistical transformations," arXiv:2603.07935,
2026, which used mean-pooled XLS-R-53 embeddings with a lightweight classifier and target-aware
statistical adaptation.

- **No source code from that project is included or adapted here.** The shared idea is the design
  choice of frozen, mean-pooled XLS-R-53 embeddings with a logistic-regression classifier.
- **Original to this repository:** the source-only augmentation protocol (codec, noise and channel
  families with identical rules for both classes), the pre-registered design and decision rule, the
  frozen manifests with per-clip SHA-256, layer and regularisation selection on source validation only,
  the paired speaker-cluster bootstrap, the duration/silence shortcut diagnostic, the post hoc
  fixed-layer analysis and the verification scripts.
- Results reported by Thani *et al.* are **not** reproduced or claimed here.

## Third-party software (used as dependencies, not copied)

| Package | Use | Licence |
|---|---|---|
| PyTorch 2.4.1 | XLS-R inference | BSD-3-Clause |
| Hugging Face Transformers 4.44.2 | loading `facebook/wav2vec2-large-xlsr-53` | Apache-2.0 |
| scikit-learn 1.5.2 | standardisation and logistic regression | BSD-3-Clause |
| NumPy, SciPy | numerics, filters, resampling, ranking | BSD-3-Clause |
| Matplotlib | figures | Matplotlib licence (PSF-based) |
| soundfile, tqdm | audio I/O, progress bars | BSD-3-Clause, MIT/MPL-2.0 |
| imageio-ffmpeg | bundled ffmpeg binary for decoding and codec augmentation | BSD-2-Clause (ffmpeg binary: LGPL/GPL) |

## Model and datasets

- XLS-R-53: A. Conneau *et al.*, Proc. Interspeech 2021; `facebook/wav2vec2-large-xlsr-53` (Apache-2.0).
- ASVspoof 2019 LA: X. Wang *et al.*, Computer Speech & Language 64, 2020; doi:10.7488/ds/2555 (ODC-By).
- In-the-Wild: N. M. Müller *et al.*, Proc. Interspeech 2022; huggingface.co/datasets/mueller91/In-The-Wild (CC BY-SA 4.0).
- Silence trimming follows the shortcut reported by N. M. Müller *et al.*, "Speech is silver, silence is
  golden: What do ASVspoof-trained models really learn?," Proc. ASVspoof 2021 Workshop.

## Acknowledgement

The author thanks Dr Priyanka Singh and Gagandeep Singh for supervision and feedback.
