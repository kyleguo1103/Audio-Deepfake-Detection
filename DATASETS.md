# Datasets and model

No audio is stored in this repository. Obtain each item from its official source and keep its licence.

## ASVspoof 2019 Logical Access (source domain)

- **Use here:** source training pool from the LA *train* partition (1,500 bona fide + 250 spoofed clips
  for each attack A01–A06) and source validation from the speaker-disjoint LA *dev* partition
  (1,000 bona fide + 1,000 spoofed). The *eval* partition is not used.
- **Obtain:** `LA.zip` from Edinburgh DataShare, record 10283/3336, doi:10.7488/ds/2555
  (https://datashare.ed.ac.uk/handle/10283/3336). An extracted `LA` folder with the official layout
  (`ASVspoof2019_LA_cm_protocols/`, `ASVspoof2019_LA_train/flac/`, `ASVspoof2019_LA_dev/flac/`) also works.
- **Licence:** Open Data Commons Attribution License (ODC-By).
- **Cite:** X. Wang *et al.*, "ASVspoof 2019: A large-scale public database of synthesized, converted
  and replayed speech," *Computer Speech & Language*, vol. 64, 101114, 2020; and the DataShare record above.

## In-the-Wild (target domain)

- **Use here:** all 31,779 clips (19,963 bona fide, 11,816 spoofed; 54 speaker labels in `meta.csv`),
  used only for the final evaluation. No target label is used for fitting, selection, thresholding
  or calibration.
- **Obtain:** `release_in_the_wild.zip` from https://huggingface.co/datasets/mueller91/In-The-Wild.
  Keep it zipped. Expected size 8,161,489,023 bytes and SHA-256
  `46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a`; step 0 and step 1 check both.
- **Licence:** CC BY-SA 4.0 (as listed on the dataset page).
- **Cite:** N. M. Müller *et al.*, "Does audio deepfake detection generalize?," in *Proc. Interspeech*,
  2022, pp. 2783–2787.

## XLS-R-53 (frozen feature extractor)

- `facebook/wav2vec2-large-xlsr-53`, Hugging Face commit `c3f9d884181a224a6ac87bf8885c84d1cff3384f`,
  downloaded automatically by step 2 (about 1.27 GB). Licence: Apache-2.0.
- **Cite:** A. Conneau *et al.*, "Unsupervised cross-lingual representation learning for speech
  recognition," in *Proc. Interspeech*, 2021, pp. 2426–2430.

## Derived files included in this repository

| Files | Content | Licence |
|---|---|---|
| `pilot_la_to_itw/outputs/manifests/source_*.csv` | ASVspoof clip IDs, labels, speaker and attack IDs, SHA-256 of each selected clip | ODC-By (derived from ASVspoof 2019) |
| `pilot_la_to_itw/outputs/manifests/target_itw.csv`, `outputs/run_*/predictions_target.csv` | In-the-Wild clip IDs, labels, speaker names from `meta.csv`, SHA-256, model logits | CC BY-SA 4.0 (derived from In-the-Wild) |
| `outputs/run_*/predictions_source_validation.csv` | ASVspoof clip IDs, labels, model logits | ODC-By |
| Aggregate metrics, bootstrap results, figures | Numbers computed in this study | Free to reuse with attribution to this repository and the datasets above |

## Not included

- Any audio, augmented audio, XLS-R embeddings or model weights.
- Datasets that are not part of this pilot. Restricted datasets such as DeepSpeak, if used in later
  studies, will be documented here with access instructions only and never uploaded.
