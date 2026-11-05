# Confirmatory Follow-up: ASVspoof 2019 LA → DeepSpeak v2.0

> **Frozen on 2026-10-24 (Australia/Brisbane), before any DeepSpeak media was downloaded, any DeepSpeak XLS-R embedding was extracted, or any DeepSpeak model score was viewed.**  
> Machine-readable design: `config/confirm_config.json`  
> Config SHA-256: `c2fa249e79253d928624292c690a81c52db33b7271a6237558434e377546d5fd`  
> Frozen target manifest SHA-256: `9f3b894ffaebb6bec3827bd7f516307fd93b9321803d577f0390fdd6055f5b7d`

## 1. Status and scope

This is a **confirmatory follow-up on a new target domain**, not a claim that the complete study was designed before any earlier result existed. The completed ASVspoof 2019 LA → In-the-Wild pilot produced an exploratory finding that additive noise was more consistent than the other tested augmentations across XLS-R layers. The present study freezes one layer, one regularisation value and one new target protocol before acquiring or scoring any DeepSpeak media.

DeepSpeak v2.0 is used only as an independent target. It is not partitioned again, and no DeepSpeak example is used to fit the detector, choose a layer, choose `C`, choose a threshold or calibrate scores.

## 2. Confirmatory research question

When the source data, preprocessing, frozen XLS-R encoder, hidden-state layer and classifier regularisation are held constant, does additive noise augmentation during ASVspoof 2019 LA source training reduce EER on the official DeepSpeak v2.0 test identities relative to no augmentation?

### Hypothesis and primary contrast

- Control: `none` — clean source training clips only.
- Treatment: `noise` — the same clean clips plus one deterministic additive-noise copy of every selected source clip.
- Primary statistic: mean over the three source-subsampling seeds of `EER_noise - EER_none` on the frozen DeepSpeak target.
- Confirmation rule: improvement is confirmed only when the entire two-sided 95% paired identity-cluster bootstrap confidence interval lies below zero. A confidence interval crossing zero is reported as inconclusive.

## 3. What is held fixed

| Component | Frozen choice |
|---|---|
| Source train/validation | The existing ASVspoof 2019 LA train pool and speaker-disjoint dev validation manifests from the completed pilot |
| Source subsets | Stratified 80% of the source pool, seeds 42, 1337 and 2026; augmented copies follow their parent clip |
| Preprocessing | Decode to mono 16 kHz; trim leading/trailing silence at the previously defined threshold; retain the first 6 s; per-utterance zero-mean/unit-variance normalisation |
| Encoder | Frozen `facebook/wav2vec2-large-xlsr-53`, resolved commit `c3f9d884181a224a6ac87bf8885c84d1cff3384f` |
| Hidden state | **Layer 10 for both conditions** |
| Classifier | StandardScaler + class-balanced logistic regression, `lbfgs`, `max_iter=3000` |
| Regularisation | **C = 0.1 for both conditions** |
| Noise | White, pink or brown; SNR 5–20 dB; one copy per source clip; the original deterministic pilot mapping with augmentation seed 7841 |
| Threshold for thresholded secondary metrics | Each fitted model's EER threshold determined on ASVspoof source validation only |

Layer 10 was chosen after the earlier In-the-Wild exploratory layer analysis. It therefore cannot support a claim of result-blind layer selection in that earlier target. Its role here is different: it is fixed before DeepSpeak scoring so layer selection cannot explain a new DeepSpeak difference. `C=0.1` is deliberately shared by both conditions, removing condition-specific regularisation as a confound.

## 4. Frozen DeepSpeak target

The official identity split is retained. Only rows assigned to the official `test` identities are eligible.

| Label | Inclusion rule | Count |
|---|---|---:|
| Real audio | Official test identity and `script-type` in `{scripted-short, unscripted}` | 1,324 |
| Fake audio | Both source and target identity in the official test split; `audio-config` in `{elevenlabs, playht, speechify}`; `script-type` in `{scripted-short, unscripted}` | 476 |
| **Total** | Frozen before media acquisition | **1,800** |

The 476 fake-audio clips contain 158 ElevenLabs, 158 PlayHT and 160 Speechify examples. Rows from manipulated videos whose `audio-config=real` are excluded because their audio is not fake. Real `no-speech` and `scripted-long` rows are excluded to match the content types available in the fake-audio pool.

The frozen manifest contains 56 official test identities and is stored at `outputs/manifests/target_deepspeak_v2_test.csv`. This file is a restricted-data derivative and must not be published.

## 5. Evaluation and uncertainty

For each source-subsampling seed, two classifiers are fitted: `none` and `noise`. They differ only in whether the deterministic noise copies are added to source training. Both use layer 10 and `C=0.1`.

- Primary metric: target EER from raw decision scores.
- Secondary metrics: ROC AUC, fake-positive PR AUC, balanced accuracy and macro F1 at the source-validation EER threshold.
- Primary uncertainty: 10,000 paired bootstrap resamples of the 56 target identities. Within a replicate, the same sampled identities are used for both conditions and all three seeds; metrics are averaged over seeds before the condition difference is calculated.
- Provider, generation-engine and script-type breakdowns are exploratory. They cannot replace or redefine the primary result.

The class imbalance is retained rather than artificially rebalancing the target. EER is the primary metric because it is threshold-free and not determined by the observed class prevalence; balanced accuracy is reported rather than plain accuracy as the main thresholded summary.

## 6. Acquisition and evidence rules

The repository page displayed revision prefix `dd0d22e`. Before the first media shard, the acquisition script resolved and recorded the full immutable commit as `dd0d22e5546e6ed26c9c2599d8dc345d2760ad7c` in `outputs/acquisition/revision_record.json`. No media was downloaded by that API lookup.

Because all 25 archives total roughly 134 GB and the data drive cannot safely hold them together, acquisition is sequential:

1. download one archive into an isolated temporary cache;
2. find only files named in the frozen manifest;
3. extract one selected video at a time and decode its audio losslessly to mono 16 kHz FLAC;
4. record archive, video and audio hashes plus exact lineage;
5. save the shard record;
6. remove that archive cache before downloading the next archive.

The script must inspect all shards because a shard number is only a storage division, not an experimental subset. The official full-loader command is not used.

## 7. Permitted claims and limitations

If the decision rule is met, the result supports additive-noise augmentation for this fixed ASVspoof → DeepSpeak transfer using this frozen XLS-R layer and linear probe. It does not establish that noise is best for every layer, encoder, source dataset, fake generator or deployment condition.

Known limitations remain:

- layer 10 was motivated by prior In-the-Wild exploration;
- only one frozen augmentation draw per source clip is used, so this study does not estimate sensitivity to alternative noise draws;
- DeepSpeak fake clips share participants and recording lineage, so clip-level independence is false; identity-cluster uncertainty is therefore primary;
- DeepSpeak contains video manipulations, but this protocol evaluates only audio labels determined by `audio-config`;
- target labels are necessarily used to calculate final evaluation metrics, but never for model or threshold selection.

Any change after this freeze must be documented as a dated protocol amendment before the affected result is interpreted.

## 中文摘要

这次实验不是重新“证明”上一轮 In-the-Wild 结果，而是在一个尚未评分的新目标域 DeepSpeak 上做确认。源域仍为 ASVspoof 2019 LA；只比较 `none` 与 `noise`；两者共同固定 XLS-R 第 10 层和 `C=0.1`。DeepSpeak 官方 test identity 中符合音频标签规则的 1,800 个样本已经冻结。只有当 `EER_noise - EER_none` 的 95% identity-cluster bootstrap 区间完全低于 0，才称噪声增强得到确认；否则如实报告为不确定或有害。

