# Pilot Pre-registration: ASVspoof 2019 LA → In-the-Wild, Controlled Audio Augmentation

> Frozen: 2026-10-09 23:10 (Australia/Brisbane), **before any data was downloaded, any embedding was extracted, or any result existed.**
> Machine-readable design: `config/pilot_config.json`, SHA-256 `a26f7873f8b54d9224894bae0ab2ccf7b6588e3a1cc42b51e90a44b859f39782`. Every run records this hash; a different hash means the design was changed and must be reported as a deviation.
> 中文：本文件在下载数据、提取特征和得到任何结果**之前**冻结。后续若修改设计，必须在结果中写明偏离。

## 1. Why this pilot exists / 为什么做这个试点

The supervisor confirmed the direction *Improving Cross-Domain Generalization of XLS-R-Based Audio Deepfake Detection through Controlled Audio Augmentation* (2026-09-21). DeepSpeak v2.0 access is not yet evidenced, so this pilot substitutes **ASVspoof 2019 LA** as the source domain to test the method and the full pipeline on real data now. It is **not** the final thesis experiment and does not replace the DeepSpeak → In-the-Wild plan.

中文：导师已认可研究方向，但 DeepSpeak 访问尚无证据。本试点用 ASVspoof 2019 LA 代替源域，先在真实数据上检验方法和整套流程；它不是最终论文实验，也不取代 DeepSpeak → In-the-Wild 的计划。

## 2. Research question for the pilot / 试点问题

When an XLS-R + logistic-regression detector is trained only on ASVspoof 2019 LA, does adding pre-declared waveform augmentation (codec, noise, channel) to the source training audio lower its EER on the unseen In-the-Wild dataset, compared with the identical pipeline without augmentation?

## 3. Data / 数据

| Role | Dataset | Selection (seeded, fixed before results) |
|---|---|---|
| Source train pool | ASVspoof 2019 LA **train** | 1,500 bona fide + 250 spoof per attack A01–A06 (1,500) = 3,000 clips, 20 speakers |
| Source validation | ASVspoof 2019 LA **dev** (speaker-disjoint from train) | 1,000 bona fide + 1,000 spoof spread over A01–A06 |
| Target (final evaluation only) | In-the-Wild (`release_in_the_wild.zip`) | **All clips**. A balanced core subset (3,000 per label, seed 2026) is extracted first and is used **only** if the full extraction cannot finish; whichever is reported is stated in the results. |

Labels: `bonafide`/`bona-fide` → `real` (0); `spoof` → `fake` (1). Step 1 hashes every selected file and freezes the manifests; later steps refuse to run if a manifest changes.

## 4. Pipeline / 流程

1. Decode → mono 16 kHz → trim leading/trailing silence (frame energy > −40 dB of clip max; fall back to untrimmed if < 0.5 s remains) → keep the first 6.0 s, no padding.
   Silence trimming is applied to every set because ASVspoof 2019 LA is known to leak the label through silence duration (Müller et al., 2021).
2. **Augmentation (source train only, waveform level, same rules for real and fake, global seed 7841):** one copy per family per clip.
   - codec: MP3 / AAC (24–96 kbps) or Opus (12–48 kbps), via ffmpeg;
   - noise: white / pink / brown at SNR 5–20 dB;
   - channel: telephone band (300–3,400 Hz, 8 kHz), low-pass 3–7 kHz, 3-band parametric EQ (±8 dB), or synthetic reverb (RT60 0.2–0.7 s).
3. Features: `facebook/wav2vec2-large-xlsr-53` (frozen), per-utterance normalisation, mean-pooled over time for all 25 hidden states; the resolved model commit is recorded.
4. Classifier: StandardScaler + LogisticRegression (class-balanced).

## 5. Conditions / 实验条件

| Condition | Training data |
|---|---|
| `none` (control) | clean source clips |
| `codec`, `noise`, `channel` | clean + one augmented copy from that family |
| **`mix` (primary treatment)** | clean + one augmented copy, family fixed per clip by seed (size-matched to single-family conditions) |
| `all` | clean + all three augmented copies |

## 6. Selection and evaluation rules / 选择与评估规则

- Hidden-state layer ∈ {0, 2, …, 24} and C ∈ {0.001, 0.01, 0.1, 1} are chosen **per condition on source validation only** (minimum log-loss; ties → lower EER → smaller C → smaller layer), using the seed-42 training subset.
- Seeds 42, 1337, 2026: each draws a stratified 80% subset of the source pool; augmented copies follow their parent clip.
- Decision threshold = source-validation EER point; target labels are never used for fitting, selection, thresholding or calibration.
- Target scored once per seed. Ranking metrics use classifier logits.

**Primary endpoint:** target EER, `mix` − `none`, averaged over seeds, with a 95% **speaker-cluster** bootstrap CI (1,000 resamples, paired across conditions).
**Decision rule:** improvement is supported only if the whole CI is below 0; harm if the whole CI is above 0; otherwise inconclusive.

Secondary: ROC AUC, PR AUC, balanced accuracy and macro F1 at the source threshold, per-family ablation, `all`, in-domain source-validation metrics, and a duration/silence-only shortcut baseline. An **exploratory** per-layer target profile is computed only after the primary results are saved and is never used for selection.

## 7. What this pilot can and cannot claim / 可以和不可以声称的内容

- Can: a real-data, pre-registered test of whether these augmentations help this source → target transfer, with uncertainty.
- Cannot: conclusions about DeepSpeak, other targets, other detectors, or the final thesis result. Augmentation randomness uses one fixed seed, so seed variation reflects training-subset sampling, not alternative augmentation draws. ASVspoof 2019 LA attacks (2019-era TTS/VC) are older than In-the-Wild fakes.
- A null or negative result is reported as it is.

## Addendum A — 2026-10-10, before any result: source audio location / 补记 A：源域音频位置（仍在任何结果之前）

The student already holds an extracted copy of the official ASVspoof 2019 LA release at `D:\7841\LA`, so the pipeline reads that folder (`--la-source D:\7841\LA`) instead of downloading `LA.zip`. Files are resolved by their official relative paths (`ASVspoof2019_LA_cm_protocols/…`, `ASVspoof2019_LA_{train,dev}/flac/…`). **Sampling, labels, seeds and every design choice are unchanged; the config hash is unchanged.**

Checks made on 2026-10-10:

- CM protocols: train 25,380 rows (2,580 bona fide; 3,800 per attack A01–A06), dev 24,844 (2,548; 3,716 per attack), eval 71,237 — the official counts. The duplicate train protocol at the folder root is byte-identical to the one in `ASVspoof2019_LA_cm_protocols`.
- Audio counts: train 25,380; dev 24,986 = 24,844 + 142 ASV enrolment files; eval 71,933 = 71,237 + 696 ASV enrolment files (enrolment IDs taken from the ASV `trn` protocols).
- 21 randomly chosen files from train/dev/eval decode cleanly as FLAC, 16 kHz, mono, 16-bit.
- The root-level `flac/` folder (25,380 files) and `ASVspoof2019_LA_train.zip` duplicate the training set and are not used.
- Train and dev speakers are disjoint (20 and 20 speaker IDs).

Limitation: the copy was checked by counts and samples, not byte-for-byte against the distributor's checksum. Step 1 records the SHA-256 of every selected file and stops if any selected file is missing.

中文：学生已有官方 ASVspoof 2019 LA 的解压副本，试点直接读取 `D:\7841\LA`，不再下载 `LA.zip`。抽样、标签、种子和全部设计不变，配置哈希不变。核查结果：协议行数与官方一致；音频数量与协议加 ASV 注册文件数量完全吻合；抽查 21 个文件均可正常解码；根目录的 `flac/` 和 `ASVspoof2019_LA_train.zip` 是训练集重复副本，不使用。

## Addendum B — 2026-10-10, before any result: target archive location / 补记 B：目标域压缩包位置

`release_in_the_wild.zip` is stored at `D:\7841\raw\` and read in place (`--itw-source`). Step 1 refuses a file whose size differs from the official 8,161,489,023 bytes and compares the whole-file SHA-256 with the official Hugging Face LFS value `46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a`. Design unchanged.

中文：目标域压缩包放在 `D:\7841\raw\` 并直接读取；第 1 步会核对其大小与官方 SHA-256，不一致即停止。设计不变。
