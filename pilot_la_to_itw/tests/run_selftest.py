"""End-to-end software self-test on tiny synthetic zips (no download, no XLS-R, no GPU).

Builds fake LA.zip / release_in_the_wild.zip with the real archive layouts, then runs
steps 1-4 with the FAKE spectral encoder. Numbers produced here are meaningless.

    python tests/run_selftest.py            # uses a temp folder, deletes it afterwards
    python tests/run_selftest.py --keep     # keep the temp folder to inspect outputs
    python tests/run_selftest.py --workers 2  # also test multi-process audio loading (needs torch)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot_common import eer_and_threshold, ffmpeg_exe, roc_auc  # noqa: E402

SR = 16000


def synth(kind: str, rng: np.random.Generator, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(100, 220)
    if kind == "real":
        sig = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 8))
        sig *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t) ** 2
        sig += 0.05 * rng.standard_normal(n)
    else:
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / (k ** 0.5) for k in range(1, 12))
        sig += 0.02 * rng.standard_normal(n)
    sig = 0.3 * sig / np.max(np.abs(sig))
    lead = np.zeros(int(rng.uniform(0.1, 0.6) * SR))
    return np.concatenate([lead, sig, np.zeros(int(0.2 * SR))]).astype(np.float32)


def encode(signal: np.ndarray, fmt: str, ffmpeg: str) -> bytes:
    codec = ["-c:a", "flac", "-f", "flac"] if fmt == "flac" else ["-c:a", "pcm_s16le", "-f", "wav"]
    proc = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                           "-i", "pipe:0", *codec, "pipe:1"], input=signal.tobytes(), capture_output=True, check=True)
    return proc.stdout


def make_la(path: Path, ffmpeg: str, rng) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for part, prefix, letter, speakers in (("train", "trn", "T", ["LA_0079", "LA_0080", "LA_0081"]),
                                               ("dev", "trl", "D", ["LA_0069", "LA_0070"])):
            lines = []
            for i in range(60):
                utt = f"LA_{letter}_{1000000 + i}"
                spk = speakers[i % len(speakers)]
                if i % 3 == 0:
                    lines.append(f"{spk} {utt} - - bonafide")
                    kind = "real"
                else:
                    lines.append(f"{spk} {utt} - A0{1 + (i // 3) % 6} spoof")
                    kind = "fake"
                zf.writestr(f"LA/ASVspoof2019_LA_{part}/flac/{utt}.flac", encode(synth(kind, rng, rng.uniform(1.0, 3.0)), "flac", ffmpeg))
            zf.writestr(f"LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.{prefix}.txt", "\n".join(lines) + "\n")


def make_itw(path: Path, ffmpeg: str, rng) -> None:
    rows = ["file,speaker,label"]
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(60):
            label = "bona-fide" if i % 2 else "spoof"
            rows.append(f"{i}.wav,Speaker {i % 7},{label}")
            sig = synth("real" if label == "bona-fide" else "fake", rng, rng.uniform(1.0, 8.0))
            sig = sig + 0.03 * rng.standard_normal(len(sig)).astype(np.float32)
            zf.writestr(f"release_in_the_wild/{i}.wav", encode(sig, "wav", ffmpeg))
        zf.writestr("release_in_the_wild/meta.csv", "\n".join(rows) + "\n")


def unit_tests() -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 500)
        p = np.clip(rng.normal(0.4 + 0.2 * y, 0.2), 0, 1)
        eer, thr = eer_and_threshold(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        k = np.nanargmin(np.abs(fpr - (1 - tpr)))
        assert abs(eer - (fpr[k] + 1 - tpr[k]) / 2) < 0.01, (eer, fpr[k], 1 - tpr[k])
        assert abs(roc_auc(y, p) - roc_auc_score(y, p)) < 1e-9
        pred = p >= thr
        far, frr = np.mean(pred[y == 0]), np.mean(~pred[y == 1])
        assert abs((far + frr) / 2 - eer) < 0.02
    assert eer_and_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))[0] == 0.0
    print("unit tests: EER / AUC helpers OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    unit_tests()
    os.environ["PILOT_SKIP_KNOWN_ARCHIVE_CHECK"] = "1"  # tiny fake archives reuse the official file names
    ffmpeg = ffmpeg_exe()
    tmp = Path(tempfile.mkdtemp(prefix="pilot_selftest_"))
    try:
        rng = np.random.default_rng(1)
        (tmp / "raw").mkdir()
        make_la(tmp / "raw" / "LA.zip", ffmpeg, rng)
        # Mimic the student's extracted copy (D:\\7841\\LA): official sub-folders at the root, plus a
        # duplicate top-level flac/ folder and a duplicate train protocol that must be ignored.
        la_dir = tmp / "LA_extracted"
        with zipfile.ZipFile(tmp / "raw" / "LA.zip") as zf:
            for name in zf.namelist():
                rel = name[len("LA/"):]
                dest = la_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))
                if rel.startswith("ASVspoof2019_LA_train/flac/"):
                    dup = la_dir / "flac" / Path(rel).name
                    dup.parent.mkdir(parents=True, exist_ok=True)
                    dup.write_bytes(zf.read(name))
        (la_dir / "ASVspoof2019.LA.cm.train.trn.txt").write_bytes(
            (la_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt").read_bytes())
        make_itw(tmp / "raw" / "release_in_the_wild.zip", ffmpeg, rng)
        cfg = json.loads((ROOT / "config" / "pilot_config.json").read_text(encoding="utf-8"))
        test_cfg = copy.deepcopy(cfg)
        test_cfg["paths"]["output_dir"] = "{data_root}/outputs"
        test_cfg["source"]["train_pool"].update({"n_bonafide": 12, "n_spoof_per_attack": 3})
        test_cfg["source"]["validation"].update({"n_bonafide": 8, "n_spoof_total": 14})
        test_cfg["target"]["core_subset"]["n_per_label"] = 10
        test_cfg["feature_extractor"]["shard_size"] = 7
        test_cfg["selection"]["candidate_C"] = [0.1, 1.0]
        cfg_path = tmp / "selftest_config.json"
        cfg_path.write_text(json.dumps(test_cfg, indent=2), encoding="utf-8")
        common = ["--config", str(cfg_path), "--data-root", str(tmp)]
        py = sys.executable
        steps = [
            [py, str(ROOT / "src" / "step0_check_env.py"), *common, "--la-source", str(la_dir)],
            [py, str(ROOT / "src" / "step1_build_manifests.py"), *common, "--la-source", str(la_dir)],
            [py, str(ROOT / "src" / "step2_extract_xlsr.py"), *common, "--fake-model", "--workers", str(args.workers),
             "--only", "source_val/clean,source_train/clean,source_train/codec"],
            [py, str(ROOT / "src" / "step2_extract_xlsr.py"), *common, "--fake-model", "--workers", str(args.workers)],
            [py, str(ROOT / "src" / "step3_train_eval.py"), *common, "--allow-fake-model", "--bootstrap-reps", "50"],
        ]
        for i, cmd in enumerate(steps):
            print(f"\n=== self-test step {i}: {Path(cmd[1]).name} ===", flush=True)
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0 and i != 0:  # step 0 may report missing torch/transformers in a test env
                print(f"SELF-TEST FAILED at {Path(cmd[1]).name}")
                return 1
        print("\n=== negative check: step 3 must refuse fake-encoder embeddings without --allow-fake-model ===")
        refused = subprocess.run([py, str(ROOT / "src" / "step3_train_eval.py"), *common], cwd=ROOT, capture_output=True, text=True)
        assert refused.returncode != 0 and "FAKE" in (refused.stdout + refused.stderr), refused.stdout + refused.stderr
        print("refused as expected")
        print("\n=== negative check: editing a frozen manifest must be detected ===")
        manifest = tmp / "outputs" / "manifests" / "target_itw.csv"
        original = manifest.read_bytes()
        manifest.write_bytes(original.replace(b"spoof", b"bona-fide", 1))
        tampered = subprocess.run([py, str(ROOT / "src" / "step3_train_eval.py"), *common, "--allow-fake-model"], cwd=ROOT, capture_output=True, text=True)
        assert tampered.returncode != 0 and "changed after freezing" in (tampered.stdout + tampered.stderr)
        manifest.write_bytes(original)
        print("tampering detected as expected")
        print("\n=== equivalence check: LA.zip and the extracted folder must give identical source manifests ===")
        zip_cfg = copy.deepcopy(test_cfg)
        zip_cfg["paths"]["output_dir"] = "{data_root}/outputs_zipcheck"
        zip_cfg_path = tmp / "selftest_config_zip.json"
        zip_cfg_path.write_text(json.dumps(zip_cfg, indent=2), encoding="utf-8")
        zrun = subprocess.run([py, str(ROOT / "src" / "step1_build_manifests.py"), "--config", str(zip_cfg_path),
                               "--data-root", str(tmp), "--skip-zip-hash"], cwd=ROOT, capture_output=True, text=True)
        assert zrun.returncode == 0, zrun.stdout + zrun.stderr
        for name in ("source_train_pool.csv", "source_validation.csv"):
            def key(path):
                import csv as _csv
                with open(path, encoding="utf-8") as h:
                    return [(r["sample_id"], r["label"], r["sha256"]) for r in _csv.DictReader(h)]
            assert key(tmp / "outputs" / "manifests" / name) == key(tmp / "outputs_zipcheck" / "manifests" / name), name
        print("identical sample IDs, labels and audio hashes")
        run_dir = sorted((tmp / "outputs").glob("run_*"))[-1]
        for name in ["RESULTS.md", "summary.json", "predictions_target.csv", "figures/fig1_target_eer_by_condition.png",
                     "figures/fig2_target_roc.png", "figures/fig3_target_score_distributions.png",
                     "figures/fig4_layer_profile_exploratory.png", "layer_profile_EXPLORATORY.csv", "shortcut_diagnostic.json"]:
            assert (run_dir / name).exists(), name
        print(f"\nSELF-TEST PASSED (software only; scores are meaningless). Outputs: {run_dir}")
        return 0
    finally:
        if args.keep:
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
