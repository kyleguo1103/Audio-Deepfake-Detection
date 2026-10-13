"""Recompute the reported pilot numbers from the committed raw results.

No audio, embeddings or GPU are needed. The script reads the frozen configuration and the
sample-level predictions in pilot_la_to_itw/outputs/<run>/ and checks that

  1. the configuration and manifests match the SHA-256 values recorded by the run;
  2. per-seed target EERs recomputed from the saved logits reproduce summary.json;
  3. the pre-registered paired speaker-cluster bootstrap CI (mix - none) is reproduced
     with the frozen bootstrap seed (1,000 resamples by default);
  4. source-validation EERs reproduce summary.json;
  5. the exploratory matched-layer averages and layer counts reproduce
     layer_averaged_summary.json, and the layer-selection decomposition used in the paper.

Matched-layer confidence intervals need the per-layer logits, which are not saved; they can only be
re-derived by re-running src/explore_fixed_layer.py on extracted embeddings (see REPRODUCE.md).

Usage (from the repository root):
    python scripts/verify_results.py                 # full check, about 1-3 minutes on a laptop CPU
    python scripts/verify_results.py --reps 200      # quicker; bootstrap CI only approximately equal
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "pilot_la_to_itw"
sys.path.insert(0, str(PILOT / "src"))

from pilot_common import eer_and_threshold, roc_auc  # noqa: E402

CONDITIONS = ["none", "codec", "noise", "channel", "mix", "all"]
SEEDS = [42, 1337, 2026]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    y = np.array([1 if r["label"] == "fake" else 0 for r in rows])
    speakers = np.array([r["speaker_id"] for r in rows])
    scores = {c: np.array([[float(r[f"score_{c}_seed{s}"]) for r in rows] for s in SEEDS]) for c in CONDITIONS}
    return y, speakers, scores


def paired_speaker_bootstrap(y, scores, speakers, control, reps, seed):
    """Same resampling scheme as step3_train_eval.bootstrap(kind='speaker_cluster')."""
    rng = np.random.default_rng(seed)
    by_speaker = {s: np.flatnonzero(speakers == s) for s in np.unique(speakers)}
    speaker_list = list(by_speaker)
    eer = {c: [] for c in scores}
    for _ in range(reps):
        picked = rng.choice(len(speaker_list), size=len(speaker_list), replace=True)
        idx = np.concatenate([by_speaker[speaker_list[p]] for p in picked])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for c in scores:
            eer[c].append(np.mean([eer_and_threshold(yy, p[idx])[0] for p in scores[c]]))
    ctrl = np.array(eer[control])
    return {c: [float(np.percentile(np.array(v) - ctrl, 2.5)), float(np.percentile(np.array(v) - ctrl, 97.5))]
            for c, v in eer.items()}


class Checker:
    def __init__(self):
        self.failures = 0

    def check(self, label, got, want, tol):
        ok = abs(got - want) <= tol
        self.failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: recomputed {got:.4f} vs recorded {want:.4f} (tol {tol})")
        return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run folder name (default: outputs/LATEST_RUN.txt)")
    parser.add_argument("--reps", type=int, default=None, help="bootstrap resamples (default: frozen value, 1000)")
    args = parser.parse_args()

    cfg_path = PILOT / "config" / "pilot_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    run_name = args.run or (PILOT / "outputs" / "LATEST_RUN.txt").read_text(encoding="utf-8").strip()
    run = PILOT / "outputs" / run_name
    record = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    summary = {c["condition"]: c for c in json.loads((run / "summary.json").read_text(encoding="utf-8"))["conditions"]}
    chk = Checker()

    print(f"Run: {run_name}")
    print("1. Frozen configuration and manifests")
    for label, got, want in [("config", sha256(cfg_path), record["config_sha256"])] + [
        (name, sha256(PILOT / "outputs" / "manifests" / f), record["manifest_sha256"][name])
        for name, f in [("source_train", "source_train_pool.csv"), ("source_val", "source_validation.csv"),
                        ("target", "target_itw.csv")]
    ]:
        ok = got == want
        chk.failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} SHA-256 {got[:16]}...")

    print("2. Target EER per condition (mean of 3 seeds, from saved logits)")
    y, speakers, scores = load_predictions(run / "predictions_target.csv")
    print(f"  target clips: {len(y)} (fake {int(y.sum())}, real {int(len(y) - y.sum())}), "
          f"speakers: {len(np.unique(speakers))}")
    eers = {}
    for c in CONDITIONS:
        eers[c] = float(np.mean([eer_and_threshold(y, s)[0] for s in scores[c]]))
        chk.check(f"{c:8s} target EER", eers[c], summary[c]["target_eer_mean"], 1e-4)
        chk.check(f"{c:8s} target AUC", float(np.mean([roc_auc(y, s) for s in scores[c]])),
                  summary[c]["target_roc_auc_mean"], 1e-4)
    control, treatment = cfg["primary_comparison"]["control"], cfg["primary_comparison"]["treatment"]
    chk.check(f"primary delta ({treatment} - {control})", eers[treatment] - eers[control],
              summary[treatment]["delta_eer_vs_none"], 1e-4)

    print("3. Pre-registered paired speaker-cluster bootstrap")
    reps = args.reps or cfg["evaluation"]["bootstrap_reps"]
    seed = cfg["evaluation"]["bootstrap_seed"]
    print(f"  {reps} resamples, seed {seed} (this is the slow part)...")
    cis = paired_speaker_bootstrap(y, scores, speakers, control, reps, seed)
    tol = 0.002 if reps == cfg["evaluation"]["bootstrap_reps"] else 0.02
    for c in CONDITIONS[1:]:
        lo, hi = cis[c]
        want_lo, want_hi = summary[c]["delta_eer_ci_speaker"]
        chk.check(f"{c:8s} delta CI lower", lo, want_lo, tol)
        chk.check(f"{c:8s} delta CI upper", hi, want_hi, tol)
    lo, hi = cis[treatment]
    verdict = "improvement supported" if hi < 0 else ("harm" if lo > 0 else "inconclusive")
    print(f"  decision rule on recomputed CI [{100 * lo:.2f}, {100 * hi:.2f}] pp -> {verdict}")

    print("4. Source-validation EER")
    yv, _, sv = load_predictions(run / "predictions_source_validation.csv")
    for c in CONDITIONS:
        chk.check(f"{c:8s} source-val EER", float(np.mean([eer_and_threshold(yv, s)[0] for s in sv[c]])),
                  summary[c]["source_val_eer_mean"], 1e-4)

    print("5. Exploratory matched-layer analysis (post hoc)")
    ex = run / "exploratory_fixed_layer"
    rows = list(csv.DictReader((ex / "per_layer_summary.csv").open(encoding="utf-8", newline="")))
    lay = json.loads((ex / "layer_averaged_summary.json").read_text(encoding="utf-8"))
    table = {(r["condition"], int(r["layer"])): r for r in rows}
    layers = lay["layers"]
    for c in CONDITIONS:
        avg = float(np.mean([float(table[(c, L)]["target_eer_mean"]) for L in layers]))
        chk.check(f"{c:8s} layer-averaged EER", avg, lay["conditions"][c]["layer_averaged_target_eer"], 1e-6)
        if c != "none":
            better = sum(float(table[(c, L)]["target_eer_mean"]) < float(table[("none", L)]["target_eer_mean"])
                         for L in layers)
            ok = better == lay["conditions"][c]["layers_better_than_none"]
            chk.failures += 0 if ok else 1
            ci = lay["conditions"][c]["delta_ci_speaker_cluster"]
            print(f"  [{'PASS' if ok else 'FAIL'}] {c:8s} better at {better}/13 layers; recorded 95% CI "
                  f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp (needs embeddings to recompute)")
    t = lambda c, L: 100 * float(table[(c, L)]["target_eer_mean"])  # noqa: E731
    print(f"  decomposition: none L2 {t('none', 2):.1f}% -> none L20 {t('none', 20):.1f}% "
          f"({t('none', 20) - t('none', 2):+.1f} pp); mix L20 {t('mix', 20):.1f}% "
          f"({t('mix', 20) - t('none', 20):+.1f} pp vs none at L20)")

    print()
    print("ALL CHECKS PASSED" if chk.failures == 0 else f"{chk.failures} CHECK(S) FAILED")
    return 0 if chk.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
