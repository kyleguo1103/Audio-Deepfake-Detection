"""EXPLORATORY sensitivity analysis (added after the primary results were saved; not pre-registered).

Question: is the primary gain (mix vs none) caused by augmentation itself, or by the two conditions
selecting different XLS-R layers on source validation?

For every candidate layer and every condition, C is still chosen on SOURCE VALIDATION ONLY (same rule as
step 3), the three seeds are refit, and the target is scored. Comparing conditions at the SAME layer removes
layer selection from the comparison. Results go to <run>/exploratory_fixed_layer/ and never replace the
primary result.

Pre-stated reading rule (written before this analysis was run):
  * augmentation effect "robust to layer choice" only if the layer-averaged EER difference (mix - none)
    has a 95% speaker-cluster bootstrap CI entirely below 0 AND mix is better at a majority of layers;
  * otherwise the primary gain is described as depending on layer selection.

Usage: python src/explore_fixed_layer.py [--run outputs/run_YYYYmmdd_HHMMSS] [--bootstrap-reps 1000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from augment import FAMILIES
from pilot_common import (LABEL_TO_INT, eer_and_threshold, full_metrics, load_config, load_manifest, now_utc,
                          roc_auc, verify_manifest_hashes, write_csv, write_json)
from step3_train_eval import TrainData, align, load_embeddings, make_model, stratified_subset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--run", help="run directory (default: outputs/LATEST_RUN.txt)")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    t0 = time.time()
    cfg = load_config(args.config, args.data_root)
    verify_manifest_hashes(cfg)
    out_root = cfg["_paths"]["output_dir"]
    run_dir = Path(args.run) if args.run else out_root / (out_root / "LATEST_RUN.txt").read_text().strip()
    out_dir = run_dir / "exploratory_fixed_layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = cfg["_paths"]["embedding_dir"]

    pool = load_manifest(cfg, "source_train")
    val_manifest = load_manifest(cfg, "source_val")
    target_manifest = sorted(load_manifest(cfg, "target"), key=lambda r: int(r["order_index"]))
    train_sets = {v: load_embeddings(emb_dir, "source_train", v) for v in ("clean",) + FAMILIES}
    val_set = load_embeddings(emb_dir, "source_val", "clean")
    target_set = load_embeddings(emb_dir, "target", "clean")
    data = TrainData(cfg, pool, train_sets)
    val_rows = align(val_manifest, val_set, "source_val")
    tgt_rows = align(target_manifest, target_set, "target")
    y_val = np.array([LABEL_TO_INT[r["label"]] for r in val_manifest])
    y_tgt = np.array([LABEL_TO_INT[r["label"]] for r in target_manifest])
    speakers = np.array([r["speaker_id"] for r in target_manifest])
    sel = cfg["selection"]
    layers = sel["candidate_layers"]
    seeds = cfg["seeds"]
    conditions = list(cfg["conditions"])
    subsets = {s: stratified_subset(data.labels, cfg["train_subset_fraction"], s) for s in seeds}

    rows, scores = [], {}  # scores[(cond, layer)] = [n_seeds, n_target]
    for layer in layers:
        Xv = val_set.layer(layer, val_rows)
        Xt = target_set.layer(layer, tgt_rows)
        for cond in conditions:
            best = None
            X, y = data.build(cond, subsets[sel["seed_used_for_selection"]], layer)
            for C in sel["candidate_C"]:
                pv = make_model(C, cfg).fit(X, y).decision_function(Xv)
                m = full_metrics(y_val, pv, 0.0)
                key = (round(m["log_loss"], 6), round(m["eer"], 6), C)
                if best is None or key < best[0]:
                    best = (key, C)
            C = best[1]
            per_seed = []
            for s in seeds:
                X, y = data.build(cond, subsets[s], layer)
                model = make_model(C, cfg).fit(X, y)
                pv = model.decision_function(Xv)
                _, thr = eer_and_threshold(y_val, pv)
                pt = model.decision_function(Xt)
                mt = full_metrics(y_tgt, pt, thr)
                rows.append({"layer": layer, "condition": cond, "C": C, "seed": s,
                             "val_eer": eer_and_threshold(y_val, pv)[0], "target_eer": mt["eer"],
                             "target_auc": mt["roc_auc"], "target_bal_acc_src_thr": mt["balanced_accuracy"]})
                per_seed.append(pt)
            scores[(cond, layer)] = np.array(per_seed)
            eers = [r["target_eer"] for r in rows if r["layer"] == layer and r["condition"] == cond]
            print(f"layer {layer:2d} {cond:8s} C={C:<6} target EER {np.mean(eers):.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        write_csv(out_dir / "per_layer_per_seed.csv", rows, list(rows[0]))

    # summary table: mean over seeds
    summary = []
    for layer in layers:
        base = np.mean([r["target_eer"] for r in rows if r["layer"] == layer and r["condition"] == "none"])
        for cond in conditions:
            sub = [r for r in rows if r["layer"] == layer and r["condition"] == cond]
            e = np.mean([r["target_eer"] for r in sub])
            summary.append({"layer": layer, "condition": cond, "C": sub[0]["C"],
                            "target_eer_mean": e, "delta_vs_none_same_layer": e - base,
                            "target_auc_mean": np.mean([r["target_auc"] for r in sub]),
                            "bal_acc_src_thr_mean": np.mean([r["target_bal_acc_src_thr"] for r in sub]),
                            "val_eer_mean": np.mean([r["val_eer"] for r in sub])})
    write_csv(out_dir / "per_layer_summary.csv", summary, list(summary[0]))

    # layer-averaged EER per condition + paired speaker-cluster bootstrap vs none
    rng = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"])
    by_speaker = {s: np.flatnonzero(speakers == s) for s in np.unique(speakers)}
    spk = list(by_speaker)

    def layer_avg(idx, cond):
        yy = y_tgt[idx]
        return np.mean([np.mean([eer_and_threshold(yy, p[idx])[0] for p in scores[(cond, L)]]) for L in layers])

    full_idx = np.arange(len(y_tgt))
    point = {c: layer_avg(full_idx, c) for c in conditions}
    boot = {c: [] for c in conditions}
    for b in range(args.bootstrap_reps):
        pick = rng.choice(len(spk), size=len(spk), replace=True)
        idx = np.concatenate([by_speaker[spk[p]] for p in pick])
        if y_tgt[idx].min() == y_tgt[idx].max():
            continue
        for c in conditions:
            boot[c].append(layer_avg(idx, c))
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{args.bootstrap_reps} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    result = {"created_at_utc": now_utc(), "status": "EXPLORATORY_POST_HOC", "layers": layers,
              "bootstrap_reps_used": len(boot["none"]), "conditions": {}}
    for c in conditions:
        d = np.array(boot[c]) - np.array(boot["none"])
        wins = sum(1 for L in layers
                   if next(s for s in summary if s["layer"] == L and s["condition"] == c)["delta_vs_none_same_layer"] < 0)
        result["conditions"][c] = {
            "layer_averaged_target_eer": point[c],
            "layer_averaged_delta_vs_none": point[c] - point["none"],
            "delta_ci_speaker_cluster": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if c != "none" else None,
            "layers_better_than_none": wins if c != "none" else None,
        }
    mix = result["conditions"]["mix"]
    robust = mix["delta_ci_speaker_cluster"][1] < 0 and mix["layers_better_than_none"] > len(layers) / 2
    result["reading"] = ("augmentation effect robust to layer choice" if robust
                         else "primary gain depends on layer selection; fixed-layer effect not robust")
    result["minutes"] = round((time.time() - t0) / 60, 1)
    write_json(out_dir / "layer_averaged_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
