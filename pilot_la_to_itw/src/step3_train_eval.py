"""Step 3: source-only training, validation-only selection, single final evaluation on In-the-Wild.

Order of operations (fixed in config/pilot_config.json before any result):
  1. verify frozen manifests and that embeddings are complete;
  2. for every condition, pick (layer, C) on SOURCE VALIDATION ONLY (seed 42 subset);
  3. refit for each seed, set the decision threshold at the source-validation EER point;
  4. score the target once, save sample-level predictions, metrics and paired bootstrap CIs;
  5. diagnostics (duration/silence shortcut baseline) and, only after the primary results
     are written, an exploratory per-layer profile that is never used for selection.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from augment import FAMILIES, mix_family
from pilot_common import (LABEL_TO_INT, eer_and_threshold, full_metrics, load_config, load_manifest, now_utc,
                          read_csv, roc_auc, verify_manifest_hashes, write_csv, write_json)

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------- loading

class EmbeddingSet:
    def __init__(self, matrix: np.ndarray, info: list[dict[str, str]]):
        self.matrix = matrix  # float16 [n, L, D]
        self.info = info
        self.index = {row["sample_id"]: i for i, row in enumerate(info)}

    def layer(self, layer: int, rows: np.ndarray | None = None) -> np.ndarray:
        data = self.matrix[:, layer, :] if rows is None else self.matrix[rows, layer, :]
        return data.astype(np.float32)


def load_embeddings(embedding_dir: Path, set_name: str, variant: str) -> EmbeddingSet | None:
    folder = embedding_dir / set_name / variant
    shards = sorted(folder.glob("shard_[0-9][0-9][0-9][0-9].npy"))
    if not shards:
        return None
    matrices, info = [], []
    for shard in shards:
        csv_path = shard.with_suffix(".csv")
        if not csv_path.exists():
            break
        matrices.append(np.load(shard))
        info.extend(read_csv(csv_path))
    if not matrices:
        return None
    return EmbeddingSet(np.concatenate(matrices, axis=0), info)


def align(manifest: list[dict[str, str]], emb: EmbeddingSet, what: str) -> np.ndarray:
    missing = [r["sample_id"] for r in manifest if r["sample_id"] not in emb.index]
    if missing:
        raise SystemExit(f"{what}: {len(missing)} manifest clips have no embedding (first: {missing[0]}). Run step 2 again.")
    return np.array([emb.index[r["sample_id"]] for r in manifest], dtype=np.int64)


# ----------------------------------------------------------------------------- model

def make_model(C: float, cfg) -> "Pipeline":
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = cfg["classifier"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, class_weight=clf["class_weight"], solver=clf["solver"], max_iter=clf["max_iter"])),
    ])


def stratified_subset(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        k = int(round(fraction * len(idx)))
        keep.append(np.sort(rng.choice(idx, size=k, replace=False)))
    return np.sort(np.concatenate(keep))


class TrainData:
    """Builds the training matrix of any condition for any pool subset."""

    def __init__(self, cfg, pool_manifest, train_sets: dict[str, EmbeddingSet]):
        self.cfg = cfg
        self.labels = np.array([LABEL_TO_INT[r["label"]] for r in pool_manifest], dtype=np.int64)
        self.sample_ids = [r["sample_id"] for r in pool_manifest]
        self.rows = {v: align(pool_manifest, s, f"source_train/{v}") for v, s in train_sets.items()}
        self.sets = train_sets
        self.mix = np.array([mix_family(sid, cfg) for sid in self.sample_ids])

    def build(self, condition: str, subset: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray]:
        parts, labels = [], []
        for item in self.cfg["conditions"][condition]:
            if item == "mix_one_family_per_clip":
                for family in FAMILIES:
                    chosen = subset[self.mix[subset] == family]
                    if len(chosen):
                        parts.append(self.sets[family].layer(layer, self.rows[family][chosen]))
                        labels.append(self.labels[chosen])
            else:
                parts.append(self.sets[item].layer(layer, self.rows[item][subset]))
                labels.append(self.labels[subset])
        return np.concatenate(parts), np.concatenate(labels)


# ----------------------------------------------------------------------------- bootstrap

def bootstrap(y, probs: dict[str, np.ndarray], speakers: np.ndarray, control: str, reps: int, seed: int, kind: str):
    """probs[cond] = classifier logits, shape [n_seeds, n]. Returns per-condition and paired-difference CIs (seed-averaged metrics)."""
    rng = np.random.default_rng(seed)
    conds = list(probs)
    by_speaker = {s: np.flatnonzero(speakers == s) for s in np.unique(speakers)}
    speaker_list = list(by_speaker)
    by_label = [np.flatnonzero(y == v) for v in (0, 1)]
    eer = {c: [] for c in conds}
    auc = {c: [] for c in conds}
    for _ in range(reps):
        if kind == "speaker_cluster":
            picked = rng.choice(len(speaker_list), size=len(speaker_list), replace=True)
            idx = np.concatenate([by_speaker[speaker_list[p]] for p in picked])
        else:
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in by_label])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for c in conds:
            eer[c].append(np.mean([eer_and_threshold(yy, p[idx])[0] for p in probs[c]]))
            auc[c].append(np.mean([roc_auc(yy, p[idx]) for p in probs[c]]))
    out = {}
    for c in conds:
        e, a = np.array(eer[c]), np.array(auc[c])
        de, da = e - np.array(eer[control]), a - np.array(auc[control])
        out[c] = {
            "eer_ci": [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))],
            "auc_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "delta_eer_ci": [float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))],
            "delta_auc_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "p_delta_eer_ge_0": float(np.mean(de >= 0)),
            "reps_used": int(len(e)),
        }
    return out


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--allow-fake-model", action="store_true", help="self-test only")
    parser.add_argument("--bootstrap-reps", type=int, default=0, help="override (self-test only)")
    args = parser.parse_args()
    t0 = time.time()
    cfg = load_config(args.config, args.data_root)
    manifest_hashes = verify_manifest_hashes(cfg)
    emb_dir = cfg["_paths"]["embedding_dir"]
    extraction_record = json.loads((emb_dir / "extraction_record.json").read_text(encoding="utf-8"))
    fake_runs = [r for r in extraction_record["runs"] if r.get("fake_model")]
    if fake_runs and not args.allow_fake_model:
        raise SystemExit("These embeddings were produced by the FAKE self-test encoder. Delete them and run step 2 for real.")
    if any(r.get("config_sha256") != cfg["_config_sha256"] for r in extraction_record["runs"]):
        raise SystemExit("Embeddings were extracted under a different config hash - report as a deviation and re-extract.")
    codecs_available = sorted({c for r in extraction_record["runs"] for c in r.get("codec_encoders_available", [])})
    encoders_used = {json.dumps(r["encoder"], sort_keys=True) for r in extraction_record["runs"]}
    if len(encoders_used) > 1:
        raise SystemExit(f"Embeddings were produced with different encoders/devices: {encoders_used}")

    # ---------------- data
    pool = load_manifest(cfg, "source_train")
    val_manifest = load_manifest(cfg, "source_val")
    target_manifest = sorted(load_manifest(cfg, "target"), key=lambda r: int(r["order_index"]))
    train_sets = {}
    for variant in ("clean",) + FAMILIES:
        loaded = load_embeddings(emb_dir, "source_train", variant)
        if loaded is None:
            raise SystemExit(f"missing embeddings for source_train/{variant}")
        train_sets[variant] = loaded
    val_set = load_embeddings(emb_dir, "source_val", "clean")
    target_set = load_embeddings(emb_dir, "target", "clean")
    if val_set is None or target_set is None:
        raise SystemExit("missing validation or target embeddings")

    target_scope = "all_clips"
    if any(r["sample_id"] not in target_set.index for r in target_manifest):
        core = [r for r in target_manifest if r["in_core_subset"] == "1"]
        if cfg["target"]["core_subset"]["use_only_if_full_extraction_incomplete"] and all(r["sample_id"] in target_set.index for r in core):
            target_manifest, target_scope = core, "preregistered_core_subset"
            print("NOTE: target extraction incomplete -> using the pre-registered balanced core subset (as declared).")
        else:
            raise SystemExit("target embeddings incomplete and core subset not finished - resume step 2")

    data = TrainData(cfg, pool, train_sets)
    val_rows = align(val_manifest, val_set, "source_val")
    tgt_rows = align(target_manifest, target_set, "target")
    y_val = np.array([LABEL_TO_INT[r["label"]] for r in val_manifest])
    y_tgt = np.array([LABEL_TO_INT[r["label"]] for r in target_manifest])
    tgt_speakers = np.array([r["speaker_id"] for r in target_manifest])
    n_layers = train_sets["clean"].matrix.shape[1]
    sel = cfg["selection"]
    layers = [l for l in sel["candidate_layers"] if l < n_layers]
    seeds = cfg["seeds"]
    conditions = list(cfg["conditions"])
    subsets = {s: stratified_subset(data.labels, cfg["train_subset_fraction"], s) for s in seeds}

    run_dir = cfg["_paths"]["output_dir"] / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run_dir}")
    print(f"Target scope: {target_scope}  (n={len(y_tgt)}, real={int((y_tgt == 0).sum())}, fake={int((y_tgt == 1).sum())})")

    # ---------------- selection on source validation only
    selection_rows, selected = [], {}
    sel_seed = sel["seed_used_for_selection"]
    for cond in conditions:
        best = None
        for layer in layers:
            X, y = data.build(cond, subsets[sel_seed], layer)
            Xv = val_set.layer(layer, val_rows)
            for C in sel["candidate_C"]:
                model = make_model(C, cfg).fit(X, y)
                p = model.decision_function(Xv)
                m = full_metrics(y_val, p, 0.0)
                row = {"condition": cond, "layer": layer, "C": C, "n_train": len(y),
                       "val_log_loss": m["log_loss"], "val_eer": m["eer"], "val_auc": m["roc_auc"]}
                selection_rows.append(row)
                key = (round(m["log_loss"], 6), round(m["eer"], 6), C, layer)
                if best is None or key < best[0]:
                    best = (key, layer, C)
        selected[cond] = {"layer": best[1], "C": best[2]}
        print(f"  selected for {cond:8s}: layer {best[1]:2d}, C={best[2]}  (source-validation log-loss {best[0][0]:.4f})")
    write_csv(run_dir / "selection_grid_source_validation.csv", selection_rows, list(selection_rows[0]))
    write_json(run_dir / "selected_configuration.json", selected)

    # ---------------- final fits and single target evaluation
    metric_rows = []
    probs_tgt: dict[str, np.ndarray] = {}
    probs_val: dict[str, np.ndarray] = {}
    for cond in conditions:
        layer, C = selected[cond]["layer"], selected[cond]["C"]
        Xv = val_set.layer(layer, val_rows)
        Xt = target_set.layer(layer, tgt_rows)
        pv_all, pt_all = [], []
        for s in seeds:
            X, y = data.build(cond, subsets[s], layer)
            model = make_model(C, cfg).fit(X, y)
            pv = model.decision_function(Xv)
            _, thr = eer_and_threshold(y_val, pv)
            pt = model.decision_function(Xt)
            for split, yy, pp in (("source_validation", y_val, pv), ("target", y_tgt, pt)):
                metric_rows.append({"condition": cond, "seed": s, "split": split, "layer": layer, "C": C,
                                    "n_train": len(y), **full_metrics(yy, pp, thr)})
            pv_all.append(pv)
            pt_all.append(pt)
        probs_val[cond], probs_tgt[cond] = np.array(pv_all), np.array(pt_all)
        tgt_eers = [r["eer"] for r in metric_rows if r["condition"] == cond and r["split"] == "target"]
        print(f"  {cond:8s} target EER per seed: {', '.join(f'{e:.4f}' for e in tgt_eers)}")
    write_csv(run_dir / "metrics_per_seed.csv", metric_rows, list(metric_rows[0]))

    pred_rows = []
    for i, r in enumerate(target_manifest):
        row = {"sample_id": r["sample_id"], "speaker_id": r["speaker_id"], "label": r["label"]}
        for cond in conditions:
            for k, s in enumerate(seeds):
                row[f"score_{cond}_seed{s}"] = f"{probs_tgt[cond][k, i]:.6f}"
        pred_rows.append(row)
    write_csv(run_dir / "predictions_target.csv", pred_rows, list(pred_rows[0]))
    val_pred_rows = []
    for i, r in enumerate(val_manifest):
        row = {"sample_id": r["sample_id"], "speaker_id": r["speaker_id"], "attack_id": r["attack_id"], "label": r["label"]}
        for cond in conditions:
            for k, s in enumerate(seeds):
                row[f"score_{cond}_seed{s}"] = f"{probs_val[cond][k, i]:.6f}"
        val_pred_rows.append(row)
    write_csv(run_dir / "predictions_source_validation.csv", val_pred_rows, list(val_pred_rows[0]))

    # ---------------- paired bootstrap on the target
    ev = cfg["evaluation"]
    reps = args.bootstrap_reps or ev["bootstrap_reps"]
    control = cfg["primary_comparison"]["control"]
    treatment = cfg["primary_comparison"]["treatment"]
    print(f"  bootstrapping target ({reps} reps x {len(ev['bootstrap_types'])} schemes)...")
    boot = {kind: bootstrap(y_tgt, probs_tgt, tgt_speakers, control, reps, ev["bootstrap_seed"], kind)
            for kind in ev["bootstrap_types"]}
    write_json(run_dir / "bootstrap_target.json", boot)

    summary = []
    for cond in conditions:
        rows = [r for r in metric_rows if r["condition"] == cond and r["split"] == "target"]
        vrows = [r for r in metric_rows if r["condition"] == cond and r["split"] == "source_validation"]
        entry = {"condition": cond, "layer": selected[cond]["layer"], "C": selected[cond]["C"], "n_train": rows[0]["n_train"]}
        for metric in ("eer", "roc_auc", "pr_auc_fake", "balanced_accuracy", "macro_f1", "accuracy", "recall_fake", "recall_real"):
            values = np.array([r[metric] for r in rows])
            entry[f"target_{metric}_mean"] = float(values.mean())
            entry[f"target_{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        entry["source_val_eer_mean"] = float(np.mean([r["eer"] for r in vrows]))
        entry["source_val_auc_mean"] = float(np.mean([r["roc_auc"] for r in vrows]))
        primary = boot[ev["primary_ci"]][cond]
        clip = boot.get("clip_stratified", {}).get(cond, primary)
        entry.update({
            "target_eer_ci_speaker": primary["eer_ci"], "target_auc_ci_speaker": primary["auc_ci"],
            "delta_eer_vs_none": entry["target_eer_mean"] - 0.0,
            "delta_eer_ci_speaker": primary["delta_eer_ci"], "delta_eer_ci_clip": clip["delta_eer_ci"],
            "delta_auc_ci_speaker": primary["delta_auc_ci"],
        })
        summary.append(entry)
    base = next(e for e in summary if e["condition"] == control)
    for e in summary:
        e["delta_eer_vs_none"] = e["target_eer_mean"] - base["target_eer_mean"]
        e["delta_auc_vs_none"] = e["target_roc_auc_mean"] - base["target_roc_auc_mean"]

    prim = next(e for e in summary if e["condition"] == treatment)
    lo, hi = prim["delta_eer_ci_speaker"]
    verdict = "improvement_supported" if hi < 0 else ("harm_supported" if lo > 0 else "inconclusive")
    write_json(run_dir / "summary.json", {"target_scope": target_scope, "primary_comparison": cfg["primary_comparison"],
                                          "decision_rule": ev["decision_rule"], "primary_verdict": verdict,
                                          "conditions": summary})
    print(f"  PRIMARY ({treatment} vs {control}): delta EER = {prim['delta_eer_vs_none']:+.4f}, "
          f"95% speaker-cluster CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ---------------- shortcut diagnostic: can duration / silence alone separate the classes?
    shortcut = shortcut_diagnostic(cfg, data, train_sets, subsets[sel_seed], val_set, val_rows, y_val,
                                   target_set, tgt_rows, y_tgt)
    write_json(run_dir / "shortcut_diagnostic.json", shortcut)

    # ---------------- exploratory layer profile (after primary results are saved)
    profile = []
    for cond in (control, treatment):
        C = selected[cond]["C"]
        for layer in layers:
            X, y = data.build(cond, subsets[sel_seed], layer)
            model = make_model(C, cfg).fit(X, y)
            pv = model.decision_function(val_set.layer(layer, val_rows))
            pt = model.decision_function(target_set.layer(layer, tgt_rows))
            profile.append({"condition": cond, "layer": layer, "C": C,
                            "val_eer": eer_and_threshold(y_val, pv)[0], "target_eer": eer_and_threshold(y_tgt, pt)[0]})
    write_csv(run_dir / "layer_profile_EXPLORATORY.csv", profile, list(profile[0]))

    # ---------------- run record
    import sklearn

    record = {
        "created_at_utc": now_utc(), "study_id": cfg["study_id"], "config_sha256": cfg["_config_sha256"],
        "manifest_sha256": manifest_hashes, "target_scope": target_scope,
        "encoder": json.loads(next(iter(encoders_used))),
        "fake_model_self_test": bool(fake_runs),
        "n_source_pool": len(pool), "n_source_val": len(y_val), "n_target": len(y_tgt),
        "seeds": seeds, "selection": selected, "bootstrap_reps": reps, "codec_encoders_available": codecs_available,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    write_json(run_dir / "run_record.json", record)
    (cfg["_paths"]["output_dir"] / "LATEST_RUN.txt").write_text(run_dir.name, encoding="utf-8")

    from step4_report import build_report

    build_report(run_dir, cfg)
    print(f"Done in {record['minutes']} min. Results: {run_dir}")
    return 0


def shortcut_diagnostic(cfg, data, train_sets, subset, val_set, val_rows, y_val, target_set, tgt_rows, y_tgt):
    def feats(es: EmbeddingSet, rows: np.ndarray, cols):
        return np.array([[float(es.info[i][c]) for c in cols] for i in rows])

    clean_rows = data.rows["clean"][subset]
    y_train = data.labels[subset]
    out = {}
    for name, cols in {"raw_duration_and_silence": ["raw_seconds", "lead_silence_s", "trail_silence_s"],
                       "model_input_duration_only": ["used_seconds"]}.items():
        model = make_model(1.0, cfg).fit(feats(train_sets["clean"], clean_rows, cols), y_train)
        pv = model.decision_function(feats(val_set, val_rows, cols))
        pt = model.decision_function(feats(target_set, tgt_rows, cols))
        out[name] = {"features": cols,
                     "source_val_eer": eer_and_threshold(y_val, pv)[0], "source_val_auc": roc_auc(y_val, pv),
                     "target_eer": eer_and_threshold(y_tgt, pt)[0], "target_auc": roc_auc(y_tgt, pt)}

    def describe(es, rows, y, col):
        values = np.array([float(es.info[i][col]) for i in rows])
        return {lab: {"mean": float(values[y == v].mean()), "median": float(np.median(values[y == v]))}
                for lab, v in (("real", 0), ("fake", 1))}

    out["distributions"] = {
        split: {col: describe(es, rows, y, col) for col in ("raw_seconds", "lead_silence_s", "trail_silence_s", "used_seconds")}
        for split, es, rows, y in (("source_train", train_sets["clean"], clean_rows, y_train),
                                    ("source_val", val_set, val_rows, y_val), ("target", target_set, tgt_rows, y_tgt))
    }
    out["note"] = ("Silence is trimmed before XLS-R in every condition. These scores show how much label signal "
                   "duration/silence metadata alone would carry; high source scores mean the source has a known shortcut.")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
