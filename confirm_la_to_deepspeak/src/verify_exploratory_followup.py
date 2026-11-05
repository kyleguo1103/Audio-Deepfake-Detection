"""Independently recalculate the saved exploratory follow-up metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PILOT_SRC = PROJECT.parent / "pilot_la_to_itw" / "src"
sys.path.insert(0, str(PILOT_SRC))
from pilot_common import eer_and_threshold, roc_auc  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap(y, scores, identities, conditions, reps, seed):
    rng = np.random.default_rng(seed)
    identity_list = np.unique(identities)
    by_identity = {identity: np.flatnonzero(identities == identity) for identity in identity_list}
    eers = {condition: np.empty(reps, dtype=np.float64) for condition in conditions}
    aucs = {condition: np.empty(reps, dtype=np.float64) for condition in conditions}
    used = 0
    for _ in range(reps):
        selected = rng.choice(identity_list, size=len(identity_list), replace=True)
        idx = np.concatenate([by_identity[item] for item in selected])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for condition in conditions:
            eers[condition][used] = np.mean(
                [eer_and_threshold(yy, per_seed[idx])[0] for per_seed in scores[condition]]
            )
            aucs[condition][used] = np.mean(
                [roc_auc(yy, per_seed[idx]) for per_seed in scores[condition]]
            )
        used += 1
    output = {"reps_used": used, "conditions": {}}
    base_eer = eers["none"][:used]
    base_auc = aucs["none"][:used]
    for condition in conditions:
        condition_eer = eers[condition][:used]
        condition_auc = aucs[condition][:used]
        output["conditions"][condition] = {
            "eer_ci": [float(np.percentile(condition_eer, 2.5)), float(np.percentile(condition_eer, 97.5))],
            "auc_ci": [float(np.percentile(condition_auc, 2.5)), float(np.percentile(condition_auc, 97.5))],
            "delta_eer_vs_none_ci": [
                float(np.percentile(condition_eer - base_eer, 2.5)),
                float(np.percentile(condition_eer - base_eer, 97.5)),
            ],
            "delta_auc_vs_none_ci": [
                float(np.percentile(condition_auc - base_auc, 2.5)),
                float(np.percentile(condition_auc - base_auc, 97.5)),
            ],
        }
    return output


def close(a, b, tolerance=1e-12):
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tolerance, rtol=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    predictions = read_csv(run_dir / "predictions_target.csv")
    metrics = read_csv(run_dir / "metrics_per_seed.csv")
    conditions = list(summary["conditions"])
    seeds = [int(value) for value in summary["seeds"]]
    y = np.array([1 if row["label"] == "fake" else 0 for row in predictions], dtype=np.int64)
    identities = np.array([row["identity"] for row in predictions])
    scores = {
        condition: np.stack(
            [
                np.array([float(row[f"score_{condition}_seed{seed}"]) for row in predictions])
                for seed in seeds
            ]
        )
        for condition in conditions
    }

    checks = []
    for condition in conditions:
        calculated_eers = [eer_and_threshold(y, row)[0] for row in scores[condition]]
        calculated_aucs = [roc_auc(y, row) for row in scores[condition]]
        saved_rows = [
            row for row in metrics if row["condition"] == condition and row["split"] == "target"
        ]
        saved_rows.sort(key=lambda row: seeds.index(int(row["seed"])))
        saved_eers = [float(row["eer"]) for row in saved_rows]
        saved_aucs = [float(row["roc_auc"]) for row in saved_rows]
        checks.append(
            {
                "name": f"{condition}_per_seed_eer",
                "passed": close(calculated_eers, saved_eers, 2e-9),
            }
        )
        checks.append(
            {
                "name": f"{condition}_per_seed_auc",
                "passed": close(calculated_aucs, saved_aucs, 2e-9),
            }
        )
        checks.append(
            {
                "name": f"{condition}_mean_eer",
                "passed": close(np.mean(calculated_eers), summary["conditions"][condition]["eer_mean"], 2e-9),
            }
        )

    saved_boot = summary["bootstrap"]
    calculated_boot = bootstrap(
        y,
        scores,
        identities,
        conditions,
        int(saved_boot["reps_requested"]),
        7841,
    )
    checks.append(
        {
            "name": "bootstrap_reps_used",
            "passed": calculated_boot["reps_used"] == int(saved_boot["reps_used"]),
        }
    )
    for condition in conditions:
        for field in ("eer_ci", "auc_ci", "delta_eer_vs_none_ci", "delta_auc_vs_none_ci"):
            checks.append(
                {
                    "name": f"{condition}_{field}",
                    "passed": close(
                        calculated_boot["conditions"][condition][field],
                        saved_boot["conditions"][condition][field],
                    ),
                }
            )
    passed = all(check["passed"] for check in checks)
    result = {
        "status": "PASS" if passed else "FAIL",
        "source_run": run_dir.name,
        "classification": "verification_of_exploratory_result",
        "n_predictions": len(predictions),
        "conditions": conditions,
        "seeds": seeds,
        "checks": checks,
    }
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    failed = [check["name"] for check in checks if not check["passed"]]
    lines = [
        "# Exploratory follow-up verification",
        "",
        f"- Status: **{result['status']}**",
        f"- Source run: `{run_dir.name}`",
        f"- Saved predictions independently recalculated: {len(predictions)}",
        f"- Conditions: {', '.join(conditions)}",
        f"- Source seeds: {', '.join(str(seed) for seed in seeds)}",
        "- Recalculated quantities: per-seed EER/AUC, condition means, and all 10,000 paired identity-cluster intervals.",
        "",
        "Verification confirms numerical reproducibility only. It does not change the exploratory status of the follow-up.",
    ]
    if failed:
        lines.extend(["", "Failed checks:", *[f"- {name}" for name in failed]])
    (run_dir / "VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed": failed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
