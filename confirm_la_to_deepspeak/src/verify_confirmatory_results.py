"""Recalculate the frozen primary DeepSpeak result from saved predictions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PILOT_SRC = PROJECT.parent / "pilot_la_to_itw" / "src"
sys.path.insert(0, str(PILOT_SRC))
from pilot_common import eer_and_threshold  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap(y, none, noise, identities, reps, seed):
    rng = np.random.default_rng(seed)
    identity_list = np.unique(identities)
    groups = {identity: np.flatnonzero(identities == identity) for identity in identity_list}
    delta = []
    for _ in range(reps):
        selected = rng.choice(identity_list, size=len(identity_list), replace=True)
        idx = np.concatenate([groups[item] for item in selected])
        yy = y[idx]
        none_eer = np.mean([eer_and_threshold(yy, row[idx])[0] for row in none])
        noise_eer = np.mean([eer_and_threshold(yy, row[idx])[0] for row in noise])
        delta.append(noise_eer - none_eer)
    values = np.asarray(delta)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> int:
    latest = (PROJECT / "outputs" / "LATEST_RUN.txt").read_text(encoding="utf-8").strip()
    run_dir = PROJECT / "outputs" / latest
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(run_dir / "predictions_target.csv")
    seeds = [int(item) for item in summary["seeds"]]
    label = np.array([0 if row["label"] == "real" else 1 for row in rows], dtype=np.int64)
    identities = np.array([row["identity"] for row in rows])
    none = np.array([[float(row[f"score_none_seed{seed}"]) for row in rows] for seed in seeds])
    noise = np.array([[float(row[f"score_noise_seed{seed}"]) for row in rows] for seed in seeds])
    none_eer = float(np.mean([eer_and_threshold(label, row)[0] for row in none]))
    noise_eer = float(np.mean([eer_and_threshold(label, row)[0] for row in noise]))
    delta = noise_eer - none_eer
    ci = bootstrap(label, none, noise, identities, summary["bootstrap"]["reps_requested"], 7841)
    checks = {
        "prediction_rows_1800": len(rows) == 1800,
        "identities_56": len(np.unique(identities)) == 56,
        "none_eer_matches": abs(none_eer - summary["conditions"]["none"]["eer_mean"]) < 1e-9,
        "noise_eer_matches": abs(noise_eer - summary["conditions"]["noise"]["eer_mean"]) < 1e-9,
        "delta_matches": abs(delta - summary["delta_eer_noise_minus_none"]) < 1e-9,
        "bootstrap_ci_matches": np.allclose(ci, summary["delta_eer_ci_identity_cluster"], atol=1e-9, rtol=0),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    record = {
        "status": status,
        "run": latest,
        "recalculated_none_eer": none_eer,
        "recalculated_noise_eer": noise_eer,
        "recalculated_delta_eer": delta,
        "recalculated_delta_ci": ci,
        "checks": checks,
    }
    (run_dir / "verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    lines = [
        "# Confirmatory result verification",
        "",
        f"- Status: **{status}**",
        f"- Saved prediction rows: {len(rows)}",
        f"- Target identities: {len(np.unique(identities))}",
        f"- Recalculated no-augmentation EER: {100 * none_eer:.2f}%",
        f"- Recalculated noise EER: {100 * noise_eer:.2f}%",
        f"- Recalculated difference: {100 * delta:+.2f} percentage points",
        f"- Recalculated 95% identity-cluster CI: [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] percentage points",
        "",
        "Every check recomputes the primary result from saved sample-level predictions. No model fitting occurs in this verification step.",
    ]
    (run_dir / "VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
