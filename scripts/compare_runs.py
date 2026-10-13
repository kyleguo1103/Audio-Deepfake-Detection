"""Compare a re-run with the reported run, condition by condition.

    python scripts/compare_runs.py run_YYYYMMDD_HHMMSS                 # vs the reported run
    python scripts/compare_runs.py run_A run_B

Prints the selected layer/C and target EER of both runs and their difference. Small differences are
expected from different ffmpeg builds (codec augmentation), BLAS libraries and GPU fp16 arithmetic;
a different selected layer is the most likely cause of a large difference (see README).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "pilot_la_to_itw" / "outputs"
REFERENCE = "run_20261011_113206"


def load(run: str) -> dict:
    data = json.loads((OUTPUTS / run / "summary.json").read_text(encoding="utf-8"))
    return {c["condition"]: c for c in data["conditions"]}


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    new, ref = (sys.argv[1], REFERENCE) if len(sys.argv) == 2 else (sys.argv[2], sys.argv[1])
    a, b = load(ref), load(new)
    print(f"reference: {ref}\nnew:       {new}\n")
    print(f"{'condition':10s} {'layer/C ref':>12s} {'layer/C new':>12s} {'EER ref %':>10s} {'EER new %':>10s} {'diff pp':>8s}")
    for cond in a:
        ra, rb = a[cond], b.get(cond)
        if rb is None:
            print(f"{cond:10s} missing in new run")
            continue
        print(f"{cond:10s} {str(ra['layer']) + '/' + str(ra['C']):>12s} {str(rb['layer']) + '/' + str(rb['C']):>12s} "
              f"{100 * ra['target_eer_mean']:10.2f} {100 * rb['target_eer_mean']:10.2f} "
              f"{100 * (rb['target_eer_mean'] - ra['target_eer_mean']):+8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
