"""Step 0: check that this computer can run the pilot. Changes nothing."""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import sys
from pathlib import Path

from pilot_common import ffmpeg_encoders, ffmpeg_exe, known_archive_problem, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--la-source", help="LA.zip or extracted ASVspoof 2019 LA folder")
    parser.add_argument("--itw-source", help="release_in_the_wild.zip or extracted folder")
    args = parser.parse_args()
    cfg = load_config(args.config, args.data_root)
    problems: list[str] = []

    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    for name in ["numpy", "scipy", "sklearn", "matplotlib", "soundfile", "transformers", "tqdm", "imageio_ffmpeg"]:
        try:
            module = importlib.import_module(name)
            print(f"{name:12s}: {getattr(module, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:12s}: MISSING ({exc.__class__.__name__})")
            problems.append(f"install {name}")

    try:
        import torch

        print(f"torch       : {torch.__version__}  CUDA build: {torch.version.cuda}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU         : {props.name}  VRAM {props.total_memory / 1024 ** 3:.1f} GB")
        else:
            print("GPU         : NOT AVAILABLE to torch (CPU only - extraction will be slow)")
            problems.append("torch cannot see the NVIDIA GPU (install the CUDA build of torch)")
    except Exception as exc:  # noqa: BLE001
        print(f"torch       : MISSING ({exc})")
        problems.append("install torch")

    try:
        exe = ffmpeg_exe()
        encoders = ffmpeg_encoders(exe)
        wanted = [o["encoder"] for o in cfg["augmentation"]["families"]["codec"]["options"]]
        have = [e for e in wanted if e in encoders]
        print(f"ffmpeg      : {exe}")
        print(f"codecs      : available {have}; missing {[e for e in wanted if e not in encoders]}")
        if not have:
            problems.append("no usable codec encoder in ffmpeg")
    except Exception as exc:  # noqa: BLE001
        print(f"ffmpeg      : MISSING ({exc})")
        problems.append("install imageio-ffmpeg")

    paths = cfg["_paths"]
    print(f"data_root   : {paths['data_root']}")
    sources = {"LA source": Path(args.la_source) if args.la_source else paths["la_zip"],
               "ITW source": Path(args.itw_source) if args.itw_source else paths["itw_zip"]}
    for key, path in sources.items():
        if path.is_dir():
            print(f"{key:12s}: OK  {path}  (extracted folder)")
        elif path.exists():
            issue = known_archive_problem(path)
            if issue:
                print(f"{key:12s}: PROBLEM  {issue}")
                problems.append(issue)
            else:
                print(f"{key:12s}: OK  {path}  ({path.stat().st_size / 1024 ** 3:.2f} GB zip)")
        else:
            print(f"{key:12s}: NOT FOUND at {path}")
            problems.append(f"{key}: put the data at {path} or pass its location")
    try:
        anchor = paths["data_root"] if paths["data_root"].exists() else paths["data_root"].anchor
        free = shutil.disk_usage(anchor).free / 1024 ** 3
        print(f"free space  : {free:.1f} GB on {anchor}")
        if free < 8:
            problems.append("less than 8 GB free for embeddings")
    except Exception:  # noqa: BLE001
        pass

    print()
    if problems:
        print("NOT READY:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("READY: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
