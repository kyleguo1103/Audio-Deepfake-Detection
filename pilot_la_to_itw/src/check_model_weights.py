"""Check that XLS-R's positional-convolution weights were really loaded from the checkpoint.

transformers prints "Some weights ... were not initialized: pos_conv_embed.conv.parametrizations.weight.original0/1"
with newer torch versions. This is often only a naming warning (torch converts the old weight_g / weight_v
names while loading), but if the weights were truly left random every embedding would be affected.
This script compares the loaded weights with the checkpoint file. It reads only the local cache.

    python src\\check_model_weights.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\7841_data\hf_cache")

import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import Wav2Vec2Model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_common import load_config  # noqa: E402

MODEL_ID = "facebook/wav2vec2-large-xlsr-53"


def main() -> int:
    cfg = load_config()
    record_path = cfg["_paths"]["embedding_dir"] / "extraction_record.json"
    commit = None
    if record_path.exists():
        runs = json.loads(record_path.read_text(encoding="utf-8"))["runs"]
        commit = runs[-1]["encoder"].get("resolved_commit")
    print(f"HF_HOME={os.environ['HF_HOME']}  commit={commit}  torch={torch.__version__}")

    model = Wav2Vec2Model.from_pretrained(MODEL_ID, revision=commit, local_files_only=True).eval()
    ckpt_path = hf_hub_download(MODEL_ID, "pytorch_model.bin", revision=commit, local_files_only=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    g_key = next(k for k in state if k.endswith("pos_conv_embed.conv.weight_g"))
    v_key = next(k for k in state if k.endswith("pos_conv_embed.conv.weight_v"))
    g, v = state[g_key].float(), state[v_key].float()
    expected = torch._weight_norm(v, g, 2)  # wav2vec2 uses weight_norm(conv, dim=2)
    conv = model.encoder.pos_conv_embed.conv
    loaded = conv.weight.detach().float()
    diff = (loaded - expected).abs().max().item()
    print(f"checkpoint keys : {g_key}, {v_key}")
    print(f"max |loaded - checkpoint| of positional-conv weight = {diff:.3e}")

    # second check: every other checkpoint tensor that maps onto the model is identical
    model_state = model.state_dict()
    compared, mismatched = 0, []
    for key, value in state.items():
        name = key[len("wav2vec2."):] if key.startswith("wav2vec2.") else key
        if name in model_state and model_state[name].shape == value.shape:
            compared += 1
            if not torch.equal(model_state[name].float(), value.float()):
                mismatched.append(name)
    print(f"other tensors compared: {compared}, mismatched: {len(mismatched)} {mismatched[:3]}")

    if diff < 1e-5 and not mismatched:
        print("PASS: all XLS-R weights match the checkpoint; the warning was only about parameter names.")
        return 0
    print("FAIL: weights differ from the checkpoint - do NOT run step 3; send this output to Claude.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
