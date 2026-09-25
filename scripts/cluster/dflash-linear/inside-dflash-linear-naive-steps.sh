#!/bin/bash
# 3-step dflash_linear training smoke inside naqin/primus-specforge:v0.5.14-rocm700-mi35x.
# backend=auto selects chunked FLA on GPU when the kernel is installed.
set -euo pipefail

cd /workspace/SpecForge

echo "=== mmap fallback (pickle-protocol ckpts) ==="
python3 - <<'PY'
from pathlib import Path
p = Path("/workspace/SpecForge/specforge/runtime/data_plane/feature_store.py")
text = p.read_text(encoding="utf-8")
old = "return torch.load(path, weights_only=False, mmap=True)"
new = "return torch.load(path, weights_only=False)"
if old not in text:
    print("load_feature_file mmap line not found; leaving file unchanged")
else:
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("patched", p)
PY

echo "=== pip install -e . --no-deps ==="
pip install -e . --no-deps

echo "=== torch / training-path preflight ==="
python3 - <<'PY'
import torch
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash_linear import DFlashLinearDecoderLayer
from specforge.modeling.draft.linear_context import (
    fla_available,
    resolve_scan_backend,
)

print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "naive-steps smoke needs a GPU"
resolved = resolve_scan_backend("auto", on_cuda=True, num_anchors=512)
print("auto_backend", resolved, "fla_available", fla_available("gdn"))
assert resolved == ("fla" if fla_available("gdn") else "naive")

config = AutoDraftModelConfig.from_file(
    "/workspace/SpecForge/configs/qwen3.5-4b-dflash-linear.json"
)
model = AutoDraftModel.from_config(config)
layer = model.layers[0]
assert isinstance(layer, DFlashLinearDecoderLayer)
assert layer.settings["backend"] == "auto"
assert hasattr(layer, "horizon_embed")
scan = layer.context_scan
print(
    "layers",
    len(model.layers),
    "horizon",
    tuple(layer.horizon_embed.weight.shape),
    "A_log",
    tuple(scan.A_log.shape),
    "dt_bias",
    tuple(scan.dt_bias.shape),
)
assert tuple(scan.A_log.shape) == tuple(scan.dt_bias.shape)
PY

echo "=== shard listing ==="
python3 - <<'PY'
import os
root = os.environ["HIDDEN_STATES_PATH"]
n = 0
for dirpath, dirnames, filenames in os.walk(root):
    n += sum(1 for name in filenames if name.endswith(".ckpt"))
print("ckpt_count", n, "root", root)
if n < 10:
    raise SystemExit(f"too few ckpts in shard: {n}")
PY

TRAIN_DIR="${RUN_DIR}/train-linear"
mkdir -p "${TRAIN_DIR}" "${RUN_DIR}/sf-cache"

echo "=== train dflash_linear (max_steps=3, num_anchors=512, naive tape) ==="
specforge train \
  --config /workspace/SpecForge/scripts/cluster/dflash-linear/qwen3.5-4b-dflash-linear-naive-steps.yaml \
  "data.hidden_states_path=${HIDDEN_STATES_PATH}" \
  "data.cache_dir=${RUN_DIR}/sf-cache" \
  training.max_steps=3 \
  training.num_epochs=1 \
  training.num_anchors=512 \
  training.save_interval=1 \
  training.log_interval=1 \
  model.use_liger_kernel=false \
  "output_dir=${TRAIN_DIR}" \
  2>&1 | tee "${RUN_DIR}/train-linear.log"

python3 - <<PY
import math
import re
from pathlib import Path

run = Path("${RUN_DIR}")
text = (run / "train-linear.log").read_text(encoding="utf-8", errors="replace")
losses = [float(x) for x in re.findall(r"'ce_loss':\s*([0-9.eE+-]+)", text)]
if len(losses) < 1:
    raise SystemExit("no ce_loss in train-linear.log")
if any(not math.isfinite(v) for v in losses):
    raise SystemExit(f"non-finite ce_loss {losses}")
if any(not (1.0 < v < 30.0) for v in losses):
    raise SystemExit(f"ce_loss outside smoke band {losses}")
ckpt = list((run / "train-linear").rglob("training_state.pt"))
ckpt += list((run / "train-linear").rglob("*.safetensors"))
if not ckpt:
    raise SystemExit("no checkpoint under train-linear")
report = (
    f"ce_losses={losses}\n"
    f"n_logged_steps={len(losses)}\n"
    f"ckpt={ckpt[0]}\n"
)
(run / "naive-steps.env").write_text(report, encoding="utf-8")
print(report)
PY
