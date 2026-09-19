#!/bin/bash
# 1-epoch dflash_linear train inside naqin/primus-specforge:v0.5.14-rocm700-mi35x.
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

echo "=== tensorboard (SummaryWriter) ==="
pip install tensorboard

echo "=== install flash-linear-attention[rocm] without replacing image torch ==="
# Do not pip-install torch from download.pytorch.org/whl/rocm7.2: this image
# already has a working ROCm torch. The [rocm] extra only pins torch>=2.7.0.
python3 - <<'PY'
import torch
print("torch_before_fla", torch.__version__, "hip", getattr(torch.version, "hip", None))
PY
pip install einops
pip install 'flash-linear-attention[rocm]'

echo "=== torch / FLA / dataset / tensorboard preflight ==="
python3 - <<'PY'
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from specforge.modeling.draft.linear_context import (
    SCAN_CHUNK_SIZE,
    fla_available,
    resolve_scan_backend,
)
from specforge.runtime.data_plane.offline_reader import list_feature_files

print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("summary_writer", SummaryWriter)
print("scan_chunk_size", SCAN_CHUNK_SIZE)
assert torch.cuda.is_available(), "1-epoch train needs GPUs"
assert fla_available("gdn"), "1-epoch train requires FLA GDN on GPU"
resolved = resolve_scan_backend("auto", on_cuda=True, num_anchors=512)
print("auto_backend", resolved, "fla_available", fla_available("gdn"))
assert resolved == "fla", f"expected FLA scan backend, got {resolved!r}"
root = os.environ["HIDDEN_STATES_PATH"]
n = len(list_feature_files(root))
print("feature_files", n, "root", root)
if n < 10000:
    raise SystemExit(f"expected ~40k feature files, got {n}")
PY

TRAIN_DIR="${RUN_DIR}/train"
mkdir -p "${TRAIN_DIR}" "${RUN_DIR}/sf-cache"

echo "=== train dflash_linear one epoch on valid-links ==="
specforge train \
  --config /workspace/SpecForge/scripts/cluster/dflash-linear/qwen3.5-4b-dflash-linear-1epoch.yaml \
  "data.hidden_states_path=${HIDDEN_STATES_PATH}" \
  "data.cache_dir=${RUN_DIR}/sf-cache" \
  "output_dir=${TRAIN_DIR}" \
  training.max_steps=627 \
  tracking.report_to=tensorboard \
  model.use_liger_kernel=false \
  2>&1 | tee "${RUN_DIR}/train.log"
