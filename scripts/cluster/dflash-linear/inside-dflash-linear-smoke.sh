#!/bin/bash
# Runs inside naqin/primus-specforge:v0.5.14-rocm700-mi35x.
# 1-step linear-context GDN smoke, then stock DFlash on the same shard.
set -euo pipefail

cd /workspace/SpecForge

echo "=== specforge checkout ==="
python3 - <<'PY'
from pathlib import Path
for rel in (
    "specforge/modeling/draft/dflash_linear.py",
    "specforge/algorithms/dflash_linear/model.py",
    "specforge/algorithms/dflash_linear/providers.py",
    "configs/qwen3.5-4b-dflash-linear.json",
    "configs/qwen3.5-4b-dflash.json",
):
    path = Path("/workspace/SpecForge") / rel
    assert path.is_file(), path
    print("ok", path)
text = Path("/workspace/SpecForge/specforge/algorithms/dflash_linear/model.py").read_text(
    encoding="utf-8"
)
assert "class OnlineDFlashLinearModel" in text
print("online_wrapper_ok")
PY

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

echo "=== install flash-linear-attention[rocm] without replacing image torch ==="
python3 - <<'PY'
import torch
print("torch_before_fla", torch.__version__, "hip", getattr(torch.version, "hip", None))
PY
pip install einops
pip install 'flash-linear-attention[rocm]'
python3 - <<'PY'
import torch
import fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from specforge.modeling.draft.linear_context import fla_available
print("torch_after_fla", torch.__version__, "hip", getattr(torch.version, "hip", None))
print("fla_version", getattr(fla, "__version__", "?"))
print("chunk_gated_delta_rule", chunk_gated_delta_rule)
print("fla_gdn", fla_available("gdn"))
assert torch.cuda.is_available(), "smoke needs a GPU"
assert fla_available("gdn"), "GDN FLA backend unavailable after install"
PY

echo "=== draft architecture preflight ==="
python3 - <<'PY'
from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.dflash_linear import DFlashLinearDecoderLayer, DFlashLinearDraftModel

config = AutoDraftModelConfig.from_file(
    "/workspace/SpecForge/configs/qwen3.5-4b-dflash-linear.json"
)
model = AutoDraftModel.from_config(config)
assert isinstance(model, DFlashLinearDraftModel)
assert isinstance(model.layers[0], DFlashLinearDecoderLayer)
print(
    "layers",
    len(model.layers),
    "variant",
    model.layers[0].settings["variant"],
    "backend",
    model.layers[0].settings["backend"],
)
PY

echo "=== shard listing ==="
python3 - <<'PY'
import os
root = "/shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links/s00/rows_0-2000"
n = 0
for dirpath, dirnames, filenames in os.walk(root):
    n += sum(1 for name in filenames if name.endswith(".ckpt"))
print("ckpt_count", n, "root", root)
if n < 10:
    raise SystemExit(f"too few ckpts in shard: {n}")
PY

LINEAR_DIR="${RUN_DIR}/train-linear"
DFLASH_DIR="${RUN_DIR}/train-dflash"
mkdir -p "${LINEAR_DIR}" "${DFLASH_DIR}" "${RUN_DIR}/sf-cache"

echo "=== train dflash_linear (max_steps=1) ==="
specforge train \
  --config /workspace/SpecForge/scripts/cluster/dflash-linear/qwen3.5-4b-dflash-linear-smoke.yaml \
  "data.hidden_states_path=${HIDDEN_STATES_PATH}" \
  "data.cache_dir=${RUN_DIR}/sf-cache" \
  training.max_steps=1 \
  training.num_epochs=1 \
  training.save_interval=1 \
  training.log_interval=1 \
  model.use_liger_kernel=false \
  "output_dir=${LINEAR_DIR}" \
  2>&1 | tee "${RUN_DIR}/train-linear.log"

echo "=== train stock dflash (max_steps=1) ==="
specforge train \
  --config /workspace/SpecForge/scripts/cluster/dflash-linear/qwen3.5-4b-dflash-smoke.yaml \
  "data.hidden_states_path=${HIDDEN_STATES_PATH}" \
  "data.cache_dir=${RUN_DIR}/sf-cache" \
  training.max_steps=1 \
  training.num_epochs=1 \
  training.save_interval=1 \
  training.log_interval=1 \
  model.use_liger_kernel=false \
  "output_dir=${DFLASH_DIR}" \
  2>&1 | tee "${RUN_DIR}/train-dflash.log"

python3 - <<PY
import math
import re
from pathlib import Path

run = Path("${RUN_DIR}")


def parse_ce(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"'ce_loss':\s*([0-9.eE+-]+)", text)
    if not matches:
        raise SystemExit(f"no ce_loss in {path}")
    value = float(matches[-1])
    if not math.isfinite(value):
        raise SystemExit(f"non-finite ce_loss {value} in {path}")
    if not (1.0 < value < 30.0):
        raise SystemExit(f"ce_loss {value} outside smoke band (1, 30) in {path}")
    return value


def find_ckpt(root: Path) -> Path:
    ckpt = list(root.rglob("training_state.pt"))
    ckpt += list(root.rglob("*.safetensors"))
    if not ckpt:
        raise SystemExit(f"no checkpoint under {root}")
    return ckpt[0]


linear_ce = parse_ce(run / "train-linear.log")
dflash_ce = parse_ce(run / "train-dflash.log")
linear_ckpt = find_ckpt(run / "train-linear")
dflash_ckpt = find_ckpt(run / "train-dflash")
report = (
    f"linear_ce_loss={linear_ce}\n"
    f"dflash_ce_loss={dflash_ce}\n"
    f"delta_linear_minus_dflash={linear_ce - dflash_ce}\n"
    f"identity_ce_loss_job141000=12.871541023254395\n"
    f"linear_ckpt={linear_ckpt}\n"
    f"dflash_ckpt={dflash_ckpt}\n"
)
(run / "compare.env").write_text(report, encoding="utf-8")
print(report)
PY
