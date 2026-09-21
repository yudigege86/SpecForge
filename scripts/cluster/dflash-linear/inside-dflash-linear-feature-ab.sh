#!/bin/bash
# A/B SGLang capture features vs HuggingFace hidden_states (no draft).
set -euo pipefail

cd /workspace/SpecForge

echo "=== pip install -e . --no-deps ==="
pip install -e . --no-deps

echo "=== torch preflight ==="
python3 - <<'PY'
import torch
print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "feature A/B needs a GPU"
PY

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:?HIDDEN_STATES_PATH is required}"
AB_N="${AB_N:-8}"
AB_PY="${AB_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_feature_ab.py}"

mkdir -p "${RUN_DIR}"
test -d "${HIDDEN_STATES_PATH}"
test -f "${AB_PY}"

echo "=== feature A/B n=${AB_N} ==="
python3 "${AB_PY}" \
  --target "${TARGET_MODEL}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --out "${RUN_DIR}/dflash_linear_feature_ab.json" \
  --n "${AB_N}" \
  --max-length 2048 \
  --target-layer-ids 1,8,15,22,29 \
  2>&1 | tee "${RUN_DIR}/ab.log"

python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = os.environ["RUN_DIR"]
report = json.loads(Path(run_dir, "dflash_linear_feature_ab.json").read_text())
lines = [
    "# Linear-context DFlash feature A/B",
    "",
    f"n: {report.get('n')}",
    f"target_layer_ids: {report.get('target_layer_ids')}",
    f"extract_offset: {report.get('extract_offset')}",
    f"width_match: {report.get('width_match')}",
    f"cosine_flat_mean: {report.get('cosine_flat_mean')}",
    f"cosine_token_mean: {report.get('cosine_token_mean')}",
    f"capture_norm_mean: {report.get('capture_norm_mean')}",
    f"hf_norm_mean: {report.get('hf_norm_mean')}",
    f"elapsed_s: {report.get('elapsed_s')}",
    "",
    "Capture ckpts vs HF use_cache=False hidden_states. No draft.",
    "",
]
Path(run_dir, "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY
echo "PASS: dflash_linear feature A/B"
echo "run_dir=${RUN_DIR}"
