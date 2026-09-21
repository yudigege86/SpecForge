#!/bin/bash
# Capture replay: exported linear draft on valid-links ckpts (no HF hidden states).
set -euo pipefail

cd /workspace/SpecForge

echo "=== pip install -e . --no-deps ==="
pip install -e . --no-deps

echo "=== install flash-linear-attention[rocm] without replacing image torch ==="
python3 - <<'PY'
import torch
print("torch_before_fla", torch.__version__, "hip", getattr(torch.version, "hip", None))
PY
pip install einops
pip install 'flash-linear-attention[rocm]'

echo "=== torch / FLA preflight ==="
python3 - <<'PY'
import torch
from specforge.modeling.draft.linear_context import fla_available, resolve_scan_backend
print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "capture replay needs a GPU"
assert fla_available("gdn"), "capture replay requires FLA GDN on GPU"
resolved = resolve_scan_backend("auto", on_cuda=True, num_anchors=512)
print("auto_backend", resolved)
assert resolved == "fla", f"expected FLA scan backend, got {resolved!r}"
PY

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_HF="${DRAFT_HF:?DRAFT_HF is required}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:?HIDDEN_STATES_PATH is required}"
REPLAY_N="${REPLAY_N:-32}"
REPLAY_PY="${REPLAY_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_capture_replay.py}"

mkdir -p "${RUN_DIR}"
test -f "${DRAFT_HF}/config.json"
test -d "${HIDDEN_STATES_PATH}"
test -f "${REPLAY_PY}"

python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${DRAFT_HF}/config.json").read_text())
print("architectures", cfg.get("architectures"))
if "DFlashLinearDraftModel" not in (cfg.get("architectures") or []):
    raise SystemExit("reused export is not DFlashLinearDraftModel")
PY

echo "=== capture replay n=${REPLAY_N} ==="
python3 "${REPLAY_PY}" \
  --draft "${DRAFT_HF}" \
  --target "${TARGET_MODEL}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --out "${RUN_DIR}/dflash_linear_replay.json" \
  --n "${REPLAY_N}" \
  --max-length 2048 \
  --num-anchors 512 \
  --loss-decay-gamma 7 \
  --attention-backend sdpa \
  --seed 42 \
  2>&1 | tee "${RUN_DIR}/replay.log"

python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = os.environ["RUN_DIR"]
report = json.loads(Path(run_dir, "dflash_linear_replay.json").read_text())
ref = report.get("training_reference") or {}
lines = [
    "# Linear-context DFlash capture replay",
    "",
    f"n: {report.get('n')}",
    f"skipped: {report.get('skipped')}",
    f"ce_loss_micro: {report.get('ce_loss_micro')}",
    f"acc_micro: {report.get('acc_micro')}",
    f"expected_accepted_length_micro: {report.get('expected_accepted_length_micro')}",
    f"ce_loss_mean: {report.get('ce_loss_mean')}",
    f"acc_mean: {report.get('acc_mean')}",
    f"expected_accepted_length_mean: {report.get('expected_accepted_length_mean')}",
    f"elapsed_s: {report.get('elapsed_s')}",
    "",
    "Training step-620 reference (same capture distribution):",
    f"ce: {ref.get('ce')}  acc: {ref.get('acc')}  eal: {ref.get('expected_accepted_length')}",
    "",
    "Hidden states are SGLang capture ckpts. No HF target generate.",
    "",
]
Path(run_dir, "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY
echo "PASS: dflash_linear capture replay"
echo "run_dir=${RUN_DIR}"
