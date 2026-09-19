#!/bin/bash
# Offline teacher-forced MAL for dflash_linear (no SGLang, no target KV cache).
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
assert torch.cuda.is_available(), "MAL eval needs a GPU"
assert fla_available("gdn"), "MAL eval requires FLA GDN on GPU"
resolved = resolve_scan_backend("auto", on_cuda=True, num_anchors=1)
print("auto_backend", resolved)
assert resolved == "fla", f"expected FLA scan backend, got {resolved!r}"
PY

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_CONFIG="${DRAFT_CONFIG:-/workspace/SpecForge/configs/qwen3.5-4b-dflash-linear.json}"
EXPORT_DIR="${EXPORT_DIR:-${RUN_DIR}/draft_hf}"
EMBEDDING_KEY="${EMBEDDING_KEY:-model.language_model.embed_tokens.weight}"
EVAL_JSONL="${EVAL_JSONL:?EVAL_JSONL is required}"
EVAL_N="${EVAL_N:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SERVE_PY="${SERVE_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_serve.py}"
DRAFT_HF="${DRAFT_HF:-}"

mkdir -p "${RUN_DIR}"
test -s "${EVAL_JSONL}"
test -f "${SERVE_PY}"
grep -q "def acceptance_along_sequence" /workspace/SpecForge/specforge/modeling/draft/dflash_linear.py

if [[ -n "${DRAFT_HF}" && -f "${DRAFT_HF}/config.json" ]]; then
  EXPORT_DIR="${DRAFT_HF}"
  echo "=== reusing exported draft ${EXPORT_DIR} ==="
  python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${EXPORT_DIR}/config.json").read_text())
print("architectures", cfg.get("architectures"))
if "DFlashLinearDraftModel" not in (cfg.get("architectures") or []):
    raise SystemExit("reused export is not DFlashLinearDraftModel")
PY
else
  CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required when DRAFT_HF is unset}"
  test -f "${CHECKPOINT}/training_state.pt"
  test -f "${DRAFT_CONFIG}"
  mkdir -p "${EXPORT_DIR}"
  echo "=== specforge export --to hf ==="
  specforge export \
    --to hf \
    --checkpoint "${CHECKPOINT}" \
    --draft-config "${DRAFT_CONFIG}" \
    --output-dir "${EXPORT_DIR}" \
    --embedding-source "${TARGET_MODEL}" \
    --embedding-key "${EMBEDDING_KEY}" \
    2>&1 | tee "${RUN_DIR}/export.log"
  python3 /workspace/SpecForge/scripts/gates/normalize_dflash_export.py \
    --config "${EXPORT_DIR}/config.json" \
    --block-size "${BLOCK_SIZE}" \
    2>&1 | tee "${RUN_DIR}/normalize.log"
  python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${EXPORT_DIR}/config.json").read_text())
print("architectures", cfg.get("architectures"))
if "DFlashLinearDraftModel" not in (cfg.get("architectures") or []):
    raise SystemExit("export config is not DFlashLinearDraftModel")
PY
fi

echo "=== offline MAL on ${EVAL_N} holdout prompts ==="
python3 "${SERVE_PY}" mal \
  --target "${TARGET_MODEL}" \
  --draft "${EXPORT_DIR}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --out "${RUN_DIR}/dflash_linear_mal.json" \
  --n "${EVAL_N}" \
  --max-new-tokens 64 \
  --no-ignore-eos \
  2>&1 | tee "${RUN_DIR}/mal.log"

python3 - <<'PY'
import json
from pathlib import Path
import os
run_dir = os.environ["RUN_DIR"]
report = json.loads(Path(run_dir, "dflash_linear_mal.json").read_text())
lines = [
    "# Linear-context DFlash offline MAL",
    "",
    f"n: {report.get('n')}",
    f"spec_accept_length_mean: {report.get('spec_accept_length_mean')}",
    f"spec_accept_length_p50: {report.get('spec_accept_length_p50')}",
    f"block_accept_length_mean: {report.get('block_accept_length_mean')}",
    f"empty_outputs: {report.get('empty_outputs')}",
    f"elapsed_s: {report.get('elapsed_s')}",
    "",
    "Method: teacher-forced MAL on vanilla greedy trajectories.",
    "Target hidden states come from one full use_cache=False forward (Qwen3.5 hybrid).",
    "Stock SGLang DFLASH is not used.",
    "",
]
Path(run_dir, "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY
echo "PASS: dflash_linear offline MAL"
echo "run_dir=${RUN_DIR}"
