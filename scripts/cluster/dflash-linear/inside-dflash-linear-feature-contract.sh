#!/bin/bash
# Cosine check: 0.5.19 live DFLASH aux vs the existing 40k training capture.
set -euo pipefail
cd /workspace/SpecForge
pip install -e . --no-deps
pip install datasets pandas tiktoken requests
bash /workspace/SpecForge/scripts/apply_sglang_spec_capture_patch.sh --target v0.5.19 || \
  echo "WARN: spec-capture patch not applied"

EVAL_PY="${EVAL_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_eval.py}"
TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required}"
DRAFT_HF="${DRAFT_HF:?DRAFT_HF is required}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links}"
EVAL_N="${EVAL_N:-16}"
GATE="${FEATURE_CONTRACT_GATE:-0.99}"
mkdir -p "${RUN_DIR}"

python3 "${EVAL_PY}" feature-contract-check \
  --target "${TARGET_MODEL}" \
  --draft "${DRAFT_HF}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --out "${RUN_DIR}/feature_contract_check.json" \
  --n "${EVAL_N}" \
  --gate "${GATE}" \
  2>&1 | tee "${RUN_DIR}/feature_contract_check.log"
echo "run_dir=${RUN_DIR}"
