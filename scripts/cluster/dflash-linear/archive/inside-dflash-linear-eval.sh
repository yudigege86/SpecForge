#!/bin/bash
# Export the 1-epoch dflash_linear checkpoint, serve vanilla vs spec, eval holdout.
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
assert torch.cuda.is_available(), "eval needs a GPU"
assert fla_available("gdn"), "eval requires FLA GDN on GPU"
resolved = resolve_scan_backend("auto", on_cuda=True, num_anchors=1)
print("auto_backend", resolved)
assert resolved == "fla", f"expected FLA scan backend, got {resolved!r}"
PY

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_CONFIG="${DRAFT_CONFIG:-/workspace/SpecForge/configs/qwen3.5-4b-dflash-linear.json}"
EXPORT_DIR="${EXPORT_DIR:-${RUN_DIR}/draft_hf}"
EMBEDDING_KEY="${EMBEDDING_KEY:-model.language_model.embed_tokens.weight}"
EVAL_JSONL="${EVAL_JSONL:?EVAL_JSONL is required}"
EVAL_N="${EVAL_N:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
SERVER_PORT="${SERVER_PORT:-30000}"
BASE="http://127.0.0.1:${SERVER_PORT}"
SERVE_PY="${SERVE_PY:-/workspace/SpecForge/scripts/eval/dflash_linear_serve.py}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-600}"
SERVER_PID=""

mkdir -p "${EXPORT_DIR}" "${RUN_DIR}"
test -f "${CHECKPOINT}/training_state.pt"
test -s "${EVAL_JSONL}"
test -f "${DRAFT_CONFIG}"
test -f "${SERVE_PY}"

echo "=== specforge export --to hf ==="
specforge export \
  --to hf \
  --checkpoint "${CHECKPOINT}" \
  --draft-config "${DRAFT_CONFIG}" \
  --output-dir "${EXPORT_DIR}" \
  --embedding-source "${TARGET_MODEL}" \
  --embedding-key "${EMBEDDING_KEY}" \
  2>&1 | tee "${RUN_DIR}/export.log"

test -f "${EXPORT_DIR}/config.json"

echo "=== normalize linear export (keep DFlashLinearDraftModel) ==="
python3 /workspace/SpecForge/scripts/gates/normalize_dflash_export.py \
  --config "${EXPORT_DIR}/config.json" \
  --block-size "${BLOCK_SIZE}" \
  2>&1 | tee "${RUN_DIR}/normalize.log"

python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${EXPORT_DIR}/config.json").read_text())
print("architectures", cfg.get("architectures"))
print("auto_map", cfg.get("auto_map"))
print("linear_context", (cfg.get("dflash_config") or {}).get("linear_context"))
if "DFlashLinearDraftModel" not in (cfg.get("architectures") or []):
    raise SystemExit("export config is not DFlashLinearDraftModel")
if "DFlashDraftModel" in (cfg.get("architectures") or []) and len(cfg.get("architectures") or []) == 1:
    raise SystemExit("normalize rewrote the linear draft to stock DFlashDraftModel")
PY

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

start_server() {
  local mode="$1"
  local log="$2"
  stop_server
  local argv=(
    python3 "${SERVE_PY}" serve
    --target "${TARGET_MODEL}"
    --mode "${mode}"
    --host 127.0.0.1
    --port "${SERVER_PORT}"
  )
  if [[ "${mode}" == "spec" ]]; then
    argv+=(--draft "${EXPORT_DIR}")
  fi
  printf '%s\n' "${argv[@]}" > "${log%.log}.argv"
  echo "=== serve ${argv[*]} ==="
  "${argv[@]}" >"${log}" 2>&1 &
  SERVER_PID=$!
  python3 "${SERVE_PY}" wait --base "${BASE}" --timeout 900
}

start_server vanilla "${RUN_DIR}/vanilla-server.log"
python3 "${SERVE_PY}" eval \
  --base "${BASE}" \
  --label vanilla \
  --out "${RUN_DIR}/vanilla.json" \
  --target "${TARGET_MODEL}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --n "${EVAL_N}" \
  --warmup 2 \
  --max-new-tokens 64 \
  --timeout "${BENCH_TIMEOUT}" \
  --no-ignore-eos \
  2>&1 | tee "${RUN_DIR}/vanilla-bench.log"
stop_server

start_server spec "${RUN_DIR}/dflash-linear-server.log"
python3 "${SERVE_PY}" eval \
  --base "${BASE}" \
  --label dflash_linear \
  --out "${RUN_DIR}/dflash_linear.json" \
  --target "${TARGET_MODEL}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --n "${EVAL_N}" \
  --warmup 2 \
  --max-new-tokens 64 \
  --timeout "${BENCH_TIMEOUT}" \
  --no-ignore-eos \
  2>&1 | tee "${RUN_DIR}/dflash-linear-bench.log"
stop_server

COMPARE_ARGS=(
  --vanilla "${RUN_DIR}/vanilla.json"
  --spec "${RUN_DIR}/dflash_linear.json"
  --out "${RUN_DIR}/summary.md"
  --min-speedup "${MIN_SPEEDUP:-1.00}"
  --min-accept "${MIN_ACCEPT:-1.05}"
)
if [[ "${FAIL_ON_GATES:-0}" == "1" ]]; then
  COMPARE_ARGS+=(--fail-on-gates)
fi
python3 "${SERVE_PY}" compare "${COMPARE_ARGS[@]}"
echo "PASS: dflash_linear export/serve/eval"
echo "run_dir=${RUN_DIR}"
