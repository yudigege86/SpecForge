#!/bin/bash
# Sequential M1 gates: smoke, then HE n=16 at batch 1/4 and overlap on/off.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="/shared_nfs/naqin/Linear-Context-DFlash"
export DRAFT_HF="${DRAFT_HF:-${ROOT}/eval-1epoch/20260918T213552Z/draft_hf}"
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
export EVAL_DATASET="${EVAL_DATASET:-humaneval}"
export EVAL_N="${EVAL_N:-16}"
export STOCK_MAL_JSON="${STOCK_MAL_JSON:-}"
GATE_ROOT="${RESULTS_DIR:-${ROOT}/mal-eval/m1-gates}"
mkdir -p "${GATE_ROOT}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="${GATE_ROOT}/${STAMP}/summary.json"
mkdir -p "$(dirname "${SUMMARY}")"
exec > >(tee "${GATE_ROOT}/${STAMP}/driver.log") 2>&1

echo "M1 driver stamp=${STAMP} draft=${DRAFT_HF}"
status=0

run_one() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  if "$@"; then
    echo "PASS ${name}"
    echo "{\"name\":\"${name}\",\"ok\":true}" >> "${GATE_ROOT}/${STAMP}/steps.jsonl"
  else
    echo "FAIL ${name}"
    echo "{\"name\":\"${name}\",\"ok\":false}" >> "${GATE_ROOT}/${STAMP}/steps.jsonl"
    status=1
  fi
}

run_one smoke \
  env RESULTS_DIR="${GATE_ROOT}/${STAMP}/smoke" \
  bash "${SCRIPT_DIR}/run-dflash-linear-sglang-smoke.sh"

if [[ -z "${STOCK_MAL_JSON}" ]]; then
  echo "STOCK_MAL_JSON unset; skipping stock-vs-linear id compare"
fi

for batch in 1 4; do
  for overlap in 1 0; do
    name="parity-bs${batch}-ov${overlap}"
    run_one "${name}" \
      env \
        RESULTS_DIR="${GATE_ROOT}/${STAMP}/${name}" \
        MAX_RUNNING_REQUESTS="${batch}" \
        OVERLAP_SCHEDULE="${overlap}" \
        SGLANG_DFLASH_LINEAR_SHADOW_CHECK=1 \
        STOCK_MAL_JSON="${STOCK_MAL_JSON}" \
      bash "${SCRIPT_DIR}/run-dflash-linear-parity.sh"
  done
done

echo "M1 driver finished status=${status} log=${GATE_ROOT}/${STAMP}"
exit "${status}"
