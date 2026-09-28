#!/bin/bash
# Inside the SGLang ROCm container: cache the target, convert Open-PerfectBlend, split shards.
set -euo pipefail

cd /workspace/SpecForge

echo "=== pip install data prep extras without touching torch/sglang ==="
python3 -m pip install --no-cache-dir datasets huggingface_hub openai tqdm

echo "=== cache ${MODEL_PATH} ==="
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ["MODEL_PATH"]
path = snapshot_download(model, resume_download=True)
print("model_snapshot", path)
PY

TRAIN_JSONL="${PROMPTS_DIR}/perfectblend_train.jsonl"
mkdir -p "${PROMPTS_DIR}"
row_count=0
if [[ -f "${TRAIN_JSONL}" ]]; then
  row_count="$(awk 'END {print NR}' "${TRAIN_JSONL}")"
fi

if [[ "${row_count}" -lt 1000000 ]]; then
  if [[ -f "${TRAIN_JSONL}" ]]; then
    echo "existing ${TRAIN_JSONL} has only ${row_count} rows; rebuilding"
    rm -f "${TRAIN_JSONL}"
  fi
  echo "=== prepare_data.py --dataset perfectblend ==="
  python3 scripts/prepare_data.py \
    --dataset perfectblend \
    --output-path "${PROMPTS_DIR}"
  row_count="$(awk 'END {print NR}' "${TRAIN_JSONL}")"
fi

echo "perfectblend_rows=${row_count}"
if [[ "${row_count}" -lt 1000000 ]]; then
  echo "FAIL: expected ~1.42M Open-PerfectBlend rows, got ${row_count}"
  exit 1
fi

echo "=== split shards NUM_SHARDS=${NUM_SHARDS} SMOKE_ROWS=${SMOKE_ROWS} ==="
python3 /opt/perfectblend-regen/split_shards.py \
  --input "${TRAIN_JSONL}" \
  --out-dir "${PROMPTS_DIR}" \
  --num-shards "${NUM_SHARDS}" \
  --smoke-rows "${SMOKE_ROWS}"

ls -lh "${PROMPTS_DIR}/num_shards" "${PROMPTS_DIR}/row_count" \
  "${PROMPTS_DIR}/shard-smoke.jsonl" "${PROMPTS_DIR}/shard-0000.jsonl"
echo "PASS: inside-prepare"
