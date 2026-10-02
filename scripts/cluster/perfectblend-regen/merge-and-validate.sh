#!/bin/bash
# Concatenate successful shard JSONL and validate the reasoning contract.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if [[ -f "${PROMPTS_DIR}/num_shards" ]]; then
  NUM_SHARDS="$(tr -d '[:space:]' < "${PROMPTS_DIR}/num_shards")"
fi

MERGED="${REGEN_DIR}/perfectblend_qwen35_4b_regen.jsonl"
mkdir -p "${REGEN_DIR}"

missing=0
i=0
while [[ "${i}" -lt "${NUM_SHARDS}" ]]; do
  shard="$(printf '%04d' "${i}")"
  if [[ ! -f "${REGEN_DIR}/shard-${shard}.done" ]]; then
    echo "missing ${REGEN_DIR}/shard-${shard}.done"
    missing=$((missing + 1))
  fi
  i=$((i + 1))
done
if [[ "${missing}" -ne 0 ]]; then
  echo "FAIL: ${missing} shards are not done"
  exit 1
fi

python3 - "${REGEN_DIR}" "${NUM_SHARDS}" "${MERGED}" <<'PY'
import json
import sys
from pathlib import Path

regen = Path(sys.argv[1])
num_shards = int(sys.argv[2])
merged = Path(sys.argv[3])

def compact(path: Path) -> list[dict]:
    latest = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            row_id = row.get("id")
            if row_id is None or row.get("status") != "success":
                continue
            latest[str(row_id)] = row
    return list(latest.values())

merged.parent.mkdir(parents=True, exist_ok=True)
with merged.open("w", encoding="utf-8") as out:
    for i in range(num_shards):
        src = regen / f"shard-{i:04d}.jsonl"
        rows = compact(src)
        if not rows:
            raise SystemExit(f"FAIL: missing success file {src}")
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {merged}")
PY

python3 "${SPECFORGE_SRC}/scripts/validate_regenerated_data.py" \
  --data-path "${MERGED}" \
  --expect-reasoning \
  --strict-think-markers

input_rows="$(tr -d '[:space:]' < "${PROMPTS_DIR}/row_count")"
merged_rows="$(awk 'END {print NR+0}' "${MERGED}")"
echo "merged=${MERGED}"
echo "input_rows=${input_rows} merged_success_rows=${merged_rows}"
echo "PASS: merge-and-validate"
