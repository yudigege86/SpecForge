#!/bin/bash
# CPU-side context-first GDN layer tests inside the Primus ROCm image.
set -euo pipefail

cd /workspace/SpecForge

echo "=== specforge install --no-deps ==="
pip install -e . --no-deps

echo "=== torch ==="
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
PY

echo "=== linear-context layer tests ==="
python3 -m unittest \
  tests.test_algorithms.test_dflash_linear \
  tests.test_modeling.test_linear_context_scan \
  tests.test_modeling.test_dflash_linear_layer \
  tests.test_modeling.test_draft_registry \
  -v \
  2>&1 | tee "${RUN_DIR}/unittest.log"
