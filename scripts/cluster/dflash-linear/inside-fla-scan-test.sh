#!/bin/bash
# Install flash-linear-attention[rocm] on the existing MI355X image torch
# and run GDN/KDA scan/gather parity tests.
# https://github.com/fla-org/flash-linear-attention#installation
set -euo pipefail

cd /workspace/SpecForge

echo "=== torch (leave the image wheel in place) ==="
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("gcn_arch", torch.cuda.get_device_properties(0).gcnArchName)
PY

echo "=== install flash-linear-attention[rocm] ==="
# Do not pip-install torch from download.pytorch.org/whl/rocm7.2: this image
# already has a working ROCm torch. The [rocm] extra only pins torch>=2.7.0.
pip install einops
pip install 'flash-linear-attention[rocm]'

echo "=== FLA import ==="
python3 - <<'PY'
import fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("fla_file", fla.__file__)
print("fla_version", getattr(fla, "__version__", "?"))
print("chunk_gated_delta_rule", chunk_gated_delta_rule)
try:
    from fla.ops.kda import chunk_kda
    print("chunk_kda", chunk_kda)
except Exception as exc:
    print("chunk_kda_unavailable", type(exc).__name__, exc)
PY

echo "=== scan/gather tests ==="
export LINEAR_CONTEXT_REQUIRE_FLA=1
python3 -m unittest tests.test_modeling.test_linear_context_scan -v \
  2>&1 | tee "${RUN_DIR}/unittest.log"
