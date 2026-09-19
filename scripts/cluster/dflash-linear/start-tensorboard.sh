#!/bin/bash
set -euo pipefail
export PATH="/home/naqin/.local/bin:${PATH}"
# NFS flock on ~/.cache/uv deadlocks across leftover uv processes.
export UV_CACHE_DIR="/tmp/naqin-uv-cache"
mkdir -p "${UV_CACHE_DIR}"
ROOT="/shared_nfs/naqin/Linear-Context-DFlash/train-1epoch"
# Newest stamp that actually wrote events (skip failed launches).
LOGDIR="$(ls -td "${ROOT}"/2026*/train/runs 2>/dev/null | head -1)"
if [[ -z "${LOGDIR}" ]]; then
  echo "no TensorBoard event dirs under ${ROOT}" >&2
  exit 1
fi
LOGDIR="$(dirname "${LOGDIR}")"
echo "hostname=$(hostname) logdir=${LOGDIR}"
exec /home/naqin/.local/bin/uv run --with tensorboard python -u -m tensorboard.main \
  --logdir="${LOGDIR}" \
  --host 127.0.0.1 \
  --port 6006
