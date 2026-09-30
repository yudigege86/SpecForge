#!/bin/bash
# Serve TensorBoard for the 1-epoch tree. Bind 127.0.0.1 and run this in the
# SAME ssh -tt -L 127.0.0.1:6006:127.0.0.1:6006 session (see spur-mi355-cluster
# skill). Override TB_HOST=0.0.0.0 only if you know you need it.
set -euo pipefail
export PATH="/home/naqin/.local/bin:${PATH}"
# NFS flock on ~/.cache/uv deadlocks across leftover uv processes.
export UV_CACHE_DIR="/tmp/naqin-uv-cache"
mkdir -p "${UV_CACHE_DIR}"
ROOT="${TB_LOGDIR:-/shared_nfs/naqin/Linear-Context-DFlash/train-1epoch}"
HOST="${TB_HOST:-127.0.0.1}"
PORT="${TB_PORT:-6006}"
if [[ ! -d "${ROOT}" ]]; then
  echo "missing logdir ${ROOT}" >&2
  exit 1
fi
echo "hostname=$(hostname) logdir=${ROOT} bind=${HOST}:${PORT}"
exec /home/naqin/.local/bin/uv run --with tensorboard python -u -m tensorboard.main \
  --logdir="${ROOT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --reload_interval 30 \
  --load_fast false
