#!/bin/bash
# Respawns watch-and-merge.sh if it exits, so login-shell SIGHUP/set -e
# does not leave the queue unfilled.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="${LOG_ROOT:-/shared_nfs/naqin/Linear-Context-DFlash/perfectblend-qwen35-4b/logs}"
mkdir -p "${LOG_ROOT}"
echo $$ > "${LOG_ROOT}/keep-watch.pid"
while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keep-watch start" >> "${LOG_ROOT}/watch.out"
  bash "${SCRIPT_DIR}/watch-and-merge.sh" >> "${LOG_ROOT}/watch.out" 2>&1 || true
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keep-watch restart" >> "${LOG_ROOT}/watch.out"
  sleep 5
done
