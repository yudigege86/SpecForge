# Bind-mount linear DFLASH overlay files over the image package.
# shellcheck disable=SC2034
SGLANG_SRC="${SGLANG_SRC:-/shared_nfs/naqin/Linear-Context-DFlash/sglang}"
_SGLANG_PKG="/sgl-workspace/sglang/python/sglang"
OVERLAY_MOUNTS=()
_overlay_add() {
  local src="${SGLANG_SRC}/$1"
  local dest="$2"
  if [[ -f "${src}" ]]; then
    OVERLAY_MOUNTS+=(-v "${src}:${dest}:ro")
    echo "overlay_mount ${src} -> ${dest}"
  fi
}
_overlay_add python/sglang/srt/models/dflash_linear.py "${_SGLANG_PKG}/srt/models/dflash_linear.py"
_overlay_add python/sglang/srt/speculative/dflash_linear_state.py "${_SGLANG_PKG}/srt/speculative/dflash_linear_state.py"
_overlay_add python/sglang/srt/speculative/dflash_linear_worker_v2.py "${_SGLANG_PKG}/srt/speculative/dflash_linear_worker_v2.py"
_overlay_add python/sglang/srt/speculative/dflash_worker_v2.py "${_SGLANG_PKG}/srt/speculative/dflash_worker_v2.py"
_overlay_add python/sglang/srt/speculative/spec_info.py "${_SGLANG_PKG}/srt/speculative/spec_info.py"
_overlay_add python/sglang/srt/speculative/dflash_utils.py "${_SGLANG_PKG}/srt/speculative/dflash_utils.py"
_overlay_add python/sglang/srt/arg_groups/speculative_hook.py "${_SGLANG_PKG}/srt/arg_groups/speculative_hook.py"
unset -f _overlay_add
unset _SGLANG_PKG
