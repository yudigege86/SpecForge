"""GDN/KDA prefix-state scan used by linear-context DFlash.

Stock DFlash concatenates target-context K/V with the B-token block, so
attention and cache are O(L). Linear-context DFlash instead:

1. runs one causal GDN or KDA scan over the verified target-feature sequence;
2. gathers the prefix state ``S_{p-1}`` for each sampled anchor ``p``;
3. lets later layers read that fixed-size state instead of the full context KV.

This module is the scan/gather primitive. It does not yet replace draft
attention. The recurrence matches the paper:

``S_t = (I - β_t k_t k_t^T) D_t S_{t-1} + β_t k_t v_t^T``

with scalar decay ``D_t = α_t I`` for GDN and channel-wise
``D_t = diag(α_t)`` for KDA.

Gates follow native GDN/KDA: ``g = -exp(A_log) * softplus(a + dt_bias)``
(log-space ``α``), matching flash-linear-attention's ``use_gate_in_kernel``.
GDN uses per-head ``A_log`` and ``dt_bias`` with shape ``[H]``. KDA uses
native shapes: ``A_log`` is ``[H]`` (broadcast over the key dim) and
``dt_bias`` is ``[H*K]``. Both stay FP32; the scan itself runs in the
feature dtype (BF16 in the default recipe).

On MI355X, install the ROCm extra against the image torch — do not overlay a
CUDA wheel or reinstall torch from the PyTorch ROCm index:

``pip install flash-linear-attention[rocm]``

See https://github.com/fla-org/flash-linear-attention#installation
"""

from __future__ import annotations

import inspect
import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

LinearContextVariant = Literal["gdn", "kda"]
LinearContextBackend = Literal["auto", "naive", "fla"]

# FLA chunk kernels use 64; gather never materializes more than this many
# per-token states at once (``T/C`` times less memory than a full tape).
SCAN_CHUNK_SIZE = 64

# Native GDN/KDA timescale init. ``A ~ U(0.25, 16)`` and
# ``dt ~ LogU(1e-5, 0.1)`` so some channels retain tens of thousands of
# tokens (32K-scale) while others stay short-memory. Stock FLA uses
# ``dt_min=1e-3``; that alone gives a ~700-token half-life at ``A=1``.
GDN_A_INIT_MIN = 0.25
GDN_A_INIT_MAX = 16.0
GDN_DT_MIN = 1e-5
GDN_DT_MAX = 0.1
GDN_DT_INIT_FLOOR = 1e-5


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)


def _inverse_softplus(dt: torch.Tensor) -> torch.Tensor:
    return dt + torch.log(-torch.expm1(-dt))


def _init_A_log(num_channels: int) -> nn.Parameter:
    values = torch.empty(num_channels).uniform_(GDN_A_INIT_MIN, GDN_A_INIT_MAX)
    parameter = nn.Parameter(values.log())
    parameter._no_weight_decay = True
    return parameter


def _init_dt_bias(num_channels: int) -> nn.Parameter:
    dt = torch.exp(
        torch.rand(num_channels) * (math.log(GDN_DT_MAX) - math.log(GDN_DT_MIN))
        + math.log(GDN_DT_MIN)
    ).clamp(min=GDN_DT_INIT_FLOOR)
    parameter = nn.Parameter(_inverse_softplus(dt))
    parameter._no_weight_decay = True
    return parameter


def _import_fla_chunk(variant: LinearContextVariant):
    if variant == "gdn":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return chunk_gated_delta_rule
    from fla.ops.kda import chunk_kda

    return chunk_kda


def fla_available(variant: LinearContextVariant = "gdn") -> bool:
    try:
        _import_fla_chunk(variant)
    except ImportError:
        return False
    return True


def _fla_accepts_initial_state(chunk_fn) -> bool:
    try:
        return "initial_state" in inspect.signature(chunk_fn).parameters
    except (TypeError, ValueError):
        return True


def gated_delta_step(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply one causal GDN/KDA update.

    Args:
        state: ``[B, H, K, V]`` recurrent matrix ``S_{t-1}``.
        key: ``[B, H, K]``.
        value: ``[B, H, V]``.
        log_decay: GDN ``[B, H]`` or KDA ``[B, H, K]`` log-space ``α``.
        beta: ``[B, H]`` write strength in ``(0, 1)``.
    """

    alpha = log_decay.exp()
    if alpha.ndim == 2:
        decayed = state * alpha[:, :, None, None]
    elif alpha.ndim == 3:
        decayed = state * alpha[:, :, :, None]
    else:
        raise ValueError(
            f"log_decay must be [B, H] (GDN) or [B, H, K] (KDA), got {tuple(log_decay.shape)}"
        )
    beta = beta[:, :, None, None]
    key_t_state = torch.einsum("bhk,bhkv->bhv", key, decayed)
    write = torch.einsum("bhk,bhv->bhkv", key, value)
    erase = torch.einsum("bhk,bhv->bhkv", key, key_t_state)
    return decayed + beta * (write - erase)


def gated_delta_scan(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    return_all_states: bool = False,
    normalize_qk: bool = False,
) -> torch.Tensor:
    """Causal GDN/KDA scan over a target-feature sequence.

    Args:
        key / value: ``[B, T, H, D]``.
        log_decay: GDN ``[B, T, H]`` or KDA ``[B, T, H, K]``.
        beta: ``[B, T, H]``.
        initial_state: optional ``[B, H, K, V]``. Defaults to zeros.
        return_all_states: if True, return ``[B, T, H, K, V]`` states after
            every token. Otherwise return only the final state ``[B, H, K, V]``.
    """

    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"key/value must be [B, T, H, D], got {tuple(key.shape)} and {tuple(value.shape)}"
        )
    batch, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if normalize_qk:
        key = _l2norm(key)
    if initial_state is None:
        state = key.new_zeros(batch, num_heads, key_dim, value_dim)
    else:
        state = initial_state
    states = (
        key.new_empty(batch, seq_len, num_heads, key_dim, value_dim)
        if return_all_states
        else None
    )
    for time in range(seq_len):
        state = gated_delta_step(
            state,
            key[:, time],
            value[:, time],
            log_decay[:, time],
            beta[:, time],
        )
        if states is not None:
            states[:, time] = state
    if states is not None:
        return states
    return state


def _align_scan_tensors(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Force recurrent tensors onto ``key.dtype``.

    ``A_log`` / ``dt_bias`` stay FP32; the projected gate is cast to the
    feature dtype before the scan so BF16 naive ``alpha * state`` and
    ``einsum`` do not mix FP32 and BF16.
    """

    dtype = key.dtype
    value = value.to(dtype=dtype)
    log_decay = log_decay.to(dtype=dtype)
    beta = beta.to(dtype=dtype)
    if initial_state is not None:
        initial_state = initial_state.to(dtype=dtype)
    return key, value, log_decay, beta, initial_state


def _validate_anchor_positions(
    anchor_positions: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
) -> None:
    if anchor_positions.ndim != 2:
        raise ValueError(
            f"anchor_positions must be [B, A], got {tuple(anchor_positions.shape)}"
        )
    if anchor_positions.shape[0] != batch:
        raise ValueError(
            "anchor_positions batch does not match features: "
            f"{tuple(anchor_positions.shape)} vs batch={batch}"
        )
    out_of_range = (anchor_positions < 0) | (anchor_positions > seq_len)
    if bool(out_of_range.any()):
        raise ValueError(
            "anchor_positions must satisfy 0 <= p <= seq_len "
            f"({seq_len}); got min={int(anchor_positions.min())} "
            f"max={int(anchor_positions.max())}"
        )


def gather_prefix_states(
    states_after: torch.Tensor,
    anchor_positions: torch.Tensor,
) -> torch.Tensor:
    """Gather ``S_{p-1}`` from per-token post-update states.

    ``states_after[:, t]`` is the recurrent state after consuming target
    token ``t`` (0-based). For an anchor at position ``p``, the block may
    only see the prefix ``[0, p)``, which is ``states_after[:, p - 1]``
    when ``p > 0`` and zeros when ``p == 0``.
    """

    if states_after.ndim != 5:
        raise ValueError(
            f"states_after must be [B, T, H, K, V], got {tuple(states_after.shape)}"
        )
    batch, seq_len, num_heads, key_dim, value_dim = states_after.shape
    _validate_anchor_positions(
        anchor_positions, batch=batch, seq_len=seq_len
    )
    prefix_index = (anchor_positions - 1).clamp(min=-1)
    gathered = states_after.new_zeros(
        batch, anchor_positions.shape[1], num_heads, key_dim, value_dim
    )
    valid = prefix_index >= 0
    safe_index = prefix_index.clamp(min=0, max=seq_len - 1)
    selected = torch.gather(
        states_after,
        1,
        safe_index[:, :, None, None, None].expand(
            -1, -1, num_heads, key_dim, value_dim
        ),
    )
    return torch.where(valid[:, :, None, None, None], selected, gathered)


def _chunked_naive_scan_and_gather(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    anchor_positions: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    normalize_qk: bool = False,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> torch.Tensor:
    """Gather ``S_{p-1}`` while only materializing a ``chunk_size`` state tape."""

    batch, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    _validate_anchor_positions(anchor_positions, batch=batch, seq_len=seq_len)
    gathered = key.new_zeros(
        batch, anchor_positions.shape[1], num_heads, key_dim, value_dim
    )
    prefix_index = anchor_positions - 1
    running = initial_state
    if running is None:
        running = key.new_zeros(batch, num_heads, key_dim, value_dim)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        tape = gated_delta_scan(
            key[:, start:end],
            value[:, start:end],
            log_decay[:, start:end],
            beta[:, start:end],
            initial_state=running,
            return_all_states=True,
            normalize_qk=normalize_qk,
        )
        running = tape[:, -1]
        in_chunk = (prefix_index >= start) & (prefix_index < end)
        if not bool(in_chunk.any()):
            continue
        local = (prefix_index - start).clamp(min=0)
        selected = torch.gather(
            tape,
            1,
            local[:, :, None, None, None].expand(
                -1, -1, num_heads, key_dim, value_dim
            ),
        )
        gathered = torch.where(in_chunk[:, :, None, None, None], selected, gathered)
    return gathered


def _fla_accepts_cu_seqlens(chunk_fn) -> bool:
    try:
        return "cu_seqlens" in inspect.signature(chunk_fn).parameters
    except (TypeError, ValueError):
        return False


def _concat_varlen(tensor: torch.Tensor, batch_index, lengths) -> torch.Tensor:
    pieces = [
        tensor[batch, :length]
        for batch, length in zip(batch_index.tolist(), lengths.tolist())
    ]
    return torch.cat(pieces, dim=0).unsqueeze(0)


def _fla_varlen_final_states(
    chunk_fn,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    batch_index: torch.Tensor,
    lengths: torch.Tensor,
    *,
    normalize_qk: bool,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    """One FLA launch for many prefixes via ``cu_seqlens`` (batch flattened to 1)."""

    packed_key = _concat_varlen(key, batch_index, lengths)
    packed_value = _concat_varlen(value, batch_index, lengths)
    packed_decay = _concat_varlen(log_decay, batch_index, lengths)
    packed_beta = _concat_varlen(beta, batch_index, lengths)
    dummy_query = torch.zeros_like(packed_key)
    cu_seqlens = torch.zeros(
        lengths.numel() + 1, dtype=torch.long, device=key.device
    )
    cu_seqlens[1:] = lengths.to(dtype=torch.long).cumsum(0)
    kwargs = {
        "scale": 1.0,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": normalize_qk,
        "cu_seqlens": cu_seqlens,
    }
    if initial_state is None:
        num_heads, key_dim = key.shape[2], key.shape[3]
        value_dim = value.shape[-1]
        h0 = key.new_zeros(lengths.numel(), num_heads, key_dim, value_dim)
    else:
        h0 = initial_state[batch_index]
    kwargs["initial_state"] = h0
    _output, final_state = chunk_fn(
        dummy_query,
        packed_key,
        packed_value,
        packed_decay,
        packed_beta,
        **kwargs,
    )
    if final_state is None:
        raise RuntimeError("FLA chunk kernel returned no final state")
    return final_state


def _fla_prefix_state(
    chunk_fn,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    normalize_qk: bool = False,
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dummy_query = torch.zeros_like(key)
    kwargs = {
        "scale": 1.0,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": normalize_qk,
    }
    if initial_state is not None:
        kwargs["initial_state"] = initial_state
    _output, final_state = chunk_fn(
        dummy_query,
        key,
        value,
        log_decay,
        beta,
        **kwargs,
    )
    if final_state is None:
        raise RuntimeError("FLA chunk kernel returned no final state")
    return final_state


def _fla_chunked_scan_and_gather(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    anchor_positions: torch.Tensor,
    *,
    variant: LinearContextVariant,
    normalize_qk: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> torch.Tensor:
    """GPU default: one FLA launch per window, chained through ``initial_state``.

    When the kernel accepts ``cu_seqlens``, every in-window prefix plus the
    full window (for the next ``running`` state) is packed into that launch.
    Otherwise unique local lengths still share a batched kernel, not a
    per-anchor full-prefix rescan. Host packing is O(anchors in the window);
    kernel count is O(seq_len / chunk_size).
    """

    chunk_fn = _import_fla_chunk(variant)
    batch, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    _validate_anchor_positions(anchor_positions, batch=batch, seq_len=seq_len)
    if initial_state is not None and not _fla_accepts_initial_state(chunk_fn):
        return _chunked_naive_scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchor_positions,
            initial_state=initial_state,
            normalize_qk=normalize_qk,
            chunk_size=chunk_size,
        )
    use_varlen = _fla_accepts_cu_seqlens(chunk_fn) and _fla_accepts_initial_state(
        chunk_fn
    )
    gathered = key.new_zeros(
        batch, anchor_positions.shape[1], num_heads, key_dim, value_dim
    )
    prefix_index = anchor_positions - 1
    running = initial_state
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_key = key[:, start:end]
        chunk_value = value[:, start:end]
        chunk_decay = log_decay[:, start:end]
        chunk_beta = beta[:, start:end]
        in_chunk = (prefix_index >= start) & (prefix_index < end)
        window = end - start
        if use_varlen:
            local = (prefix_index - start).clamp(min=0)
            row, col = torch.nonzero(in_chunk, as_tuple=True)
            full_batch = torch.arange(batch, device=key.device)
            full_length = torch.full(
                (batch,), window, device=key.device, dtype=torch.long
            )
            if row.numel():
                pack_batch = torch.cat([full_batch, row])
                pack_length = torch.cat([full_length, local[row, col] + 1])
            else:
                pack_batch = full_batch
                pack_length = full_length
            try:
                finals = _fla_varlen_final_states(
                    chunk_fn,
                    chunk_key,
                    chunk_value,
                    chunk_decay,
                    chunk_beta,
                    pack_batch,
                    pack_length,
                    normalize_qk=normalize_qk,
                    initial_state=running,
                )
            except TypeError:
                use_varlen = False
            else:
                running = finals[:batch]
                if row.numel():
                    gathered[row, col] = finals[batch:]
                continue
        next_state = None
        if bool(in_chunk.any()):
            local = (prefix_index - start).clamp(min=0)
            lengths = torch.unique(local[in_chunk])
            last_index = window - 1
            for length in lengths.tolist():
                row_needs = in_chunk & (local == length)
                ht = _fla_prefix_state(
                    chunk_fn,
                    chunk_key[:, : length + 1],
                    chunk_value[:, : length + 1],
                    chunk_decay[:, : length + 1],
                    chunk_beta[:, : length + 1],
                    normalize_qk=normalize_qk,
                    initial_state=running,
                )
                gathered = torch.where(
                    row_needs[:, :, None, None, None],
                    ht[:, None],
                    gathered,
                )
                if length == last_index:
                    next_state = ht
        if next_state is None:
            next_state = _fla_prefix_state(
                chunk_fn,
                chunk_key,
                chunk_value,
                chunk_decay,
                chunk_beta,
                normalize_qk=normalize_qk,
                initial_state=running,
            )
        running = next_state
    return gathered


def _fla_scan_and_gather(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    anchor_positions: torch.Tensor,
    *,
    variant: LinearContextVariant,
    normalize_qk: bool = False,
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FLA gather used by ``backend='fla'`` and GPU ``auto``."""

    return _fla_chunked_scan_and_gather(
        key,
        value,
        log_decay,
        beta,
        anchor_positions,
        variant=variant,
        normalize_qk=normalize_qk,
        initial_state=initial_state,
    )


def resolve_scan_backend(
    backend: LinearContextBackend,
    *,
    on_cuda: bool = False,
    num_anchors: int = 0,
    has_initial_state: bool = False,
    variant: LinearContextVariant = "gdn",
) -> Literal["naive", "fla"]:
    """Pick the scan implementation used to gather ``S_{p-1}``.

    ``auto`` uses chunked FLA whenever a GPU and the matching FLA kernel
    are available. Otherwise it is the chunked naive scan (peak tape
    ``SCAN_CHUNK_SIZE``, not ``seq_len``). Neither path is the old
    per-anchor full-prefix FLA loop.
    """

    del num_anchors, has_initial_state
    want_fla = backend in {"auto", "fla"} and on_cuda and fla_available(variant)
    if backend == "naive":
        return "naive"
    if backend in {"auto", "fla"}:
        return "fla" if want_fla else "naive"
    raise ValueError(f"unknown linear-context backend {backend!r}")


def scan_and_gather(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    anchor_positions: torch.Tensor,
    *,
    variant: LinearContextVariant = "gdn",
    backend: LinearContextBackend = "auto",
    initial_state: Optional[torch.Tensor] = None,
    normalize_qk: bool = False,
) -> torch.Tensor:
    """Scan target features and gather ``S_{p-1}`` at ``anchor_positions``.

    Returns ``[B, A, H, K, V]``. ``anchor_positions`` is ``[B, A]`` with
    0-based token indices; position ``0`` gathers the zero / initial state.
    """

    backend = resolve_scan_backend(
        backend,
        on_cuda=bool(key.is_cuda),
        num_anchors=int(anchor_positions.shape[1]),
        has_initial_state=initial_state is not None,
        variant=variant,
    )
    key, value, log_decay, beta, initial_state = _align_scan_tensors(
        key, value, log_decay, beta, initial_state
    )
    if backend == "fla":
        return _fla_scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchor_positions,
            variant=variant,
            normalize_qk=normalize_qk,
            initial_state=initial_state,
        )
    if backend != "naive":
        raise ValueError(f"unknown linear-context backend {backend!r}")
    return _chunked_naive_scan_and_gather(
        key,
        value,
        log_decay,
        beta,
        anchor_positions,
        initial_state=initial_state,
        normalize_qk=normalize_qk,
    )


class LinearContextScan(nn.Module):
    """Trainable GDN/KDA projections plus prefix-state gather.

    ``forward(target_hidden, anchor_positions)`` maps fused target features
    ``[B, L, C]`` to gathered states ``[B, A, H, K, V]``. Recurrent weights
    stay inside the autograd graph, as the paper requires.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        *,
        variant: LinearContextVariant = "gdn",
        backend: LinearContextBackend = "auto",
        normalize_qk: bool = False,
    ) -> None:
        super().__init__()
        if variant not in ("gdn", "kda"):
            raise ValueError(f"variant must be 'gdn' or 'kda', got {variant!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.variant: LinearContextVariant = variant
        self.backend: LinearContextBackend = backend
        self.normalize_qk = normalize_qk
        self.k_proj = nn.Linear(hidden_size, num_heads * key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * value_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # Native FLA: GDN ``A_log``/``dt_bias`` are per head ``[H]``. KDA uses
        # ``A_log`` per head ``[H]`` broadcast over the key dim and ``dt_bias``
        # per head/key channel ``[H*K]``. ``g_proj`` matches the gate tensor.
        if variant == "gdn":
            self.g_proj = nn.Linear(hidden_size, num_heads, bias=False)
            self.A_log = _init_A_log(num_heads)
            self.dt_bias = _init_dt_bias(num_heads)
        else:
            self.g_proj = nn.Linear(hidden_size, num_heads * key_dim, bias=False)
            self.A_log = _init_A_log(num_heads)
            self.dt_bias = _init_dt_bias(num_heads * key_dim)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        # Gate timescale parameters stay FP32 under ``module.to(bf16)``.
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()
        return out

    def _project(
        self, target_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = target_hidden.shape
        dtype = target_hidden.dtype
        key = self.k_proj(target_hidden).view(
            batch, seq_len, self.num_heads, self.key_dim
        )
        value = self.v_proj(target_hidden).view(
            batch, seq_len, self.num_heads, self.value_dim
        )
        beta = torch.sigmoid(self.beta_proj(target_hidden))
        gate = self.g_proj(target_hidden)
        if self.variant == "kda":
            gate = gate.view(batch, seq_len, self.num_heads, self.key_dim)
            scale = self.A_log.float().exp().view(1, 1, self.num_heads, 1)
            shift = self.dt_bias.float().view(1, 1, self.num_heads, self.key_dim)
        else:
            scale = self.A_log.float().exp().view(1, 1, self.num_heads)
            shift = self.dt_bias.float().view(1, 1, self.num_heads)
        log_decay = (-scale * F.softplus(gate.float() + shift)).to(dtype=dtype)
        return key, value, log_decay, beta

    def forward(
        self,
        target_hidden: torch.Tensor,
        anchor_positions: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key, value, log_decay, beta = self._project(target_hidden)
        return scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchor_positions,
            variant=self.variant,
            backend=self.backend,
            initial_state=initial_state,
            normalize_qk=self.normalize_qk,
        )


__all__ = [
    "GDN_A_INIT_MAX",
    "GDN_A_INIT_MIN",
    "GDN_DT_MAX",
    "GDN_DT_MIN",
    "LinearContextScan",
    "SCAN_CHUNK_SIZE",
    "fla_available",
    "gated_delta_scan",
    "gated_delta_step",
    "resolve_scan_backend",
    "gather_prefix_states",
    "scan_and_gather",
]
