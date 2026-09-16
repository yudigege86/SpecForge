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
``D_t = diag(α_t)`` for KDA. Gates are stored in log space
(``α = exp(g)``), matching flash-linear-attention's default ``g``.

On MI355X, install the ROCm extra against the image torch — do not overlay a
CUDA wheel or reinstall torch from the PyTorch ROCm index:

``pip install flash-linear-attention[rocm]``

See https://github.com/fla-org/flash-linear-attention#installation
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

LinearContextVariant = Literal["gdn", "kda"]
LinearContextBackend = Literal["auto", "naive", "fla"]


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)


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
    states = []
    for time in range(seq_len):
        state = gated_delta_step(
            state,
            key[:, time],
            value[:, time],
            log_decay[:, time],
            beta[:, time],
        )
        if return_all_states:
            states.append(state)
    if return_all_states:
        return torch.stack(states, dim=1)
    return state


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
    if anchor_positions.ndim != 2:
        raise ValueError(
            f"anchor_positions must be [B, A], got {tuple(anchor_positions.shape)}"
        )
    batch, seq_len, num_heads, key_dim, value_dim = states_after.shape
    if anchor_positions.shape[0] != batch:
        raise ValueError(
            "anchor_positions batch does not match states_after: "
            f"{tuple(anchor_positions.shape)} vs {tuple(states_after.shape)}"
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


def _naive_scan_and_gather(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    anchor_positions: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    normalize_qk: bool = False,
) -> torch.Tensor:
    states = gated_delta_scan(
        key,
        value,
        log_decay,
        beta,
        initial_state=initial_state,
        return_all_states=True,
        normalize_qk=normalize_qk,
    )
    return gather_prefix_states(states, anchor_positions)


def _fla_prefix_state(
    chunk_fn,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    normalize_qk: bool = False,
) -> torch.Tensor:
    dummy_query = torch.zeros_like(key)
    _output, final_state = chunk_fn(
        dummy_query,
        key,
        value,
        log_decay,
        beta,
        scale=1.0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=normalize_qk,
    )
    if final_state is None:
        raise RuntimeError("FLA chunk kernel returned no final state")
    return final_state


def _fla_scan_and_gather(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    anchor_positions: torch.Tensor,
    *,
    variant: LinearContextVariant,
    normalize_qk: bool = False,
) -> torch.Tensor:
    """Gather prefix states by running an FLA chunk scan on each prefix.

    FLA's chunk kernels expose only a final state, so each gathered
    ``S_{p-1}`` is the final state of ``c_1…c_{p-1}``. Duplicate prefix
    work is acceptable here; training can later switch to a segmented
    scan once the layer is wired.
    """

    chunk_fn = _import_fla_chunk(variant)
    batch, _seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    num_anchors = anchor_positions.shape[1]
    gathered = key.new_zeros(batch, num_anchors, num_heads, key_dim, value_dim)
    for batch_idx in range(batch):
        for anchor_idx in range(num_anchors):
            prefix = int(anchor_positions[batch_idx, anchor_idx].item())
            if prefix <= 0:
                continue
            ht = _fla_prefix_state(
                chunk_fn,
                key[batch_idx : batch_idx + 1, :prefix],
                value[batch_idx : batch_idx + 1, :prefix],
                log_decay[batch_idx : batch_idx + 1, :prefix],
                beta[batch_idx : batch_idx + 1, :prefix],
                normalize_qk=normalize_qk,
            )
            gathered[batch_idx, anchor_idx] = ht[0]
    return gathered


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

    if backend == "auto":
        use_fla = (
            key.is_cuda
            and initial_state is None
            and fla_available(variant)
        )
        backend = "fla" if use_fla else "naive"
    if backend == "fla":
        if initial_state is not None:
            return _naive_scan_and_gather(
                key,
                value,
                log_decay,
                beta,
                anchor_positions,
                initial_state=initial_state,
                normalize_qk=normalize_qk,
            )
        return _fla_scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchor_positions,
            variant=variant,
            normalize_qk=normalize_qk,
        )
    if backend != "naive":
        raise ValueError(f"unknown linear-context backend {backend!r}")
    return _naive_scan_and_gather(
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
        if variant == "gdn":
            self.g_proj = nn.Linear(hidden_size, num_heads, bias=False)
        else:
            self.g_proj = nn.Linear(hidden_size, num_heads * key_dim, bias=False)

    def _project(
        self, target_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = target_hidden.shape
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
        log_decay = F.logsigmoid(gate)
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
    "LinearContextScan",
    "fla_available",
    "gated_delta_scan",
    "gated_delta_step",
    "gather_prefix_states",
    "scan_and_gather",
]
