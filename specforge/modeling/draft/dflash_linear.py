"""Linear-context DFlash: GDN/KDA prefix state plus dense B-token attention.

Stock DFlash concatenates target-context K/V with the draft block. This module
instead:

1. scans fused target features into a fixed-size GDN (default) or KDA state;
2. gathers ``S_{p-1}`` for each sampled anchor;
3. reads that state before building local Q/K/V (context-first);
4. mixes only inside the B-token block.

Serving uses ``spec_generate``: one B-token block at the current decode
position, with a full-prefix GDN/KDA scan instead of concatenated draft KV.
Stock SGLang ``--speculative-algorithm DFLASH`` cannot load this architecture.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    GradientCheckpointingLayer,
    Qwen3Config,
)

from .dflash import (
    DFlashDraftModel,
    apply_rotary_pos_emb,
    extract_context_feature,
    project_draft_logits,
    sample,
    target_input_embeddings,
    target_lm_head,
)
from .dflash_kernels import DFlashKernels
from .linear_context import LinearContextScan, _l2norm
from .registry import register_draft


LINEAR_CONTEXT_VARIANTS = ("gdn", "kda")
LINEAR_CONTEXT_INJECTIONS = ("gated_residual", "independent", "qkv_conditioning")


def resolve_linear_context_settings(config: Qwen3Config) -> dict:
    """Defaults for the recurrent context memory; JSON may override any field."""

    method = dict(getattr(config, "dflash_config", None) or {})
    settings = dict(method.get("linear_context") or {})
    head_dim = int(
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    )
    kv_heads = int(config.num_key_value_heads)
    settings.setdefault("variant", "gdn")
    settings.setdefault("injection", "gated_residual")
    settings.setdefault("context_residual", True)
    settings.setdefault("num_heads", min(8, kv_heads))
    settings.setdefault("key_dim", min(64, head_dim))
    settings.setdefault("value_dim", settings["key_dim"])
    settings.setdefault("normalize_qk", True)
    settings.setdefault("backend", "auto")
    if settings["variant"] not in LINEAR_CONTEXT_VARIANTS:
        raise ValueError(
            "linear_context.variant must be one of "
            f"{LINEAR_CONTEXT_VARIANTS}, got {settings['variant']!r}"
        )
    if settings["injection"] not in LINEAR_CONTEXT_INJECTIONS:
        raise ValueError(
            "linear_context.injection must be one of "
            f"{LINEAR_CONTEXT_INJECTIONS}, got {settings['injection']!r}"
        )
    settings["context_residual"] = bool(settings["context_residual"])
    return settings


def _reject_linear_sliding_window(config: Qwen3Config) -> None:
    """Recurrent prefix scan is full-context; sliding DFlash configs are invalid."""

    if bool(getattr(config, "use_sliding_window", False)):
        raise ValueError(
            "DFlashLinearDraftModel does not support sliding-window configs; "
            "the recurrent scan is full-prefix"
        )
    layer_types = list(getattr(config, "layer_types", None) or [])
    if any(str(layer_type) == "sliding_attention" for layer_type in layer_types):
        raise ValueError(
            "DFlashLinearDraftModel does not support sliding_attention layers; "
            "the recurrent scan is full-prefix"
        )


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


_input_embeddings = target_input_embeddings
_lm_head = target_lm_head
_project_logits = project_draft_logits


def _forward_target(
    target: nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    *,
    logits_to_keep: Optional[int] = None,
):
    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
        "output_hidden_states": True,
    }
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = logits_to_keep
    try:
        return target(**kwargs)
    except TypeError:
        kwargs.pop("logits_to_keep", None)
        return target(**kwargs)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _reshape_block_rope(
    tensor: torch.Tensor,
    batch: int,
    num_anchors: int,
    block: int,
) -> torch.Tensor:
    """Map model-level RoPE caches onto independent [B*A, block, dim] blocks."""

    packed = num_anchors * block
    if tensor.dim() == 4:
        if tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        elif tensor.shape[2] == packed:
            tensor = tensor.squeeze(1)
        else:
            tensor = tensor.reshape(batch, packed, tensor.shape[-1])
    return tensor.reshape(batch, num_anchors, block, tensor.shape[-1]).reshape(
        batch * num_anchors, block, tensor.shape[-1]
    )


class BlockLocalAttention(nn.Module):
    """Dense bidirectional attention over one DFlash block. No context KV."""

    def __init__(
        self,
        config: Qwen3Config,
        kernels: DFlashKernels,
        *,
        retrieved_bias: bool = False,
    ):
        super().__init__()
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        bias = bool(config.attention_bias)
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=bias
        )
        if retrieved_bias:
            self.q_r_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.head_dim, bias=False
            )
            self.k_r_proj = nn.Linear(
                config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
            self.v_r_proj = nn.Linear(
                config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
        else:
            self.q_r_proj = None
            self.k_r_proj = None
            self.v_r_proj = None
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        retrieved: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if retrieved is not None:
            if self.q_r_proj is None:
                raise ValueError(
                    "BlockLocalAttention received retrieved features but was "
                    "built without retrieved_bias"
                )
            query = query + self.q_r_proj(retrieved)
            key = key + self.k_r_proj(retrieved)
            value = value + self.v_r_proj(retrieved)
        query = query.view(batch, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)


class DFlashLinearDecoderLayer(GradientCheckpointingLayer):
    """GDN/KDA prefix read, configurable injection, then dense B×B mixing.

    Layer-1 noise tokens are ``[anchor | MASK…]``, so identical MASK embeddings
    would otherwise issue the same context query. A learned per-offset horizon
    embedding is added only on the retrieve path; local B×B mixing still sees
    the unmodified block tokens.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        del layer_idx
        self.hidden_size = int(config.hidden_size)
        if block_size is None:
            block_size = getattr(config, "block_size", None)
        self.block_size = int(block_size)
        self.settings = resolve_linear_context_settings(config)
        self.input_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.context_scan = LinearContextScan(
            hidden_size=config.hidden_size,
            num_heads=int(self.settings["num_heads"]),
            key_dim=int(self.settings["key_dim"]),
            value_dim=int(self.settings["value_dim"]),
            variant=self.settings["variant"],
            backend=self.settings["backend"],
            normalize_qk=bool(self.settings["normalize_qk"]),
        )
        recurrent_out = int(self.settings["num_heads"]) * int(self.settings["value_dim"])
        self.context_query = nn.Linear(
            config.hidden_size,
            int(self.settings["num_heads"]) * int(self.settings["key_dim"]),
            bias=False,
        )
        self.horizon_embed = nn.Embedding(self.block_size, config.hidden_size)
        self.context_read_proj = nn.Linear(recurrent_out, config.hidden_size, bias=False)
        if self.settings["injection"] == "gated_residual":
            self.inject_gate = nn.Linear(
                2 * config.hidden_size, config.hidden_size, bias=True
            )
            self.inject_value = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
        else:
            self.inject_gate = None
            self.inject_value = None
        if self.settings["context_residual"]:
            self.context_residual = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
        else:
            self.context_residual = None
        self.self_attn = BlockLocalAttention(
            config,
            kernels,
            retrieved_bias=self.settings["injection"] == "qkv_conditioning",
        )
        self.mlp = kernels.make_mlp(config)

    def _context_read(
        self,
        normalized: torch.Tensor,
        prefix_state: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_anchors, block, _ = normalized.shape
        num_heads = int(self.settings["num_heads"])
        key_dim = int(self.settings["key_dim"])
        offsets = torch.arange(block, device=normalized.device)
        query_in = normalized + self.horizon_embed(offsets)
        query = self.context_query(query_in).view(
            batch, num_anchors, block, num_heads, key_dim
        )
        if self.settings["normalize_qk"]:
            query = _l2norm(query)
        # query: [B, A, block, H, K]; prefix_state: [B, A, H, K, V]
        retrieved = torch.einsum("baqhk,bahkv->baqhv", query, prefix_state)
        retrieved = retrieved.reshape(batch, num_anchors, block, -1)
        return self.context_read_proj(retrieved)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fused_target: torch.Tensor,
        anchor_positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, packed, hidden_size = hidden_states.shape
        block = self.block_size
        if packed % block:
            raise ValueError(
                f"packed draft length {packed} is not divisible by block_size {block}"
            )
        num_anchors = packed // block
        if tuple(anchor_positions.shape) != (batch, num_anchors):
            raise ValueError(
                "anchor_positions must have shape (batch, packed // block_size) "
                f"= ({batch}, {num_anchors}), got {tuple(anchor_positions.shape)}"
            )
        if block_keep_mask is not None and tuple(block_keep_mask.shape) != (
            batch,
            num_anchors,
        ):
            raise ValueError(
                "block_keep_mask must have shape (batch, packed // block_size) "
                f"= ({batch}, {num_anchors}), got {tuple(block_keep_mask.shape)}"
            )
        blocks = hidden_states.view(batch, num_anchors, block, hidden_size)
        prefix_state = self.context_scan(fused_target, anchor_positions)
        residual = blocks
        normalized = self.input_layernorm(blocks)
        retrieved = self._context_read(normalized, prefix_state)
        injection = self.settings["injection"]
        attn_retrieved = None
        if injection == "gated_residual":
            gate = torch.sigmoid(
                self.inject_gate(torch.cat([normalized, retrieved], dim=-1))
            )
            conditioned = normalized + gate * self.inject_value(retrieved)
        elif injection == "independent":
            conditioned = normalized
        elif injection == "qkv_conditioning":
            conditioned = normalized
            attn_retrieved = retrieved
        else:
            raise ValueError(f"unhandled linear_context.injection {injection!r}")

        flat = conditioned.reshape(batch * num_anchors, block, hidden_size)
        retrieved_flat = (
            None
            if attn_retrieved is None
            else attn_retrieved.reshape(batch * num_anchors, block, hidden_size)
        )
        cos, sin = position_embeddings
        cos = _reshape_block_rope(cos, batch, num_anchors, block)
        sin = _reshape_block_rope(sin, batch, num_anchors, block)
        attn = self.self_attn(flat, (cos, sin), retrieved=retrieved_flat).view(
            batch, num_anchors, block, hidden_size
        )
        hidden = residual + attn
        if self.context_residual is not None:
            hidden = hidden + self.context_residual(retrieved)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        if block_keep_mask is not None:
            hidden = hidden * block_keep_mask.to(dtype=hidden.dtype)[:, :, None, None]
        return hidden.view(batch, packed, hidden_size)


@register_draft
class DFlashLinearDraftModel(DFlashDraftModel):
    """DFlash backbone with recurrent context memory and dense block attention."""

    _no_split_modules = ["DFlashLinearDecoderLayer"]
    decoder_layer_class = DFlashLinearDecoderLayer

    def __init__(self, config, dflash_kernels=None) -> None:
        _reject_linear_sliding_window(config)
        super().__init__(config, dflash_kernels=dflash_kernels)

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        return DFlashLinearDecoderLayer(
            config,
            layer_idx,
            kernels,
            block_size=self.block_size,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del attention_mask, past_key_values, use_cache, kwargs
        if noise_embedding is None or target_hidden is None:
            raise ValueError(
                "DFlashLinearDraftModel.forward requires noise_embedding and target_hidden"
            )
        if anchor_positions is None:
            raise ValueError(
                "DFlashLinearDraftModel.forward requires anchor_positions "
                "for prefix-state gather"
            )
        hidden_states = noise_embedding
        packed = hidden_states.shape[1]
        if position_ids.shape[1] != packed:
            raise ValueError(
                "DFlashLinearDraftModel expects draft-only position_ids of "
                f"length {packed}, got {position_ids.shape[1]}"
            )
        fused_target = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                fused_target=fused_target,
                anchor_positions=anchor_positions,
                position_embeddings=position_embeddings,
                block_keep_mask=block_keep_mask,
            )
        return self.norm(hidden_states)

    def _teacher_force_block(
        self,
        *,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
    ) -> torch.Tensor:
        return self(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            anchor_positions=torch.full(
                (1, 1), start, dtype=torch.long, device=position_ids.device
            ),
        )

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        """Draft one B-token block from prefix state ``S_{p-1}``, then verify.

        Unlike stock DFlash, context is not cached as draft KV. Each step scans
        the full verified target-feature prefix and gathers one anchor at the
        current decode position.
        """
        self.eval()
        self.last_acceptance_lengths = []
        device = _module_device(target)
        embed_tokens = _input_embeddings(target)
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=device
        ).unsqueeze(0)
        past_key_values_target = DynamicCache()

        output = _forward_target(
            target,
            input_ids,
            position_ids[:, :num_input_tokens],
            past_key_values_target,
            logits_to_keep=1,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits if output.logits.ndim == 3 else output.logits.unsqueeze(1),
            temperature,
        )
        prefix_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = embed_tokens(block_output_ids)
            anchor_positions = torch.full(
                (1, 1), start, dtype=torch.long, device=device
            )
            draft_hidden = self(
                target_hidden=prefix_hidden,
                noise_embedding=noise_embedding,
                position_ids=block_position_ids,
                anchor_positions=anchor_positions,
            )
            block_output_ids[:, 1:] = self._sample_draft_tokens(
                target,
                draft_hidden,
                block_output_ids,
            )

            output = _forward_target(
                target,
                block_output_ids,
                block_position_ids,
                past_key_values_target,
            )
            posterior = sample(
                output.logits if output.logits.ndim == 3 else output.logits.unsqueeze(1),
                temperature,
            )
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            prefix_hidden = torch.cat(
                [
                    prefix_hidden,
                    extract_context_feature(
                        output.hidden_states, self.target_layer_ids
                    )[:, : acceptance_length + 1, :],
                ],
                dim=1,
            )
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids_t
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        self.last_acceptance_lengths = acceptance_lengths
        return output_ids


__all__ = [
    "BlockLocalAttention",
    "DFlashLinearDecoderLayer",
    "DFlashLinearDraftModel",
    "LINEAR_CONTEXT_INJECTIONS",
    "LINEAR_CONTEXT_VARIANTS",
    "resolve_linear_context_settings",
]
