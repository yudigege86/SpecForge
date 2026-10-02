"""Linear-context DFlash with DFlash2 local convolution and path selection.

The GDN/KDA prefix read is unchanged. Grouped conv wraps only B×B attention
and the MLP; context retrieval is not convolved. The candidate selector is
the same module stock DFlash2 uses.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from .dflash2 import CandidateSelector, DFlashGroupedConv
from .dflash_kernels import DFlashKernels
from .dflash_linear import (
    DFlashLinearDecoderLayer,
    DFlashLinearDraftModel,
    _reshape_block_rope,
)
from .registry import register_draft


def _require_dflash2_int(method_config: dict, key: str) -> int:
    value = method_config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"DFlash2LinearDraftModel requires dflash_config.{key}")
    return value


class DFlash2LinearDecoderLayer(DFlashLinearDecoderLayer):
    """Linear decoder with DFlash2 conv around local attention and MLP."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
        *,
        attention_conv: DFlashGroupedConv,
        mlp_conv: DFlashGroupedConv,
        block_size: Optional[int] = None,
    ) -> None:
        super().__init__(config, layer_idx, kernels, block_size=block_size)
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        fused_target: torch.Tensor,
        anchor_positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_keep_mask: Optional[torch.Tensor] = None,
        prefix_state: Optional[torch.Tensor] = None,
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
        if prefix_state is None:
            prefix_state = self.context_scan(fused_target, anchor_positions)
        elif tuple(prefix_state.shape[:2]) != (batch, num_anchors):
            raise ValueError(
                "prefix_state must have shape "
                f"(batch, num_anchors, H, K, V)=({batch}, {num_anchors}, ...), "
                f"got {tuple(prefix_state.shape)}"
            )
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

        packed_conditioned = conditioned.reshape(batch, packed, hidden_size)
        packed_conditioned, attn_kernel = self.attention_conv.prepare(
            packed_conditioned
        )
        flat = packed_conditioned.reshape(batch * num_anchors, block, hidden_size)
        retrieved_flat = (
            None
            if attn_retrieved is None
            else attn_retrieved.reshape(batch * num_anchors, block, hidden_size)
        )
        cos, sin = position_embeddings
        cos = _reshape_block_rope(cos, batch, num_anchors, block)
        sin = _reshape_block_rope(sin, batch, num_anchors, block)
        attn = self.self_attn(flat, (cos, sin), retrieved=retrieved_flat)
        attn = self.attention_conv.finish(
            attn.reshape(batch, packed, hidden_size),
            attn_kernel,
        ).view(batch, num_anchors, block, hidden_size)
        hidden = residual + attn
        if self.context_residual is not None:
            hidden = hidden + self.context_residual(retrieved)

        residual_mlp = hidden
        mlp_input, mlp_kernel = self.mlp_conv.prepare(
            self.post_attention_layernorm(hidden).reshape(batch, packed, hidden_size)
        )
        mlp_out = self.mlp_conv.finish(self.mlp(mlp_input), mlp_kernel)
        hidden = residual_mlp + mlp_out.view(batch, num_anchors, block, hidden_size)
        if block_keep_mask is not None:
            hidden = hidden * block_keep_mask.to(dtype=hidden.dtype)[:, :, None, None]
        return hidden.view(batch, packed, hidden_size)


@register_draft
class DFlash2LinearDraftModel(DFlashLinearDraftModel):
    """Linear-context DFlash plus DFlash2 local conv and candidate selector."""

    _no_split_modules = ["DFlash2LinearDecoderLayer"]
    decoder_layer_class = DFlash2LinearDecoderLayer

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, DFlashGroupedConv):
            nn.init.zeros_(module.kernel_projection.weight)

    def _dflash2_config(self) -> dict:
        return dict(getattr(self.config, "dflash_config", None) or {})

    def _build_decoder_layer(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> nn.Module:
        method_config = self._dflash2_config()
        taps = _require_dflash2_int(method_config, "conv_kernel_size")
        group_size = _require_dflash2_int(method_config, "conv_group_size")

        def grouped_conv() -> DFlashGroupedConv:
            return DFlashGroupedConv(
                hidden_size=int(config.hidden_size),
                block_size=self.block_size,
                taps=taps,
                group_size=group_size,
            )

        return self.decoder_layer_class(
            config,
            layer_idx,
            kernels,
            attention_conv=grouped_conv(),
            mlp_conv=grouped_conv(),
            block_size=self.block_size,
        )

    def _init_draft_head(self, config: Qwen3Config, dflash_config: dict) -> None:
        selector_rank = _require_dflash2_int(dflash_config, "selector_rank")
        selector_top_k = _require_dflash2_int(dflash_config, "selector_top_k")
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=selector_rank,
            top_k=selector_top_k,
            initializer_range=float(config.initializer_range),
        )

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        method_config = self._dflash2_config()
        transformed = logits.float() * float(
            method_config.get("output_multiplier", 1.0)
        )
        softcap = method_config.get("final_logit_softcapping")
        if softcap is not None:
            softcap = float(softcap)
            if softcap <= 0:
                raise ValueError("DFlash2 final_logit_softcapping must be > 0")
            transformed = torch.tanh(transformed / softcap) * softcap
        return transformed

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        hidden = draft_hidden[:, -self.block_size + 1 :, :]
        unary_logits = self.transform_unary_logits(target.lm_head(hidden))
        unary_topk, candidate_ids = unary_logits.topk(
            self.candidate_selector.top_k,
            dim=-1,
        )
        return self.candidate_selector.greedy_path(
            candidate_ids=candidate_ids,
            unary_logits=unary_topk,
            hidden_states=hidden,
            anchor_token_ids=block_output_ids[:, 0],
        )


__all__ = [
    "DFlash2LinearDecoderLayer",
    "DFlash2LinearDraftModel",
]
