"""Training wrapper for linear-context DFlash.

Stock ``OnlineDFlashModel`` packs Flex/SDPA masks over the full target
prefix. Linear-context training scans once, then runs independent B-token
blocks, so this subclass drops the context attention mask.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel


class OnlineDFlashLinearModel(OnlineDFlashModel):
    """DFlash objective with scan/gather context instead of O(L) draft KV."""

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            max_valid_anchors=max_valid_anchors,
        )
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        output_hidden = self.draft_model(
            position_ids=draft_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden
