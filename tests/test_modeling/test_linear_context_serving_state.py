"""Serving-state tests: masked B-block commits match a full prefix scan."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
from specforge.modeling.draft.linear_context import (
    commit_block_masked,
    gated_delta_scan,
    scan_and_gather,
)


TINY = {
    "architectures": ["DFlashLinearDraftModel"],
    "model_type": "qwen3",
    "block_size": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "num_target_layers": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
    "layer_types": ["full_attention"],
    "dflash_config": {
        "mask_token_id": 0,
        "target_layer_ids": [1],
        "linear_context": {
            "variant": "gdn",
            "injection": "gated_residual",
            "num_heads": 2,
            "key_dim": 16,
            "value_dim": 16,
            "normalize_qk": True,
            "backend": "naive",
        },
    },
}


def _tiny_model():
    payload = json.loads(json.dumps(TINY))
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle)
    try:
        config = AutoDraftModelConfig.from_file(path)
        return AutoDraftModel.from_config(config)
    finally:
        os.unlink(path)


def _random_inputs(*, batch=2, seq_len=20, heads=2, key_dim=4, value_dim=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn(batch, seq_len, heads, key_dim, generator=generator)
    value = torch.randn(batch, seq_len, heads, value_dim, generator=generator)
    log_decay = F.logsigmoid(torch.randn(batch, seq_len, heads, generator=generator))
    beta = torch.rand(batch, seq_len, heads, generator=generator).clamp(0.05, 0.95)
    return key, value, log_decay, beta


class CommitBlockMaskedTest(unittest.TestCase):
    def test_prefill_plus_masked_commits_equals_full_prefix_scan(self):
        key, value, log_decay, beta = _random_inputs(seq_len=24, batch=3, seed=7)
        batch, seq_len, heads, key_dim = key.shape
        value_dim = value.shape[-1]
        block = 4
        prefill = 4
        state = gated_delta_scan(
            key[:, :prefill],
            value[:, :prefill],
            log_decay[:, :prefill],
            beta[:, :prefill],
        )
        consumed = torch.full((batch,), prefill, dtype=torch.long)
        generator = torch.Generator().manual_seed(1)

        def _gather_block(tensor, starts):
            index = starts[:, None] + torch.arange(block, device=tensor.device)
            index = index.clamp(max=seq_len - 1)
            extra = (1,) * (tensor.ndim - 2)
            index = index.reshape(batch, block, *extra).expand(-1, -1, *tensor.shape[2:])
            return torch.gather(tensor, 1, index)

        for _ in range(4):
            remaining = (seq_len - consumed).clamp(min=0)
            if int(remaining.max()) <= 0:
                break
            commit = torch.randint(0, block + 1, (batch,), generator=generator)
            commit[0] = 0
            commit[-1] = min(block, int(remaining[-1]))
            commit = torch.minimum(commit, remaining)
            state = commit_block_masked(
                state,
                _gather_block(key, consumed),
                _gather_block(value, consumed),
                _gather_block(log_decay, consumed),
                _gather_block(beta, consumed),
                commit,
            )
            consumed = consumed + commit

        expected = []
        for row in range(batch):
            length = int(consumed[row])
            expected.append(
                gated_delta_scan(
                    key[row : row + 1, :length],
                    value[row : row + 1, :length],
                    log_decay[row : row + 1, :length],
                    beta[row : row + 1, :length],
                )[0]
            )
        torch.testing.assert_close(state, torch.stack(expected), atol=1e-5, rtol=1e-5)

    def test_identity_mask_leaves_state_unchanged_when_commit_is_zero(self):
        key, value, log_decay, beta = _random_inputs(seq_len=4, batch=1, seed=3)
        initial = gated_delta_scan(key[:, :2], value[:, :2], log_decay[:, :2], beta[:, :2])
        updated = commit_block_masked(
            initial,
            key[:, 2:],
            value[:, 2:],
            log_decay[:, 2:],
            beta[:, 2:],
            torch.zeros(1, dtype=torch.long),
        )
        torch.testing.assert_close(updated, initial)


class IncrementalStateMatchesTeacherForceTest(unittest.TestCase):
    def test_incremental_prefix_state_matches_scan_gather(self):
        model = _tiny_model()
        model.eval()
        torch.manual_seed(0)
        seq_len, start, block = 16, 9, 4
        target_hidden = torch.randn(1, seq_len, 64)
        fused = model.hidden_norm(model.fc(target_hidden))
        layer = model.layers[0]
        scan = layer.context_scan
        key, value, log_decay, beta = scan._project(fused)
        state = fused.new_zeros(1, scan.num_heads, scan.key_dim, scan.value_dim)
        cursor = 0
        while cursor < start:
            take = min(3, start - cursor)
            state = commit_block_masked(
                state,
                key[:, cursor : cursor + take],
                value[:, cursor : cursor + take],
                log_decay[:, cursor : cursor + take],
                beta[:, cursor : cursor + take],
                torch.full((1,), take, dtype=torch.long),
                normalize_qk=scan.normalize_qk,
            )
            cursor += take
        gathered = scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            torch.tensor([[start]]),
            variant="gdn",
            backend="naive",
            normalize_qk=scan.normalize_qk,
        )
        torch.testing.assert_close(state, gathered[:, 0], atol=1e-5, rtol=1e-5)

        noise = torch.randn(1, block, 64)
        position_ids = torch.arange(start, start + block).unsqueeze(0)
        with torch.no_grad():
            from_scan = model._teacher_force_block(
                target_hidden=target_hidden[:, :start, :],
                noise_embedding=noise,
                position_ids=position_ids,
                start=start,
            )
            from_state = model(
                position_ids=position_ids,
                noise_embedding=noise,
                target_hidden=target_hidden[:, :start, :],
                anchor_positions=torch.tensor([[start]]),
                prefix_states=[gathered],
            )
        torch.testing.assert_close(from_state, from_scan, atol=1e-5, rtol=1e-5)
