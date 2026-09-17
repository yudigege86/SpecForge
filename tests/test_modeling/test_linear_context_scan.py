"""Numerical tests for linear-context GDN/KDA scan and prefix-state gather."""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "specforge"
    / "modeling"
    / "draft"
    / "linear_context.py"
)
_SPEC = importlib.util.spec_from_file_location("linear_context", _MODULE_PATH)
_LINEAR_CONTEXT = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LINEAR_CONTEXT)

LinearContextScan = _LINEAR_CONTEXT.LinearContextScan
LOG_DECAY_BIAS_INIT = _LINEAR_CONTEXT.LOG_DECAY_BIAS_INIT
fla_available = _LINEAR_CONTEXT.fla_available
gated_delta_scan = _LINEAR_CONTEXT.gated_delta_scan
gather_prefix_states = _LINEAR_CONTEXT.gather_prefix_states
resolve_scan_backend = _LINEAR_CONTEXT.resolve_scan_backend
scan_and_gather = _LINEAR_CONTEXT.scan_and_gather


def _explicit_step(state, key, value, log_decay, beta):
    """Paper recurrence with explicit ``I - βkk^T`` matrices, one batch/head."""

    alpha = log_decay.exp()
    key_dim = key.shape[-1]
    identity = torch.eye(key_dim, dtype=state.dtype, device=state.device)
    if alpha.ndim == 0:
        decay = alpha * identity
    else:
        decay = torch.diag(alpha)
    kk = torch.outer(key, key)
    left = identity - beta * kk
    write = beta * torch.outer(key, value)
    return left @ (decay @ state) + write


def _random_inputs(variant, *, batch=2, seq_len=8, heads=2, key_dim=4, value_dim=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn(batch, seq_len, heads, key_dim, generator=generator)
    value = torch.randn(batch, seq_len, heads, value_dim, generator=generator)
    if variant == "gdn":
        log_decay = F.logsigmoid(torch.randn(batch, seq_len, heads, generator=generator))
    else:
        log_decay = F.logsigmoid(
            torch.randn(batch, seq_len, heads, key_dim, generator=generator)
        )
    beta = torch.rand(batch, seq_len, heads, generator=generator).clamp(0.05, 0.95)
    return key, value, log_decay, beta


class GatedDeltaScanTest(unittest.TestCase):
    def test_gdn_scan_matches_explicit_python_recurrence(self):
        key, value, log_decay, beta = _random_inputs("gdn")
        states = gated_delta_scan(
            key, value, log_decay, beta, return_all_states=True
        )
        batch, seq_len, heads, key_dim, value_dim = states.shape
        state = torch.zeros(key_dim, value_dim, dtype=key.dtype)
        for batch_idx in range(batch):
            for head in range(heads):
                state.zero_()
                for time in range(seq_len):
                    state = _explicit_step(
                        state,
                        key[batch_idx, time, head],
                        value[batch_idx, time, head],
                        log_decay[batch_idx, time, head],
                        beta[batch_idx, time, head],
                    )
                    torch.testing.assert_close(
                        states[batch_idx, time, head],
                        state,
                        atol=1e-6,
                        rtol=1e-5,
                    )

    def test_kda_scan_matches_explicit_python_recurrence(self):
        key, value, log_decay, beta = _random_inputs("kda")
        states = gated_delta_scan(
            key, value, log_decay, beta, return_all_states=True
        )
        _batch, seq_len, _heads, key_dim, value_dim = states.shape
        state = torch.zeros(key_dim, value_dim, dtype=key.dtype)
        for batch_idx in range(states.shape[0]):
            for head in range(states.shape[2]):
                state = torch.zeros_like(state)
                for time in range(seq_len):
                    state = _explicit_step(
                        state,
                        key[batch_idx, time, head],
                        value[batch_idx, time, head],
                        log_decay[batch_idx, time, head],
                        beta[batch_idx, time, head],
                    )
                    torch.testing.assert_close(
                        states[batch_idx, time, head],
                        state,
                        atol=1e-6,
                        rtol=1e-5,
                    )


class PrefixStateGatherTest(unittest.TestCase):
    def test_gathered_prefix_equals_sequential_scan_of_first_p_minus_one_tokens(self):
        key, value, log_decay, beta = _random_inputs("gdn", seq_len=9)
        anchors = torch.tensor([[0, 1, 4, 9], [2, 3, 6, 8]])
        gathered = scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchors,
            variant="gdn",
            backend="naive",
        )
        for batch_idx in range(anchors.shape[0]):
            for anchor_idx, position in enumerate(anchors[batch_idx].tolist()):
                if position == 0:
                    torch.testing.assert_close(
                        gathered[batch_idx, anchor_idx],
                        torch.zeros_like(gathered[batch_idx, anchor_idx]),
                    )
                    continue
                prefix_state = gated_delta_scan(
                    key[batch_idx : batch_idx + 1, :position],
                    value[batch_idx : batch_idx + 1, :position],
                    log_decay[batch_idx : batch_idx + 1, :position],
                    beta[batch_idx : batch_idx + 1, :position],
                )
                torch.testing.assert_close(
                    gathered[batch_idx, anchor_idx],
                    prefix_state[0],
                    atol=1e-6,
                    rtol=1e-5,
                )

    def test_block_cannot_see_features_at_or_after_the_anchor(self):
        key, value, log_decay, beta = _random_inputs("kda", seq_len=10, seed=1)
        anchors = torch.tensor([[3, 7], [1, 5]])
        baseline = scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchors,
            variant="kda",
            backend="naive",
        )
        for batch_idx in range(anchors.shape[0]):
            for anchor_idx, position in enumerate(anchors[batch_idx].tolist()):
                leaked_key = key.clone()
                leaked_value = value.clone()
                leaked_decay = log_decay.clone()
                leaked_beta = beta.clone()
                leaked_key[batch_idx, position:] = torch.randn_like(
                    leaked_key[batch_idx, position:]
                )
                leaked_value[batch_idx, position:] = torch.randn_like(
                    leaked_value[batch_idx, position:]
                )
                leaked_decay[batch_idx, position:] = F.logsigmoid(
                    torch.randn_like(leaked_decay[batch_idx, position:])
                )
                leaked_beta[batch_idx, position:] = torch.rand_like(
                    leaked_beta[batch_idx, position:]
                ).clamp(0.05, 0.95)
                leaked = scan_and_gather(
                    leaked_key,
                    leaked_value,
                    leaked_decay,
                    leaked_beta,
                    anchors,
                    variant="kda",
                    backend="naive",
                )
                torch.testing.assert_close(
                    leaked[batch_idx, anchor_idx],
                    baseline[batch_idx, anchor_idx],
                    atol=0,
                    rtol=0,
                )

    def test_gather_from_full_state_tape_matches_scan_and_gather(self):
        key, value, log_decay, beta = _random_inputs("gdn")
        anchors = torch.tensor([[1, 5, 8], [0, 3, 7]])
        states = gated_delta_scan(
            key, value, log_decay, beta, return_all_states=True
        )
        torch.testing.assert_close(
            gather_prefix_states(states, anchors),
            scan_and_gather(
                key,
                value,
                log_decay,
                beta,
                anchors,
                backend="naive",
            ),
        )


class LinearContextScanModuleTest(unittest.TestCase):
    def test_autograd_flows_through_projections(self):
        scan = LinearContextScan(
            hidden_size=16,
            num_heads=2,
            key_dim=4,
            value_dim=4,
            variant="gdn",
            backend="naive",
        )
        target_hidden = torch.randn(2, 6, 16, requires_grad=True)
        anchors = torch.tensor([[2, 5], [1, 4]])
        gathered = scan(target_hidden, anchors)
        gathered.sum().backward()
        self.assertIsNotNone(target_hidden.grad)
        self.assertGreater(target_hidden.grad.abs().sum().item(), 0)
        self.assertIsNotNone(scan.k_proj.weight.grad)
        self.assertGreater(scan.k_proj.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(scan.g_proj.weight.grad)
        self.assertGreater(scan.g_proj.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(scan.log_decay_bias.grad)
        self.assertGreater(scan.log_decay_bias.grad.abs().sum().item(), 0)

    def test_auto_backend_is_one_scan_tape_even_on_cuda(self):
        self.assertEqual(
            resolve_scan_backend("auto", on_cuda=True, num_anchors=64),
            "naive",
        )
        self.assertEqual(
            resolve_scan_backend("auto", on_cuda=True, num_anchors=1),
            "naive",
        )
        self.assertEqual(
            resolve_scan_backend("fla", on_cuda=True, has_initial_state=True),
            "naive",
        )
        self.assertEqual(
            resolve_scan_backend("fla", on_cuda=True, has_initial_state=False),
            "fla",
        )

    def test_decay_init_keeps_long_memory(self):
        torch.manual_seed(0)
        for variant in ("gdn", "kda"):
            scan = LinearContextScan(
                hidden_size=32,
                num_heads=2,
                key_dim=4,
                value_dim=4,
                variant=variant,
                backend="naive",
            )
            torch.nn.init.zeros_(scan.g_proj.weight)
            _, _, log_decay, _ = scan._project(torch.randn(2, 16, 32))
            alpha = log_decay.exp()
            expected = torch.sigmoid(
                torch.tensor(LOG_DECAY_BIAS_INIT, dtype=alpha.dtype)
            )
            torch.testing.assert_close(
                alpha.mean(),
                expected,
                atol=1e-5,
                rtol=1e-5,
                msg=variant,
            )
            # α^64 still > 0.5 so a short prefix does not wipe the state.
            self.assertGreater((alpha.mean() ** 64).item(), 0.5, msg=variant)

    def test_auto_scan_and_gather_does_not_call_per_anchor_fla(self):
        key, value, log_decay, beta = _random_inputs("gdn")
        anchors = torch.tensor([[1, 4, 8], [0, 3, 7]])

        def _boom(*_args, **_kwargs):
            raise AssertionError("auto must not use per-anchor FLA")

        original = _LINEAR_CONTEXT._fla_scan_and_gather
        _LINEAR_CONTEXT._fla_scan_and_gather = _boom
        try:
            gathered = scan_and_gather(
                key, value, log_decay, beta, anchors, backend="auto"
            )
        finally:
            _LINEAR_CONTEXT._fla_scan_and_gather = original
        self.assertEqual(tuple(gathered.shape), (2, 3, 2, 4, 5))

    def test_kda_module_gate_shape_is_channel_wise(self):
        scan = LinearContextScan(
            hidden_size=16,
            num_heads=2,
            key_dim=3,
            value_dim=5,
            variant="kda",
            backend="naive",
        )
        self.assertEqual(tuple(scan.g_proj.weight.shape), (2 * 3, 16))
        gathered = scan(torch.randn(1, 4, 16), torch.tensor([[0, 4]]))
        self.assertEqual(tuple(gathered.shape), (1, 2, 2, 3, 5))
        torch.testing.assert_close(
            gathered[:, 0],
            torch.zeros_like(gathered[:, 0]),
        )


def _gpu_available() -> bool:
    return bool(torch.cuda.is_available())


def _require_fla() -> bool:
    return os.environ.get("LINEAR_CONTEXT_REQUIRE_FLA") == "1"


def _move_scan_inputs(variant, device, dtype=torch.bfloat16, **kwargs):
    key, value, log_decay, beta = _random_inputs(variant, **kwargs)
    return (
        key.to(device=device, dtype=dtype),
        value.to(device=device, dtype=dtype),
        log_decay.to(device=device, dtype=dtype),
        beta.to(device=device, dtype=dtype),
    )


@unittest.skipUnless(
    _require_fla() or (_gpu_available() and fla_available("gdn")),
    "FLA GDN chunk kernel needs a GPU (CUDA or ROCm) and flash-linear-attention",
)
class FlaGatedDeltaParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _require_fla():
            if not _gpu_available():
                raise unittest.SkipTest("LINEAR_CONTEXT_REQUIRE_FLA=1 but no GPU")
            if not fla_available("gdn"):
                raise RuntimeError(
                    "LINEAR_CONTEXT_REQUIRE_FLA=1 but flash-linear-attention "
                    "GDN kernels are not importable"
                )

    def _assert_fla_matches_naive(self, variant: str) -> None:
        device = torch.device("cuda")
        key, value, log_decay, beta = _move_scan_inputs(
            variant,
            device,
            batch=1,
            seq_len=64,
            heads=2,
            key_dim=32,
            value_dim=32,
            seed=2,
        )
        anchors = torch.tensor([[16, 32, 64]], device=device)
        naive = scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchors,
            variant=variant,
            backend="naive",
            normalize_qk=True,
        )
        fla = scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchors,
            variant=variant,
            backend="fla",
            normalize_qk=True,
        )
        # bf16 FLA vs the naive tape; characterized on MI355X. Tighten only
        # after a wider device sweep — do not treat 8e-2 as a free pass.
        torch.testing.assert_close(fla.float(), naive.float(), atol=8e-2, rtol=8e-2)

    def test_fla_prefix_gather_matches_naive_gdn(self):
        self._assert_fla_matches_naive("gdn")

    @unittest.skipUnless(
        fla_available("kda"),
        "FLA KDA chunk kernel is not installed",
    )
    def test_fla_prefix_gather_matches_naive_kda(self):
        self._assert_fla_matches_naive("kda")

    def test_fla_prefix_gather_backward_is_finite(self):
        device = torch.device("cuda")
        key, value, log_decay, beta = _move_scan_inputs(
            "gdn",
            device,
            batch=1,
            seq_len=32,
            heads=2,
            key_dim=16,
            value_dim=16,
            seed=3,
        )
        key = key.detach().requires_grad_(True)
        anchors = torch.tensor([[8, 24]], device=device)
        gathered = scan_and_gather(
            key,
            value,
            log_decay,
            beta,
            anchors,
            variant="gdn",
            backend="fla",
            normalize_qk=True,
        )
        gathered.float().sum().backward()
        self.assertIsNotNone(key.grad)
        self.assertTrue(torch.isfinite(key.grad.float()).all())


if __name__ == "__main__":
    unittest.main()
