from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import get_args

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.runtime.contracts import DraftStrategyName

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    REPO_ROOT
    / "examples"
    / "configs"
    / "offline"
    / "colocated"
    / "qwen3.5-4b-dflash-linear-offline.yaml"
)
STOCK_DFLASH_RECIPE = (
    REPO_ROOT
    / "examples"
    / "configs"
    / "offline"
    / "colocated"
    / "qwen3.5-4b-dflash-offline-amd.yaml"
)
DRAFT_CONFIG = REPO_ROOT / "configs" / "qwen3.5-4b-dflash-linear.json"


def _yaml_scalar(path: Path, key: str) -> str:
    prefix = f"{key}:"
    matches = []
    for line in path.read_text().splitlines():
        content = line.split("#", 1)[0].strip()
        if not content.startswith(prefix):
            continue
        value = content[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        matches.append(value)
    if len(matches) != 1:
        raise AssertionError(f"{path} must define {key} exactly once, got {matches}")
    return matches[0]


class DFlashLinearRegistrationTest(unittest.TestCase):
    def test_algorithm_is_registered_beside_stock_dflash(self):
        registry = builtin_algorithm_registry()
        self.assertIn("dflash", registry.names)
        self.assertIn("dflash_linear", registry.names)
        self.assertIn("dflash_linear", get_args(DraftStrategyName))

        linear = registry.resolve("dflash_linear")
        self.assertEqual(
            linear.spec.draft.default_architecture,
            "DFlashLinearDraftModel",
        )
        self.assertEqual(
            linear.providers.offline_for("text").capture_layout.capture_method,
            "dflash",
        )

    def test_qwen35_4b_recipe_wires_the_linear_architecture(self):
        self.assertEqual(_yaml_scalar(RECIPE, "strategy"), "dflash_linear")
        self.assertEqual(_yaml_scalar(RECIPE, "target_model_path"), "Qwen/Qwen3.5-4B")
        self.assertEqual(
            _yaml_scalar(RECIPE, "draft_model_config"),
            "configs/qwen3.5-4b-dflash-linear.json",
        )
        payload = json.loads(DRAFT_CONFIG.read_text())
        self.assertEqual(payload["architectures"], ["DFlashLinearDraftModel"])
        self.assertEqual(
            payload["dflash_config"]["target_layer_ids"],
            [1, 8, 15, 22, 29],
        )
        self.assertEqual(payload["block_size"], 16)
        linear_context = payload["dflash_config"]["linear_context"]
        self.assertEqual(linear_context["variant"], "gdn")
        self.assertEqual(linear_context["injection"], "gated_residual")
        self.assertTrue(linear_context["context_residual"])

    def test_ablation_recipes_are_first_class_config(self):
        ablations = {
            "kda": {
                "variant": "kda",
                "injection": "gated_residual",
                "context_residual": True,
            },
            "independent": {
                "variant": "gdn",
                "injection": "independent",
                "context_residual": True,
            },
            "qkv": {
                "variant": "gdn",
                "injection": "qkv_conditioning",
                "context_residual": True,
            },
            "no-ctx-residual": {
                "variant": "gdn",
                "injection": "gated_residual",
                "context_residual": False,
            },
        }
        for name, expected in ablations.items():
            yaml_path = (
                REPO_ROOT
                / "examples"
                / "configs"
                / "offline"
                / "colocated"
                / f"qwen3.5-4b-dflash-linear-{name}-offline.yaml"
            )
            json_path = REPO_ROOT / "configs" / f"qwen3.5-4b-dflash-linear-{name}.json"
            self.assertEqual(_yaml_scalar(yaml_path, "strategy"), "dflash_linear")
            self.assertEqual(
                _yaml_scalar(yaml_path, "draft_model_config"),
                f"configs/qwen3.5-4b-dflash-linear-{name}.json",
            )
            payload = json.loads(json_path.read_text())
            linear_context = payload["dflash_config"]["linear_context"]
            self.assertEqual(payload["architectures"], ["DFlashLinearDraftModel"])
            for key, value in expected.items():
                self.assertEqual(linear_context[key], value, msg=f"{name}.{key}")

    def test_training_model_uses_independent_block_wrapper(self):
        import inspect

        from specforge.algorithms.dflash_linear.providers import build_training_model

        source = inspect.getsource(build_training_model)
        self.assertIn("OnlineDFlashLinearModel", source)
        self.assertNotIn("return build_dflash_model", source)

    def test_stock_qwen35_4b_dflash_recipe_is_unchanged(self):
        self.assertEqual(_yaml_scalar(STOCK_DFLASH_RECIPE, "strategy"), "dflash")
        self.assertEqual(
            _yaml_scalar(STOCK_DFLASH_RECIPE, "draft_model_config"),
            "configs/qwen3.5-4b-dflash.json",
        )
        self.assertEqual(
            _yaml_scalar(STOCK_DFLASH_RECIPE, "target_model_path"),
            "Qwen/Qwen3.5-4B",
        )
