from __future__ import annotations

import copy
from pathlib import Path

import yaml

from scripts.e03_a4_stability_evaluate import (
    build_candidate_fast_config,
    build_candidate_inference_config,
)
from scripts.e03_profile_fast_eval import _load_config as validate_fast_config


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_fast_eval_config_is_frozen_id_greedy_and_strict(tmp_path):
    config = yaml.safe_load((ROOT / "configs/training/e03_a4_lr_stability.yaml").read_text())
    source = yaml.safe_load((ROOT / "configs/evaluation/e03_profile_candidate.yaml").read_text())
    original = copy.deepcopy(source)
    inference_path = Path("/tmp/a4-eval/inference_config.json")
    candidate = build_candidate_fast_config(source, config, inference_path)
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False))
    validate_fast_config(candidate_path)
    assert candidate["experiment_version"] == "E03-A4-lr-stability-fastonly-v1"
    assert candidate["selection"]["expected_count"] == 224
    assert candidate["selection"]["expected_conversation_count"] == 101
    assert candidate["selection"]["expected_ids_sha256"] == config["identity"]["expected_frozen_eval_ids_sha256"]
    assert candidate["sources"]["inference_config_path"] == str(inference_path)
    assert candidate["generation"] == {"max_new_tokens": 1200, "do_sample": False}
    assert candidate["gpu_budget"]["max_seconds"] == 900
    assert candidate["adoption_gates"]["minimum_valid_coverage"] == 1.0
    assert candidate["output_root"] == config["evaluation"]["run_root"]
    assert source == original


def test_candidate_inference_snapshot_binds_new_adapter_without_changing_base():
    config = yaml.safe_load((ROOT / "configs/training/e03_a4_lr_stability.yaml").read_text())
    frozen = {
        "backend": "transformers",
        "model_path": "/frozen/base",
        "adapter": {"enabled": True, "path": "/frozen/adapter"},
        "generation": {"do_sample": True},
    }
    original = copy.deepcopy(frozen)
    identity = {"weight_sha256": "a" * 64, "config_sha256": "b" * 64}
    candidate = build_candidate_inference_config(
        frozen, config, Path("/tmp/a4-run"), identity,
    )
    assert candidate["adapter"]["path"] == str(Path("/tmp/a4-run/adapter").resolve())
    assert candidate["adapter"]["expected_adapter_weight_sha256"] == "a" * 64
    assert candidate["adapter"]["expected_adapter_config_sha256"] == "b" * 64
    assert candidate["adapter"]["expected_base_weight_sha256"] == config["base_model"]["expected_weight_sha256"]
    assert candidate["model_path"] == str((ROOT / config["base_model"]["local_path"]).resolve())
    assert candidate["generation"] == {"do_sample": False, "max_new_tokens": 1200}
    assert frozen == original
