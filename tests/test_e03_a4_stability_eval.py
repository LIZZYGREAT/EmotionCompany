from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts import e03_a4_stability_evaluate as stability_evaluator
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


@pytest.mark.parametrize(
    ("status", "threshold_failures"),
    [
        ("stability_preliminarily_passed", []),
        ("stability_failed", ["micro_precision_decline_over_0.02"]),
    ],
)
def test_compare_default_exit_means_report_generated_and_preserves_quality_result(
    monkeypatch, capsys, status, threshold_failures,
):
    result = {
        "status": status,
        "summary": "/synthetic/paired_stability_summary.json",
        "evaluation_manifest": "/synthetic/evaluation_manifest.json",
        "micro_f1_delta": 0.01,
        "threshold_failures": threshold_failures,
    }
    monkeypatch.setattr(stability_evaluator, "compare", lambda _path: result)

    exit_code = stability_evaluator.main(["compare", "--config", "synthetic.yaml"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == status
    assert output["threshold_failures"] == threshold_failures
    assert output["report_path"] == result["summary"]
    assert output["micro_f1_delta"] == result["micro_f1_delta"]
    assert output["evaluation_manifest"] == result["evaluation_manifest"]


def test_compare_strict_quality_gate_accepts_only_preliminary_pass(
    monkeypatch, capsys,
):
    passed = {
        "status": "stability_preliminarily_passed",
        "summary": "/synthetic/passed-summary.json",
        "threshold_failures": [],
    }
    monkeypatch.setattr(stability_evaluator, "compare", lambda _path: passed)
    assert stability_evaluator.main(["compare", "--require-quality-passed"]) == 0
    passed_output = json.loads(capsys.readouterr().out)
    assert passed_output["status"] == "stability_preliminarily_passed"
    assert passed_output["threshold_failures"] == []
    assert passed_output["report_path"] == passed["summary"]

    failed = {
        "status": "stability_failed",
        "summary": "/synthetic/failed-summary.json",
        "threshold_failures": ["false_positives_increased_over_10_percent"],
    }
    monkeypatch.setattr(stability_evaluator, "compare", lambda _path: failed)
    assert stability_evaluator.main(["compare", "--require-quality-passed"]) == 1
    failed_output = json.loads(capsys.readouterr().out)
    assert failed_output["status"] == "stability_failed"
    assert failed_output["threshold_failures"] == failed["threshold_failures"]
    assert failed_output["report_path"] == failed["summary"]


def test_prepare_rejects_strict_quality_flag_before_running(monkeypatch, capsys):
    def unexpected_prepare(_path):
        pytest.fail("prepare must not run with --require-quality-passed")

    monkeypatch.setattr(stability_evaluator, "prepare", unexpected_prepare)
    with pytest.raises(SystemExit) as error:
        stability_evaluator.main(["prepare", "--require-quality-passed"])

    assert error.value.code == 2
    assert "only valid with compare" in capsys.readouterr().err


def test_compare_exception_returns_structured_nonzero_result(monkeypatch, capsys):
    def fail_compare(_path):
        raise ValueError("synthetic identity mismatch")

    monkeypatch.setattr(stability_evaluator, "compare", fail_compare)

    exit_code = stability_evaluator.main(["compare", "--require-quality-passed"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["status"] == "error"
    assert output["error_type"] == "ValueError"
    assert output["error"] == "synthetic identity mismatch"
    assert output["threshold_failures"] is None
    assert output["report_path"] is None
