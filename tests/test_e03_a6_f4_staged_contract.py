"""CPU contracts for staged F4 Full training and authorization."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.training.e03_a6_f4_staged_contract import (
    sha256_file,
    validate_eval_config,
    validate_stage_authorization,
    validate_stage_transition,
    validate_staged_full_config,
    write_stage_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    train = yaml.safe_load((ROOT / "configs/training/e03_a6_f4_staged_full.yaml").read_text())
    evaluate = yaml.safe_load((ROOT / "configs/evaluation/e03_a6_f4_staged_fast.yaml").read_text())
    return train, evaluate


def test_staged_f4_configs_keep_the_full_train_and_fast_only_contract():
    train, evaluate = _configs()
    validate_staged_full_config(train)
    validate_eval_config(evaluate)
    assert train["training"]["max_optimizer_steps"] == 4048
    assert train["data"]["expected_train_rows"] == 16192
    assert train["evaluation"]["sft_dev_loss_rows"] == 0


def test_staged_f4_contract_rejects_a_changed_step_budget_or_cohort():
    train, evaluate = _configs()
    changed_train = copy.deepcopy(train)
    changed_train["training"]["max_optimizer_steps"] = 4096
    with pytest.raises(ValueError, match="fixed training plan"):
        validate_staged_full_config(changed_train)
    changed_budget = copy.deepcopy(train)
    changed_budget["quantization"]["max_cuda_memory_fraction"] = 0.99
    with pytest.raises(ValueError, match="92% CUDA budget"):
        validate_staged_full_config(changed_budget)
    changed_eval = copy.deepcopy(evaluate)
    changed_eval["cohorts"]["dev224"]["expected_count"] = 223
    with pytest.raises(ValueError, match="cohort identity"):
        validate_eval_config(changed_eval)


@pytest.mark.parametrize(
    ("from_step", "to_step"),
    [(None, 512), (512, 1024), (1024, 2048), (2048, 4048)],
)
def test_only_the_four_frozen_stage_transitions_are_valid(from_step, to_step):
    validate_stage_transition(from_step, to_step)


def test_invalid_stage_transition_is_rejected():
    with pytest.raises(ValueError, match="invalid F4 staged transition"):
        validate_stage_transition(512, 2048)


def test_stage_authorization_is_bound_to_run_node_and_checkpoint_hash(tmp_path: Path):
    path = tmp_path / "approval.json"
    checkpoint_sha = "a" * 64
    record = write_stage_authorization(
        path, run_id="E03-A6-F4-STAGED-FULL-TEST", from_step=512,
        to_step=1024, checkpoint_manifest_sha256=checkpoint_sha,
    )
    assert record["node_metrics_individually_reviewed"] is False
    assert validate_stage_authorization(
        path, run_id="E03-A6-F4-STAGED-FULL-TEST", from_step=512,
        to_step=1024, checkpoint_manifest_sha256=checkpoint_sha,
    ) == record
    with pytest.raises(ValueError, match="does not match"):
        validate_stage_authorization(
            path, run_id="E03-A6-F4-STAGED-FULL-TEST", from_step=512,
            to_step=1024, checkpoint_manifest_sha256="b" * 64,
        )
    with pytest.raises(FileExistsError):
        write_stage_authorization(
            path, run_id="E03-A6-F4-STAGED-FULL-TEST", from_step=512,
            to_step=1024, checkpoint_manifest_sha256=checkpoint_sha,
        )


def test_continuation_authorization_reads_the_registered_eval_output_root(tmp_path: Path, monkeypatch):
    from scripts import e03_a6_f4_staged_authorize as authorize_script

    run_id = "E03-A6-F4-STAGED-FULL-TEST"
    train_root = tmp_path / "training"
    eval_root = tmp_path / "evaluation"
    run_root = train_root / run_id
    eval_node = eval_root / run_id / "step-0512"
    (run_root / "trainer" / "checkpoint-512").mkdir(parents=True)
    (eval_node).mkdir(parents=True)
    checkpoint_manifest_path = run_root / "trainer" / "checkpoint-512" / "checkpoint_manifest.json"
    checkpoint_manifest_path.write_text(json.dumps({"run_id": run_id, "optimizer_step": 512}), encoding="utf-8")
    checkpoint_sha = sha256_file(checkpoint_manifest_path)
    predictions_path = eval_node / "predictions.jsonl"
    predictions_path.write_text("{}\n", encoding="utf-8")
    prediction_sha = sha256_file(predictions_path)
    identity = {"run_id": run_id, "checkpoint_step": 512, "checkpoint_manifest_sha256": checkpoint_sha}
    evaluation_path = eval_node / "evaluation.json"
    evaluation_path.write_text(json.dumps({
        "run_id": run_id,
        "checkpoint_step": 512,
        "engineering_gate": "passed",
        "progression_authorization_status": "awaiting_manual_review",
        "evaluation_identity": identity,
        "generation_protocol": {"prediction_sha256": prediction_sha},
    }), encoding="utf-8")
    (eval_node / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "status": "complete",
        "evaluation_identity": identity,
        "predictions_sha256": prediction_sha,
        "engineering_gate": "passed",
        "summary_sha256": sha256_file(evaluation_path),
    }), encoding="utf-8")
    (run_root / "manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "config_sha256": "frozen-config",
        "status": "awaiting_manual_review",
        "optimizer_steps": 512,
        "nodes": {"512": {
            "training_status": "passed",
            "checkpoint_manifest_sha256": checkpoint_sha,
            "evaluation_engineering_gate": "passed",
            "evaluation_summary_sha256": sha256_file(evaluation_path),
            "evaluation_summary_path": str(evaluation_path),
        }},
    }), encoding="utf-8")

    monkeypatch.setattr(authorize_script, "ROOT", tmp_path)
    monkeypatch.setattr(authorize_script, "_load_config", lambda _path: ({"data": {"output_root": str(train_root)}}, tmp_path / "config.yaml"))
    monkeypatch.setattr(authorize_script, "config_sha256", lambda _config: "frozen-config")
    monkeypatch.setattr(authorize_script, "_load_eval_config", lambda _path: ({"output_root": str(eval_root)}, tmp_path / "eval.yaml"))
    monkeypatch.setattr(authorize_script, "_render_node_report", lambda _summary: "authorized stage report\n")

    receipt = authorize_script.record_continuation_authorization(
        tmp_path / "config.yaml", tmp_path / "eval.yaml", run_id, 512, 1024,
        allow_staged_training=True,
    )

    evaluation = json.loads((eval_node / "evaluation.json").read_text(encoding="utf-8"))
    evaluation_manifest = json.loads((eval_node / "run_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert receipt == run_root / "approvals" / "step-0512.json"
    assert evaluation["progression_authorization_status"] == "preauthorized_by_explicit_user_instruction"
    assert evaluation_manifest["authorization_file"] == str(receipt.relative_to(tmp_path))
    assert run_manifest["nodes"]["512"]["node_metrics_individually_reviewed"] is False
    assert (eval_node / "report.md").read_text(encoding="utf-8") == "authorized stage report\n"
