from __future__ import annotations

import copy

import pytest
import yaml

from src.training.e03_a5_contract import (
    RunAuthorizationError,
    validate_a5_config,
    validate_paired_group_contract,
    verify_run_ticket,
)


def _config():
    with open("configs/experiments/e03_a5.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _paired_groups():
    common = {
        "source_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "audit_manifest_sha256": "c" * 64,
        "supervision_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "base_weight_sha256": "f" * 64,
        "initial_adapter_weight_sha256": "1" * 64,
        "initial_adapter_config_sha256": "2" * 64,
        "train_schedule_sha256": "3" * 64,
        "sample_ids_sha256": "4" * 64,
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "scheduler": "constant",
        "weight_decay": 0.0,
        "optimizer_steps": 128,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "seed": 42,
        "protocol_sha256": "5" * 64,
    }
    control = {**common, "group": "control_sft", "bce_lambda": 0.0}
    candidate = {**common, "group": "candidate_sft_bce", "bce_lambda": 0.1}
    return control, candidate


def test_a5_config_is_versioned_and_does_not_authorize_gpu_execution():
    result = validate_a5_config(_config())
    assert result["execution_authorized"] is False
    assert result["run_ticket_required"] is True
    config = _config()
    config["authorization"]["execution_enabled"] = True
    with pytest.raises(ValueError, match="cannot authorize GPU"):
        validate_a5_config(config)


def test_run_ticket_binds_stage_run_config_audit_and_budget():
    ticket = {
        "version": "E03-A5-run-ticket-v1",
        "approved": True,
        "approved_by": "user",
        "experiment_id": "E03-A5-BCE-AUX",
        "stage": "feature_extraction",
        "scope": "a5_frozen_prompt_feature_extraction",
        "run_id": "A5-FEATURES-01",
        "config_sha256": "a" * 64,
        "audit_manifest_sha256": "b" * 64,
        "budget": {"max_seconds": 1800, "max_sample_count": 16192},
    }
    facts = verify_run_ticket(
        ticket,
        stage="feature_extraction",
        run_id="A5-FEATURES-01",
        config_sha256="a" * 64,
        audit_manifest_sha256="b" * 64,
        max_sample_count=16192,
    )
    assert facts["status"] == "run_ticket_validated"
    with pytest.raises(RunAuthorizationError, match="sample count"):
        verify_run_ticket(
            ticket,
            stage="feature_extraction",
            run_id="A5-FEATURES-01",
            config_sha256="a" * 64,
            audit_manifest_sha256="b" * 64,
            max_sample_count=16193,
        )
    with pytest.raises(RunAuthorizationError, match="separate approved"):
        verify_run_ticket(None, stage="feature_extraction", run_id="x", config_sha256="a" * 64)


def test_run_ticket_rejects_stage_or_hash_replay():
    ticket = {
        "version": "E03-A5-run-ticket-v1",
        "approved": True,
        "approved_by": "user",
        "experiment_id": "E03-A5-BCE-AUX",
        "stage": "gpu_training",
        "scope": "a5_one_paired_training_group",
        "run_id": "A5-TRAIN-CONTROL-01",
        "config_sha256": "a" * 64,
        "budget": {"max_seconds": 1800, "max_optimizer_steps": 128},
    }
    with pytest.raises(RunAuthorizationError, match="stage or scope"):
        verify_run_ticket(
            ticket, stage="feature_extraction", run_id="A5-TRAIN-CONTROL-01", config_sha256="a" * 64
        )
    with pytest.raises(RunAuthorizationError, match="config SHA"):
        verify_run_ticket(
            ticket, stage="gpu_training", run_id="A5-TRAIN-CONTROL-01", config_sha256="f" * 64,
            max_optimizer_steps=128,
        )


def test_control_and_candidate_must_share_everything_except_bce_lambda():
    control, candidate = _paired_groups()
    assert validate_paired_group_contract(control, candidate)["status"] == "paired_contract_passed"
    mismatched = copy.deepcopy(candidate)
    mismatched["train_schedule_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="train_schedule_sha256"):
        validate_paired_group_contract(control, mismatched)


def test_paired_contract_rejects_nonzero_control_or_zero_candidate():
    control, candidate = _paired_groups()
    with pytest.raises(ValueError, match="Control lambda"):
        validate_paired_group_contract({**control, "bce_lambda": 0.01}, candidate)
    with pytest.raises(ValueError, match="Candidate lambda"):
        validate_paired_group_contract(control, {**candidate, "bce_lambda": 0.0})
