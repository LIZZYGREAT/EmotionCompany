from __future__ import annotations

import copy

import pytest
import yaml

from src.training.e03_a5_contract import (
    RunAuthorizationError,
    validate_a5_config,
    validate_paired_group_contract,
    verify_lambda_freeze,
    verify_probe_gate,
    verify_run_ticket,
)
from scripts.e03_a5_train import _validate_id, _verify_ticket_bindings


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


def test_probe_gate_requires_a_reviewed_report_bound_supported_decision():
    report = {
        "status": "probe_metrics_ready_for_gate_review",
        "probe_decision": "requires_predeclared_gate_and_manual_review",
        "a5_audit_manifest_sha256": "b" * 64,
    }
    probe_manifest = {"status": "probe_metrics_ready_for_gate_review", "run_id": "A5-PROBE-01"}
    dev_report = {
        "status": "dev_224_probe_scored",
        "run_id": "A5-DEV-01",
        "probe_run_id": "A5-PROBE-01",
        "probe_run_manifest_sha256": "d" * 64,
        "probe_report_sha256": "c" * 64,
        "config_sha256": "a" * 64,
        "a5_audit_manifest_sha256": "b" * 64,
        "frozen_ids_sha256": "e" * 64,
        "sample_count": 224,
        "frozen_point_count": 224,
        "frozen_conversation_count": 101,
        "gold_targets_read_after_feature_extraction": True,
        "thresholds_selected_on_dev": False,
        "threshold_source": "frozen Train-Calibration values from the bound CPU probe Run",
        "gold_targets_passed_to_model": False,
        "response_text_generated_or_saved": False,
        "full_generation_performed": False,
    }
    dev_manifest = {
        "status": "dev_224_probe_scored",
        "run_id": "A5-DEV-01",
        "probe_run_id": "A5-PROBE-01",
        "probe_run_manifest_sha256": "d" * 64,
        "probe_report_sha256": "c" * 64,
        "config_sha256": "a" * 64,
        "a5_audit_manifest_sha256": "b" * 64,
        "frozen_ids_sha256": "e" * 64,
        "run_identity_sha256": "9" * 64,
        "feature_contract": {
            "input_source": "frozen target-free id/history via prepare_sample_messages",
            "position": "prompt_length_minus_one",
            "row_count": 224,
            "gold_targets_passed_to_model": False,
            "full_generation_performed": False,
        },
        "model_identity": {"base_weights_unchanged": True, "source_adapter_unchanged": True},
    }
    gate = {
        "status": "probe_gate_frozen",
        "decision": "probe_supported",
        "approved": True,
        "approved_by": "reviewer",
        "rationale": "holdout signal cleared the predeclared criteria",
        "experiment_id": "E03-A5-BCE-AUX",
        "probe_run_id": "A5-PROBE-01",
        "config_sha256": "a" * 64,
        "audit_manifest_sha256": "b" * 64,
        "probe_report_sha256": "c" * 64,
        "probe_manifest_sha256": "d" * 64,
        "dev_run_id": "A5-DEV-01",
        "dev_probe_report_sha256": "f" * 64,
        "dev_probe_manifest_sha256": "0" * 64,
    }
    facts = verify_probe_gate(
        report, probe_manifest, dev_report, dev_manifest, gate,
        config_sha256="a" * 64,
        audit_manifest_sha256="b" * 64,
        probe_report_sha256="c" * 64,
        probe_manifest_sha256="d" * 64,
        dev_probe_report_sha256="f" * 64,
        dev_probe_manifest_sha256="0" * 64,
        frozen_ids_sha256="e" * 64,
    )
    assert facts["decision"] == "probe_supported"
    with pytest.raises(RunAuthorizationError, match="Dev 224"):
        verify_probe_gate(
            report, probe_manifest, {**dev_report, "thresholds_selected_on_dev": True}, dev_manifest, gate,
            config_sha256="a" * 64,
            audit_manifest_sha256="b" * 64,
            probe_report_sha256="c" * 64,
            probe_manifest_sha256="d" * 64,
            dev_probe_report_sha256="f" * 64,
            dev_probe_manifest_sha256="0" * 64,
            frozen_ids_sha256="e" * 64,
        )
    with pytest.raises(RunAuthorizationError, match="Dev 224"):
        verify_probe_gate(
            report, probe_manifest, {**dev_report, "run_id": "A5-DEV-OTHER"}, dev_manifest, gate,
            config_sha256="a" * 64,
            audit_manifest_sha256="b" * 64,
            probe_report_sha256="c" * 64,
            probe_manifest_sha256="d" * 64,
            dev_probe_report_sha256="f" * 64,
            dev_probe_manifest_sha256="0" * 64,
            frozen_ids_sha256="e" * 64,
        )
    with pytest.raises(RunAuthorizationError, match="probe_supported"):
        verify_probe_gate(
            report, probe_manifest, dev_report, dev_manifest, {**gate, "decision": "probe_indeterminate"},
            config_sha256="a" * 64,
            audit_manifest_sha256="b" * 64,
            probe_report_sha256="c" * 64,
            probe_manifest_sha256="d" * 64,
            dev_probe_report_sha256="f" * 64,
            dev_probe_manifest_sha256="0" * 64,
            frozen_ids_sha256="e" * 64,
        )


def test_lambda_freeze_binds_both_dry_steps_without_editing_audit_config():
    config = _config()
    freeze = {
        "status": "candidate_lambda_frozen",
        "approved": True,
        "approved_by": "reviewer",
        "rationale": "both maximum-length dry steps passed with expected gradients",
        "experiment_id": "E03-A5-BCE-AUX",
        "pair_id": "A5-PAIR-01",
        "candidate_lambda": 0.1,
        "config_sha256": "a" * 64,
        "audit_manifest_sha256": "b" * 64,
        "pair_manifest_sha256": "c" * 64,
        "probe_gate_manifest_sha256": "d" * 64,
        "control_dry_step_manifest_sha256": "e" * 64,
        "candidate_dry_step_manifest_sha256": "f" * 64,
        "both_dry_steps_passed_without_optimizer_updates": True,
    }
    facts = verify_lambda_freeze(
        config, freeze,
        config_sha256="a" * 64,
        audit_manifest_sha256="b" * 64,
        pair_id="A5-PAIR-01",
        pair_manifest_sha256="c" * 64,
        probe_gate_manifest_sha256="d" * 64,
        control_dry_step_manifest_sha256="e" * 64,
        candidate_dry_step_manifest_sha256="f" * 64,
    )
    assert facts["candidate_lambda"] == 0.1
    with pytest.raises(RunAuthorizationError, match="mismatched evidence"):
        verify_lambda_freeze(
            config, freeze,
            config_sha256="a" * 64,
            audit_manifest_sha256="b" * 64,
            pair_id="A5-PAIR-01",
            pair_manifest_sha256="c" * 64,
            probe_gate_manifest_sha256="d" * 64,
            control_dry_step_manifest_sha256="e" * 64,
            candidate_dry_step_manifest_sha256="0" * 64,
        )


def test_gpu_ticket_binding_is_group_pair_lambda_and_gate_specific():
    pair = {"pair_id": "A5-PAIR-01"}
    gate = {
        "gate_manifest_sha256": "a" * 64,
        "probe_run_manifest_sha256": "b" * 64,
    }
    ticket = {
        "pair_id": "A5-PAIR-01",
        "pair_manifest_sha256": "c" * 64,
        "probe_gate_manifest_sha256": "a" * 64,
        "probe_run_manifest_sha256": "b" * 64,
        "group": "candidate_sft_bce",
        "bce_lambda": 0.1,
        "dry_step_manifest_sha256": "d" * 64,
        "lambda_freeze_manifest_sha256": "e" * 64,
    }
    _verify_ticket_bindings(
        ticket, pair=pair, pair_sha256="c" * 64, gate_facts=gate,
        group="candidate_sft_bce", bce_lambda=0.1,
        dry_manifest_sha256="d" * 64, lambda_freeze_manifest_sha256="e" * 64,
    )
    with pytest.raises(RunAuthorizationError, match="does not bind"):
        _verify_ticket_bindings(
            ticket, pair=pair, pair_sha256="c" * 64, gate_facts=gate,
            group="control_sft", bce_lambda=0.0,
            dry_manifest_sha256="d" * 64, lambda_freeze_manifest_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="filesystem-safe"):
        _validate_id("../../outside", "run_id")
