"""Versioned A5 config and per-stage execution authorization contracts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


class RunAuthorizationError(PermissionError):
    """Raised before model/device access when a distinct Run ticket is absent."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_a5_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("version") != "E03-A5-v1" or config.get("experiment_id") != "E03-A5-BCE-AUX":
        raise ValueError("A5 config must retain the E03-A5-v1 identity")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get("execution_enabled") is not False:
        raise ValueError("A5 config cannot authorize GPU runs by itself")
    required_stages = {
        "gpu_feature_extraction": "requires_separate_run_ticket",
        "gpu_dry_step": "requires_separate_run_ticket",
        "gpu_training": "requires_separate_run_ticket",
        "gpu_candidate_generation": "requires_separate_run_ticket",
    }
    if any(authorization.get(name) != value for name, value in required_stages.items()):
        raise ValueError("every GPU A5 stage must require a separate Run ticket")
    identity = config.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("A5 config must pin source and model identities")
    source_commit = identity.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in source_commit.lower()
    ):
        raise ValueError("A5 identity.source_commit must be a pinned Git commit ID")
    for name in (
        "e01_b_manifest_sha256", "d03_manifest_sha256", "d042_manifest_sha256",
        "d043_manifest_sha256", "normalized_train_sha256", "sft_train_sha256",
        "official_train_schema_sha256", "official_output_schema_sha256", "p0_sha256",
        "tokenizer_signature_sha256", "chat_template_sha256", "base_weight_sha256",
        "initial_adapter_weight_identity_sha256", "initial_adapter_file_sha256",
        "initial_adapter_config_sha256",
    ):
        value = identity.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError(f"A5 identity.{name} must be a pinned SHA-256")
    target = config.get("target", {})
    if (
        target.get("label_order_version") != "official_train_schema_profile_order_v1"
        or target.get("target_policy") != "closed_world_gold_v1"
        or target.get("default_mask") != "all_ones_for_complete_three_dimension_gold"
        or target.get("missing_label_policy") != "masked_unknown"
        or target.get("allow_pos_weight") is not False
        or target.get("allow_replay") is not False
    ):
        raise ValueError("A5 closed-world labels and unweighted/no-replay policy changed")
    split = config.get("split", {})
    fractions = (
        split.get("fit_fraction"), split.get("calibration_fraction"),
        split.get("internal_holdout_fraction"),
    )
    if (
        split.get("version") != "e03_a5_conversation_group_stratified_v1"
        or split.get("seed") != 42
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in fractions)
        or not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("A5 conversation-group split must use the frozen v1 70/15/15 proportions")
    if config.get("probe", {}).get("feature_position") != "prompt_length_minus_one":
        raise ValueError("A5 probe must use the last Prompt token")
    proposal = config.get("training_proposal", {})
    if (
        proposal.get("control_lambda") != 0.0
        or not isinstance(proposal.get("candidate_lambda"), (int, float))
        or isinstance(proposal.get("candidate_lambda"), bool)
        or proposal.get("candidate_lambda") <= 0
        or proposal.get("resume_optimizer_state") is not False
        or proposal.get("source_adapter_is_read_only") is not True
    ):
        raise ValueError("A5 paired training proposal must isolate SFT Control and SFT+BCE Candidate")
    return {
        "status": "config_validation_passed",
        "experiment_id": "E03-A5-BCE-AUX",
        "target_policy": "closed_world_gold_v1",
        "execution_authorized": False,
        "run_ticket_required": True,
    }


def verify_run_ticket(
    ticket: Mapping[str, Any] | None,
    *,
    stage: str,
    run_id: str,
    config_sha256: str,
    audit_manifest_sha256: str | None = None,
    max_sample_count: int | None = None,
    max_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    """Validate an explicit per-Run permit before any model or GPU is loaded."""
    if not isinstance(ticket, Mapping):
        raise RunAuthorizationError(f"{stage} requires a separate approved E03-A5 Run ticket")
    if ticket.get("version") != "E03-A5-run-ticket-v1":
        raise RunAuthorizationError("Run ticket version must be E03-A5-run-ticket-v1")
    if ticket.get("approved") is not True or not isinstance(ticket.get("approved_by"), str) or not ticket["approved_by"].strip():
        raise RunAuthorizationError("Run ticket is not explicitly approved")
    expected_scope = {
        "feature_extraction": "a5_frozen_prompt_feature_extraction",
        "gpu_dry_step": "a5_single_group_gpu_dry_step",
        "gpu_training": "a5_one_paired_training_group",
        "candidate_generation": "a5_frozen_candidate_generation",
        "dev_224_probe": "a5_single_frozen_dev_224_diagnostic",
    }
    if stage not in expected_scope or ticket.get("stage") != stage or ticket.get("scope") != expected_scope[stage]:
        raise RunAuthorizationError("Run ticket stage or scope does not match the requested action")
    if ticket.get("experiment_id") != "E03-A5-BCE-AUX" or ticket.get("run_id") != run_id:
        raise RunAuthorizationError("Run ticket does not bind the requested A5 Run ID")
    if ticket.get("config_sha256") != config_sha256:
        raise RunAuthorizationError("Run ticket config SHA differs from the current frozen config")
    if audit_manifest_sha256 is not None and ticket.get("audit_manifest_sha256") != audit_manifest_sha256:
        raise RunAuthorizationError("Run ticket does not bind the passed A5.0 supervision audit")
    budget = ticket.get("budget")
    if not isinstance(budget, Mapping):
        raise RunAuthorizationError("Run ticket must contain an explicit resource budget")
    if max_sample_count is not None:
        authorized_count = budget.get("max_sample_count")
        if isinstance(authorized_count, bool) or not isinstance(authorized_count, int) or not 1 <= max_sample_count <= authorized_count:
            raise RunAuthorizationError("requested sample count exceeds the authorized Run budget")
    if max_optimizer_steps is not None:
        authorized_steps = budget.get("max_optimizer_steps")
        if isinstance(authorized_steps, bool) or not isinstance(authorized_steps, int) or not 1 <= max_optimizer_steps <= authorized_steps:
            raise RunAuthorizationError("requested optimizer steps exceed the authorized Run budget")
    max_seconds = budget.get("max_seconds")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or max_seconds <= 0:
        raise RunAuthorizationError("Run ticket must set a positive max_seconds budget")
    return {
        "status": "run_ticket_validated",
        "stage": stage,
        "scope": expected_scope[stage],
        "run_id": run_id,
        "max_seconds": float(max_seconds),
        "max_sample_count": budget.get("max_sample_count"),
        "max_optimizer_steps": budget.get("max_optimizer_steps"),
    }


def validate_paired_group_contract(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Require matched source/schedule/optimizer facts and the BCE lambda as sole difference."""
    if control.get("group") != "control_sft" or candidate.get("group") != "candidate_sft_bce":
        raise ValueError("A5 pair must contain control_sft and candidate_sft_bce groups")
    if control.get("bce_lambda") != 0.0:
        raise ValueError("A5 Control lambda must be exactly zero")
    candidate_lambda = candidate.get("bce_lambda")
    if isinstance(candidate_lambda, bool) or not isinstance(candidate_lambda, (int, float)) or candidate_lambda <= 0:
        raise ValueError("A5 Candidate lambda must be positive")
    compare_fields = (
        "source_commit", "config_sha256", "audit_manifest_sha256", "supervision_sha256",
        "split_manifest_sha256", "base_weight_sha256", "initial_adapter_weight_sha256",
        "initial_adapter_config_sha256", "train_schedule_sha256", "sample_ids_sha256",
        "optimizer", "learning_rate", "scheduler", "weight_decay", "optimizer_steps",
        "micro_batch_size", "gradient_accumulation_steps", "seed", "protocol_sha256",
    )
    differences = [field for field in compare_fields if control.get(field) != candidate.get(field)]
    if differences:
        raise ValueError(f"A5 paired groups differ outside BCE lambda/group identity: {differences}")
    return {
        "status": "paired_contract_passed",
        "same_train_schedule": True,
        "same_frozen_sources_and_initialization": True,
        "only_objective_difference": "bce_lambda",
        "control_lambda": 0.0,
        "candidate_lambda": float(candidate_lambda),
    }


def verify_probe_gate(
    probe_report: Mapping[str, Any],
    probe_manifest: Mapping[str, Any],
    dev_probe_report: Mapping[str, Any],
    dev_probe_manifest: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    config_sha256: str,
    audit_manifest_sha256: str,
    probe_report_sha256: str,
    probe_manifest_sha256: str,
    dev_probe_report_sha256: str,
    dev_probe_manifest_sha256: str,
    frozen_ids_sha256: str,
) -> dict[str, Any]:
    """Require Train probe and one frozen, target-free Dev diagnostic before joint training."""
    if probe_manifest.get("status") != "probe_metrics_ready_for_gate_review":
        raise RunAuthorizationError("A5.1 probe Run is not complete and ready for gate review")
    if probe_report.get("probe_decision") != "requires_predeclared_gate_and_manual_review":
        raise RunAuthorizationError("A5.1 probe report has an unexpected decision state")
    if probe_report.get("config_sha256") not in {None, config_sha256}:
        raise RunAuthorizationError("A5.1 probe report uses a different config")
    if probe_report.get("a5_audit_manifest_sha256") != audit_manifest_sha256:
        raise RunAuthorizationError("A5.1 probe Run does not bind the passed A5.0 audit")
    if (
        dev_probe_manifest.get("status") != "dev_224_probe_scored"
        or dev_probe_report.get("status") != "dev_224_probe_scored"
        or not isinstance(dev_probe_manifest.get("run_id"), str)
        or dev_probe_report.get("run_id") != dev_probe_manifest.get("run_id")
        or dev_probe_manifest.get("probe_run_id") != probe_manifest.get("run_id")
        or dev_probe_report.get("probe_run_id") != probe_manifest.get("run_id")
        or dev_probe_manifest.get("probe_run_manifest_sha256") != probe_manifest_sha256
        or dev_probe_manifest.get("probe_report_sha256") != probe_report_sha256
        or dev_probe_report.get("probe_run_manifest_sha256") != probe_manifest_sha256
        or dev_probe_report.get("probe_report_sha256") != probe_report_sha256
        or dev_probe_report.get("config_sha256") != config_sha256
        or dev_probe_report.get("a5_audit_manifest_sha256") != audit_manifest_sha256
        or dev_probe_manifest.get("config_sha256") != config_sha256
        or dev_probe_manifest.get("a5_audit_manifest_sha256") != audit_manifest_sha256
        or dev_probe_report.get("frozen_ids_sha256") != frozen_ids_sha256
        or dev_probe_manifest.get("frozen_ids_sha256") != frozen_ids_sha256
        or dev_probe_report.get("sample_count") != 224
        or dev_probe_report.get("frozen_point_count") != 224
        or dev_probe_report.get("frozen_conversation_count") != 101
        or dev_probe_report.get("gold_targets_read_after_feature_extraction") is not True
        or dev_probe_report.get("thresholds_selected_on_dev") is not False
        or dev_probe_report.get("threshold_source") != "frozen Train-Calibration values from the bound CPU probe Run"
        or dev_probe_report.get("gold_targets_passed_to_model") is not False
        or dev_probe_report.get("response_text_generated_or_saved") is not False
        or dev_probe_report.get("full_generation_performed") is not False
        or dev_probe_manifest.get("feature_contract", {}).get("input_source") != "frozen target-free id/history via prepare_sample_messages"
        or dev_probe_manifest.get("feature_contract", {}).get("position") != "prompt_length_minus_one"
        or dev_probe_manifest.get("feature_contract", {}).get("row_count") != 224
        or dev_probe_manifest.get("feature_contract", {}).get("gold_targets_passed_to_model") is not False
        or dev_probe_manifest.get("feature_contract", {}).get("full_generation_performed") is not False
        or not isinstance(dev_probe_manifest.get("run_identity_sha256"), str)
        or dev_probe_manifest.get("model_identity", {}).get("base_weights_unchanged") is not True
        or dev_probe_manifest.get("model_identity", {}).get("source_adapter_unchanged") is not True
    ):
        raise RunAuthorizationError("A5.1 frozen Dev 224 diagnostic is incomplete or violates its no-leakage contract")
    if (
        gate.get("status") != "probe_gate_frozen"
        or gate.get("decision") != "probe_supported"
        or gate.get("approved") is not True
        or not isinstance(gate.get("approved_by"), str)
        or not gate["approved_by"].strip()
        or not isinstance(gate.get("rationale"), str)
        or not gate["rationale"].strip()
    ):
        raise RunAuthorizationError("A5.2 requires an explicitly reviewed probe_supported gate")
    expected = {
        "experiment_id": "E03-A5-BCE-AUX",
        "probe_run_id": probe_manifest.get("run_id"),
        "config_sha256": config_sha256,
        "audit_manifest_sha256": audit_manifest_sha256,
        "probe_report_sha256": probe_report_sha256,
        "probe_manifest_sha256": probe_manifest_sha256,
        "dev_run_id": dev_probe_manifest.get("run_id"),
        "dev_probe_report_sha256": dev_probe_report_sha256,
        "dev_probe_manifest_sha256": dev_probe_manifest_sha256,
    }
    mismatches = [name for name, value in expected.items() if gate.get(name) != value]
    if mismatches:
        raise RunAuthorizationError(f"A5.1 probe gate has mismatched evidence bindings: {mismatches}")
    return {
        "status": "probe_gate_passed",
        "decision": "probe_supported",
        "approved_by": gate["approved_by"],
        "rationale": gate["rationale"],
        "dev_run_id": dev_probe_manifest.get("run_id"),
        "dev_probe_report_sha256": dev_probe_report_sha256,
        "dev_probe_manifest_sha256": dev_probe_manifest_sha256,
        "gate_sha256": canonical_sha256(gate),
    }


def verify_lambda_freeze(
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    config_sha256: str,
    audit_manifest_sha256: str,
    pair_id: str,
    pair_manifest_sha256: str,
    probe_gate_manifest_sha256: str,
    control_dry_step_manifest_sha256: str,
    candidate_dry_step_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate a reviewed lambda freeze without mutating the A5.0 config identity."""
    expected_lambda = float(config["training_proposal"]["candidate_lambda"])
    if (
        freeze.get("status") != "candidate_lambda_frozen"
        or freeze.get("approved") is not True
        or not isinstance(freeze.get("approved_by"), str)
        or not freeze["approved_by"].strip()
        or not isinstance(freeze.get("rationale"), str)
        or not freeze["rationale"].strip()
        or freeze.get("both_dry_steps_passed_without_optimizer_updates") is not True
    ):
        raise RunAuthorizationError("A5.2 training requires an explicit reviewed Candidate lambda freeze")
    expected = {
        "experiment_id": "E03-A5-BCE-AUX",
        "config_sha256": config_sha256,
        "audit_manifest_sha256": audit_manifest_sha256,
        "pair_id": pair_id,
        "pair_manifest_sha256": pair_manifest_sha256,
        "probe_gate_manifest_sha256": probe_gate_manifest_sha256,
        "control_dry_step_manifest_sha256": control_dry_step_manifest_sha256,
        "candidate_dry_step_manifest_sha256": candidate_dry_step_manifest_sha256,
    }
    mismatches = [name for name, value in expected.items() if freeze.get(name) != value]
    if mismatches:
        raise RunAuthorizationError(f"Candidate lambda freeze has mismatched evidence bindings: {mismatches}")
    value = freeze.get("candidate_lambda")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected_lambda or value <= 0:
        raise RunAuthorizationError("frozen Candidate lambda differs from the versioned A5 proposal")
    return {
        "status": "candidate_lambda_freeze_passed",
        "candidate_lambda": float(value),
        "approved_by": freeze["approved_by"],
        "rationale": freeze["rationale"],
        "freeze_sha256": canonical_sha256(freeze),
    }
