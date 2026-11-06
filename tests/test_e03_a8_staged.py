from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.training.e03_a8_contract import (
    conditional_quality_decision,
    validate_a8_training_config,
    validate_conditional_authorization,
    validate_stage_transition,
    write_conditional_authorization,
)
from src.training.e03_a8_staged import _backward_scaled_completion_loss

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm,path", [
    ("P", ROOT / "configs/training/e03_a8_p_staged.yaml"),
    ("EP", ROOT / "configs/training/e03_a8_ep_staged.yaml"),
])
def test_a8_configs_keep_fixed_training_contract(arm: str, path: Path) -> None:
    validate_a8_training_config(yaml.safe_load(path.read_text(encoding="utf-8")), arm)


@pytest.mark.parametrize("start,end", [(None, 512), (512, 1024), (1024, 2048)])
def test_allowed_a8_stage_transitions(start: int | None, end: int) -> None:
    validate_stage_transition(start, end)


def test_training_microbatch_loss_uses_the_frozen_accumulation_factor() -> None:
    class FakeLoss:
        divided_by: int | None = None
        backward_called = False

        def __truediv__(self, divisor: int) -> "FakeLoss":
            self.divided_by = divisor
            return self

        def backward(self) -> None:
            self.backward_called = True

    loss = FakeLoss()
    _backward_scaled_completion_loss(loss)
    assert loss.divided_by == 4
    assert loss.backward_called is True


@pytest.mark.parametrize("start,end", [(2048, 4048), (512, 2048), (None, 1024)])
def test_invalid_or_unapproved_a8_stage_transitions_are_rejected(start: int | None, end: int) -> None:
    with pytest.raises(ValueError):
        validate_stage_transition(start, end)


def _summary(step: int, p_tp: int, ep_tp: int, *, parse: float = 1.0, gate: str = "passed") -> dict:
    return {
        "checkpoint_step": step,
        "engineering_gate": gate,
        "cohorts": {
            "pilot160": {"strict_parse_rate": parse, "profile_33_label": {"overall_33_label_tp": p_tp}},
            "dev224": {"strict_parse_rate": parse, "profile_33_label": {"overall_33_label_tp": ep_tp}},
        },
    }


def test_single_weak_node_is_allowed_under_loose_pre2048_gate() -> None:
    current = _summary(512, 0, 0)
    decision = conditional_quality_decision(current, None)
    assert decision["decision"] == "conditionally_authorized"
    assert decision["current_zero_tp_both_cohorts"] is True


def test_two_consecutive_zero_tp_nodes_stop_for_review() -> None:
    decision = conditional_quality_decision(_summary(1024, 0, 0), _summary(512, 0, 0))
    assert decision["decision"] == "awaiting_manual_review"
    assert decision["reason"] == "two_consecutive_nodes_with_zero_tp_on_both_cohorts"


def test_quality_gate_does_not_stop_for_low_f1_or_baseline_delta_if_tp_exists() -> None:
    decision = conditional_quality_decision(_summary(1024, 1, 2), _summary(512, 0, 0))
    assert decision["decision"] == "conditionally_authorized"


def test_incomplete_eval_and_parse_failure_stop() -> None:
    incomplete = {"engineering_gate": "passed", "cohorts": {"dev224": {}}}
    assert conditional_quality_decision(incomplete, None)["reason"] == "fixed_cohort_coverage_incomplete"
    decision = conditional_quality_decision(_summary(512, 1, 1, parse=0.97), None)
    assert decision["decision"] == "awaiting_manual_review"
    assert decision["reason"] == "strict_parse_rate_below_98_percent"


def test_conditional_receipt_binds_only_the_pre_authorized_next_stage(tmp_path: Path) -> None:
    path = tmp_path / "approvals" / "step-0512_to_1024.json"
    decision = conditional_quality_decision(_summary(512, 0, 0), None)
    receipt_sha = write_conditional_authorization(
        path,
        run_id="E03-A8-test-run",
        arm="P",
        from_step=512,
        to_step=1024,
        checkpoint_manifest_sha256="a" * 64,
        evaluation_summary_sha256="b" * 64,
        previous_evaluation_summary_sha256=None,
        config_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
        code_sha256="e" * 64,
        quality_decision=decision,
    )
    assert len(receipt_sha) == 64
    validated = validate_conditional_authorization(
        path,
        run_id="E03-A8-test-run",
        arm="P",
        from_step=512,
        to_step=1024,
        checkpoint_manifest_sha256="a" * 64,
        evaluation_summary_sha256="b" * 64,
        previous_evaluation_summary_sha256=None,
        config_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
        code_sha256="e" * 64,
    )
    assert validated["quality_decision"] == decision


def test_conditional_receipt_cannot_authorize_training_after_2048(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds the frozen staged path"):
        write_conditional_authorization(
            tmp_path / "after-2048.json",
            run_id="E03-A8-test-run",
            arm="EP",
            from_step=2048,
            to_step=4048,
            checkpoint_manifest_sha256="a" * 64,
            evaluation_summary_sha256="b" * 64,
            previous_evaluation_summary_sha256="c" * 64,
            config_sha256="d" * 64,
            data_manifest_sha256="e" * 64,
            code_sha256="f" * 64,
            quality_decision={"decision": "conditionally_authorized"},
        )
