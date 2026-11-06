import pytest

from src.training.e03_a7_staged import (
    StageAuthorizationError,
    create_start_ticket,
    validate_stage_transition,
    verify_start_ticket,
    write_continuation_receipt,
    verify_continuation_receipt,
)


def test_a7_start_ticket_binds_run_arm_and_frozen_artifacts(tmp_path):
    path = tmp_path / "start.json"
    create_start_ticket(
        path,
        run_id="pair-P",
        arm="P",
        config_sha256="config",
        code_sha256="code",
        data_sha256="data",
        init_sha256="init",
        dry_run_manifest_sha256="dry",
    )
    expected = {
        "run_id": "pair-P", "arm": "P", "to_step": 512,
        "config_sha256": "config", "code_sha256": "code", "data_sha256": "data",
        "init_sha256": "init", "dry_run_manifest_sha256": "dry",
    }
    assert verify_start_ticket(path, expected=expected)["single_use"] is True
    with pytest.raises(StageAuthorizationError, match="arm"):
        verify_start_ticket(path, expected={**expected, "arm": "EP"})


def test_a7_continuation_requires_next_frozen_node_and_user_text(tmp_path):
    validate_stage_transition(512, 1024)
    with pytest.raises(ValueError, match="invalid A7 staged transition"):
        validate_stage_transition(512, 2048)
    path = tmp_path / "approval.json"
    receipt = write_continuation_receipt(
        path,
        run_id="pair-P",
        arm="P",
        from_step=512,
        to_step=1024,
        checkpoint_manifest_sha256="checkpoint",
        evaluation_manifest_sha256="evaluation",
        config_sha256="config",
        code_sha256="code",
        user_approval_text="继续 P 至 1024。",
    )
    assert receipt["status"] == "approved"
    assert verify_continuation_receipt(path, expected={"run_id": "pair-P", "arm": "P", "from_step": 512, "to_step": 1024})["single_use"]
