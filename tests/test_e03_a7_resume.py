import random

import numpy as np
import pytest
import torch
from torch import nn

from scripts.e03_a7_train import _validate_reload_compatibility_receipt, _verify_training_ledger_prefix
from src.training.e03_a7_staged import (
    capture_rng_state,
    load_training_state,
    restore_rng_state,
    save_training_state,
)


def test_a7_repair_ledger_verification_requires_exact_frozen_schedule_prefix():
    schedule = {
        "steps": [
            {"optimizer_step": 1, "epoch": 1, "sample_ids": ["a", "b", "c", "d"]},
            {"optimizer_step": 2, "epoch": 1, "sample_ids": ["e", "f", "g", "h"]},
        ]
    }
    train_rows = [
        {"optimizer_step": row["optimizer_step"], "epoch": row["epoch"], "sample_ids": list(row["sample_ids"])}
        for row in schedule["steps"]
    ]
    order_rows = [dict(row) for row in train_rows]
    assert _verify_training_ledger_prefix(schedule, train_rows, order_rows, expected_step=2) == list("abcdefgh")
    order_rows[1]["sample_ids"] = ["x", "f", "g", "h"]
    with pytest.raises(ValueError, match="differs from the frozen schedule"):
        _verify_training_ledger_prefix(schedule, train_rows, order_rows, expected_step=2)


def test_a7_reload_compatibility_is_bound_to_exact_old_and_new_code_hashes():
    preflight = {"code_sha256": "training-code"}
    receipt = {
        "status": "reload_compatibility_repaired",
        "scope": "inference_base_dtype_reload_only",
        "training_code_sha256": "training-code",
        "compatible_code_sha256": "evaluation-code",
        "config_sha256": "config",
        "optimizer_steps_added": 0,
    }
    assert _validate_reload_compatibility_receipt(
        preflight, receipt, current_code_sha="evaluation-code", config_sha="config"
    ) is None
    with pytest.raises(ValueError, match="does not authorize"):
        _validate_reload_compatibility_receipt(
            preflight, receipt, current_code_sha="different-code", config_sha="config"
        )


def _step(model, optimizer, scheduler, step_index):
    optimizer.zero_grad(set_to_none=True)
    values = torch.randn(4, 3) + random.random() + float(np.random.random())
    target = torch.randn(4, 2)
    loss = ((model(values) - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def _new(seed):
    torch.manual_seed(seed)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return model, optimizer, scheduler


def test_training_state_resume_matches_uninterrupted_synthetic_run(tmp_path):
    identity = {"config_sha256": "config", "code_sha256": "code", "data_sha256": "data"}
    torch.manual_seed(71)
    random.seed(71)
    np.random.seed(71)
    full, full_optimizer, full_scheduler = _new(71)
    full_losses = [_step(full, full_optimizer, full_scheduler, step) for step in range(4)]

    torch.manual_seed(71)
    random.seed(71)
    np.random.seed(71)
    split, split_optimizer, split_scheduler = _new(71)
    losses_prefix = [_step(split, split_optimizer, split_scheduler, step) for step in range(2)]
    state_path = tmp_path / "training_state.pt"
    consumed = [f"fit-{i}" for i in range(8)]
    save_training_state(
        state_path,
        identity=identity,
        arm="P",
        run_id="synthetic-run",
        optimizer_step=2,
        schedule_sha256="schedule",
        consumed_ids=consumed,
        model_state=split.state_dict(),
        head_state={"empty": torch.tensor([1.0])},
        optimizer_state=split_optimizer.state_dict(),
        scheduler_state=split_scheduler.state_dict(),
        rng_state=capture_rng_state(),
    )
    state = load_training_state(
        state_path,
        expected_identity=identity,
        expected_arm="P",
        expected_run_id="synthetic-run",
        expected_step=2,
        expected_schedule_sha256="schedule",
        expected_consumed_ids=consumed,
    )
    resumed, resumed_optimizer, resumed_scheduler = _new(901)
    resumed.load_state_dict(state["model_state"])
    resumed_optimizer.load_state_dict(state["optimizer_state"])
    resumed_scheduler.load_state_dict(state["scheduler_state"])
    restore_rng_state(state["rng_state"])
    losses_tail = [_step(resumed, resumed_optimizer, resumed_scheduler, step) for step in range(2, 4)]

    assert losses_prefix + losses_tail == full_losses
    assert all(torch.equal(full.state_dict()[name], resumed.state_dict()[name]) for name in full.state_dict())
    assert full_optimizer.state_dict()["param_groups"] == resumed_optimizer.state_dict()["param_groups"]
    for left, right in zip(full_optimizer.state_dict()["state"].values(), resumed_optimizer.state_dict()["state"].values()):
        for key in left:
            if isinstance(left[key], torch.Tensor):
                assert torch.equal(left[key], right[key])
            else:
                assert left[key] == right[key]


def test_training_state_rejects_arm_or_schedule_mismatch(tmp_path):
    model, optimizer, scheduler = _new(11)
    path = tmp_path / "state.pt"
    save_training_state(
        path,
        identity={"v": 1},
        arm="P",
        run_id="run",
        optimizer_step=1,
        schedule_sha256="schedule",
        consumed_ids=["a", "b", "c", "d"],
        model_state=model.state_dict(),
        head_state={"head": torch.tensor([1.0])},
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
    )
    with pytest.raises(Exception, match="Arm, Step, schedule"):
        load_training_state(
            path,
            expected_identity={"v": 1},
            expected_arm="EP",
            expected_run_id="run",
            expected_step=1,
            expected_schedule_sha256="schedule",
            expected_consumed_ids=["a", "b", "c", "d"],
        )
