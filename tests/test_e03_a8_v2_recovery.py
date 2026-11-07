from __future__ import annotations

import torch
from torch import nn

from src.training.e03_a8_v2_staged import restore_recovery_bundle, save_recovery_bundle


def _train_one_step(model, optimizer, scheduler, *, random_input=True):
    optimizer.zero_grad(set_to_none=True)
    source = torch.randn(4, 3) if random_input else torch.ones(4, 3)
    target = torch.randn(4, 2) if random_input else torch.zeros(4, 2)
    loss = (model(source) - target).pow(2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_checkpoint_recovery_restores_model_optimizer_scheduler_rng_and_cursor(tmp_path):
    torch.manual_seed(42)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    _train_one_step(model, optimizer, scheduler)

    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    torch.save(model.state_dict(), checkpoint / "model.pt")
    save_recovery_bundle(
        checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
        torch=torch,
        run_id="E03-A8-V2-test",
        arm="P",
        step=1,
        last_loss=0.5,
    )

    _train_one_step(model, optimizer, scheduler)
    expected_model = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()

    resumed_model = nn.Linear(3, 2)
    resumed_model.load_state_dict(torch.load(checkpoint / "model.pt", weights_only=False))
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _step: 1.0)
    state = restore_recovery_bundle(
        checkpoint,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        torch=torch,
        expected_run_id="E03-A8-V2-test",
        expected_arm="P",
    )
    assert state["global_step"] == 1
    assert state["sample_cursor"] == 4
    _train_one_step(resumed_model, resumed_optimizer, resumed_scheduler)

    for name, tensor in expected_model.items():
        assert torch.equal(resumed_model.state_dict()[name], tensor)
    assert resumed_optimizer.state_dict()["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, state_values in expected_optimizer["state"].items():
        for key, value in state_values.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(resumed_optimizer.state_dict()["state"][parameter_id][key], value)
            else:
                assert resumed_optimizer.state_dict()["state"][parameter_id][key] == value
    assert resumed_scheduler.state_dict() == expected_scheduler
