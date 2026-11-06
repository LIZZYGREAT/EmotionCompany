"""Synthetic sample-order and optimizer-state resume checks for staged F4."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from src.training.e03_a6_f4_staged_contract import SampleOrderRecorder, rewind_sample_order_log


def _consume(recorder: SampleOrderRecorder, values: list[str], step: int) -> None:
    for sample_id in values:
        recorder.record_micro_batch([sample_id])
    recorder.commit_optimizer_step(step)


def test_sample_order_resume_reproduces_the_uninterrupted_consumption_log(tmp_path: Path):
    ids = [f"train:synthetic:{index:02d}" for index in range(12)]
    uninterrupted_path = tmp_path / "uninterrupted.jsonl"
    uninterrupted = SampleOrderRecorder(uninterrupted_path, ids)
    _consume(uninterrupted, ids[:4], 1)
    _consume(uninterrupted, ids[4:8], 2)
    _consume(uninterrupted, ids[8:12], 3)

    resumed_path = tmp_path / "resumed.jsonl"
    interrupted = SampleOrderRecorder(resumed_path, ids)
    _consume(interrupted, ids[:4], 1)
    for sample_id in ids[4:8]:
        interrupted.record_micro_batch([sample_id])
    checkpoint_facts = interrupted.commit_optimizer_step(2)
    _consume(interrupted, ids[8:12], 3)

    # A crash can leave sampler evidence newer than the last durable checkpoint.
    rewind_sample_order_log(resumed_path, 2, tmp_path)
    resumed = SampleOrderRecorder(resumed_path, ids)
    assert resumed.checkpoint_facts(2) == checkpoint_facts
    _consume(resumed, ids[8:12], 3)
    assert resumed_path.read_bytes() == uninterrupted_path.read_bytes()
    assert resumed.checkpoint_facts(3)["sample_order_no_duplicates"] is True


def test_sample_order_recorder_rejects_duplicate_or_foreign_train_ids(tmp_path: Path):
    ids = [f"train:synthetic:{index:02d}" for index in range(4)]
    recorder = SampleOrderRecorder(tmp_path / "duplicate.jsonl", ids)
    for sample_id in (ids[0], ids[1], ids[2], ids[0]):
        recorder.record_micro_batch([sample_id])
    with pytest.raises(ValueError, match="repeated a sample"):
        recorder.commit_optimizer_step(1)

    recorder = SampleOrderRecorder(tmp_path / "foreign.jsonl", ids)
    with pytest.raises(ValueError, match="unknown F4 Train ID"):
        recorder.record_micro_batch(["train:foreign:1"])


def _assert_nested_close(left, right, torch):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key], torch)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_close(left_value, right_value, torch)
    else:
        assert left == right


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed in minimal CPU CI")
def test_cpu_synthetic_pause_checkpoint_resume_matches_continuous_adamw(tmp_path: Path):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(73)
    samples = [
        (torch.tensor([[0.1 * index, 0.2 - 0.03 * index]]), torch.tensor([[0.05 * index]]))
        for index in range(1, 7)
    ]
    sample_ids = [f"train:synthetic:{index:02d}" for index in range(6)]
    base_model = torch.nn.Linear(2, 1, bias=False)
    initial_model = copy.deepcopy(base_model.state_dict())
    initial_rng = torch.random.get_rng_state()

    def build():
        model = torch.nn.Linear(2, 1, bias=False)
        model.load_state_dict(initial_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        return model, optimizer, scheduler

    def train_range(model, optimizer, scheduler, start: int, stop: int, order: SampleOrderRecorder):
        model.train()
        for optimizer_step in range(start // 2 + 1, stop // 2 + 1):
            optimizer.zero_grad(set_to_none=True)
            batch_ids = []
            for index in range((optimizer_step - 1) * 2, optimizer_step * 2):
                inputs, targets = samples[index]
                loss = torch.nn.functional.mse_loss(model(inputs), targets) / 2
                loss.backward()
                batch_ids.append(sample_ids[index])
                order.record_micro_batch([sample_ids[index]])
            optimizer.step()
            scheduler.step()
            order.commit_optimizer_step(optimizer_step)
        return model, optimizer, scheduler

    continuous_model, continuous_optimizer, continuous_scheduler = build()
    torch.random.set_rng_state(initial_rng)
    continuous_order = SampleOrderRecorder(tmp_path / "continuous-order.jsonl", sample_ids, accumulation_steps=2)
    train_range(continuous_model, continuous_optimizer, continuous_scheduler, 0, 6, continuous_order)

    paused_model, paused_optimizer, paused_scheduler = build()
    torch.random.set_rng_state(initial_rng)
    paused_order = SampleOrderRecorder(tmp_path / "resumed-order.jsonl", sample_ids, accumulation_steps=2)
    train_range(paused_model, paused_optimizer, paused_scheduler, 0, 4, paused_order)
    checkpoint = {
        "model": copy.deepcopy(paused_model.state_dict()),
        "optimizer": copy.deepcopy(paused_optimizer.state_dict()),
        "scheduler": copy.deepcopy(paused_scheduler.state_dict()),
        "torch_rng_state": torch.random.get_rng_state(),
        "next_sample_index": 4,
    }
    checkpoint_path = tmp_path / "synthetic-checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    del paused_model, paused_optimizer, paused_scheduler

    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resumed_model, resumed_optimizer, resumed_scheduler = build()
    resumed_model.load_state_dict(restored["model"])
    resumed_optimizer.load_state_dict(restored["optimizer"])
    resumed_scheduler.load_state_dict(restored["scheduler"])
    torch.random.set_rng_state(restored["torch_rng_state"])
    resumed_order = SampleOrderRecorder(tmp_path / "resumed-order.jsonl", sample_ids, accumulation_steps=2)
    train_range(resumed_model, resumed_optimizer, resumed_scheduler, restored["next_sample_index"], 6, resumed_order)

    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=1e-6, atol=1e-7)
    _assert_nested_close(continuous_optimizer.state_dict(), resumed_optimizer.state_dict(), torch)
    assert continuous_scheduler.state_dict() == resumed_scheduler.state_dict()
    assert continuous_order.path.read_bytes() == resumed_order.path.read_bytes()
