from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.data.e01_b_devft_final import (
    build_dev_schedule,
    validate_dev_schedule,
    validate_optimizer_state_layout,
)


def test_dev_schedule_covers_1821_ids_with_a_singleton_tail() -> None:
    ids = [f"dev-{index}" for index in range(1821)]
    first = build_dev_schedule(ids, seed=42, group_size=4)
    second = build_dev_schedule(ids, seed=42, group_size=4)

    assert first == second
    assert len(first) == 456
    assert [len(group["sample_ids"]) for group in first[:-1]] == [4] * 455
    assert first[-1]["sample_ids"] and len(first[-1]["sample_ids"]) == 1
    assert [group["global_step"] for group in first] == list(range(4049, 4505))
    validate_dev_schedule(first, ids, group_size=4)


def test_schedule_rejects_duplicate_or_missing_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_dev_schedule(["a", "a"], seed=42, group_size=4)

    schedule = build_dev_schedule(["a", "b", "c", "d", "e"], seed=42, group_size=4)
    schedule[0]["sample_ids"][0] = "outside"
    with pytest.raises(ValueError, match="outside"):
        validate_dev_schedule(schedule, ["a", "b", "c", "d", "e"], group_size=4)


def test_optimizer_layout_preserves_empty_second_trainer_group() -> None:
    parameters = [("lora.a", SimpleNamespace(shape=(2, 3))), ("lora.b", SimpleNamespace(shape=(3, 2)))]
    state = {
        "param_groups": [{"params": [10, 11]}, {"params": []}],
        "state": {
            10: {"step": 4048, "exp_avg": SimpleNamespace(shape=(2, 3)), "exp_avg_sq": SimpleNamespace(shape=(2, 3))},
            11: {"step": 4048, "exp_avg": SimpleNamespace(shape=(3, 2)), "exp_avg_sq": SimpleNamespace(shape=(3, 2))},
        },
    }

    facts = validate_optimizer_state_layout(state, [parameters, []], expected_step=4048)

    assert facts["parameter_group_count"] == 2
    assert facts["parameters_per_group"] == [2, 0]
    assert facts["optimizer_state_count"] == 2


def test_optimizer_layout_rejects_reordered_shape_mismatch() -> None:
    parameters = [("lora.a", SimpleNamespace(shape=(2, 3))), ("lora.b", SimpleNamespace(shape=(3, 2)))]
    state = {
        "param_groups": [{"params": [10, 11]}, {"params": []}],
        "state": {
            10: {"step": 4048, "exp_avg": SimpleNamespace(shape=(3, 2)), "exp_avg_sq": SimpleNamespace(shape=(3, 2))},
            11: {"step": 4048, "exp_avg": SimpleNamespace(shape=(2, 3)), "exp_avg_sq": SimpleNamespace(shape=(2, 3))},
        },
    }

    with pytest.raises(ValueError, match="shape differs"):
        validate_optimizer_state_layout(state, [parameters, []], expected_step=4048)
