"""Identity and schedule helpers for the E01-B Train+Dev final run."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence


def canonical_sha256(value: Any) -> str:
    import json

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dev_schedule(
    sample_ids: Sequence[str],
    *,
    seed: int = 42,
    group_size: int = 4,
    parent_global_step: int = 4048,
) -> list[dict[str, Any]]:
    if not sample_ids or any(not isinstance(value, str) or not value for value in sample_ids):
        raise ValueError("Dev schedule requires non-empty string sample IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Dev schedule input contains duplicate sample IDs")
    if group_size <= 0 or parent_global_step < 0:
        raise ValueError("group_size must be positive and parent_global_step non-negative")

    shuffled = list(sample_ids)
    random.Random(seed).shuffle(shuffled)
    schedule = []
    for offset in range(0, len(shuffled), group_size):
        local_step = len(schedule) + 1
        schedule.append({
            "dev_local_step": local_step,
            "global_step": parent_global_step + local_step,
            "sample_ids": shuffled[offset:offset + group_size],
        })
    validate_dev_schedule(
        schedule,
        sample_ids,
        group_size=group_size,
        parent_global_step=parent_global_step,
    )
    return schedule


def validate_dev_schedule(
    schedule: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
    *,
    group_size: int = 4,
    parent_global_step: int = 4048,
) -> None:
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected Dev IDs must be non-empty and unique")
    expected_steps = (len(expected_ids) + group_size - 1) // group_size
    if len(schedule) != expected_steps:
        raise ValueError(f"schedule has {len(schedule)} steps; expected {expected_steps}")

    seen: list[str] = []
    for index, group in enumerate(schedule, 1):
        if group.get("dev_local_step") != index:
            raise ValueError(f"Dev-local step is not continuous at {index}")
        if group.get("global_step") != parent_global_step + index:
            raise ValueError(f"global step is not continuous at Dev-local step {index}")
        ids = group.get("sample_ids")
        expected_size = min(group_size, len(expected_ids) - len(seen))
        if not isinstance(ids, list) or len(ids) != expected_size:
            raise ValueError(f"Dev-local step {index} has the wrong actual group size")
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(f"Dev-local step {index} contains an invalid sample ID")
        seen.extend(ids)

    if len(seen) != len(expected_ids) or len(set(seen)) != len(expected_ids):
        raise ValueError("Dev schedule repeats or omits an allowed sample ID")
    if set(seen) != set(expected_ids):
        raise ValueError("Dev schedule contains an ID outside the allowed Dev set")


def validate_optimizer_state_layout(
    state_dict: Mapping[str, Any],
    named_parameter_groups: Sequence[Sequence[tuple[str, Any]]],
    *,
    expected_step: int,
) -> dict[str, Any]:
    """Verify saved optimizer positions map to the recreated Trainer parameter order."""
    saved_groups = state_dict.get("param_groups")
    saved_state = state_dict.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, Mapping):
        raise ValueError("optimizer state is missing param_groups or state")
    if len(saved_groups) != len(named_parameter_groups):
        raise ValueError("recreated AdamW parameter group count differs from the saved checkpoint")

    observed_names: list[list[str]] = []
    observed_steps: set[int] = set()
    mapped_state_count = 0
    for group_index, (saved_group, named_group) in enumerate(zip(saved_groups, named_parameter_groups)):
        saved_ids = saved_group.get("params")
        if not isinstance(saved_ids, list) or len(saved_ids) != len(named_group):
            raise ValueError(f"AdamW parameter count differs in group {group_index}")
        observed_names.append([name for name, _parameter in named_group])
        for parameter_id, (name, parameter) in zip(saved_ids, named_group):
            state = saved_state.get(parameter_id)
            if not isinstance(state, Mapping):
                raise ValueError(f"AdamW state is missing parameter {name}")
            shape = tuple(parameter.shape)
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = state.get(moment_name)
                if moment is None or tuple(moment.shape) != shape:
                    raise ValueError(f"AdamW {moment_name} shape differs for parameter {name}")
            step_value = state.get("step")
            if hasattr(step_value, "item"):
                step_value = step_value.item()
            if int(step_value) != expected_step:
                raise ValueError(f"AdamW state step differs for parameter {name}")
            observed_steps.add(int(step_value))
            mapped_state_count += 1

    if mapped_state_count != len(saved_state) or observed_steps != {expected_step}:
        raise ValueError("AdamW state contains unmapped parameters or inconsistent update steps")
    return {
        "parameter_group_count": len(saved_groups),
        "parameters_per_group": [len(group) for group in named_parameter_groups],
        "optimizer_state_count": mapped_state_count,
        "state_step": expected_step,
        "parameter_names_sha256": canonical_sha256(observed_names),
    }
