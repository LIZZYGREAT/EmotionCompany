"""Deterministic Train-only sampling plans for future isolated E03 experiments.

The module is CPU-only and is not imported by the historical Trainer path.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DIMENSIONS = ("personality_traits", "interests", "style")


@dataclass(frozen=True)
class SamplingDraw:
    sample_id: str
    conversation_id: str
    component: str


def _profile(row: Mapping[str, Any]) -> Mapping[str, Any]:
    target = row.get("target")
    profile = target.get("user_profile") if isinstance(target, Mapping) else row.get("user_profile")
    if not isinstance(profile, Mapping) or set(profile) != set(DIMENSIONS):
        raise ValueError("Train sampling metadata must include all three Gold profile dimensions")
    for dimension in DIMENSIONS:
        labels = profile[dimension]
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError(f"invalid Train profile label metadata in {dimension}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate Train profile labels in {dimension}")
    return profile


def _train_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        conversation_id = row.get("conversation_id")
        split = row.get("source_split")
        if split is None and isinstance(row.get("source"), Mapping):
            split = row["source"].get("split")
        if split != "train":
            raise ValueError(f"row {row_number} is not from Train; Dev/Val cannot enter a sampling plan")
        if not isinstance(sample_id, str) or not sample_id or sample_id in indexed:
            raise ValueError(f"row {row_number} has a missing or duplicate sample_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"row {row_number} has a missing conversation_id")
        _profile(row)
        indexed[sample_id] = row
    if not indexed:
        raise ValueError("Train sampling pool is empty")
    return indexed


def build_sampling_plan(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    draw_count: int,
    seed: int,
    source_train_sha256: str,
    replay_sample_ids: Sequence[str] = (),
    replay_fraction: float = 0.0,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
) -> tuple[list[SamplingDraw], dict[str, Any]]:
    """Create a reproducible schedule, with explicit Train-only Replay draws."""
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 1:
        raise ValueError("draw_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("sampling seed must be an integer")
    if (
        isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size < 1
        or isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1
    ):
        raise ValueError("micro_batch_size and gradient_accumulation_steps must be positive integers")
    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    if draw_count % effective_batch_size:
        raise ValueError("draw_count must align to complete optimizer steps")
    if (
        not isinstance(source_train_sha256, str)
        or len(source_train_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_train_sha256.lower())
    ):
        raise ValueError("source_train_sha256 must be a SHA-256 hex digest")
    if not isinstance(replay_fraction, (int, float)) or isinstance(replay_fraction, bool) or not 0 <= replay_fraction <= 0.5:
        raise ValueError("replay_fraction must be in [0, 0.5]")
    indexed = _train_index(train_rows)
    replay_ids = list(replay_sample_ids)
    if len(replay_ids) != len(set(replay_ids)):
        raise ValueError("replay_sample_ids must not contain duplicates")
    if any(sample_id not in indexed for sample_id in replay_ids):
        raise ValueError("Replay pool must be a subset of the original Train sample IDs")
    replay_draw_count = round(draw_count * replay_fraction)
    if replay_draw_count and not replay_ids:
        raise ValueError("Replay draws require an explicit non-empty Train replay pool")
    natural_draw_count = draw_count - replay_draw_count
    if natural_draw_count > len(indexed):
        raise ValueError("natural no-replacement draw count exceeds the unique Train row count")

    rng = random.Random(seed)
    natural_ids = list(indexed)
    rng.shuffle(natural_ids)
    draws = [
        SamplingDraw(sample_id, str(indexed[sample_id]["conversation_id"]), "natural")
        for sample_id in natural_ids[:natural_draw_count]
    ]
    for _ in range(replay_draw_count):
        sample_id = rng.choice(replay_ids)
        draws.append(SamplingDraw(sample_id, str(indexed[sample_id]["conversation_id"]), "replay"))
    rng.shuffle(draws)

    sample_exposures = Counter(draw.sample_id for draw in draws)
    conversation_exposures = Counter(draw.conversation_id for draw in draws)
    component_counts = Counter(draw.component for draw in draws)
    empty_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {dimension: Counter() for dimension in DIMENSIONS}
    for draw in draws:
        profile = _profile(indexed[draw.sample_id])
        is_empty = all(not profile[dimension] for dimension in DIMENSIONS)
        empty_counts["all_empty" if is_empty else "nonempty"] += 1
        for dimension in DIMENSIONS:
            label_counts[dimension].update(profile[dimension])

    schedule_identity = [
        {"sample_id": draw.sample_id, "conversation_id": draw.conversation_id, "component": draw.component}
        for draw in draws
    ]
    schedule_sha = hashlib.sha256(
        json.dumps(schedule_identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "version": "e03-train-sampling-plan-v1",
        "source_split": "train",
        "source_train_sha256": source_train_sha256.lower(),
        "sampling_seed": seed,
        "draw_count": draw_count,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "optimizer_step_count": draw_count // effective_batch_size,
        "unique_sample_count": len(sample_exposures),
        "repeated_sample_draw_count": sum(count - 1 for count in sample_exposures.values()),
        "replay_pool_unique_sample_count": len(replay_ids),
        "replay_fraction_configured": float(replay_fraction),
        "replay_draw_count": component_counts.get("replay", 0),
        "replay_fraction_actual": component_counts.get("replay", 0) / draw_count,
        "natural_draw_count": component_counts.get("natural", 0),
        "sampling_schedule_sha256": schedule_sha,
        "sample_exposures": dict(sorted(sample_exposures.items())),
        "conversation_exposures": dict(sorted(conversation_exposures.items())),
        "gold_profile_all_empty_draws": empty_counts.get("all_empty", 0),
        "gold_profile_nonempty_draws": empty_counts.get("nonempty", 0),
        "profile_positive_negative_distribution": {
            "all_empty_draws": empty_counts.get("all_empty", 0),
            "nonempty_draws": empty_counts.get("nonempty", 0),
            "scope": "profile-level empty/non-empty; BCE missing-label values are not inferred",
        },
        "gold_label_draw_counts": {
            dimension: dict(sorted(counts.items())) for dimension, counts in label_counts.items()
        },
        "dev_rows_read": 0,
        "val_rows_read": 0,
    }
    return draws, summary


def validate_bce_targets_and_mask(
    targets: Sequence[int | None],
    label_mask: Sequence[int | bool],
    *,
    expected_label_count: int = 33,
) -> tuple[list[float], list[float]]:
    """Validate BCE vector/mask shape without treating missing labels as negatives."""
    if len(targets) != expected_label_count or len(label_mask) != expected_label_count:
        raise ValueError(f"BCE targets and label_mask must each have length {expected_label_count}")
    numeric_targets: list[float] = []
    numeric_mask: list[float] = []
    for index, (target, mask) in enumerate(zip(targets, label_mask)):
        if not isinstance(mask, (int, bool)) or int(mask) not in (0, 1):
            raise ValueError(f"label_mask[{index}] must be binary")
        active = bool(mask)
        if target is None:
            if active:
                raise ValueError(f"missing target at index {index} must have label_mask=0")
            numeric_targets.append(0.0)
            numeric_mask.append(0.0)
            continue
        if isinstance(target, bool) or not isinstance(target, int) or target not in (0, 1):
            raise ValueError(f"BCE target at index {index} must be 0, 1, or None")
        if not active:
            raise ValueError(f"known target at index {index} must have label_mask=1")
        numeric_targets.append(float(target))
        numeric_mask.append(1.0)
    return numeric_targets, numeric_mask
