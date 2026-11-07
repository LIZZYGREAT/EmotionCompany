"""Deterministic A8-V2 schedule with a 75/25 positive/empty first window."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.e03_a8_sft import canonical_sha256
from src.shared.e03_a8_protocol import PROFILE_DIMS

SCHEDULE_ALGORITHM = "a8-v2-greedy-stratified-75-25-v1"
FIRST_WINDOW_ROWS = 2048
FIRST_WINDOW_NONEMPTY = 1536
FIRST_WINDOW_EMPTY = 512


def _length_bucket(length: int, low_cut: int, high_cut: int) -> str:
    if length <= low_cut:
        return "short"
    if length <= high_cut:
        return "medium"
    return "long"


def _select_group(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    group_name: str,
    low_cut: int,
    high_cut: int,
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"A8-V2 needs {count} {group_name} samples but only has {len(candidates)}")
    for row in candidates:
        row["_length_bucket"] = _length_bucket(int(row["history_message_count"]), low_cut, high_cut)

    all_conv_counts = Counter(str(row["conversation_id"]) for row in candidates)
    selected_conv_counts: Counter[str] = Counter()
    selected_label_counts: Counter[tuple[str, str]] = Counter()
    selected_dimension_counts: Counter[str] = Counter()
    selected_bucket_counts: Counter[str] = Counter()
    label_totals = Counter(
        (dimension, label)
        for row in candidates
        for dimension in PROFILE_DIMS
        for label in row["profile"][dimension]
    )
    dimension_totals = Counter(
        dimension
        for row in candidates
        for dimension in PROFILE_DIMS
        if row["profile"][dimension]
    )
    dimension_targets = {
        dimension: count * dimension_totals[dimension] / max(len(candidates), 1)
        for dimension in PROFILE_DIMS
    }
    label_targets = {
        pair: count * math.sqrt(total) / sum(math.sqrt(value) for value in label_totals.values())
        for pair, total in label_totals.items()
    }
    bucket_targets = {bucket: count / 3 for bucket in ("short", "medium", "long")}

    rng = random.Random(seed)
    tie_order = list(range(len(candidates)))
    rng.shuffle(tie_order)
    tie_rank = {index: rank for rank, index in enumerate(tie_order)}
    row_index = {id(row): index for index, row in enumerate(candidates)}
    remaining = {str(row["sample_id"]): row for row in candidates}
    chosen: list[dict[str, Any]] = []
    all_conversations = set(all_conv_counts)

    while len(chosen) < count:
        slots_left = count - len(chosen)
        unused_conversations = all_conversations - set(selected_conv_counts)
        require_new_conversation = len(unused_conversations) >= slots_left
        eligible = [
            row for row in remaining.values()
            if not require_new_conversation or selected_conv_counts[str(row["conversation_id"])] == 0
        ]
        if not eligible:
            eligible = list(remaining.values())

        def score(row: dict[str, Any]) -> tuple[float, int]:
            dimensions = [dimension for dimension in PROFILE_DIMS if row["profile"][dimension]]
            dimension_score = sum(
                max(0.0, dimension_targets[dimension] - selected_dimension_counts[dimension])
                / max(dimension_targets[dimension], 1.0)
                for dimension in dimensions
            ) / max(len(dimensions), 1)
            label_scores = []
            for dimension in PROFILE_DIMS:
                labels = row["profile"][dimension]
                if labels:
                    label_scores.append(
                        sum(
                            max(0.0, label_targets[(dimension, label)] - selected_label_counts[(dimension, label)])
                            / max(label_targets[(dimension, label)], 0.5)
                            for label in labels
                        ) / len(labels)
                    )
            label_score = sum(label_scores) / max(len(label_scores), 1)
            bucket = str(row["_length_bucket"])
            bucket_score = max(0.0, bucket_targets[bucket] - selected_bucket_counts[bucket]) / bucket_targets[bucket]
            conversation = str(row["conversation_id"])
            novel_score = 1.0 if selected_conv_counts[conversation] == 0 else 0.0
            repeat_penalty = selected_conv_counts[conversation] / max(all_conv_counts[conversation], 1)
            value = 0.38 * dimension_score + 0.42 * label_score + 0.18 * bucket_score + 0.20 * novel_score - 0.10 * repeat_penalty
            return value, -tie_rank[row_index[id(row)]]

        selected = max(eligible, key=score)
        chosen.append(selected)
        del remaining[str(selected["sample_id"])]
        conversation = str(selected["conversation_id"])
        selected_conv_counts[conversation] += 1
        selected_bucket_counts[str(selected["_length_bucket"])] += 1
        for dimension in PROFILE_DIMS:
            labels = selected["profile"][dimension]
            if labels:
                selected_dimension_counts[dimension] += 1
                selected_label_counts.update((dimension, label) for label in labels)
    return chosen


def build_a8_v2_schedule(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    accumulation_steps: int = 4,
    first_window_rows: int = FIRST_WINDOW_ROWS,
    first_window_nonempty: int = FIRST_WINDOW_NONEMPTY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return optimizer schedule and an auditable first-window summary.

    `candidates` are paired to the frozen effective Train by sample_id and
    contain the official Gold profile plus already-selected common-history
    lengths. No Gold value is changed and every Train row occurs once.
    """
    rows = [dict(row) for row in candidates]
    ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("A8-V2 schedule candidates require unique non-empty sample IDs")
    if accumulation_steps <= 0 or len(rows) % accumulation_steps:
        raise ValueError("A8-V2 row count must divide evenly by gradient accumulation")
    if first_window_rows <= 0 or first_window_rows % accumulation_steps or not 0 <= first_window_nonempty <= first_window_rows:
        raise ValueError("A8-V2 first-window size must be positive and step-aligned")
    if first_window_rows > len(rows):
        raise ValueError("A8-V2 first window exceeds the frozen Train set")
    for row in rows:
        profile = row.get("profile")
        if not isinstance(profile, Mapping) or set(profile) != set(PROFILE_DIMS):
            raise ValueError(f"{row.get('sample_id')}: A8-V2 Gold profile dimensions differ")
        if not isinstance(row.get("conversation_id"), str) or not row["conversation_id"]:
            raise ValueError(f"{row.get('sample_id')}: A8-V2 conversation ID is missing")
        if not isinstance(row.get("history_message_count"), int) or row["history_message_count"] <= 0:
            raise ValueError(f"{row.get('sample_id')}: A8-V2 common-history length is invalid")
        for dimension in PROFILE_DIMS:
            labels = profile[dimension]
            if not isinstance(labels, list) or len(labels) != len(set(labels)):
                raise ValueError(f"{row.get('sample_id')}: malformed Gold labels for {dimension}")

    ordered_lengths = sorted(int(row["history_message_count"]) for row in rows)
    low_cut = ordered_lengths[max(0, math.floor(0.33 * (len(ordered_lengths) - 1)))]
    high_cut = ordered_lengths[max(0, math.floor(0.67 * (len(ordered_lengths) - 1)))]
    positives = [row for row in rows if any(row["profile"][dimension] for dimension in PROFILE_DIMS)]
    empties = [row for row in rows if not any(row["profile"][dimension] for dimension in PROFILE_DIMS)]
    empty_count = first_window_rows - first_window_nonempty
    first_positive = _select_group(
        positives, first_window_nonempty, seed=seed, group_name="nonempty", low_cut=low_cut, high_cut=high_cut
    ) if first_window_nonempty else []
    first_empty = _select_group(
        empties, empty_count, seed=seed ^ 0x5A17, group_name="all-empty", low_cut=low_cut, high_cut=high_cut
    ) if empty_count else []
    first_ids = {row["sample_id"] for row in first_positive + first_empty}
    if len(first_ids) != first_window_rows:
        raise RuntimeError("A8-V2 first-window selection duplicated an ID")
    remainder = [row for row in rows if row["sample_id"] not in first_ids]
    random.Random(seed ^ 0xA803).shuffle(remainder)
    batch_rng = random.Random(seed ^ 0xA804)
    first_window: list[dict[str, Any]] = []
    step_count = first_window_rows // accumulation_steps
    for step_index in range(step_count):
        positive_start = math.floor(step_index * first_window_nonempty / step_count)
        positive_end = math.floor((step_index + 1) * first_window_nonempty / step_count)
        empty_start = math.floor(step_index * empty_count / step_count)
        empty_end = math.floor((step_index + 1) * empty_count / step_count)
        batch = first_positive[positive_start:positive_end] + first_empty[empty_start:empty_end]
        if len(batch) != accumulation_steps:
            raise ValueError("A8-V2 first-window positive/empty ratio cannot fill each optimizer step")
        batch_rng.shuffle(batch)
        first_window.extend(batch)
    ordered_ids = [str(row["sample_id"]) for row in first_window + remainder]
    schedule = [
        {"optimizer_step": index + 1, "sample_ids": ordered_ids[index * accumulation_steps:(index + 1) * accumulation_steps]}
        for index in range(len(ordered_ids) // accumulation_steps)
    ]

    by_id = {str(row["sample_id"]): row for row in rows}
    first_rows = [by_id[sample_id] for sample_id in ordered_ids[:first_window_rows]]
    observed_labels = {
        label for row in rows for dimension in PROFILE_DIMS for label in row["profile"][dimension]
    }
    label_counts = Counter({label: 0 for label in observed_labels})
    label_counts.update(
        label for row in first_rows for dimension in PROFILE_DIMS for label in row["profile"][dimension]
    )
    dimension_coverage = {
        dimension: sum(bool(row["profile"][dimension]) for row in first_rows)
        for dimension in PROFILE_DIMS
    }
    conv_counts = Counter(str(row["conversation_id"]) for row in first_rows)
    history_buckets = Counter(str(row.get("_length_bucket", "unassigned")) for row in first_positive + first_empty)
    for row in rows:
        row.pop("_length_bucket", None)
    audit = {
        "algorithm": SCHEDULE_ALGORITHM,
        "seed": seed,
        "gradient_accumulation_steps": accumulation_steps,
        "total_train_rows": len(rows),
        "total_schedule_steps": len(schedule),
        "first_window_rows": first_window_rows,
        "first_window_nonempty_rows": sum(any(first_rows[i]["profile"][dim] for dim in PROFILE_DIMS) for i in range(len(first_rows))),
        "first_window_all_empty_rows": sum(not any(first_rows[i]["profile"][dim] for dim in PROFILE_DIMS) for i in range(len(first_rows))),
        "first_window_dimension_positive_sample_counts": dimension_coverage,
        "first_window_profile_label_counts": dict(sorted(label_counts.items())),
        "first_window_unique_conversations": len(conv_counts),
        "first_window_repeated_conversation_samples": sum(count - 1 for count in conv_counts.values()),
        "first_window_max_samples_per_conversation": max(conv_counts.values(), default=0),
        "first_window_history_length_buckets": dict(sorted(history_buckets.items())),
        "first_window_sample_ids_sha256": canonical_sha256(ordered_ids[:first_window_rows]),
        "remainder_sample_ids_sha256": canonical_sha256(ordered_ids[first_window_rows:]),
        "ordered_sample_ids_sha256": canonical_sha256(ordered_ids),
        "schedule_sha256": canonical_sha256(schedule),
        "first_window_ids_complement_remainder": set(ordered_ids[:first_window_rows]).isdisjoint(ordered_ids[first_window_rows:])
        and set(ordered_ids[:first_window_rows]) | set(ordered_ids[first_window_rows:]) == set(ids),
    }
    return schedule, audit
