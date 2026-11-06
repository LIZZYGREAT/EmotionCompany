"""Train-only, schema-driven 33-label supervision for E03-A5.

This module deliberately keeps the D0.3 records and D0.4.3 token rows read-only.
It joins them by ``sample_id`` and emits only hashes and the derived target.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DIMENSIONS = ("personality_traits", "interests", "style")
LABEL_ORDER_VERSION = "official_train_schema_profile_order_v1"
TARGET_POLICY = "closed_world_gold_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    """Yield a strict JSON object and the hash of its original JSONL record."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON at column {exc.colno}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            record_bytes = line.rstrip("\r\n").encode("utf-8")
            yield line_number, value, hashlib.sha256(record_bytes).hexdigest()


def schema_label_order(
    train_schema: Mapping[str, Any], output_schema: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Read profile enums in official order and optionally cross-check inference schema."""
    try:
        profile = train_schema["$defs"]["profile"]["properties"]
        enums = {
            dimension: list(profile[dimension]["items"]["enum"])
            for dimension in DIMENSIONS
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("official Train schema lacks the three profile enum arrays") from exc
    for dimension, values in enums.items():
        if not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"official schema {dimension} enum must be non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"official schema {dimension} enum contains duplicate values")
    if [len(enums[name]) for name in DIMENSIONS] != [10, 12, 11]:
        raise ValueError("official profile dimensions must contain 10, 12, and 11 labels")
    if output_schema is not None:
        try:
            output_profile = output_schema["properties"]["user_profile"]["properties"]
            output_enums = {
                dimension: list(output_profile[dimension]["items"]["enum"])
                for dimension in DIMENSIONS
            }
        except (KeyError, TypeError) as exc:
            raise ValueError("official output schema lacks the three profile enum arrays") from exc
        if output_enums != enums:
            raise ValueError("Train and inference profile enum order differs")
    return [f"{dimension}:{label}" for dimension in DIMENSIONS for label in enums[dimension]], enums


def profile_to_target(
    profile: Any, enums: Mapping[str, Sequence[str]], *, sample_id: str,
) -> tuple[list[int], list[int], dict[str, list[str]]]:
    """Convert an explicitly complete Gold profile into a closed-world target."""
    if not isinstance(profile, dict) or set(profile) != set(DIMENSIONS):
        raise ValueError(f"{sample_id}: Gold user_profile must contain exactly the three dimensions")
    y: list[int] = []
    normalized: dict[str, list[str]] = {}
    for dimension in DIMENSIONS:
        values = profile[dimension]
        if not isinstance(values, list):
            raise ValueError(f"{sample_id}: Gold {dimension} must be an array, including when empty")
        if any(not isinstance(label, str) for label in values):
            raise ValueError(f"{sample_id}: Gold {dimension} contains a non-string label")
        if len(values) != len(set(values)):
            raise ValueError(f"{sample_id}: Gold {dimension} contains duplicate labels")
        allowed = list(enums[dimension])
        illegal = sorted(set(values) - set(allowed))
        if illegal:
            raise ValueError(f"{sample_id}: Gold {dimension} contains labels outside official enum: {illegal}")
        selected = set(values)
        normalized[dimension] = list(values)
        y.extend(int(label in selected) for label in allowed)
    mask = [1] * 33
    if len(y) != 33:
        raise AssertionError("internal label order must produce exactly 33 targets")
    return y, mask, normalized


def validate_sft_row(row: Mapping[str, Any], *, max_sequence_length: int = 2048) -> dict[str, int | str]:
    """Check the pre-tokenized Completion-only contract without retokenizing."""
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("SFT row requires a non-empty sample_id")
    input_ids = row.get("input_ids")
    labels = row.get("labels")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError(f"{sample_id}: input_ids must be a non-empty list")
    if len(input_ids) > max_sequence_length:
        raise ValueError(f"{sample_id}: sequence exceeds S={max_sequence_length}")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in input_ids):
        raise ValueError(f"{sample_id}: input_ids must contain non-negative integers")
    if not isinstance(labels, list) or len(labels) != len(input_ids):
        raise ValueError(f"{sample_id}: labels must align with input_ids")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in labels):
        raise ValueError(f"{sample_id}: labels must contain integers")
    prompt_length = row.get("prompt_length")
    completion_length = row.get("completion_length")
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or not 0 < prompt_length < len(input_ids):
        raise ValueError(f"{sample_id}: prompt_length is invalid")
    if (
        isinstance(completion_length, bool)
        or not isinstance(completion_length, int)
        or completion_length != len(input_ids) - prompt_length
    ):
        raise ValueError(f"{sample_id}: completion_length does not align with prompt_length")
    if labels[:prompt_length] != [-100] * prompt_length:
        raise ValueError(f"{sample_id}: Prompt labels are not fully masked")
    if labels[prompt_length:] != input_ids[prompt_length:]:
        raise ValueError(f"{sample_id}: Completion labels differ from the input suffix")
    if row.get("source_split") != "train":
        raise ValueError(f"{sample_id}: only D0.4.3 Train rows are eligible")
    return {
        "sample_id": sample_id,
        "sequence_length": len(input_ids),
        "prompt_length": prompt_length,
        "completion_length": completion_length,
    }


def _normalized_source_facts(row: Mapping[str, Any], *, path: Path, line_number: int) -> tuple[str, str, int, dict[str, Any]]:
    sample_id = row.get("sample_id")
    conversation_id = row.get("conversation_id")
    source = row.get("source")
    history = row.get("history")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{path}:{line_number}: normalized Train requires sample_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError(f"{sample_id}: normalized Train requires conversation_id")
    if not isinstance(source, dict) or source.get("split") != "train":
        raise ValueError(f"{sample_id}: D0.3 source split must be train")
    if not isinstance(history, list):
        raise ValueError(f"{sample_id}: normalized history must be a list")
    prediction_index = source.get("prediction_point_index")
    if isinstance(prediction_index, bool) or not isinstance(prediction_index, int) or prediction_index < 0:
        raise ValueError(f"{sample_id}: D0.3 prediction_point_index is missing or invalid")
    return sample_id, conversation_id, len(history), source


def _profile_summary(
    rows: Sequence[Mapping[str, Any]], label_order: Sequence[str], enums: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    positive = [0] * len(label_order)
    cooccurrence = [[0] * len(label_order) for _ in label_order]
    empty_by_dimension = {dimension: 0 for dimension in DIMENSIONS}
    all_empty = 0
    labels_per_row: list[int] = []
    per_history: dict[str, dict[str, Any]] = {}
    for row in rows:
        y = row["y"]
        profile = row["_profile"]
        for index, value in enumerate(y):
            positive[index] += value
        active_indices = [index for index, value in enumerate(y) if value == 1]
        for left, right in itertools.combinations(active_indices, 2):
            cooccurrence[left][right] += 1
            cooccurrence[right][left] += 1
        for dimension in DIMENSIONS:
            if not profile[dimension]:
                empty_by_dimension[dimension] += 1
        if all(not profile[dimension] for dimension in DIMENSIONS):
            all_empty += 1
        labels_per_row.append(sum(y))
        history_key = str(row["_history_message_count"])
        bucket = per_history.setdefault(history_key, {
            "sample_count": 0,
            "label_count_total": 0,
            "positive_by_dimension": {dimension: 0 for dimension in DIMENSIONS},
        })
        bucket["sample_count"] += 1
        bucket["label_count_total"] += sum(y)
        cursor = 0
        for dimension in DIMENSIONS:
            width = len(enums[dimension])
            bucket["positive_by_dimension"][dimension] += sum(y[cursor:cursor + width])
            cursor += width
    n = len(rows)
    per_label = [
        {
            "label": label,
            "positive_count": positive[index],
            "negative_count": n - positive[index],
            "positive_rate": positive[index] / n if n else 0.0,
        }
        for index, label in enumerate(label_order)
    ]
    cooccurrence_candidates = []
    for left in range(len(label_order)):
        for right in range(left + 1, len(label_order)):
            joint = cooccurrence[left][right]
            union = positive[left] + positive[right] - joint
            jaccard = joint / union if union else 0.0
            if joint >= 10 and jaccard >= 0.25:
                cooccurrence_candidates.append({
                    "left": label_order[left],
                    "right": label_order[right],
                    "cooccurring_samples": joint,
                    "jaccard": jaccard,
                    "interpretation": "co-selection_candidate_for_manual_semantic_review",
                })
    cooccurrence_candidates.sort(
        key=lambda item: (-item["jaccard"], -item["cooccurring_samples"], item["left"], item["right"])
    )
    per_dimension: dict[str, Any] = {}
    cursor = 0
    for dimension in DIMENSIONS:
        width = len(enums[dimension])
        positives = sum(positive[cursor:cursor + width])
        per_dimension[dimension] = {
            "label_count": width,
            "positive_count": positives,
            "positive_rate_per_label_position": positives / (n * width) if n else 0.0,
            "empty_sample_count": empty_by_dimension[dimension],
            "empty_sample_rate": empty_by_dimension[dimension] / n if n else 0.0,
        }
        cursor += width
    for bucket in per_history.values():
        bucket["mean_labels_per_sample"] = (
            bucket["label_count_total"] / bucket["sample_count"] if bucket["sample_count"] else 0.0
        )
    return {
        "sample_count": n,
        "conversation_count": len({str(row["conversation_id"]) for row in rows}),
        "all_three_dimensions_empty_count": all_empty,
        "all_three_dimensions_empty_rate": all_empty / n if n else 0.0,
        "mean_labels_per_sample": sum(labels_per_row) / n if n else 0.0,
        "per_dimension": per_dimension,
        "per_label": per_label,
        "rare_support_labels_below_50_positives": [
            {"label": item["label"], "positive_count": item["positive_count"]}
            for item in per_label if item["positive_count"] < 50
        ],
        "cooccurrence_candidates": cooccurrence_candidates[:20],
        "by_history_message_count": dict(sorted(per_history.items(), key=lambda item: int(item[0]))),
    }


def build_supervision(
    normalized_train_path: Path,
    sft_train_path: Path,
    label_order: Sequence[str],
    enums: Mapping[str, Sequence[str]],
    *,
    max_sequence_length: int = 2048,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact sample-id join of normalized Train targets with usable SFT Train rows."""
    normalized_by_id: dict[str, dict[str, Any]] = {}
    normalized_count = 0
    histories: dict[str, list[tuple[int, str, str, dict[str, list[str]]]]] = defaultdict(list)
    identical_history_profiles: dict[str, set[str]] = defaultdict(set)
    for line_number, row, record_sha in iter_jsonl(normalized_train_path):
        normalized_count += 1
        sample_id, conversation_id, history_count, source = _normalized_source_facts(
            row, path=normalized_train_path, line_number=line_number
        )
        if sample_id in normalized_by_id:
            raise ValueError(f"D0.3 normalized Train has duplicate sample_id: {sample_id}")
        y, mask, profile = profile_to_target(
            row.get("target", {}).get("user_profile") if isinstance(row.get("target"), dict) else None,
            enums,
            sample_id=sample_id,
        )
        profile_sha = canonical_sha256(profile)
        history_sha = canonical_sha256(row["history"])
        normalized_by_id[sample_id] = {
            "conversation_id": conversation_id,
            "history_message_count": history_count,
            "source": source,
            "y": y,
            "mask": mask,
            "profile": profile,
            "profile_sha256": profile_sha,
            "history_sha256": history_sha,
            "record_sha256": record_sha,
        }
        identical_history_profiles[history_sha].add(profile_sha)
        histories[conversation_id].append((
            int(source["prediction_point_index"]), sample_id, history_sha, profile
        ))

    supervision: list[dict[str, Any]] = []
    seen_sft: set[str] = set()
    for line_number, row, sft_record_sha in iter_jsonl(sft_train_path):
        facts = validate_sft_row(row, max_sequence_length=max_sequence_length)
        sample_id = str(facts["sample_id"])
        if sample_id in seen_sft:
            raise ValueError(f"D0.4.3 SFT Train has duplicate sample_id: {sample_id}")
        seen_sft.add(sample_id)
        normalized = normalized_by_id.get(sample_id)
        if normalized is None:
            raise ValueError(f"SFT sample_id is absent from normalized D0.3 Train: {sample_id}")
        if row.get("conversation_id") != normalized["conversation_id"]:
            raise ValueError(f"{sample_id}: conversation_id differs across the exact sample_id join")
        source = row.get("source")
        if not isinstance(source, dict) or source != normalized["source"]:
            raise ValueError(f"{sample_id}: D0.3 source metadata differs across the SFT join")
        supervision.append({
            "sample_id": sample_id,
            "conversation_id": normalized["conversation_id"],
            "prompt_length": facts["prompt_length"],
            "label_order_version": LABEL_ORDER_VERSION,
            "target_policy": TARGET_POLICY,
            "y": normalized["y"],
            "mask": normalized["mask"],
            "source_sha256": {
                "normalized_record": normalized["record_sha256"],
                "sft_record": sft_record_sha,
            },
            "_profile": normalized["profile"],
            "_history_message_count": normalized["history_message_count"],
            "_prediction_point_index": normalized["source"]["prediction_point_index"],
            "_history_sha256": normalized["history_sha256"],
        })
    if len(seen_sft) != len(supervision):
        raise AssertionError("internal SFT join count mismatch")
    excluded_ids = sorted(set(normalized_by_id) - seen_sft)
    if len(normalized_by_id) != normalized_count:
        raise AssertionError("normalized Train id count changed while building supervision")

    ordered = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in sorted(supervision, key=lambda item: item["sample_id"])
    ]

    # Rehydrate only the target diagnostics; original histories and texts are never copied.
    diagnostic_rows: list[dict[str, Any]] = []
    for row in supervision:
        source = normalized_by_id[row["sample_id"]]
        diagnostic_rows.append({
            **row,
            "_profile": source["profile"],
            "_history_message_count": source["history_message_count"],
        })
    summary = _profile_summary(diagnostic_rows, label_order, enums)

    conversation_rows: dict[str, list[tuple[int, str, str, dict[str, list[str]]]]] = defaultdict(list)
    for conversation_id, records in histories.items():
        conversation_rows[conversation_id] = records
    multi_point = changed_profiles = unchanged_transitions = 0
    added_labels = removed_labels = 0
    for records in conversation_rows.values():
        records.sort(key=lambda item: (item[0], item[1]))
        if len(records) > 1:
            multi_point += 1
        for left, right in zip(records, records[1:]):
            left_labels = {f"{dimension}:{label}" for dimension in DIMENSIONS for label in left[3][dimension]}
            right_labels = {f"{dimension}:{label}" for dimension in DIMENSIONS for label in right[3][dimension]}
            if left_labels == right_labels:
                unchanged_transitions += 1
            else:
                changed_profiles += 1
            added_labels += len(right_labels - left_labels)
            removed_labels += len(left_labels - right_labels)
    report = {
        "join": {
            "normalized_train_rows": normalized_count,
            "normalized_unique_sample_ids": len(normalized_by_id),
            "sft_train_rows": len(seen_sft),
            "joined_rows": len(supervision),
            "excluded_normalized_rows": len(excluded_ids),
            "excluded_sample_ids": excluded_ids,
            "join_key": "sample_id",
            "join_exact": len(seen_sft) == len(supervision) and set(seen_sft) <= set(normalized_by_id),
            "all_joined_source_splits_train": True,
            "sft_token_contract_validated": True,
        },
        "target": {
            "label_order_version": LABEL_ORDER_VERSION,
            "target_policy": TARGET_POLICY,
            "label_order": list(label_order),
            "mask_policy": "all_ones_for_complete_three_dimension_gold",
            "mask_zero_count": 0,
            "gold_empty_rows_retained": True,
            "profile_stats": summary,
        },
        "annotation_evolution": {
            "conversation_count": len(conversation_rows),
            "conversations_with_multiple_prediction_points": multi_point,
            "adjacent_prediction_point_transitions": changed_profiles + unchanged_transitions,
            "adjacent_profile_changed_transitions": changed_profiles,
            "adjacent_profile_unchanged_transitions": unchanged_transitions,
            "labels_added_between_adjacent_points": added_labels,
            "labels_removed_between_adjacent_points": removed_labels,
            "identical_history_hash_count": len(identical_history_profiles),
            "strict_identical_history_profile_conflict_count": sum(
                1 for values in identical_history_profiles.values() if len(values) > 1
            ),
        },
        "rows": ordered,
    }
    return ordered, report


def freeze_conversation_splits(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 42,
    fit_fraction: float = 0.70, calibration_fraction: float = 0.15,
) -> dict[str, Any]:
    """Deterministically stratify whole conversations and report per-label support.

    Labels supported by at least three distinct conversations are first spread so
    that each partition has positive support when the group-size constraints allow.
    No row-level split or model result is used to choose the partitions.
    """
    if not rows:
        raise ValueError("cannot split an empty supervision table")
    if not (0 < fit_fraction < 1 and 0 < calibration_fraction < 1 and fit_fraction + calibration_fraction < 1):
        raise ValueError("split fractions must be positive and sum to less than one")
    by_conversation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id, conversation_id = row.get("sample_id"), row.get("conversation_id")
        if not isinstance(sample_id, str) or not isinstance(conversation_id, str):
            raise ValueError("every supervision row needs sample_id and conversation_id")
        by_conversation[conversation_id].append(row)
    conversations = sorted(by_conversation)
    rng = random.Random(seed)
    n = len(conversations)
    fit_count = max(1, math.floor(n * fit_fraction + 0.5))
    calibration_count = max(1, math.floor(n * calibration_fraction + 0.5))
    if fit_count + calibration_count >= n:
        raise ValueError("not enough conversations for three non-empty partitions")
    names = ("fit", "calibration", "internal_holdout")
    fractions = (fit_fraction, calibration_fraction, 1.0 - fit_fraction - calibration_fraction)
    capacities = (fit_count, calibration_count, n - fit_count - calibration_count)
    tie_order = list(conversations)
    rng.shuffle(tie_order)
    tie_rank = {conversation_id: index for index, conversation_id in enumerate(tie_order)}
    conversation_presence: dict[str, set[int]] = {}
    for conversation_id, conversation_rows in by_conversation.items():
        presence: set[int] = set()
        for row in conversation_rows:
            target = row.get("y")
            if not isinstance(target, list) or len(target) != 33:
                raise ValueError("all supervision rows must have aligned 33-label targets")
            presence.update(index for index, value in enumerate(target) if value == 1)
        conversation_presence[conversation_id] = presence
    label_supporting_conversations = {
        index: {conversation_id for conversation_id in conversations if index in conversation_presence[conversation_id]}
        for index in range(33)
    }
    desired_label_groups = {
        index: [len(label_supporting_conversations[index]) * fraction for fraction in fractions]
        for index in range(33)
    }
    assignment: dict[str, int] = {}
    partition_sizes = [0, 0, 0]
    assigned_label_groups = [[0] * 33 for _ in names]

    def place(conversation_id: str, partition_index: int) -> None:
        assignment[conversation_id] = partition_index
        partition_sizes[partition_index] += 1
        for label_index in conversation_presence[conversation_id]:
            assigned_label_groups[partition_index][label_index] += 1

    rare_labels = sorted(range(33), key=lambda index: (len(label_supporting_conversations[index]), index))
    for label_index in rare_labels:
        candidates = sorted(
            label_supporting_conversations[label_index] - assignment.keys(),
            key=lambda conversation_id: (tie_rank[conversation_id], conversation_id),
        )
        for conversation_id in candidates:
            available = [index for index in range(3) if partition_sizes[index] < capacities[index]]
            if not available:
                raise AssertionError("conversation split capacity exhausted before all groups were assigned")
            if len(label_supporting_conversations[label_index]) >= 3:
                missing = [index for index in available if assigned_label_groups[index][label_index] == 0]
            else:
                missing = []
            choices = missing or available
            if missing:
                partition_index = max(
                    choices,
                    key=lambda index: ((capacities[index] - partition_sizes[index]) / capacities[index], -index),
                )
            else:
                partition_index = max(
                    choices,
                    key=lambda index: (
                        (desired_label_groups[label_index][index] - assigned_label_groups[index][label_index])
                        / max(desired_label_groups[label_index][index], 1.0),
                        (capacities[index] - partition_sizes[index]) / capacities[index],
                        -index,
                    ),
                )
            place(conversation_id, partition_index)

    remaining_conversations = [conversation_id for conversation_id in conversations if conversation_id not in assignment]
    remaining_conversations.sort(key=lambda conversation_id: (tie_rank[conversation_id], conversation_id))
    for conversation_id in remaining_conversations:
        available = [index for index in range(3) if partition_sizes[index] < capacities[index]]
        partition_index = max(
            available,
            key=lambda index: (
                (capacities[index] - partition_sizes[index]) / capacities[index],
                -index,
            ),
        )
        place(conversation_id, partition_index)
    groups = {
        name: sorted(
            (conversation_id for conversation_id, partition_index in assignment.items() if partition_index == index),
            key=lambda conversation_id: (tie_rank[conversation_id], conversation_id),
        )
        for index, name in enumerate(names)
    }
    split_rows = {
        name: sorted(
            (row for conversation_id in conv_ids for row in by_conversation[conversation_id]),
            key=lambda row: str(row["sample_id"]),
        )
        for name, conv_ids in groups.items()
    }
    all_ids = [str(row["sample_id"]) for split in split_rows.values() for row in split]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != {str(row["sample_id"]) for row in rows}:
        raise ValueError("conversation split lost or duplicated sample IDs")
    conv_sets = {name: set(values) for name, values in groups.items()}
    for left, left_values in conv_sets.items():
        for right, right_values in conv_sets.items():
            if left < right and left_values & right_values:
                raise ValueError("conversation_id leaked across partitions")
    width = len(rows[0].get("y", []))
    if width != 33 or any(len(row.get("y", [])) != width for row in rows):
        raise ValueError("all supervision rows must have aligned 33-label targets")
    support = {}
    for name, partition in split_rows.items():
        positive = [0] * width
        for row in partition:
            for index, value in enumerate(row["y"]):
                positive[index] += int(value)
        support[name] = {
            "sample_count": len(partition),
            "conversation_count": len(groups[name]),
            "positive_counts": positive,
            "insufficient_support_indices": [index for index, count in enumerate(positive) if count == 0],
        }
    return {
        "version": "e03_a5_conversation_group_stratified_v1",
        "seed": seed,
        "fractions": {
            "fit": fit_fraction,
            "calibration": calibration_fraction,
            "internal_holdout": 1.0 - fit_fraction - calibration_fraction,
        },
        "splits": {
            name: {
                "conversation_ids": list(groups[name]),
                "sample_ids": [str(row["sample_id"]) for row in split_rows[name]],
                "sample_ids_sha256": canonical_sha256([str(row["sample_id"]) for row in split_rows[name]]),
                "conversation_ids_sha256": canonical_sha256(groups[name]),
            }
            for name in ("fit", "calibration", "internal_holdout")
        },
        "support": support,
        "no_conversation_leakage": True,
        "sample_ids_unique_and_complete": True,
    }


def target_counts(rows: Iterable[Mapping[str, Any]], width: int = 33) -> list[int]:
    counts = [0] * width
    for row in rows:
        target = row.get("y")
        if not isinstance(target, list) or len(target) != width:
            raise ValueError(f"expected a {width}-element y vector")
        for index, value in enumerate(target):
            if value not in (0, 1) or isinstance(value, bool):
                raise ValueError("y must contain binary integers")
            counts[index] += value
    return counts
