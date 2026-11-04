"""Pure aggregation helpers for the E03-A3 profile diagnosis.

The module accepts already joined, in-memory rows and emits aggregate counts.
It does not read files, load a model, tokenize text, or write sample-level data.
"""
from __future__ import annotations

import math
import random
import statistics
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


DIMENSIONS = ("personality_traits", "interests", "style")
LENGTH_BUCKETS = (
    (1, 4, "1-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, None, "17+"),
)
GOLD_COUNT_BUCKETS = ("0", "1", "2", "3+")
PROFILE_TRANSITIONS = ("fn_to_tp", "tp_to_fn", "fp_to_tn", "tn_to_fp")


def length_bucket(message_count: int) -> str:
    if isinstance(message_count, bool) or not isinstance(message_count, int) or message_count < 1:
        raise ValueError("history message count must be a positive integer")
    for lower, upper, name in LENGTH_BUCKETS:
        if message_count >= lower and (upper is None or message_count <= upper):
            return name
    raise AssertionError("unreachable history length")


def gold_count_bucket(label_count: int) -> str:
    if isinstance(label_count, bool) or not isinstance(label_count, int) or label_count < 0:
        raise ValueError("Gold label count must be a non-negative integer")
    return str(label_count) if label_count <= 2 else "3+"


def validate_history(history: Any) -> int:
    """Validate native history shape and return its message count."""
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a non-empty list")
    for index, message in enumerate(history):
        if not isinstance(message, Mapping):
            raise ValueError(f"history[{index}] must be an object")
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"history[{index}] has an invalid role")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"history[{index}] has non-string content")
    if history[-1].get("role") != "user":
        raise ValueError("history must include the current user as its last message")
    return len(history)


def join_by_id(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    left_key: str = "id",
    right_key: str = "id",
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """Join two row sets by an explicit ID, independent of their file order."""
    def index(rows: Sequence[Mapping[str, Any]], key: str, side: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{side} row has a missing/non-string join ID")
            if value in result:
                raise ValueError(f"{side} rows contain a duplicate join ID")
            result[value] = row
        return result

    left = index(left_rows, left_key, "left")
    right = index(right_rows, right_key, "right")
    if set(left) != set(right):
        raise ValueError("joined row ID sets differ")
    return [(key, left[key], right[key]) for key in left]


def freeze_conversation_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    target_points: int = 224,
    min_points: int = 192,
    max_points: int = 256,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Choose a deterministic, Gold/history-stratified Dev subset by conversation.

    Selection examines only IDs, conversation groups, history lengths, and Gold
    profile strata. Predictions and model metrics are deliberately ignored.
    Every prediction point from a selected conversation is kept together.
    """
    if not (0 < min_points <= target_points <= max_points):
        raise ValueError("evaluation subset bounds must satisfy min <= target <= max")
    by_conversation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        eval_id = row.get("eval_id")
        conversation_id = row.get("conversation_id")
        if not isinstance(eval_id, str) or not eval_id or eval_id in seen_ids:
            raise ValueError("evaluation rows must have unique non-empty eval IDs")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("evaluation rows must have non-empty conversation IDs")
        seen_ids.add(eval_id)
        by_conversation[conversation_id].append(row)
    if not by_conversation:
        raise ValueError("cannot freeze an empty evaluation set")

    flags: dict[str, set[str]] = {}
    priorities: dict[str, str] = {}
    for conversation_id, conversation_rows in by_conversation.items():
        present: set[str] = set()
        for row in conversation_rows:
            present.add("short_history" if row["history_length"] <= 4 else "long_history")
            tag_count = sum(len(row["gold"][dimension]) for dimension in DIMENSIONS)
            present.add(f"gold_count_{gold_count_bucket(tag_count)}")
            if tag_count == 0:
                present.add("gold_empty")
            else:
                present.add("gold_nonempty")
            for dimension in DIMENSIONS:
                if row["gold"][dimension]:
                    present.add(f"dimension_{dimension}")
        flags[conversation_id] = present
        priorities[conversation_id] = hashlib.sha256(f"{seed}:{conversation_id}".encode("utf-8")).hexdigest()

    required = {
        "short_history",
        "long_history",
        "gold_empty",
        "gold_nonempty",
        *(f"dimension_{dimension}" for dimension in DIMENSIONS),
    }
    available = set().union(*flags.values())
    missing_features = required - available
    if missing_features:
        raise ValueError(f"Dev does not contain required evaluation strata: {sorted(missing_features)}")

    selected_conversations: set[str] = set()
    covered: set[str] = set()
    while not required <= covered:
        missing = required - covered
        candidates = [
            conversation_id for conversation_id in by_conversation
            if conversation_id not in selected_conversations and flags[conversation_id] & missing
        ]
        if not candidates:
            raise ValueError("could not cover all required Dev evaluation strata")
        candidates.sort(key=lambda conversation_id: (
            -len(flags[conversation_id] & missing),
            len(by_conversation[conversation_id]),
            priorities[conversation_id],
        ))
        chosen = candidates[0]
        selected_conversations.add(chosen)
        covered |= flags[chosen]
        if sum(len(by_conversation[key]) for key in selected_conversations) > max_points:
            raise ValueError("required whole-conversation strata exceed the evaluation size limit")

    point_count = sum(len(by_conversation[key]) for key in selected_conversations)
    fill_order = sorted(
        (key for key in by_conversation if key not in selected_conversations),
        key=lambda key: priorities[key],
    )
    for conversation_id in fill_order:
        if point_count >= target_points and point_count >= min_points:
            break
        conversation_size = len(by_conversation[conversation_id])
        if point_count + conversation_size > max_points:
            continue
        selected_conversations.add(conversation_id)
        point_count += conversation_size
    if point_count < min_points or point_count > max_points:
        raise ValueError(f"whole-conversation Dev selection has {point_count} points outside the configured bounds")

    selected = [
        row for conversation_id in selected_conversations for row in by_conversation[conversation_id]
    ]
    selected.sort(key=lambda row: row["eval_id"])
    ids = [row["eval_id"] for row in selected]
    digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    summary = {
        "status": "frozen",
        "selection_method": "seeded_hash_order_whole_conversations_with_strata_coverage_v1",
        "seed": seed,
        "target_points": target_points,
        "min_points": min_points,
        "max_points": max_points,
        "selected_points": len(selected),
        "selected_conversations": len(selected_conversations),
        "ids_sha256": digest,
        "history_length_bucket_counts": dict(sorted(Counter(length_bucket(row["history_length"]) for row in selected).items())),
        "gold_all_empty_points": sum(not any(row["gold"][dimension] for dimension in DIMENSIONS) for row in selected),
        "gold_nonempty_points": sum(any(row["gold"][dimension] for dimension in DIMENSIONS) for row in selected),
        "gold_label_count_bucket_counts": dict(sorted(Counter(
            gold_count_bucket(sum(len(row["gold"][dimension]) for dimension in DIMENSIONS)) for row in selected
        ).items())),
        "dimension_nonempty_point_counts": {
            dimension: sum(bool(row["gold"][dimension]) for row in selected)
            for dimension in DIMENSIONS
        },
        "required_strata_covered": sorted(required),
        "selection_uses_predictions": False,
        "conversation_groups_split": False,
    }
    return selected, summary


def retained_message_count(original_count: int, removed_count: int) -> int:
    if isinstance(original_count, bool) or not isinstance(original_count, int) or original_count < 1:
        raise ValueError("original history message count must be positive")
    if isinstance(removed_count, bool) or not isinstance(removed_count, int):
        raise ValueError("removed history message count must be an integer")
    if removed_count < 0 or removed_count >= original_count:
        raise ValueError("T1 must retain the current user message")
    return original_count - removed_count


def validate_profile(profile: Any, label_enums: Mapping[str, Sequence[str]], *, field: str = "profile") -> dict[str, set[str]]:
    if not isinstance(profile, Mapping) or set(profile) != set(DIMENSIONS):
        raise ValueError(f"{field} must contain exactly the three profile dimensions")
    normalized: dict[str, set[str]] = {}
    for dimension in DIMENSIONS:
        values = profile[dimension]
        if not isinstance(values, list) or any(not isinstance(label, str) for label in values):
            raise ValueError(f"{field}.{dimension} must be a list of strings")
        if len(values) != len(set(values)):
            raise ValueError(f"{field}.{dimension} contains duplicate labels")
        allowed = set(label_enums[dimension])
        if not set(values) <= allowed:
            raise ValueError(f"{field}.{dimension} contains a label outside the fixed enum")
        normalized[dimension] = set(values)
    return normalized


def _f1(tp: int | float, fp: int | float, fn: int | float) -> float:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _distribution(values: Iterable[int | float]) -> dict[str, int | float | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "mean": None, "min": None, "p50": None, "max": None}
    ordered = sorted(numbers)
    p50 = ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)]
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "min": min(numbers),
        "p50": p50,
        "max": max(numbers),
    }


def _profile_counts(
    rows: Sequence[Mapping[str, Any]],
    label_enums: Mapping[str, Sequence[str]],
    *,
    gold_key: str = "gold",
    pred_key: str | None = "pred",
) -> dict[str, Any]:
    label_counts = {
        dimension: {
            label: {"gold_support": 0, "predicted_count": 0, "tp": 0, "fp": 0, "fn": 0}
            for label in label_enums[dimension]
        }
        for dimension in DIMENSIONS
    }
    dimension_counts = {
        dimension: {
            "gold_nonempty_points": 0,
            "pred_nonempty_points": 0,
            "gold_nonempty_pred_empty_points": 0,
            "gold_nonempty_pred_no_hit_points": 0,
            "exact_match_points": 0,
            "gold_label_count": 0,
            "predicted_label_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for dimension in DIMENSIONS
    }
    gold_tag_histogram: Counter[str] = Counter()
    predicted_tag_histogram: Counter[str] = Counter()
    total_tp = total_fp = total_fn = 0
    gold_all_empty = pred_all_empty = gold_nonempty_pred_all_empty = 0
    overall_exact = 0

    for row in rows:
        gold = row[gold_key]
        gold_count = sum(len(gold[dimension]) for dimension in DIMENSIONS)
        gold_tag_histogram[gold_count_bucket(gold_count)] += 1
        gold_all_empty += gold_count == 0
        pred = row.get(pred_key) if pred_key is not None else None
        if pred is not None:
            predicted_count = sum(len(pred[dimension]) for dimension in DIMENSIONS)
            predicted_tag_histogram[gold_count_bucket(predicted_count)] += 1
            pred_all_empty += predicted_count == 0
            gold_nonempty_pred_all_empty += gold_count > 0 and predicted_count == 0
            overall_exact += all(gold[dimension] == pred[dimension] for dimension in DIMENSIONS)

        for dimension in DIMENSIONS:
            counts = dimension_counts[dimension]
            expected = gold[dimension]
            counts["gold_label_count"] += len(expected)
            counts["gold_nonempty_points"] += bool(expected)
            if pred is None:
                for label in label_enums[dimension]:
                    label_counts[dimension][label]["gold_support"] += label in expected
                continue

            actual = pred[dimension]
            counts["predicted_label_count"] += len(actual)
            counts["pred_nonempty_points"] += bool(actual)
            counts["gold_nonempty_pred_empty_points"] += bool(expected) and not actual
            counts["gold_nonempty_pred_no_hit_points"] += bool(expected) and not bool(expected & actual)
            counts["exact_match_points"] += expected == actual
            for label in label_enums[dimension]:
                item = label_counts[dimension][label]
                has_gold = label in expected
                has_pred = label in actual
                item["gold_support"] += has_gold
                item["predicted_count"] += has_pred
                item["tp"] += has_gold and has_pred
                item["fp"] += not has_gold and has_pred
                item["fn"] += has_gold and not has_pred
                counts["tp"] += has_gold and has_pred
                counts["fp"] += not has_gold and has_pred
                counts["fn"] += has_gold and not has_pred

    if pred_key is not None:
        for dimension in DIMENSIONS:
            total_tp += dimension_counts[dimension]["tp"]
            total_fp += dimension_counts[dimension]["fp"]
            total_fn += dimension_counts[dimension]["fn"]
    dimensions: dict[str, Any] = {}
    for dimension, counts in dimension_counts.items():
        support = counts["gold_label_count"]
        predictions = counts["predicted_label_count"]
        dim_labels = label_counts[dimension]
        dim_tp, dim_fp, dim_fn = counts["tp"], counts["fp"], counts["fn"]
        dim_f1_values = [_f1(x["tp"], x["fp"], x["fn"]) for x in dim_labels.values()]
        dimensions[dimension] = {
            **counts,
            "gold_nonempty_rate": counts["gold_nonempty_points"] / len(rows) if rows else 0.0,
            "pred_nonempty_rate": counts["pred_nonempty_points"] / len(rows) if rows and pred_key is not None else None,
            "gold_nonempty_pred_empty_rate": (
                counts["gold_nonempty_pred_empty_points"] / counts["gold_nonempty_points"]
                if counts["gold_nonempty_points"] and pred_key is not None else None
            ),
            "gold_nonempty_pred_no_hit_rate": (
                counts["gold_nonempty_pred_no_hit_points"] / counts["gold_nonempty_points"]
                if counts["gold_nonempty_points"] and pred_key is not None else None
            ),
            "micro_precision": dim_tp / (dim_tp + dim_fp) if dim_tp + dim_fp else 0.0,
            "micro_recall": dim_tp / support if support else 0.0,
            "micro_f1": _f1(dim_tp, dim_fp, dim_fn),
            "macro_f1": statistics.fmean(dim_f1_values) if dim_f1_values else 0.0,
            "exact_match_rate": counts["exact_match_points"] / len(rows) if rows and pred_key is not None else None,
            "labels": dim_labels,
        }

    total_gold = sum(dimension_counts[dimension]["gold_label_count"] for dimension in DIMENSIONS)
    return {
        "point_count": len(rows),
        "conversation_count": len({row["conversation_id"] for row in rows}),
        "target_turn_index": _distribution(
            row["target_turn_index"] for row in rows if isinstance(row.get("target_turn_index"), int)
        ),
        "history_message_count": _distribution(row["history_length"] for row in rows),
        "t1_retained_message_count": _distribution(row["retained_history_length"] for row in rows),
        "prompt_token_count": _distribution(
            row["prompt_tokens"] for row in rows if isinstance(row.get("prompt_tokens"), int)
        ),
        "t1_clipped_points": sum(row["removed_history_messages"] > 0 for row in rows),
        "t1_clipped_rate": (
            sum(row["removed_history_messages"] > 0 for row in rows) / len(rows) if rows else 0.0
        ),
        "gold_all_empty_points": gold_all_empty,
        "gold_all_empty_rate": gold_all_empty / len(rows) if rows else 0.0,
        "gold_nonempty_points": len(rows) - gold_all_empty,
        "gold_label_count_distribution": {key: gold_tag_histogram.get(key, 0) for key in GOLD_COUNT_BUCKETS},
        "gold_average_label_count": total_gold / len(rows) if rows else 0.0,
        "pred_all_empty_points": pred_all_empty if pred_key is not None else None,
        "pred_all_empty_rate": pred_all_empty / len(rows) if rows and pred_key is not None else None,
        "gold_nonempty_pred_all_empty_points": gold_nonempty_pred_all_empty if pred_key is not None else None,
        "gold_nonempty_pred_all_empty_rate": (
            gold_nonempty_pred_all_empty / (len(rows) - gold_all_empty)
            if pred_key is not None and len(rows) > gold_all_empty else None
        ),
        "predicted_label_count_distribution": (
            {key: predicted_tag_histogram.get(key, 0) for key in GOLD_COUNT_BUCKETS}
            if pred_key is not None else None
        ),
        "predicted_average_label_count": (
            sum(dimension_counts[d]["predicted_label_count"] for d in DIMENSIONS) / len(rows)
            if rows and pred_key is not None else None
        ),
        "profile_tp": total_tp if pred_key is not None else None,
        "profile_fp": total_fp if pred_key is not None else None,
        "profile_fn": total_fn if pred_key is not None else None,
        "profile_micro_precision": total_tp / (total_tp + total_fp) if pred_key is not None and total_tp + total_fp else (0.0 if pred_key is not None else None),
        "profile_micro_recall": total_tp / total_gold if pred_key is not None and total_gold else (0.0 if pred_key is not None else None),
        "profile_micro_f1": _f1(total_tp, total_fp, total_fn) if pred_key is not None else None,
        "profile_macro_f1": (
            statistics.fmean(
                _f1(item["tp"], item["fp"], item["fn"])
                for dimension in DIMENSIONS
                for item in label_counts[dimension].values()
            ) if pred_key is not None else None
        ),
        "profile_exact_match_points": overall_exact if pred_key is not None else None,
        "profile_exact_match_rate": overall_exact / len(rows) if rows and pred_key is not None else None,
        "dimensions": dimensions,
    }


def length_breakdown(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    label_enums: Mapping[str, Sequence[str]],
    *,
    length_field: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for _, _, bucket in LENGTH_BUCKETS:
        train_group = [row for row in train_rows if length_bucket(row[length_field]) == bucket]
        dev_group = [row for row in dev_rows if length_bucket(row[length_field]) == bucket]
        strata: dict[str, Any] = {}
        for gold_bucket in GOLD_COUNT_BUCKETS:
            group = [
                row for row in dev_group
                if gold_count_bucket(sum(len(row["gold"][d]) for d in DIMENSIONS)) == gold_bucket
            ]
            strata[gold_bucket] = {
                "point_count": len(group),
                "conversation_count": len({row["conversation_id"] for row in group}),
                "dimensions": _profile_counts(group, label_enums)["dimensions"] if group else None,
            }
        output[bucket] = {
            "train": _profile_counts(train_group, label_enums, pred_key=None),
            "dev": _profile_counts(dev_group, label_enums),
            "dev_by_gold_label_count": strata,
        }
    return output


def clipping_breakdown(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    label_enums: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for clipped in (False, True):
        key = "clipped" if clipped else "not_clipped"
        train_group = [row for row in train_rows if (row["removed_history_messages"] > 0) == clipped]
        dev_group = [row for row in dev_rows if (row["removed_history_messages"] > 0) == clipped]
        output[key] = {
            "train": _profile_counts(train_group, label_enums, pred_key=None),
            "dev": _profile_counts(dev_group, label_enums),
        }
    return output


def transition_counts(
    rows: Sequence[Mapping[str, Any]],
    label_enums: Mapping[str, Sequence[str]],
    *,
    before_key: str,
    after_key: str,
    group_field: str | None = None,
) -> dict[str, Any]:
    def empty() -> dict[str, dict[str, Counter[str]]]:
        return {
            dimension: {
                label: Counter({transition: 0 for transition in PROFILE_TRANSITIONS})
                for label in label_enums[dimension]
            }
            for dimension in DIMENSIONS
        }

    def add_counts(target: dict[str, dict[str, Counter[str]]], row: Mapping[str, Any]) -> None:
        before, after, gold = row[before_key], row[after_key], row["gold"]
        for dimension in DIMENSIONS:
            for label in label_enums[dimension]:
                was = label in before[dimension]
                now = label in after[dimension]
                positive = label in gold[dimension]
                if positive and not was and now:
                    target[dimension][label]["fn_to_tp"] += 1
                elif positive and was and not now:
                    target[dimension][label]["tp_to_fn"] += 1
                elif not positive and was and not now:
                    target[dimension][label]["fp_to_tn"] += 1
                elif not positive and not was and now:
                    target[dimension][label]["tn_to_fp"] += 1

    total = empty()
    grouped: dict[str, dict[str, dict[str, Counter[str]]]] = {}
    group_counts: Counter[str] = Counter()
    for row in rows:
        add_counts(total, row)
        if group_field is not None:
            name = row[group_field]
            if name not in grouped:
                grouped[name] = empty()
            add_counts(grouped[name], row)
            group_counts[name] += 1

    def render(counts: dict[str, dict[str, Counter[str]]]) -> dict[str, Any]:
        return {
            dimension: {
                label: {name: int(value) for name, value in sorted(transitions.items())}
                for label, transitions in labels.items()
            }
            for dimension, labels in counts.items()
        }

    return {
        "overall": render(total),
        "by_group": {
            key: {"point_count": group_counts[key], "labels": render(value)}
            for key, value in sorted(grouped.items())
        },
    }


def opportunity_space(
    *,
    total_tp: int,
    total_fp: int,
    total_fn: int,
    group_fn: int,
    recovery_fractions: Sequence[float] = (0.25, 0.50, 1.0),
    added_fp_per_recovered_tp: Sequence[float] = (0.0, 0.5, 1.0),
) -> dict[str, Any]:
    if min(total_tp, total_fp, total_fn, group_fn) < 0 or group_fn > total_fn:
        raise ValueError("TP/FP/FN and subgroup FN counts are inconsistent")
    scenarios: list[dict[str, Any]] = []
    for fraction in recovery_fractions:
        if not 0 <= fraction <= 1:
            raise ValueError("recovery fraction must be between zero and one")
        recovered = group_fn * fraction
        for fp_ratio in added_fp_per_recovered_tp:
            if fp_ratio < 0:
                raise ValueError("added FP ratio must be non-negative")
            extra_fp = recovered * fp_ratio
            scenarios.append({
                "fraction_of_group_fn_recovered": fraction,
                "recovered_tp_theoretical": recovered,
                "added_fp_per_recovered_tp": fp_ratio,
                "added_fp_theoretical": extra_fp,
                "micro_f1": _f1(total_tp + recovered, total_fp + extra_fp, total_fn - recovered),
            })
    gold_positives = total_tp + total_fn
    target_tp = math.ceil((total_fp + gold_positives) / 3)
    return {
        "group_fn": group_fn,
        "total_gold_positive_labels": gold_positives,
        "required_additional_tp_for_micro_f1_0_50_if_fp_fixed": max(0, target_tp - total_tp),
        "scenarios": scenarios,
    }


def cluster_bootstrap_recall_gap(
    rows: Sequence[Mapping[str, Any]],
    *,
    short_history_max: int = 4,
    repetitions: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Conversation-cluster bootstrap for short-minus-long profile label recall."""
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    by_conversation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_conversation[row["conversation_id"]].append(row)
    conversation_ids = sorted(by_conversation)

    def collect(sampled_ids: Sequence[str]) -> tuple[int, int, int, int]:
        short_tp = short_fn = long_tp = long_fn = 0
        for conversation_id in sampled_ids:
            for row in by_conversation[conversation_id]:
                is_short = row["history_length"] <= short_history_max
                for dimension in DIMENSIONS:
                    gold, pred = row["gold"][dimension], row["pred"][dimension]
                    tp = len(gold & pred)
                    fn = len(gold - pred)
                    if is_short:
                        short_tp += tp
                        short_fn += fn
                    else:
                        long_tp += tp
                        long_fn += fn
        return short_tp, short_fn, long_tp, long_fn

    original = collect(conversation_ids)
    short_recall = original[0] / (original[0] + original[1]) if original[0] + original[1] else 0.0
    long_recall = original[2] / (original[2] + original[3]) if original[2] + original[3] else 0.0
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(repetitions):
        sample = [rng.choice(conversation_ids) for _ in conversation_ids]
        short_tp, short_fn, long_tp, long_fn = collect(sample)
        if short_tp + short_fn and long_tp + long_fn:
            differences.append(short_tp / (short_tp + short_fn) - long_tp / (long_tp + long_fn))
    if not differences:
        interval = None
    else:
        ordered = sorted(differences)
        low = ordered[max(0, math.ceil(0.025 * len(ordered)) - 1)]
        high = ordered[max(0, math.ceil(0.975 * len(ordered)) - 1)]
        interval = [low, high]
    return {
        "estimand": "micro_label_recall_short_minus_long",
        "short_history_max_messages": short_history_max,
        "seed": seed,
        "requested_repetitions": repetitions,
        "valid_repetitions": len(differences),
        "conversation_count": len(conversation_ids),
        "short_recall": short_recall,
        "long_recall": long_recall,
        "difference": short_recall - long_recall,
        "percentile_95_interval": interval,
        "interpretation": "descriptive_uncertainty_only_not_a_causal_test",
    }
