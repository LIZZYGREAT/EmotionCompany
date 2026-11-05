"""Paired 224-point profile/emotion summaries for the E03-A4 stability screen."""
from __future__ import annotations

import random
from collections import defaultdict
from math import ceil
from typing import Any, Mapping, Sequence

from src.evaluation.metrics import emotion_metrics, profile_metrics


PROFILE_DIMENSIONS = ("personality_traits", "interests", "style")
QUALITY_THRESHOLDS = {
    "profile_micro_f1_min_delta": -0.01,
    "micro_precision_min_delta": -0.02,
    "fp_max_relative_increase": 0.10,
    "emotion_accuracy_min_delta": -0.02,
}


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("bootstrap quantiles require at least one replicate")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _profile_summary(
    indices: Sequence[int],
    gold: Sequence[Mapping[str, Sequence[str]]],
    predicted: Sequence[Mapping[str, Sequence[str]]],
    enums: dict[str, list[str]],
) -> dict[str, Any]:
    result = profile_metrics(
        [dict(gold[index]) for index in indices],
        [dict(predicted[index]) for index in indices],
        enums,
    )
    return {
        "sample_count": result["per_dimension"][PROFILE_DIMENSIONS[0]]["sample_count"],
        "overall_33_label_tp": result["overall_33_label_tp"],
        "overall_33_label_fp": result["overall_33_label_fp"],
        "overall_33_label_fn": result["overall_33_label_fn"],
        "overall_33_label_micro_precision": result["overall_33_label_micro_precision"],
        "overall_33_label_micro_recall": result["overall_33_label_micro_recall"],
        "overall_33_label_micro_f1": result["overall_33_label_micro_f1"],
        "overall_33_label_macro_f1": result["overall_33_label_macro_f1"],
        "per_dimension": {
            dimension: {
                key: result["per_dimension"][dimension][key]
                for key in (
                    "local_micro_precision", "local_micro_recall", "local_micro_f1",
                    "tp", "fp", "fn",
                )
            }
            for dimension in PROFILE_DIMENSIONS
        },
    }


def _emotion_accuracy(
    indices: Sequence[int], gold: Sequence[str], predicted: Sequence[str], labels: list[str],
) -> float:
    metrics = emotion_metrics(
        [gold[index] for index in indices],
        [predicted[index] for index in indices],
        labels,
    )
    return float(metrics["accuracy"])


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float:
    return float(right[key]) - float(left[key])


def paired_stability_summary(
    *,
    ids: Sequence[str],
    conversation_ids: Sequence[str],
    history_lengths: Sequence[int],
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    baseline_profiles: Sequence[Mapping[str, Sequence[str]]],
    candidate_profiles: Sequence[Mapping[str, Sequence[str]]],
    gold_emotions: Sequence[str],
    baseline_emotions: Sequence[str],
    candidate_emotions: Sequence[str],
    profile_enums: dict[str, list[str]],
    emotion_labels: list[str],
    seed: int = 42,
    bootstrap_repetitions: int = 1000,
    short_history_max_messages: int = 4,
) -> dict[str, Any]:
    """Compare fixed-denominator predictions and resample complete conversations."""
    size = len(ids)
    aligned = (
        size > 0
        and all(len(values) == size for values in (
            conversation_ids, history_lengths, gold_profiles, baseline_profiles,
            candidate_profiles, gold_emotions, baseline_emotions, candidate_emotions,
        ))
        and len(set(ids)) == size
    )
    if not aligned:
        raise ValueError("A4 paired evaluation arrays must have the same length and unique IDs")
    if isinstance(bootstrap_repetitions, bool) or not isinstance(bootstrap_repetitions, int) or bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if any(not isinstance(value, str) or not value for value in conversation_ids):
        raise ValueError("every frozen evaluation ID must map to a conversation cluster")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in history_lengths):
        raise ValueError("history_lengths must be positive message counts")
    for name, profiles in (("Gold", gold_profiles), ("E01-B", baseline_profiles), ("A4", candidate_profiles)):
        for sample_id, profile in zip(ids, profiles):
            if not isinstance(profile, Mapping) or set(profile) != set(PROFILE_DIMENSIONS):
                raise ValueError(f"{name} profile for {sample_id} does not have the three frozen dimensions")
            for dimension in PROFILE_DIMENSIONS:
                labels = profile[dimension]
                if (
                    not isinstance(labels, Sequence)
                    or isinstance(labels, (str, bytes))
                    or any(not isinstance(label, str) or label not in profile_enums[dimension] for label in labels)
                    or len(labels) != len(set(labels))
                ):
                    raise ValueError(f"{name} profile for {sample_id} has invalid {dimension} labels")
    if any(label not in emotion_labels for label in (*gold_emotions, *baseline_emotions, *candidate_emotions)):
        raise ValueError("A4 paired emotion arrays contain a label outside the frozen schema enum")

    indices = list(range(size))
    baseline = _profile_summary(indices, gold_profiles, baseline_profiles, profile_enums)
    candidate = _profile_summary(indices, gold_profiles, candidate_profiles, profile_enums)
    baseline_emotion_accuracy = _emotion_accuracy(indices, gold_emotions, baseline_emotions, emotion_labels)
    candidate_emotion_accuracy = _emotion_accuracy(indices, gold_emotions, candidate_emotions, emotion_labels)
    profile_delta = {
        key: _delta(baseline, candidate, key)
        for key in (
            "overall_33_label_micro_precision", "overall_33_label_micro_recall",
            "overall_33_label_micro_f1", "overall_33_label_macro_f1",
        )
    }
    profile_delta.update({
        "overall_33_label_tp": candidate["overall_33_label_tp"] - baseline["overall_33_label_tp"],
        "overall_33_label_fp": candidate["overall_33_label_fp"] - baseline["overall_33_label_fp"],
        "overall_33_label_fn": candidate["overall_33_label_fn"] - baseline["overall_33_label_fn"],
    })
    profile_delta["per_dimension"] = {
        dimension: {
            key: _delta(baseline["per_dimension"][dimension], candidate["per_dimension"][dimension], key)
            if key.startswith("local_")
            else candidate["per_dimension"][dimension][key] - baseline["per_dimension"][dimension][key]
            for key in (
                "local_micro_precision", "local_micro_recall", "local_micro_f1", "tp", "fp", "fn",
            )
        }
        for dimension in PROFILE_DIMENSIONS
    }
    fp_increase_fraction = (
        (candidate["overall_33_label_fp"] - baseline["overall_33_label_fp"])
        / baseline["overall_33_label_fp"]
        if baseline["overall_33_label_fp"]
        else (0.0 if candidate["overall_33_label_fp"] == 0 else None)
    )

    short_indices = [index for index, length in enumerate(history_lengths) if length <= short_history_max_messages]
    long_indices = [index for index, length in enumerate(history_lengths) if length > short_history_max_messages]
    history_groups: dict[str, Any] = {}
    for group_name, group_indices in (("short_le_4_messages", short_indices), ("long_gt_4_messages", long_indices)):
        if not group_indices:
            raise ValueError(f"the frozen cohort has no samples in {group_name}")
        history_groups[group_name] = {
            "baseline": _profile_summary(group_indices, gold_profiles, baseline_profiles, profile_enums),
            "candidate": _profile_summary(group_indices, gold_profiles, candidate_profiles, profile_enums),
        }
        history_groups[group_name]["candidate_minus_baseline_micro_f1"] = _delta(
            history_groups[group_name]["baseline"],
            history_groups[group_name]["candidate"],
            "overall_33_label_micro_f1",
        )

    clusters: dict[str, list[int]] = defaultdict(list)
    for index, conversation_id in enumerate(conversation_ids):
        clusters[conversation_id].append(index)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    bootstrap_values: dict[str, list[float]] = {
        "micro_f1": [], "micro_precision": [], "micro_recall": [],
        "macro_f1": [], "fp_count_delta": [], "emotion_accuracy": [],
    }
    for _replicate in range(bootstrap_repetitions):
        selected_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        selected_indices = [index for cluster_id in selected_clusters for index in clusters[cluster_id]]
        left = _profile_summary(selected_indices, gold_profiles, baseline_profiles, profile_enums)
        right = _profile_summary(selected_indices, gold_profiles, candidate_profiles, profile_enums)
        bootstrap_values["micro_f1"].append(_delta(left, right, "overall_33_label_micro_f1"))
        bootstrap_values["micro_precision"].append(_delta(left, right, "overall_33_label_micro_precision"))
        bootstrap_values["micro_recall"].append(_delta(left, right, "overall_33_label_micro_recall"))
        bootstrap_values["macro_f1"].append(_delta(left, right, "overall_33_label_macro_f1"))
        bootstrap_values["fp_count_delta"].append(float(right["overall_33_label_fp"] - left["overall_33_label_fp"]))
        bootstrap_values["emotion_accuracy"].append(
            _emotion_accuracy(selected_indices, gold_emotions, candidate_emotions, emotion_labels)
            - _emotion_accuracy(selected_indices, gold_emotions, baseline_emotions, emotion_labels)
        )
    confidence_intervals = {
        metric: {
            "lower_95": _quantile(values, 0.025),
            "upper_95": _quantile(values, 0.975),
            "replicates": bootstrap_repetitions,
            "unit": "conversation_cluster",
        }
        for metric, values in bootstrap_values.items()
    }

    threshold_failures: list[str] = []
    if profile_delta["overall_33_label_micro_f1"] < QUALITY_THRESHOLDS["profile_micro_f1_min_delta"]:
        threshold_failures.append("profile_micro_f1_decline_over_0.01")
    if profile_delta["overall_33_label_micro_precision"] < QUALITY_THRESHOLDS["micro_precision_min_delta"]:
        threshold_failures.append("micro_precision_decline_over_0.02")
    fp_failed = (
        candidate["overall_33_label_fp"] > baseline["overall_33_label_fp"] * (1 + QUALITY_THRESHOLDS["fp_max_relative_increase"])
        if baseline["overall_33_label_fp"]
        else candidate["overall_33_label_fp"] > 0
    )
    if fp_failed:
        threshold_failures.append("false_positives_increased_over_10_percent")
    emotion_delta = candidate_emotion_accuracy - baseline_emotion_accuracy
    if emotion_delta < QUALITY_THRESHOLDS["emotion_accuracy_min_delta"]:
        threshold_failures.append("emotion_accuracy_decline_over_0.02")

    return {
        "status": "stability_failed" if threshold_failures else "stability_preliminarily_passed",
        "decision_scope": "engineering_stop_gate_on_one_frozen_224_point_dev_proxy; not a statistical non-inferiority proof",
        "sample_count": size,
        "conversation_count": len(clusters),
        "short_history_max_messages": short_history_max_messages,
        "baseline": {
            "profile": baseline,
            "emotion_accuracy_fixed_denominator": baseline_emotion_accuracy,
            "valid_coverage": 1.0,
        },
        "candidate": {
            "profile": candidate,
            "emotion_accuracy_fixed_denominator": candidate_emotion_accuracy,
            "valid_coverage": 1.0,
        },
        "candidate_minus_baseline": {
            "profile": profile_delta,
            "emotion_accuracy_fixed_denominator": emotion_delta,
            "false_positive_relative_increase": fp_increase_fraction,
        },
        "history_groups": history_groups,
        "conversation_cluster_bootstrap": {
            "seed": seed,
            "confidence_intervals": confidence_intervals,
            "interpretation": "descriptive paired uncertainty intervals for the frozen proxy cohort",
        },
        "quality_thresholds": dict(QUALITY_THRESHOLDS),
        "threshold_failures": threshold_failures,
        "full_json_and_reply_smoke": "not_run",
    }
