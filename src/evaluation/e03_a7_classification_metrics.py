"""Fixed-cohort Profile/Emotion scoring and paired cluster summaries for E03-A7."""

from __future__ import annotations

import random
from collections import defaultdict
from math import ceil
from typing import Any, Mapping, Sequence

from src.evaluation.metrics import emotion_metrics, profile_metrics


DIMENSIONS = ("personality_traits", "interests", "style")


def _counts(probabilities: Sequence[Sequence[float]], targets: Sequence[Sequence[int]], threshold: float) -> dict[str, Any]:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("calibration arrays must have equal non-zero row counts")
    tp = fp = fn = 0
    for probs, gold in zip(probabilities, targets):
        if len(probs) != 33 or len(gold) != 33:
            raise ValueError("Profile probability and target rows must have 33 values")
        for probability, target in zip(probs, gold):
            if not 0 <= float(probability) <= 1 or target not in (0, 1):
                raise ValueError("Profile calibration values must be probabilities and binary targets")
            predicted = float(probability) >= threshold
            tp += predicted and target == 1
            fp += predicted and target == 0
            fn += not predicted and target == 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn), "micro_precision": precision, "micro_recall": recall, "micro_f1": f1}


def choose_calibration_threshold(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[int]],
    *,
    thresholds: Sequence[float],
    minimum_precision: float,
) -> dict[str, Any]:
    """Choose a single global threshold from Train-Calibration only."""
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("A7 threshold grid must contain probabilities strictly between zero and one")
    candidates = [_counts(probabilities, targets, float(value)) for value in sorted(set(thresholds))]
    eligible = [row for row in candidates if row["micro_precision"] >= minimum_precision]
    if not eligible:
        return {
            "status": "threshold_constraint_unsatisfied",
            "threshold": None,
            "minimum_precision": float(minimum_precision),
            "threshold_grid": list(sorted(set(float(value) for value in thresholds))),
            "best_observed_precision": max(row["micro_precision"] for row in candidates),
            "candidates": candidates,
        }
    best = max(eligible, key=lambda row: (row["micro_f1"], row["micro_precision"], row["threshold"]))
    return {
        "status": "calibration_threshold_selected",
        "threshold": best["threshold"],
        "minimum_precision": float(minimum_precision),
        "threshold_grid": list(sorted(set(float(value) for value in thresholds))),
        "selection_rule": "maximize_micro_f1_then_precision_then_higher_threshold",
        "selected_metrics": best,
        "candidates": candidates,
    }


def profile_from_probabilities(
    probabilities: Sequence[Sequence[float]],
    *,
    threshold: float,
    enums: Mapping[str, Sequence[str]],
) -> list[dict[str, list[str]]]:
    if not 0 <= threshold <= 1:
        raise ValueError("Profile threshold must be in [0, 1]")
    if sum(len(enums[name]) for name in DIMENSIONS) != 33:
        raise ValueError("official Profile schema must contain exactly 33 labels")
    predictions: list[dict[str, list[str]]] = []
    for row in probabilities:
        if len(row) != 33:
            raise ValueError("Profile probability rows must have 33 labels")
        profile: dict[str, list[str]] = {}
        cursor = 0
        for dimension in DIMENSIONS:
            labels = list(enums[dimension])
            values = row[cursor : cursor + len(labels)]
            if any(not 0 <= float(value) <= 1 for value in values):
                raise ValueError("Profile values must be probabilities in [0, 1]")
            profile[dimension] = [label for label, value in zip(labels, values) if float(value) >= threshold]
            cursor += len(labels)
        predictions.append(profile)
    return predictions


def average_precision_by_label(
    probabilities: Sequence[Sequence[float]], targets: Sequence[Sequence[int]], labels: Sequence[str],
) -> dict[str, Any]:
    """Compute non-interpolated AP with ties grouped at each unique probability."""
    if len(probabilities) != len(targets) or not probabilities or len(labels) != 33:
        raise ValueError("AP arrays require aligned rows and 33 frozen label names")
    columns = [[] for _ in range(33)]
    for probs, gold in zip(probabilities, targets):
        if len(probs) != 33 or len(gold) != 33:
            raise ValueError("AP rows must have 33 values")
        for index, (score, target) in enumerate(zip(probs, gold)):
            columns[index].append((float(score), int(target)))

    result: dict[str, Any] = {}
    for index, label in enumerate(labels):
        support = sum(target for _score, target in columns[index])
        if support == 0:
            result[label] = {"status": "insufficient_support", "support": 0, "average_precision": None}
            continue
        ordered = sorted(columns[index], key=lambda item: -item[0])
        seen = true_positive = 0
        area = 0.0
        cursor = 0
        while cursor < len(ordered):
            score = ordered[cursor][0]
            end = cursor
            group_positive = 0
            while end < len(ordered) and ordered[end][0] == score:
                group_positive += ordered[end][1]
                end += 1
            seen += end - cursor
            true_positive += group_positive
            area += (group_positive / support) * (true_positive / seen)
            cursor = end
        result[label] = {"status": "computed", "support": support, "average_precision": area}
    flat = [item for column in columns for item in column]
    support = sum(target for _score, target in flat)
    if support:
        ordered = sorted(flat, key=lambda item: -item[0])
        seen = true_positive = 0
        area = 0.0
        cursor = 0
        while cursor < len(ordered):
            score = ordered[cursor][0]
            end = cursor
            group_positive = 0
            while end < len(ordered) and ordered[end][0] == score:
                group_positive += ordered[end][1]
                end += 1
            seen += end - cursor
            true_positive += group_positive
            area += (group_positive / support) * (true_positive / seen)
            cursor = end
        micro = area
    else:
        micro = None
    return {"micro_average_precision": micro, "labels": result}


def summarize_profile_pair(
    *,
    ids: Sequence[str],
    conversation_ids: Sequence[str],
    gold: Sequence[Mapping[str, Sequence[str]]],
    baseline: Sequence[Mapping[str, Sequence[str]]],
    candidate: Sequence[Mapping[str, Sequence[str]]],
    enums: Mapping[str, Sequence[str]],
    seed: int = 42,
    bootstrap_repetitions: int = 1000,
) -> dict[str, Any]:
    """Compare a fixed cohort and bootstrap complete conversations."""
    size = len(ids)
    if (
        size == 0 or len(set(ids)) != size
        or any(len(values) != size for values in (conversation_ids, gold, baseline, candidate))
        or any(not isinstance(value, str) or not value for value in conversation_ids)
    ):
        raise ValueError("A7 paired Profile rows must align to unique IDs and conversation clusters")
    if bootstrap_repetitions < 1:
        raise ValueError("A7 paired bootstrap repetitions must be positive")
    base_metric = profile_metrics(list(gold), list(baseline), {key: list(enums[key]) for key in DIMENSIONS})
    cand_metric = profile_metrics(list(gold), list(candidate), {key: list(enums[key]) for key in DIMENSIONS})

    clusters: dict[str, list[int]] = defaultdict(list)
    for index, conversation_id in enumerate(conversation_ids):
        clusters[conversation_id].append(index)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {"micro_f1": [], "micro_precision": [], "micro_recall": [], "macro_f1": []}
    for _replicate in range(bootstrap_repetitions):
        selected_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        selected = [index for cluster_id in selected_clusters for index in clusters[cluster_id]]
        left = profile_metrics(
            [gold[index] for index in selected], [baseline[index] for index in selected],
            {key: list(enums[key]) for key in DIMENSIONS},
        )
        right = profile_metrics(
            [gold[index] for index in selected], [candidate[index] for index in selected],
            {key: list(enums[key]) for key in DIMENSIONS},
        )
        for key, field in (
            ("micro_f1", "overall_33_label_micro_f1"),
            ("micro_precision", "overall_33_label_micro_precision"),
            ("micro_recall", "overall_33_label_micro_recall"),
            ("macro_f1", "overall_33_label_macro_f1"),
        ):
            deltas[key].append(float(right[field] - left[field]))

    def quantile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        return float(ordered[max(0, min(len(ordered) - 1, ceil(probability * len(ordered)) - 1))])

    report = {
        "sample_count": size,
        "conversation_count": len(cluster_ids),
        "baseline": base_metric,
        "candidate": cand_metric,
        "candidate_minus_baseline": {
            "micro_precision": cand_metric["overall_33_label_micro_precision"] - base_metric["overall_33_label_micro_precision"],
            "micro_recall": cand_metric["overall_33_label_micro_recall"] - base_metric["overall_33_label_micro_recall"],
            "micro_f1": cand_metric["overall_33_label_micro_f1"] - base_metric["overall_33_label_micro_f1"],
            "macro_f1": cand_metric["overall_33_label_macro_f1"] - base_metric["overall_33_label_macro_f1"],
            "tp": cand_metric["overall_33_label_tp"] - base_metric["overall_33_label_tp"],
            "fp": cand_metric["overall_33_label_fp"] - base_metric["overall_33_label_fp"],
            "fn": cand_metric["overall_33_label_fn"] - base_metric["overall_33_label_fn"],
        },
        "cluster_bootstrap": {
            "unit": "complete_conversation",
            "seed": seed,
            "repetitions": bootstrap_repetitions,
            "candidate_minus_baseline_delta": {
                key: {"mean": sum(values) / len(values), "ci_2_5": quantile(values, 0.025), "ci_97_5": quantile(values, 0.975)}
                for key, values in deltas.items()
            },
        },
    }
    return report


def compare_arm_profile_predictions(
    *,
    ids: Sequence[str],
    gold: Sequence[Mapping[str, Sequence[str]]],
    p_predictions: Sequence[Mapping[str, Sequence[str]]],
    ep_predictions: Sequence[Mapping[str, Sequence[str]]],
    enums: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not (len(ids) == len(gold) == len(p_predictions) == len(ep_predictions)):
        raise ValueError("A7 P/EP comparison arrays must align")
    p_correct_ep_wrong: dict[str, int] = defaultdict(int)
    ep_correct_p_wrong: dict[str, int] = defaultdict(int)
    p_metric = profile_metrics(list(gold), list(p_predictions), {key: list(enums[key]) for key in DIMENSIONS})
    ep_metric = profile_metrics(list(gold), list(ep_predictions), {key: list(enums[key]) for key in DIMENSIONS})
    p_tp = p_metric["overall_33_label_tp"]
    ep_tp = ep_metric["overall_33_label_tp"]
    p_fp = p_metric["overall_33_label_fp"]
    ep_fp = ep_metric["overall_33_label_fp"]
    for gold_row, p_row, ep_row in zip(gold, p_predictions, ep_predictions):
        for dimension in DIMENSIONS:
            for label in enums[dimension]:
                expected = label in gold_row[dimension]
                p_value = label in p_row[dimension]
                ep_value = label in ep_row[dimension]
                if p_value == expected and ep_value != expected:
                    p_correct_ep_wrong[f"{dimension}:{label}"] += 1
                if ep_value == expected and p_value != expected:
                    ep_correct_p_wrong[f"{dimension}:{label}"] += 1
    return {
        "sample_count": len(ids),
        "p_profile_metrics": p_metric,
        "ep_profile_metrics": ep_metric,
        "ep_minus_p": {
            "micro_f1": ep_metric["overall_33_label_micro_f1"] - p_metric["overall_33_label_micro_f1"],
            "micro_precision": ep_metric["overall_33_label_micro_precision"] - p_metric["overall_33_label_micro_precision"],
            "micro_recall": ep_metric["overall_33_label_micro_recall"] - p_metric["overall_33_label_micro_recall"],
            "tp": ep_tp - p_tp,
            "fp": ep_fp - p_fp,
            "fn": ep_metric["overall_33_label_fn"] - p_metric["overall_33_label_fn"],
        },
        "p_correct_ep_wrong_label_counts": dict(sorted(p_correct_ep_wrong.items())),
        "ep_correct_p_wrong_label_counts": dict(sorted(ep_correct_p_wrong.items())),
        "profile_only_emotion_metrics": "not_applicable",
    }


def emotion_summary(gold: Sequence[str], logits: Sequence[Sequence[float]], labels: Sequence[str]) -> dict[str, Any]:
    if len(gold) != len(logits):
        raise ValueError("EP emotion rows do not align")
    if any(len(row) != len(labels) for row in logits):
        raise ValueError("EP emotion logits do not align to the frozen enum")
    predictions = [labels[max(range(len(row)), key=lambda index: float(row[index]))] for row in logits]
    return {"predictions": predictions, "metrics": emotion_metrics(list(gold), predictions, list(labels))}
