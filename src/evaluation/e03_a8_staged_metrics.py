"""Fixed-denominator A8 classification metrics and conversation-cluster bootstrap."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Sequence

from src.evaluation.metrics import emotion_metrics, profile_metrics

PROFILE_DIMS = ("personality_traits", "interests", "style")


def score_a8_cohort(
    cohort: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    enums: Mapping[str, list[str]],
    *,
    arm: str,
) -> dict[str, Any]:
    ids = list(cohort["ids"])
    if set(ids) - set(references) or set(ids) - set(predictions):
        raise ValueError("A8 fixed cohort is missing a Reference or prediction")
    gold_targets = [references[eval_id]["target"] for eval_id in ids]
    rows = [predictions[eval_id] for eval_id in ids]
    valid = [row.get("status") == "valid" and isinstance(row.get("parsed"), Mapping) for row in rows]
    gold_profiles = [
        {dim: list(target["user_profile"][dim]) for dim in PROFILE_DIMS}
        for target in gold_targets
    ]
    predicted_profiles = []
    predicted_emotions = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    eos_count = 0
    generated_tokens = []
    nonempty_count = 0
    nonempty_by_dim = Counter()
    false_positive_labels_on_empty_gold = 0
    for target, row, is_valid in zip(gold_targets, rows, valid):
        parsed = row.get("parsed") if is_valid else None
        profile = parsed.get("user_profile") if isinstance(parsed, Mapping) else None
        if not isinstance(profile, Mapping) or any(not isinstance(profile.get(dim), list) for dim in PROFILE_DIMS):
            profile = {dim: [] for dim in PROFILE_DIMS}
        normalized = {dim: list(profile[dim]) for dim in PROFILE_DIMS}
        predicted_profiles.append(normalized)
        has_profile = any(normalized[dim] for dim in PROFILE_DIMS)
        nonempty_count += has_profile
        for dim in PROFILE_DIMS:
            nonempty_by_dim[dim] += bool(normalized[dim])
        if not any(target["user_profile"][dim] for dim in PROFILE_DIMS):
            false_positive_labels_on_empty_gold += sum(len(normalized[dim]) for dim in PROFILE_DIMS)
        predicted_emotions.append(str(parsed.get("emotion_label", "")) if isinstance(parsed, Mapping) else "")
        status_counts[str(row.get("status", "missing"))] += 1
        if row.get("reason"):
            reason_counts[str(row["reason"])] += 1
        eos_count += bool(row.get("eos_generated"))
        generated_tokens.append(int(row.get("generated_token_count", 0)))

    profile = profile_metrics(gold_profiles, predicted_profiles, dict(enums))
    emotion = (
        emotion_metrics(
            [str(target["emotion_label"]) for target in gold_targets],
            predicted_emotions,
            list(enums["emotion_label"]),
        )
        if arm == "EP"
        else {"status": "N/A", "reason": "P has no Emotion target"}
    )
    label_errors = {}
    for dim in PROFILE_DIMS:
        per_label = profile["per_dimension"][dim]["per_label"]
        label_errors[dim] = {
            "largest_false_positives": sorted(
                ({"label": label, **facts} for label, facts in per_label.items()),
                key=lambda value: (-int(value["fp"]), value["label"]),
            )[:5],
            "largest_false_negatives": sorted(
                ({"label": label, **facts} for label, facts in per_label.items()),
                key=lambda value: (-int(value["fn"]), value["label"]),
            )[:5],
        }
    total = len(ids)
    return {
        "cohort_name": cohort["name"],
        "fixed_denominator_count": total,
        "ids_sha256": cohort["ids_sha256"],
        "conversation_count": cohort["conversation_count"],
        "valid_count": sum(valid),
        "strict_parse_rate": sum(valid) / total if total else 0.0,
        "eos_completion_count": eos_count,
        "eos_completion_rate": eos_count / total if total else 0.0,
        "truncation_or_missing_eos_count": sum(not bool(row.get("eos_generated")) for row in rows),
        "status_counts": dict(sorted(status_counts.items())),
        "error_reason_counts": dict(sorted(reason_counts.items())),
        "profile_33_label": profile,
        "profile_dimensions": profile["per_dimension"],
        "nonempty_profile_prediction_count": nonempty_count,
        "empty_profile_prediction_count": total - nonempty_count,
        "empty_profile_prediction_rate": (total - nonempty_count) / total if total else 0.0,
        "predicted_nonempty_by_dimension": dict(nonempty_by_dim),
        "false_positive_labels_on_gold_empty_profiles": false_positive_labels_on_empty_gold,
        "predicted_profile_label_count_per_sample": {
            "total": sum(sum(len(profile[dim]) for dim in PROFILE_DIMS) for profile in predicted_profiles),
            "mean": sum(sum(len(profile[dim]) for dim in PROFILE_DIMS) for profile in predicted_profiles) / total if total else 0.0,
            "max": max((sum(len(profile[dim]) for dim in PROFILE_DIMS) for profile in predicted_profiles), default=0),
        },
        "gold_nonempty_profile_count": sum(any(target["user_profile"][dim] for dim in PROFILE_DIMS) for target in gold_targets),
        "generated_token_count": {
            "total": sum(generated_tokens),
            "mean": sum(generated_tokens) / total if total else 0.0,
            "max": max(generated_tokens, default=0),
        },
        "emotion": emotion,
        "label_errors": label_errors,
    }


def bootstrap_conversation_micro_f1(
    cohort: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    enums: Mapping[str, list[str]],
    *,
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    ids = list(cohort["ids"])
    by_conversation: dict[str, list[str]] = {}
    for eval_id in ids:
        conversation = cohort["mapping"][eval_id].get("conversation_id")
        if not isinstance(conversation, str) or not conversation:
            raise ValueError("A8 bootstrap requires stable conversation_id mappings")
        by_conversation.setdefault(conversation, []).append(eval_id)
    clusters = sorted(by_conversation)
    rng = random.Random(seed)
    scores = []
    for _ in range(replicates):
        sampled = [cluster for cluster in (rng.choice(clusters) for _ in clusters)]
        sampled_ids = [eval_id for cluster in sampled for eval_id in by_conversation[cluster]]
        gold, predicted = [], []
        for eval_id in sampled_ids:
            target = references[eval_id]["target"]
            prediction = predictions[eval_id]
            parsed = prediction.get("parsed") if prediction.get("status") == "valid" else None
            profile = parsed.get("user_profile", {}) if isinstance(parsed, Mapping) else {}
            gold.append({dim: list(target["user_profile"][dim]) for dim in PROFILE_DIMS})
            predicted.append({dim: list(profile.get(dim, [])) for dim in PROFILE_DIMS})
        scores.append(float(profile_metrics(gold, predicted, dict(enums))["overall_33_label_micro_f1"]))
    ordered = sorted(scores)
    lower = ordered[max(0, int(0.025 * len(ordered)))]
    upper = ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]
    return {
        "method": "conversation_cluster_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "conversation_clusters": len(clusters),
        "micro_f1_2_5_percentile": lower,
        "micro_f1_97_5_percentile": upper,
        "exploratory_only": True,
    }
