"""A8-V2 fixed cohort scoring with conversation-cluster bootstrap."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from src.evaluation.e03_a6_f4_staged_metrics import _conversation_id
from src.evaluation.e03_a8_staged_metrics import score_a8_cohort
from src.evaluation.metrics import profile_metrics
from src.shared.e03_a8_protocol import PROFILE_DIMS


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
    by_conversation: dict[str, list[str]] = defaultdict(list)
    for eval_id in ids:
        mapping = cohort["mapping"].get(eval_id)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"A8-V2 bootstrap ID map is missing: {eval_id}")
        by_conversation[_conversation_id(mapping)].append(eval_id)
    clusters = sorted(by_conversation)
    rng = random.Random(seed)
    scores = []
    for _replicate in range(replicates):
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        sampled_ids = [eval_id for cluster in sampled_clusters for eval_id in by_conversation[cluster]]
        gold, predicted = [], []
        for eval_id in sampled_ids:
            target = references[eval_id]["target"]
            prediction = predictions[eval_id]
            parsed = prediction.get("parsed") if prediction.get("status") == "valid" else None
            profile = parsed.get("user_profile", {}) if isinstance(parsed, Mapping) else {}
            gold.append({dimension: list(target["user_profile"][dimension]) for dimension in PROFILE_DIMS})
            predicted.append({dimension: list(profile.get(dimension, [])) for dimension in PROFILE_DIMS})
        scores.append(float(profile_metrics(gold, predicted, dict(enums))["overall_33_label_micro_f1"]))
    ordered = sorted(scores)
    return {
        "method": "conversation_cluster_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "conversation_clusters": len(clusters),
        "micro_f1_2_5_percentile": ordered[max(0, int(0.025 * len(ordered)))],
        "micro_f1_97_5_percentile": ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))],
        "exploratory_only": True,
    }


__all__ = ["bootstrap_conversation_micro_f1", "score_a8_cohort"]
