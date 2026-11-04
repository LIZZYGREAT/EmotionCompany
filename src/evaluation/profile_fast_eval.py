"""CPU helpers shared by the isolated E03 fast profile evaluator."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from src.evaluation.source_integrity import canonical_sha256, unique_index, validate_target_free_inputs


def select_unique_conversation_samples(
    input_rows: Sequence[Mapping[str, Any]],
    id_map_rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select one point from each session using only IDs and fixed seed."""
    if isinstance(count, bool) or count < 1 or count > 32:
        raise ValueError("fast profile selection count must be between 1 and 32")
    inputs = validate_target_free_inputs(input_rows)
    id_map = unique_index(id_map_rows, "eval_id", "ID map")
    if set(inputs) != set(id_map):
        raise ValueError("generation input and ID map ID sets differ")
    by_conversation: dict[str, list[str]] = defaultdict(list)
    for eval_id, mapping in id_map.items():
        conversation_id = mapping.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"ID map {eval_id} has no conversation_id")
        by_conversation[conversation_id].append(eval_id)
    if len(by_conversation) < count:
        raise ValueError("not enough unique conversations for the fixed fast-eval sample count")

    ranked: list[tuple[str, str, str]] = []
    for conversation_id, eval_ids in by_conversation.items():
        representative = min(
            eval_ids,
            key=lambda eval_id: hashlib.sha256(f"{seed}:point:{eval_id}".encode("utf-8")).hexdigest(),
        )
        rank = hashlib.sha256(f"{seed}:conversation:{conversation_id}".encode("utf-8")).hexdigest()
        ranked.append((rank, conversation_id, representative))
    ranked.sort()
    return [dict(inputs[eval_id]) for _rank, _conversation, eval_id in ranked[:count]]


def compare_generated_prefix(
    full_token_ids: Sequence[int],
    fast_token_ids: Sequence[int],
    full_emotion: str | None,
    fast_emotion: str | None,
    full_profile: Mapping[str, Sequence[str]] | None,
    fast_profile: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any]:
    """Compare token IDs before comparing parsed semantic fields."""
    full_ids = [int(token) for token in full_token_ids]
    fast_ids = [int(token) for token in fast_token_ids]
    token_match = len(fast_ids) <= len(full_ids) and full_ids[:len(fast_ids)] == fast_ids
    profile_match = (
        full_profile is not None
        and fast_profile is not None
        and full_emotion == fast_emotion
        and all(list(full_profile.get(key, [])) == list(fast_profile.get(key, []))
                for key in ("personality_traits", "interests", "style"))
    )
    return {
        "token_prefix_match": token_match,
        "emotion_and_profile_match": profile_match,
        "full_generated_token_count": len(full_ids),
        "fast_generated_token_count": len(fast_ids),
        "full_prefix_token_ids_sha256": canonical_sha256(fast_ids) if token_match else None,
    }


def profile_nonempty_recall(
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    predicted_profiles: Sequence[Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, int | float]]:
    if len(gold_profiles) != len(predicted_profiles):
        raise ValueError("gold and predicted profile row counts differ")
    result: dict[str, dict[str, int | float]] = {}
    for dimension in ("personality_traits", "interests", "style"):
        support = hits = 0
        for gold, predicted in zip(gold_profiles, predicted_profiles):
            is_nonempty = bool(gold[dimension])
            support += is_nonempty
            hits += is_nonempty and bool(predicted[dimension])
        result[dimension] = {
            "gold_nonempty_count": support,
            "predicted_nonempty_on_gold_nonempty_count": hits,
            "recall": hits / support if support else 0.0,
        }
    return result
