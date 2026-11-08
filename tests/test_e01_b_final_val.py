from __future__ import annotations

from src.evaluation.e01_b_final_val import (
    paired_conversation_bootstrap,
    prefix_token_ids_match,
    profile_hit_transitions,
    select_stable_ids,
)


ENUMS = {
    "personality_traits": ["warm", "curious"],
    "interests": ["music"],
    "style": ["concise"],
}


def test_prefix_equivalence_requires_exact_token_prefix() -> None:
    assert prefix_token_ids_match([1, 2, 3, 4], [1, 2, 3])
    assert not prefix_token_ids_match([1, 2, 3, 4], [1, 2, 4])
    assert not prefix_token_ids_match([1, 2], [])


def test_fixed_id_selection_is_stable_and_unique() -> None:
    ids = [f"val-{index}" for index in range(40)]
    selected = select_stable_ids(ids, count=16, seed=42, purpose="test")

    assert selected == select_stable_ids(ids, count=16, seed=42, purpose="test")
    assert len(selected) == len(set(selected)) == 16


def test_profile_hit_transitions_conserve_gold_positive_pairs() -> None:
    gold = {"a": {"personality_traits": ["warm"]}, "b": {"style": ["concise"]}}
    e01 = {"a": {"personality_traits": ["warm"]}, "b": {"style": []}}
    a8 = {"a": {"personality_traits": ["curious"]}, "b": {"style": ["concise"]}}

    result = profile_hit_transitions(gold, e01, a8)

    assert result == {
        "e01_b_only_hits": 1,
        "a8_only_hits": 1,
        "both_models_hit": 0,
        "both_models_missed": 0,
        "gold_positive_sample_label_pairs": 2,
    }


def test_conversation_bootstrap_is_paired_and_seeded() -> None:
    gold = {
        "a": {"personality_traits": ["warm"], "interests": [], "style": []},
        "b": {"personality_traits": [], "interests": ["music"], "style": []},
        "c": {"personality_traits": [], "interests": [], "style": ["concise"]},
    }
    e01 = {
        "a": {"personality_traits": ["warm"], "interests": [], "style": []},
        "b": {"personality_traits": [], "interests": [], "style": []},
        "c": {"personality_traits": [], "interests": [], "style": ["concise"]},
    }
    a8 = {
        "a": {"personality_traits": [], "interests": [], "style": []},
        "b": {"personality_traits": [], "interests": ["music"], "style": []},
        "c": {"personality_traits": [], "interests": [], "style": []},
    }
    conversations = {"a": "x", "b": "y", "c": "z"}

    first = paired_conversation_bootstrap(gold, e01, a8, conversations, ENUMS, replicates=50, seed=42)
    second = paired_conversation_bootstrap(gold, e01, a8, conversations, ENUMS, replicates=50, seed=42)

    assert first == second
    assert first["conversation_clusters"] == 3
    assert first["e01_b_minus_a8_micro_f1_95_percent_interval"][0] <= first["e01_b_minus_a8_micro_f1_95_percent_interval"][1]
