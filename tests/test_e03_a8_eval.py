from __future__ import annotations

from src.evaluation.e03_a8_staged_metrics import bootstrap_conversation_micro_f1, score_a8_cohort

ENUMS = {
    "emotion_label": ["joy", "neutral"],
    "personality_traits": ["open", "sensitive"],
    "interests": ["games", "pets"],
    "style": ["direct", "rational"],
}


def _fixture():
    ids = ["eval-1", "eval-2", "eval-3"]
    cohort = {
        "name": "dev224",
        "ids": ids,
        "ids_sha256": "fixture",
        "conversation_count": 2,
        "mapping": {
            "eval-1": {"conversation_id": "conv-a"},
            "eval-2": {"conversation_id": "conv-a"},
            "eval-3": {"conversation_id": "conv-b"},
        },
    }
    references = {
        "eval-1": {"target": {"emotion_label": "joy", "user_profile": {"personality_traits": ["open"], "interests": ["games"], "style": ["direct"]}}},
        "eval-2": {"target": {"emotion_label": "neutral", "user_profile": {"personality_traits": [], "interests": [], "style": []}}},
        "eval-3": {"target": {"emotion_label": "joy", "user_profile": {"personality_traits": ["sensitive"], "interests": [], "style": ["rational"]}}},
    }
    predictions = {
        "eval-1": {"status": "valid", "parsed": {"emotion_label": "joy", "user_profile": {"personality_traits": ["open"], "interests": ["games"], "style": ["direct"]}}, "eos_generated": True, "generated_token_count": 18},
        "eval-2": {"status": "invalid", "parsed": None, "reason": "unknown_label", "eos_generated": True, "generated_token_count": 10},
        "eval-3": {"status": "valid", "parsed": {"emotion_label": "neutral", "user_profile": {"personality_traits": ["sensitive"], "interests": [], "style": ["rational"]}}, "eos_generated": True, "generated_token_count": 21},
    }
    return cohort, references, predictions


def test_fixed_denominator_metrics_count_invalid_output_as_empty_profile() -> None:
    cohort, references, predictions = _fixture()
    result = score_a8_cohort(cohort, references, predictions, ENUMS, arm="EP")
    assert result["fixed_denominator_count"] == 3
    assert result["valid_count"] == 2
    assert result["strict_parse_rate"] == 2 / 3
    assert result["profile_33_label"]["overall_33_label_tp"] == 5
    assert result["profile_33_label"]["overall_33_label_fn"] == 0
    assert result["emotion"]["sample_count"] == 3
    assert result["eos_completion_rate"] == 1.0


def test_p_emotion_is_reported_as_not_applicable() -> None:
    cohort, references, predictions = _fixture()
    result = score_a8_cohort(cohort, references, predictions, ENUMS, arm="P")
    assert result["emotion"]["status"] == "N/A"


def test_conversation_cluster_bootstrap_is_deterministic() -> None:
    cohort, references, predictions = _fixture()
    first = bootstrap_conversation_micro_f1(cohort, references, predictions, ENUMS, replicates=25, seed=5)
    second = bootstrap_conversation_micro_f1(cohort, references, predictions, ENUMS, replicates=25, seed=5)
    assert first == second
    assert first["conversation_clusters"] == 2
    assert first["exploratory_only"] is True
