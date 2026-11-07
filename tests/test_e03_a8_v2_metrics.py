from __future__ import annotations

from src.evaluation.e03_a8_v2_staged_metrics import bootstrap_conversation_micro_f1


def test_v2_bootstrap_derives_conversation_from_stable_sample_id():
    dimensions = {
        "personality_traits": ["sensitive"],
        "interests": ["games"],
        "style": ["direct"],
    }
    cohort = {
        "ids": ["eval-a", "eval-b"],
        "mapping": {
            "eval-a": {"sample_id": "train:conversation-1:pp0"},
            "eval-b": {"sample_id": "train:conversation-2:pp0"},
        },
    }
    references = {
        eval_id: {"target": {"user_profile": {dimension: [] for dimension in dimensions}}}
        for eval_id in cohort["ids"]
    }
    predictions = {
        eval_id: {"status": "valid", "parsed": {"user_profile": {dimension: [] for dimension in dimensions}}}
        for eval_id in cohort["ids"]
    }
    enums = {
        "personality_traits": ["sensitive"],
        "interests": ["games"],
        "style": ["direct"],
    }

    result = bootstrap_conversation_micro_f1(
        cohort,
        references,
        predictions,
        enums,
        replicates=10,
        seed=42,
    )

    assert result["conversation_clusters"] == 2
    assert result["micro_f1_2_5_percentile"] == 0.0
    assert result["micro_f1_97_5_percentile"] == 0.0
