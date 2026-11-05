from __future__ import annotations

from src.evaluation.f3_pilot import (
    F3FastPrefixStopper,
    derive_smoke_selection,
    score_fast_cases,
    score_smoke_cases,
    selected_ids_sha256,
)


ENUMS = {
    "emotion_label": ["sadness", "anxiety"],
    "personality_traits": ["introverted", "sensitive"],
    "interests": ["games", "reading"],
    "style": ["brief", "direct"],
}


def _decode(ids):
    return "".join(chr(int(value)) for value in ids)


def test_f3_fast_stopper_waits_for_cross_token_r_marker_and_keeps_exact_ids():
    text = "E=sadness\nP=introverted\nI=-\nS=brief\nR=回复正文"
    ids = list(map(ord, text))
    stopper = F3FastPrefixStopper(_decode, ENUMS)
    assert stopper.inspect(ids[: text.index("\nR=") + 1]) is False
    assert stopper.inspect(ids[: text.index("\nR=") + 2]) is False
    assert stopper.inspect(ids[: text.index("\nR=") + 3]) is True
    assert stopper.result is not None and stopper.result.valid
    assert stopper.result.generated_token_ids == tuple(ids[: text.index("\nR=") + 3])
    assert stopper.result.raw_prefix == "E=sadness\nP=introverted\nI=-\nS=brief\nR="
    assert stopper.result.parsed["response_text"] == ""


def test_f3_fast_stopper_rejects_invalid_labels_at_r_boundary():
    ids = list(map(ord, "E=sadness\nP=unknown\nI=-\nS=brief\nR="))
    stopper = F3FastPrefixStopper(_decode, ENUMS)
    assert stopper.inspect(ids)
    assert stopper.result is not None
    assert stopper.result.status == "invalid"
    assert stopper.result.reason == "invalid_profile_label"


def test_f3_fast_stopper_marks_eos_before_r_as_incomplete():
    stopper = F3FastPrefixStopper(_decode, ENUMS)
    result = stopper.finish("eos", list(map(ord, "E=sadness\nP=-")))
    assert result.status == "incomplete"
    assert result.reason == "eos"


def test_smoke_selection_is_stable_and_never_splits_conversations():
    ids = [f"eval-{index}" for index in range(10)]
    sample_ids = {
        ids[0]: "train:conv-a:pp0",
        ids[1]: "train:conv-a:pp1",
        ids[2]: "train:conv-b:pp0",
        ids[3]: "train:conv-c:pp0",
        ids[4]: "train:conv-d:pp0",
        ids[5]: "train:conv-e:pp0",
        ids[6]: "train:conv-f:pp0",
        ids[7]: "train:conv-g:pp0",
        ids[8]: "train:conv-h:pp0",
        ids[9]: "train:conv-i:pp0",
    }
    selected = derive_smoke_selection(ids, sample_ids, count=6, seed=2026)
    assert selected == derive_smoke_selection(ids, sample_ids, count=6, seed=2026)
    assert len(selected) == len(set(selected)) == 6
    assert len({sample_ids[eval_id].split(":")[1] for eval_id in selected}) == 6
    assert selected_ids_sha256(selected) == selected_ids_sha256(list(reversed(selected)))


def test_fast_scoring_keeps_failures_in_fixed_denominator_as_empty():
    targets = [
        {
            "emotion_label": "sadness",
            "user_profile": {"personality_traits": ["introverted"], "interests": [], "style": []},
        },
        {
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
        },
    ]
    cases = [
        {
            "id": "a",
            "parse_status": "success",
            "parsed": {
                "emotion_label": "sadness",
                "user_profile": {"personality_traits": ["introverted"], "interests": [], "style": []},
            },
        },
        {"id": "b", "parse_status": "incomplete", "parsed": None},
    ]
    gates = {
        "min_strict_f3_parse_rate": 0.5,
        "min_cases_with_nonempty_profile_prediction": 1,
        "min_profile_true_positives": 1,
        "max_profile_false_positives": 0,
        "min_profile_micro_precision": 1.0,
        "min_emotion_accuracy": 0.5,
        "max_single_emotion_prediction_share": 1.0,
    }
    result = score_fast_cases(targets, cases, ENUMS, gates)
    profile = result["fixed_denominator_profile_metrics"]
    assert result["fixed_denominator_count"] == 2
    assert profile["overall_33_label_tp"] == 1
    assert profile["overall_33_label_fp"] == 0
    assert profile["overall_33_label_fn"] == 0
    assert result["emotion_accuracy_fixed_denominator"] == 0.5
    assert result["gate_values"]["max_single_emotion_prediction_share"] == 1.0
    assert result["gate_passed"]


def test_smoke_gate_counts_every_case_and_prefix_failure():
    cases = [
        {
            "id": "a",
            "full": {
                "strict_parse_success": True,
                "official_schema_valid": True,
                "nonempty_response": True,
                "eos_generated": True,
                "generated_token_count": 20,
            },
            "fast": {"generated_token_count": 10},
            "prefix_equivalence": {"equivalent": True},
        },
        {
            "id": "b",
            "full": {
                "strict_parse_success": True,
                "official_schema_valid": True,
                "nonempty_response": True,
                "eos_generated": False,
                "generated_token_count": 18,
            },
            "fast": {"generated_token_count": 9},
            "prefix_equivalence": {"equivalent": False},
        },
    ]
    gates = {
        "min_strict_f3_parse_rate": 0.95,
        "min_official_schema_valid_rate": 1.0,
        "min_prefix_full_equivalence_rate": 1.0,
        "min_nonempty_response_rate": 0.95,
        "min_eos_rate": 0.9,
    }
    result = score_smoke_cases(cases, gates)
    assert result["gate_values"]["prefix_full_equivalence_rate"] == 0.5
    assert not result["gate_passed"]
    assert result["full_generated_tokens_total"] == 38
