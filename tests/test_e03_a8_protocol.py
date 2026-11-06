from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.e03_a8_protocol import (
    A8ParseError,
    parse_specialist_output,
    serialize_specialist_completion,
    validate_a8_prompt_enum_order,
    validate_training_target_round_trip,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))
P_PROMPT = (ROOT / "configs/prompts/p_a8_profile.txt").read_text(encoding="utf-8")
EP_PROMPT = (ROOT / "configs/prompts/p_a8_emotion_profile.txt").read_text(encoding="utf-8")


def _target() -> dict:
    return {
        "emotion_label": "joy",
        "user_profile": {
            "personality_traits": ["sensitive", "open"],
            "interests": ["pets", "games"],
            "style": ["rational", "direct"],
        },
        "response_text": "not part of A8 completion",
        "memory_refs": [],
    }


def test_prompts_match_official_enum_order_and_field_layout() -> None:
    assert len(validate_a8_prompt_enum_order(P_PROMPT, SCHEMA, "P")["personality_traits"]) == 10
    assert len(validate_a8_prompt_enum_order(EP_PROMPT, SCHEMA, "EP")["emotion_label"]) == 16


def test_prompt_enum_reordering_is_rejected() -> None:
    changed = P_PROMPT.replace("extroverted, introverted", "introverted, extroverted")
    with pytest.raises(ValueError, match="enum/order"):
        validate_a8_prompt_enum_order(changed, SCHEMA, "P")


def test_serialization_sorts_by_official_order_and_omits_nonclassification_fields() -> None:
    p = serialize_specialist_completion(_target(), SCHEMA, "P")
    ep = serialize_specialist_completion(_target(), SCHEMA, "EP")
    assert p == "Personality=open|sensitive\nInterests=games|pets\nStyle=direct|rational"
    assert ep == "Emotion=joy\n" + p
    assert "emotion" not in p.lower()
    assert "response" not in ep.lower()
    assert "memory_refs" not in ep


@pytest.mark.parametrize("arm,raw", [
    ("P", "Personality=-\nInterests=-\nStyle=-"),
    ("EP", "Emotion=neutral\nPersonality=-\nInterests=-\nStyle=-"),
])
def test_empty_profiles_round_trip(arm: str, raw: str) -> None:
    assert parse_specialist_output(raw, SCHEMA, arm)["user_profile"] == {
        "personality_traits": [], "interests": [], "style": []
    }


@pytest.mark.parametrize("arm,raw,code", [
    ("P", "personality=open\nInterests=-\nStyle=-", "unknown_field"),
    ("P", "Personality:open\nInterests=-\nStyle=-", "invalid_assignment"),
    ("P", "Personality=open\nInterests=-\nStyle=-\nextra", "extra_text"),
    ("P", "Personality=open\nInterests=-\nStyle=direct, rational", "invalid_separator"),
    ("P", "Personality=open|open\nInterests=-\nStyle=-", "duplicate_label"),
    ("P", "Personality=sensitive|open\nInterests=-\nStyle=-", "noncanonical_label_order"),
    ("P", "Personality=open\nInterests=unlisted\nStyle=-", "unknown_label"),
    ("EP", "Emotion=joy|care\nPersonality=-\nInterests=-\nStyle=-", "invalid_emotion"),
    ("P", "Emotion=joy\nPersonality=-\nInterests=-\nStyle=-", "extra_field"),
    ("EP", "Emotion=joy\nPersonality=-\nInterests=-", "incomplete"),
    ("P", "Personality=-\r\nInterests=-\nStyle=-", "invalid_line_ending"),
])
def test_strict_parser_rejects_invalid_outputs(arm: str, raw: str, code: str) -> None:
    with pytest.raises(A8ParseError) as error:
        parse_specialist_output(raw, SCHEMA, arm)
    assert error.value.code == code


@pytest.mark.parametrize("arm", ["P", "EP"])
def test_training_gold_round_trip_is_lossless_for_selected_fields(arm: str) -> None:
    parsed = validate_training_target_round_trip(_target(), SCHEMA, arm)
    assert parsed["user_profile"]["interests"] == ["games", "pets"]
    assert ("emotion_label" in parsed) is (arm == "EP")
