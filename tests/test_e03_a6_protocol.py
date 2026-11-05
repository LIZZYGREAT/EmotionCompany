"""CPU-only regression tests for the isolated E03-A6 F3 text protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.shared.f3_protocol import (
    F3ParseError,
    parse_f3,
    parse_f3_prediction,
    serialize_f3,
    submission_enums,
    validate_f3_prompt_enum_order,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_SCHEMA = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))
SUBMISSION_SCHEMA = json.loads(
    (ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8")
)
ENUMS = submission_enums(SUBMISSION_SCHEMA)
F3_PROMPT = (ROOT / "configs/prompts/p_a6_f3.txt").read_text(encoding="utf-8")


def _target(response: str = "合成回复") -> dict:
    return {
        "emotion_label": "anxiety",
        "user_profile": {
            "personality_traits": ["sensitive", "introverted"],
            "interests": ["programming_technology", "study_exam"],
            "style": ["direct", "brief"],
        },
        "response_text": response,
        "memory_refs": ["mem_000001"],
    }


def test_f3_round_trip_is_canonical_and_lossless_for_response_text():
    response = '  第一行🙂\n第二行含 R=、"引号" 和反斜线 \\\n尾部空白 \t '
    target = _target(response)
    encoded = serialize_f3(target, DATA_SCHEMA)
    parsed = parse_f3(encoded, ENUMS)
    assert encoded.splitlines()[:4] == [
        "E=anxiety",
        "P=introverted|sensitive",
        "I=study_exam|programming_technology",
        "S=brief|direct",
    ]
    assert parsed["response_text"] == response
    assert parsed["memory_refs"] == []
    assert parsed["user_profile"] == {
        "personality_traits": ["introverted", "sensitive"],
        "interests": ["study_exam", "programming_technology"],
        "style": ["brief", "direct"],
    }


def test_f3_all_33_profile_labels_and_empty_dimensions_round_trip():
    target = _target("保留非空回复")
    for name in ("personality_traits", "interests", "style"):
        target["user_profile"][name] = list(reversed(ENUMS[name]))
    all_labels = serialize_f3(target, DATA_SCHEMA)
    parsed = parse_f3(all_labels, ENUMS)
    assert sum(len(parsed["user_profile"][name]) for name in ENUMS if name != "emotion_label") == 33
    empty = _target("回复")
    empty["user_profile"] = {name: [] for name in ("personality_traits", "interests", "style")}
    assert serialize_f3(empty, DATA_SCHEMA).splitlines()[1:4] == ["P=-", "I=-", "S=-"]
    assert parse_f3(serialize_f3(empty, DATA_SCHEMA), ENUMS)["user_profile"] == empty["user_profile"]


def test_f3_prediction_fills_id_and_passes_unchanged_submission_schema():
    prediction = parse_f3_prediction("test_000001_t1", serialize_f3(_target(), DATA_SCHEMA), ENUMS)
    Draft202012Validator(SUBMISSION_SCHEMA).validate(prediction)
    assert prediction["id"] == "test_000001_t1"
    assert prediction["memory_refs"] == []


def test_f3_prompt_label_lists_match_the_unchanged_submission_schema():
    validate_f3_prompt_enum_order(F3_PROMPT, ENUMS)
    with pytest.raises(ValueError, match="official submission schema"):
        validate_f3_prompt_enum_order(F3_PROMPT.replace("E 标签（选一个）：joy", "E 标签（选一个）：not-a-label"), ENUMS)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("E=anxiety\nP=-\nI=-\nS=-", "incomplete"),
        ("P=-\nE=anxiety\nI=-\nS=-\nR=回复", "unsupported_order"),
        ("E=not-emotion\nP=-\nI=-\nS=-\nR=回复", "invalid_emotion"),
        ("E=anxiety\nP=unknown\nI=-\nS=-\nR=回复", "invalid_profile_label"),
        ("E=anxiety\nP=sensitive|sensitive\nI=-\nS=-\nR=回复", "duplicate_label"),
        ("E=anxiety\nP=sensitive|introverted\nI=-\nS=-\nR=回复", "noncanonical_label_order"),
        ("E=anxiety\nP=-\nS=-\nI=-\nR=回复", "unsupported_order"),
    ],
)
def test_f3_parser_reports_strict_format_failures(raw: str, code: str):
    with pytest.raises(F3ParseError) as error:
        parse_f3(raw, ENUMS)
    assert error.value.code == code
