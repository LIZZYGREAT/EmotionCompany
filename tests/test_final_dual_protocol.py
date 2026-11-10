from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.data.e03_a8_sft as a8_sft
import src.inference.final_dual_protocol as final_protocol
from src.inference.final_dual_protocol import (
    PromptBuildError,
    assemble_submission,
    build_emotion_response_prompt,
    build_profile_prompt,
    parse_emotion_response_output,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8"))


class WordTokenizer:
    eos_token_id = 151645

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        body = " ".join(message["content"] for message in messages)
        return [1, *range(2, 2 + len(body.split())), 151645]


def _prompts() -> dict[str, str]:
    paths = {
        "selection": "configs/prompts/p0.txt",
        "profile": "configs/prompts/p_a8_v2_profile.txt",
        "profile_budget": "configs/prompts/p_a8_v2_emotion_profile.txt",
        "emotion_response": "configs/prompts/p_e03b.txt",
    }
    return {key: (ROOT / value).read_text(encoding="utf-8") for key, value in paths.items()}


def _sample() -> dict:
    return {
        "id": "test_000001_t5",
        "history": [
            {"turn_id": 1, "role": "user", "content": "old " * 80},
            {"turn_id": 2, "role": "assistant", "content": "reply " * 80},
            {"turn_id": 5, "role": "user", "content": "current user message"},
        ],
    }


def test_profile_prompt_uses_p0_then_joint_a8_history_budget() -> None:
    prompts = _prompts()
    result = build_profile_prompt(
        _sample(),
        WordTokenizer(),
        selection_prompt=prompts["selection"],
        profile_prompt=prompts["profile"],
        profile_budget_prompt=prompts["profile_budget"],
        prompt_cap=128,
    )
    assert result["prompt_tokens"] <= 128
    assert result["selected_start_index"] >= result["p0_selected_start_index"]
    assert result["messages"][-1]["content"] == "current user message"
    assert result["selected_message_ids"][-1] == 5
    assert len(result["prompt_token_ids_sha256"]) == 64


def test_e03b_prompt_uses_p0_history_then_rechecks_final_prompt_cap() -> None:
    prompts = _prompts()
    result = build_emotion_response_prompt(
        _sample(),
        WordTokenizer(),
        selection_prompt=prompts["selection"],
        emotion_response_prompt=prompts["emotion_response"],
        prompt_cap=128,
    )
    assert result["prompt_tokens"] <= 128
    assert result["selected_start_index"] >= result["p0_selected_start_index"]
    assert result["messages"][-1]["content"] == "current user message"
    assert result["selected_message_ids"][-1] == 5


def test_runtime_prompt_builders_skip_per_sample_hash_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_hash(*_args, **_kwargs):
        raise AssertionError("official inference must not calculate identity hashes")

    monkeypatch.setattr(final_protocol, "token_ids_sha256", unexpected_hash)
    monkeypatch.setattr(final_protocol, "history_sha256", unexpected_hash)
    monkeypatch.setattr(a8_sft, "canonical_sha256", unexpected_hash)
    prompts = _prompts()

    profile = build_profile_prompt(
        _sample(),
        WordTokenizer(),
        selection_prompt=prompts["selection"],
        profile_prompt=prompts["profile"],
        profile_budget_prompt=prompts["profile_budget"],
        prompt_cap=128,
        include_diagnostics=False,
    )
    emotion = build_emotion_response_prompt(
        _sample(),
        WordTokenizer(),
        selection_prompt=prompts["selection"],
        emotion_response_prompt=prompts["emotion_response"],
        prompt_cap=128,
        include_diagnostics=False,
    )

    assert set(profile) == {"messages", "prompt_ids"}
    assert set(emotion) == {"messages", "prompt_ids"}


def test_prompt_floor_error_is_reported_without_slicing_current_user() -> None:
    prompts = _prompts()
    sample = {"id": "test_000001_t5", "history": [{"role": "user", "content": "huge " * 500}]}
    with pytest.raises(PromptBuildError) as error:
        build_emotion_response_prompt(
            sample,
            WordTokenizer(),
            selection_prompt=prompts["selection"],
            emotion_response_prompt=prompts["emotion_response"],
            prompt_cap=128,
        )
    assert error.value.code in {"p0_prompt_floor_exceeds_cap", "e03b_prompt_floor_exceeds_cap"}


def test_e03b_split_preserves_semicolons_and_newlines_after_first_separator() -> None:
    raw = "emotion=neutral;response=First; second line\nthird;response=literal"
    parsed = parse_emotion_response_output(raw, SCHEMA)
    assert parsed == {
        "emotion_label": "neutral",
        "response_text": "First; second line\nthird;response=literal",
        "emotion_error": None,
        "response_error": None,
        "format_error": None,
    }


def test_invalid_emotion_keeps_independently_parseable_response() -> None:
    parsed = parse_emotion_response_output("emotion=unknown;response=Still a usable reply", SCHEMA)
    assert parsed["emotion_label"] == ""
    assert parsed["emotion_error"] == "invalid_emotion_label"
    assert parsed["response_text"] == "Still a usable reply"
    assert parsed["response_error"] is None


def test_missing_e03b_separator_recovers_emotion_and_visible_response() -> None:
    parsed = parse_emotion_response_output("emotion=neutral This is not the frozen format", SCHEMA)
    assert parsed["emotion_label"] == "neutral"
    assert parsed["response_text"] == "This is not the frozen format"
    assert parsed["emotion_error"] is None
    assert parsed["response_error"] is None
    assert parsed["format_error"] == "missing_response_separator"


def test_json_fallback_extracts_response_text_and_valid_emotion() -> None:
    parsed = parse_emotion_response_output(
        '{"emotion_label":"joy","response_text":"I can listen."}', SCHEMA
    )

    assert parsed["emotion_label"] == "joy"
    assert parsed["response_text"] == "I can listen."
    assert parsed["emotion_error"] is None
    assert parsed["response_error"] is None
    assert parsed["format_error"] == "json_fallback"


def test_malformed_json_fallback_extracts_response_text_field() -> None:
    parsed = parse_emotion_response_output(
        '{"response_text":"I can listen.",}', SCHEMA
    )

    assert parsed["emotion_label"] == ""
    assert parsed["response_text"] == "I can listen."
    assert parsed["emotion_error"] == "missing_emotion_label"
    assert parsed["response_error"] is None
    assert parsed["format_error"] == "json_fallback"


def test_valid_json_with_missing_fields_uses_empty_field_fallbacks() -> None:
    parsed = parse_emotion_response_output('{"emotion_label":"joy"}', SCHEMA)

    assert parsed["emotion_label"] == "joy"
    assert parsed["response_text"] == ""
    assert parsed["emotion_error"] is None
    assert parsed["response_error"] == "missing_response_text"
    assert parsed["format_error"] == "json_fallback"


def test_unparseable_output_falls_back_to_complete_cleaned_text() -> None:
    parsed = parse_emotion_response_output(
        "<think>internal reasoning</think>\n```text\nReturn valid JSON.\n```", SCHEMA
    )

    assert parsed["emotion_label"] == ""
    assert parsed["response_text"] == "Return valid JSON."
    assert parsed["emotion_error"] == "invalid_emotion_prefix"
    assert parsed["response_error"] is None
    assert parsed["format_error"] == "missing_response_separator"


@pytest.mark.parametrize(
    ("raw", "emotion", "response"),
    [
        ("emotion=joy\nresponse=I am here with you.", "joy", "I am here with you."),
        ("<think>internal reasoning</think>\n\u0060\u0060\u0060text\nemotion=neutral\nresponse=Take your time.\n\u0060\u0060\u0060", "neutral", "Take your time."),
        ("emotion=invalid; response=I can listen.", "", "I can listen."),
    ],
)
def test_missing_e03b_separator_recovers_independent_fields(raw: str, emotion: str, response: str) -> None:
    parsed = parse_emotion_response_output(raw, SCHEMA)

    assert parsed["emotion_label"] == emotion
    assert parsed["response_text"] == response
    assert parsed["format_error"] == "missing_response_separator"


def test_assembly_preserves_valid_er_when_profile_fails() -> None:
    row, diagnostics = assemble_submission(
        "test_000001_t5",
        "Personality=not_a_label\nInterests=music\nStyle=colloquial",
        "emotion=neutral;response=Hi; there",
        SCHEMA,
    )
    assert row["user_profile"] == {"personality_traits": [], "interests": [], "style": []}
    assert row["emotion_label"] == "neutral"
    assert row["response_text"] == "Hi; there"
    assert row["memory_refs"] == []
    assert diagnostics["profile_parse_error"] is not None
    assert diagnostics["response_parse_error"] is None


def test_assembly_preserves_valid_profile_and_er_response_when_emotion_is_invalid() -> None:
    profile = "Personality=introverted\nInterests=music\nStyle=colloquial"
    row, diagnostics = assemble_submission(
        "test_000001_t5", profile, "emotion=invalid;response=Hello", SCHEMA
    )
    assert row["user_profile"]["personality_traits"] == ["introverted"]
    assert row["emotion_label"] == ""
    assert row["response_text"] == "Hello"
    assert diagnostics["emotion_parse_error"] == "invalid_emotion_label"


def test_submission_id_is_preserved_without_runtime_schema_rejection() -> None:
    row, diagnostics = assemble_submission("val:bad-id", None, None, SCHEMA)

    assert row["id"] == "val:bad-id"
    assert row["user_profile"] == {"personality_traits": [], "interests": [], "style": []}
    assert row["emotion_label"] == ""
    assert row["response_text"] == ""
    assert row["memory_refs"] == []
    assert diagnostics["profile_parse_error"] == "generation_failed"
