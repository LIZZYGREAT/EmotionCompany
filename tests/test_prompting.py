"""Synthetic protocol tests plus local Qwen3 tokenizer integration checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.shared.prompting import (
    build_messages,
    encode_full_training_sequence,
    encode_inference_prompt,
    serialize_completion,
    tokenizer_signature,
    validate_prompt_enum_order,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_SCHEMA = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))
P0 = (ROOT / "configs/prompts/p0.txt").read_text(encoding="utf-8")


def _target(
    *,
    emotion: str = "anxiety",
    profile: dict | None = None,
    response: str = "合成回复",
    memories: list[str] | None = None,
) -> dict:
    return {
        "emotion_label": emotion,
        "user_profile": profile or {"personality_traits": [], "interests": [], "style": []},
        "response_text": response,
        "memory_refs": memories or [],
    }


@pytest.fixture(scope="session")
def qwen_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B",
            revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:  # pragma: no cover - expected only without the declared local cache
        pytest.skip(f"local pinned Qwen3 tokenizer is unavailable: {type(exc).__name__}: {exc}")


def test_p01_system_and_native_multi_turn_messages_preserve_roles_and_text():
    history = [
        {"role": "user", "content": "你好\n最近有点焦虑。"},
        {"role": "assistant", "content": "谢谢你愿意说出来。"},
        {"role": "user", "content": "我担心考试。"},
    ]
    messages = build_messages(history, "synthetic P0")
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert [message["content"] for message in messages[1:]] == [turn["content"] for turn in history]


def test_p02_current_user_is_required_as_last_message():
    assert build_messages([{"role": "user", "content": "当前消息"}], "P0")[-1]["role"] == "user"
    with pytest.raises(ValueError, match="last history message"):
        build_messages([
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回复"},
        ], "P0")


def test_p03_completion_json_order_and_compact_encoding_are_stable():
    target = _target()
    expected = (
        '{"emotion_label":"anxiety","user_profile":{"personality_traits":[],"interests":[],"style":[]},'
        '"response_text":"合成回复","memory_refs":[]}'
    )
    assert serialize_completion(target, DATA_SCHEMA) == expected
    assert serialize_completion(target, DATA_SCHEMA).encode("utf-8") == expected.encode("utf-8")
    assert list(json.loads(expected)) == ["emotion_label", "user_profile", "response_text", "memory_refs"]


def test_p04_profile_values_are_sorted_in_official_order_without_mutation():
    target = _target(profile={
        "personality_traits": ["sensitive", "extroverted"],
        "interests": ["music", "study_exam"],
        "style": ["rational", "brief"],
    })
    before = copy.deepcopy(target)
    rendered = serialize_completion(target, DATA_SCHEMA)
    decoded = json.loads(rendered)
    assert decoded["user_profile"] == {
        "personality_traits": ["extroverted", "sensitive"],
        "interests": ["study_exam", "music"],
        "style": ["brief", "rational"],
    }
    assert target == before


def test_p05_unicode_escaping_round_trips_and_memory_refs_are_empty_in_completion():
    response = '中文第一行\n第二行含有"引号"、反斜线 \\ 和表情🙂'
    target = _target(response=response, memories=["mem_000001"])
    before = copy.deepcopy(target)
    rendered = serialize_completion(target, DATA_SCHEMA)
    decoded = json.loads(rendered)
    assert decoded["response_text"] == response
    assert decoded["memory_refs"] == []
    assert target == before and target["memory_refs"] == ["mem_000001"]
    assert "\\n" in rendered and "\\u4e2d" not in rendered


@pytest.mark.parametrize("target", [
    _target(emotion="not_an_official_emotion"),
    _target(profile={"personality_traits": ["made_up"], "interests": [], "style": []}),
    _target(profile={"personality_traits": ["sensitive", "sensitive"], "interests": [], "style": []}),
])
def test_p06_illegal_or_duplicate_labels_are_rejected(target: dict):
    with pytest.raises(ValueError, match="official data schema"):
        serialize_completion(target, DATA_SCHEMA)


def test_prompt_enum_lists_match_official_train_schema():
    assert set(validate_prompt_enum_order(P0, DATA_SCHEMA)) == {
        "emotion_label", "personality_traits", "interests", "style"
    }
    with pytest.raises(ValueError, match="official data schema"):
        validate_prompt_enum_order(P0.replace("surprise, mixed", "surprise, unsupported, mixed"), DATA_SCHEMA)


def test_p07_real_qwen_non_thinking_template_has_exact_training_prefix(qwen_tokenizer):
    history = [
        {"role": "user", "content": "合成历史消息一。"},
        {"role": "assistant", "content": "合成历史回复。"},
        {"role": "user", "content": "合成当前消息。"},
    ]
    messages = build_messages(history, P0)
    completion = serialize_completion(_target(response='中文"引号"\n第二行🙂'), DATA_SCHEMA)
    prompt_ids = encode_inference_prompt(qwen_tokenizer, messages)
    full_ids = encode_full_training_sequence(qwen_tokenizer, messages, completion)
    assert full_ids[:len(prompt_ids)] == prompt_ids
    completion_ids = full_ids[len(prompt_ids):]
    assert completion_ids
    assert qwen_tokenizer.eos_token_id in completion_ids
    decoded = qwen_tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert decoded.startswith(completion)
    assert not qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"] == prompt_ids


def test_p08_template_or_revision_change_changes_tokenizer_signature(qwen_tokenizer):
    first = tokenizer_signature(
        qwen_tokenizer,
        model_id="Qwen/Qwen3-1.7B",
        revision="revision-a",
        tokenizer_files={"tokenizer.json": "a" * 64},
    )
    changed = copy.copy(qwen_tokenizer)
    changed.chat_template = str(qwen_tokenizer.chat_template) + "\n{# audit signature test #}"
    second = tokenizer_signature(
        changed,
        model_id="Qwen/Qwen3-1.7B",
        revision="revision-b",
        tokenizer_files={"tokenizer.json": "b" * 64},
    )
    assert first["template_sha256"] != second["template_sha256"]
    assert first["signature_sha256"] != second["signature_sha256"]
