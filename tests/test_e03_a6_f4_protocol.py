"""CPU-only regression tests for the isolated E03-A6 F4 full-field protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from participant.run_inference import parse_f4_prediction
from src.data.e03_a6_f4_sft import build_f4_training_item
from src.shared.f4_protocol import (
    F4ParseError,
    parse_f4,
    serialize_f4,
    validate_f4_prompt_enum_order,
)
from src.shared.f3_protocol import submission_enums
from src.shared.prompting import build_messages, encode_inference_prompt


ROOT = Path(__file__).resolve().parents[1]
DATA_SCHEMA = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))
SUBMISSION_SCHEMA = json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8"))
ENUMS = submission_enums(SUBMISSION_SCHEMA)
F4_PROMPT = (ROOT / "configs/prompts/p_a6_f4.txt").read_text(encoding="utf-8")
P0_PROMPT = (ROOT / "configs/prompts/p0.txt").read_text(encoding="utf-8")


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


class FakeTokenizer:
    eos_token_id = 2_000_000

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        assert enable_thinking is False
        chunks = []
        last_is_completion = bool(messages and messages[-1]["role"] == "assistant") and not add_generation_prompt
        completed = messages[:-1] if last_is_completion else messages
        for message in completed:
            chunks.append(f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n")
        if add_generation_prompt:
            chunks.append("<assistant>\n")
        elif last_is_completion:
            chunks.extend(["<assistant>\n", messages[-1]["content"]])
        ids = [ord(char) for char in "".join(chunks)]
        if last_is_completion:
            ids.append(self.eos_token_id)
        return {"input_ids": ids}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join("<eos>" if int(value) == self.eos_token_id else chr(int(value)) for value in ids)


def test_f4_round_trip_uses_full_field_names_and_preserves_response_exactly():
    response = '  第一行🙂\n第二行含 Response=、"引号" 和反斜线 \\\n尾部空白 \t '
    encoded = serialize_f4(_target(response), DATA_SCHEMA)
    assert encoded.splitlines()[:4] == [
        "Emotion=anxiety",
        "Personality=introverted|sensitive",
        "Interests=study_exam|programming_technology",
        "Style=brief|direct",
    ]
    parsed = parse_f4(encoded, ENUMS)
    assert parsed["response_text"] == response
    assert parsed["memory_refs"] == []
    Draft202012Validator(SUBMISSION_SCHEMA).validate(parse_f4_prediction("test_000001_t1", encoded, ENUMS)[0])


def test_f4_all_profile_labels_and_empty_dimensions_round_trip():
    target = _target()
    for name in ("personality_traits", "interests", "style"):
        target["user_profile"][name] = list(reversed(ENUMS[name]))
    parsed = parse_f4(serialize_f4(target, DATA_SCHEMA), ENUMS)
    assert sum(len(parsed["user_profile"][name]) for name in ("personality_traits", "interests", "style")) == 33
    target["user_profile"] = {name: [] for name in ("personality_traits", "interests", "style")}
    assert serialize_f4(target, DATA_SCHEMA).splitlines()[1:4] == [
        "Personality=-", "Interests=-", "Style=-",
    ]
    assert parse_f4(serialize_f4(target, DATA_SCHEMA), ENUMS)["user_profile"] == target["user_profile"]


def test_f4_prompt_label_lists_match_the_unchanged_submission_schema():
    validate_f4_prompt_enum_order(F4_PROMPT, ENUMS)
    with pytest.raises(ValueError, match="official submission schema"):
        validate_f4_prompt_enum_order(
            F4_PROMPT.replace("Emotion 标签（选一个）：joy", "Emotion 标签（选一个）：unknown"), ENUMS,
        )


def test_f4_inference_config_locks_its_prompt_and_preserves_p0_t1():
    from participant import run_inference as runner

    config = json.loads((ROOT / "participant/configs/e03_a6_f4_inference_config.json").read_text())
    prompt, _schema, _enums, facts = runner.load_runtime_protocol(config)
    assert facts["format_version"] == "F4"
    assert facts["version"] == "P_A6_F4"
    assert facts["selection_prompt"]["version"] == "P0"
    assert "Emotion=<情绪标签>" in prompt


def test_f4_fast_stopper_stops_after_the_full_response_marker():
    from src.evaluation.f4_pilot import F4FastPrefixStopper

    decoded = {
        (1,): "Emotion=anxiety\nPersonality=introverted|sensitive\nInterests=-\nStyle=-",
        (1, 2): "Emotion=anxiety\nPersonality=introverted|sensitive\nInterests=-\nStyle=-\nResponse=",
    }
    stopper = F4FastPrefixStopper(lambda ids: decoded.get(tuple(ids), ""), ENUMS)
    assert stopper.inspect([1]) is False
    assert stopper.inspect([1, 2]) is True
    assert stopper.result is not None and stopper.result.valid
    assert stopper.result.parsed["emotion_label"] == "anxiety"


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("Emotion=anxiety\nPersonality=-\nInterests=-\nStyle=-", "incomplete"),
        ("Personality=-\nEmotion=anxiety\nInterests=-\nStyle=-\nResponse=回复", "unsupported_order"),
        ("Emotion=unknown\nPersonality=-\nInterests=-\nStyle=-\nResponse=回复", "invalid_emotion"),
        ("Emotion=anxiety\nPersonality=unknown\nInterests=-\nStyle=-\nResponse=回复", "invalid_profile_label"),
        ("Emotion=anxiety\nPersonality=sensitive|sensitive\nInterests=-\nStyle=-\nResponse=回复", "duplicate_label"),
        ("Emotion=anxiety\nPersonality=sensitive|introverted\nInterests=-\nStyle=-\nResponse=回复", "noncanonical_label_order"),
    ],
)
def test_f4_parser_rejects_malformed_full_field_rows(raw: str, code: str):
    with pytest.raises(F4ParseError) as exc:
        parse_f4(raw, ENUMS)
    assert exc.value.code == code


def test_f4_training_item_reuses_the_frozen_p0_prefix_and_masks_prompt_labels():
    tokenizer = FakeTokenizer()
    sample = {
        "sample_id": "train:synthetic:pp0",
        "conversation_id": "synthetic",
        "source": {"split": "train", "source_line": 1},
        "history": [
            {"role": "user", "content": "之前很焦虑。"},
            {"role": "assistant", "content": "愿意说说吗？"},
            {"role": "user", "content": "我在准备考试。"},
        ],
        "target": _target("  回复原样保留。\n包括第二行。  "),
    }
    p0_ids = encode_inference_prompt(tokenizer, build_messages(sample["history"], P0_PROMPT))
    f2_item = {
        "sample_id": sample["sample_id"],
        "conversation_id": sample["conversation_id"],
        "source_split": "train",
        "source": sample["source"],
        "prompt_length": len(p0_ids),
        "completion_length": 17,
        "input_ids": p0_ids + list(range(100, 117)),
        "labels": [-100] * len(p0_ids) + list(range(100, 117)),
    }
    item, diagnostic = build_f4_training_item(
        sample,
        f2_item,
        p0_prompt=P0_PROMPT,
        f4_prompt=F4_PROMPT,
        prompt_cap=5000,
        sequence_cap=12000,
        tokenizer=tokenizer,
        official_schema=DATA_SCHEMA,
        selected_start_index=0,
    )
    assert diagnostic["p0_selected_history_matches_f2"] is True
    assert diagnostic["completion_round_trip_lossless"] is True
    assert item["labels"][:item["prompt_length"]] == [-100] * item["prompt_length"]
    assert item["labels"][item["prompt_length"]:] == item["input_ids"][item["prompt_length"]:]
    assert item["labels"][-1] == tokenizer.eos_token_id
    decoded = tokenizer.decode(item["input_ids"][item["prompt_length"]:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert decoded.startswith("Emotion=anxiety\nPersonality=introverted|sensitive\n")


def test_f4_expanded_training_config_preserves_the_frozen_pilot_budget():
    from src.training.e03_a6_contract import validate_expanded_f4_pilot_config
    import yaml

    config = yaml.safe_load((ROOT / "configs/training/e03_a6_f4_full_field_pilot.yaml").read_text(encoding="utf-8"))
    validate_expanded_f4_pilot_config(config)
