from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data import e03_a8_sft
from src.shared.e03_a8_protocol import parse_specialist_output

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **_kwargs):
        system = messages[0]["content"]
        history_tokens = sum(len(message["content"].split()) for message in messages[1:])
        prompt_base = {"p0": 3, "p": 2, "ep": 5}[system]
        return {"input_ids": list(range(prompt_base + history_tokens))}

    def decode(self, ids, **_kwargs):
        if list(ids) == [99]:
            return "<eos>"
        return ""


def test_common_history_trims_both_arms_together() -> None:
    history = [
        {"role": "assistant", "content": "a a a"},
        {"role": "user", "content": "b b b"},
        {"role": "assistant", "content": "c c"},
        {"role": "user", "content": "d"},
    ]
    selected = e03_a8_sft.select_common_history(
        history, p_prompt="p", ep_prompt="ep", tokenizer=FakeTokenizer(),
        initial_start_index=0, prompt_cap=8,
    )
    assert selected["selected_start_index"] == 2
    assert selected["selected_history"] == history[2:]
    assert selected["prompt_lengths"]["P"] <= 8
    assert selected["prompt_lengths"]["EP"] <= 8
    assert selected["extra_removed_messages"] == 2


def test_common_history_blocks_when_current_user_floor_exceeds_cap() -> None:
    history = [{"role": "user", "content": "too long current message"}]
    with pytest.raises(e03_a8_sft.PromptFloorExceeded):
        e03_a8_sft.select_common_history(
            history, p_prompt="p", ep_prompt="ep", tokenizer=FakeTokenizer(),
            initial_start_index=0, prompt_cap=2,
        )


def test_p_and_ep_preencoded_rows_share_history_and_mask_only_prompts(monkeypatch) -> None:
    def fake_encode_prompt(_tokenizer, messages):
        ids = {"p0": [1, 2, 3], "p": [4, 5], "ep": [6, 7, 8]}[messages[0]["content"]]
        return list(ids)

    def fake_full(_tokenizer, messages, completion):
        prompt = fake_encode_prompt(None, messages)
        return prompt + [20, 21, 99]

    def fake_suffix(_tokenizer, _completion, prompt_ids, full_ids, **_kwargs):
        assert list(full_ids[:len(prompt_ids)]) == list(prompt_ids)
        return list(full_ids[len(prompt_ids):])

    monkeypatch.setattr(e03_a8_sft, "encode_inference_prompt", fake_encode_prompt)
    monkeypatch.setattr(e03_a8_sft, "encode_full_training_sequence", fake_full)
    monkeypatch.setattr(e03_a8_sft, "validate_completion_suffix", fake_suffix)
    target = {
        "emotion_label": "joy",
        "user_profile": {
            "personality_traits": [], "interests": ["games"], "style": ["direct"],
        },
        "response_text": "excluded",
        "memory_refs": [],
    }
    sample = {
        "sample_id": "train:conv:sample",
        "conversation_id": "conv",
        "source": {},
        "history": [{"role": "user", "content": "current"}],
        "target": target,
    }
    f2_item = {
        "sample_id": sample["sample_id"], "conversation_id": "conv", "source_split": "train",
        "prompt_length": 3, "input_ids": [1, 2, 3, 99],
    }
    budget = {"removed_history_message_count": 0, "outcome": "full_fit", "source": {}}
    rows, audit = e03_a8_sft.build_a8_training_item(
        sample, f2_item, budget, p0_prompt="p0", p_prompt="p", ep_prompt="ep",
        prompt_cap=1280, sequence_cap=2048, tokenizer=FakeTokenizer(), official_schema=SCHEMA,
    )
    assert rows["P"]["common_history_sha256"] == rows["EP"]["common_history_sha256"]
    assert rows["P"]["labels"][:2] == [-100, -100]
    assert rows["P"]["labels"][2:] == rows["P"]["input_ids"][2:]
    assert rows["EP"]["labels"][:3] == [-100, -100, -100]
    assert rows["EP"]["labels"][3:] == rows["EP"]["input_ids"][3:]
    assert audit["p_terminal_eos_supervised"] and audit["ep_terminal_eos_supervised"]
    assert "emotion_label" not in parse_specialist_output(
        "Personality=-\nInterests=games\nStyle=direct", SCHEMA, "P"
    )


def test_sample_schedule_is_deterministic_unique_and_stage_aligned() -> None:
    ids = [f"train:conv:{i}" for i in range(16)]
    first = e03_a8_sft.make_sample_order(ids, seed=42, accumulation_steps=4)
    second = e03_a8_sft.make_sample_order(ids, seed=42, accumulation_steps=4)
    flattened = [sample_id for row in first for sample_id in row["sample_ids"]]
    assert first == second
    assert len(first) == 4
    assert len(flattened) == len(set(flattened)) == 16
    assert set(flattened) == set(ids)
