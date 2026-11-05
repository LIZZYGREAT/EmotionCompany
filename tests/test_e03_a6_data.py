"""Synthetic tests for the CPU-only F3 data encoding helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.e03_a6_sft import build_f3_training_item
from src.shared.prompting import build_messages, encode_inference_prompt


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))


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
        text = "".join(chunks)
        ids = [ord(char) for char in text]
        if last_is_completion:
            ids.append(self.eos_token_id)
        return {"input_ids": ids}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join("<eos>" if int(value) == self.eos_token_id else chr(int(value)) for value in ids)


@pytest.fixture
def rows():
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
        "target": {
            "emotion_label": "anxiety",
            "user_profile": {
                "personality_traits": ["sensitive"],
                "interests": ["study_exam"],
                "style": ["direct"],
            },
            "response_text": "  我们可以一起拆解一下。\n先看最担心的部分。  ",
            "memory_refs": [],
        },
    }
    p0 = "F2 P0 synthetic"
    f3 = "F3 synthetic"
    prompt_ids = encode_inference_prompt(tokenizer, build_messages(sample["history"], p0))
    f2_item = {
        "sample_id": sample["sample_id"],
        "conversation_id": sample["conversation_id"],
        "source_split": "train",
        "source": sample["source"],
        "prompt_length": len(prompt_ids),
        "completion_length": 17,
        "input_ids": prompt_ids + list(range(100, 117)),
        "labels": [-100] * len(prompt_ids) + list(range(100, 117)),
    }
    return sample, f2_item, p0, f3, tokenizer


def test_f3_dataset_item_reuses_exact_p0_t1_prefix_and_masks_prompt(rows):
    sample, f2_item, p0, f3, tokenizer = rows
    item, diagnostic = build_f3_training_item(
        sample,
        f2_item,
        p0_prompt=p0,
        f3_prompt=f3,
        prompt_cap=5000,
        sequence_cap=8000,
        tokenizer=tokenizer,
        official_schema=SCHEMA,
    )
    assert diagnostic["p0_selected_history_matches_f2"] is True
    assert diagnostic["selected_history_sha256"]
    assert item["sample_id"] == f2_item["sample_id"]
    assert item["prompt_length"] == diagnostic["f3_prompt_tokens"]
    assert item["labels"][:item["prompt_length"]] == [-100] * item["prompt_length"]
    assert item["labels"][item["prompt_length"]:] == item["input_ids"][item["prompt_length"]:]
    assert item["labels"][-1] == tokenizer.eos_token_id
    assert diagnostic["completion_round_trip_lossless"] is True


def test_f3_dataset_item_blocks_prompt_or_sequence_overflow(rows):
    sample, f2_item, p0, f3, tokenizer = rows
    long_f3 = f3 + (" x" * 100)
    with pytest.raises(ValueError, match="F3 prompt exceeds P"):
        build_f3_training_item(
            sample,
            f2_item,
            p0_prompt=p0,
            f3_prompt=long_f3,
            prompt_cap=f2_item["prompt_length"],
            sequence_cap=8000,
            tokenizer=tokenizer,
            official_schema=SCHEMA,
        )
    with pytest.raises(ValueError, match="exceeds S"):
        build_f3_training_item(
            sample,
            f2_item,
            p0_prompt=p0,
            f3_prompt=f3,
            prompt_cap=5000,
            sequence_cap=10,
            tokenizer=tokenizer,
            official_schema=SCHEMA,
        )
