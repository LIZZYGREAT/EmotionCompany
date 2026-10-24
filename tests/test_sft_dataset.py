"""D0.4.3 Completion-only sample construction checks with pinned Qwen tokens."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from participant.run_inference import load_runtime_tokenizer
from src.data.sft_dataset import (
    _apply_overrides,
    _derive_outcome,
    build_training_item,
    load_config,
    parse_args,
)
from src.data.sft_collator import SFTDataCollator, validate_dataset_batches
from src.data.sft_trainer_validation import validate_trainer_data_path
from src.shared.prompting import (
    build_messages,
    encode_full_training_sequence,
    encode_inference_prompt,
    select_recent_complete_suffix,
    serialize_completion,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sft_context():
    config = json.loads((ROOT / "participant/configs/inference_config.json").read_text(encoding="utf-8"))
    tokenizer, facts = load_runtime_tokenizer(config)
    prompt = (ROOT / "configs/prompts/p0.txt").read_text(encoding="utf-8")
    schema = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))
    target_schema = {
        "$schema": schema["$schema"],
        "$ref": "#/$defs/target",
        "$defs": schema["$defs"],
    }
    return tokenizer, prompt, schema, Draft202012Validator(target_schema), facts


def _sample(split="train", *, history=None, response="合成监督回复", profile=None, memories=None):
    source_split = "val" if split == "val" else "train"
    if history is None:
        history = [
            {"role": "user", "content": "之前发生了一些事情"},
            {"role": "assistant", "content": "谢谢你愿意说出来。"},
            {"role": "user", "content": "今天我有些焦虑。"},
        ]
    return {
        "sample_id": f"{source_split}:{source_split}_conv_000001:pp0",
        "conversation_id": f"{source_split}_conv_000001",
        "source_split": split,
        "source": {
            "split": source_split,
            "source_line": 1,
            "prediction_point_index": 0,
            "target_user_turn_id": 5,
            "assistant_response_turn_id": 6,
            "target_turn_index": 4,
            "response_turn_index": 5,
        },
        "history": history,
        "target": {
            "response_text": response,
            "emotion_label": "anxiety",
            "user_profile": profile or {
                "personality_traits": [], "interests": [], "style": [],
            },
            "memory_refs": memories or [],
        },
    }


def _d042_row(sample, prompt_text, prompt_cap, sequence_cap, tokenizer, schema):
    selection = select_recent_complete_suffix(sample["history"], prompt_text, prompt_cap, tokenizer)
    messages = build_messages(selection["selected_history"], prompt_text)
    completion = serialize_completion(sample["target"], schema)
    full_ids = encode_full_training_sequence(tokenizer, messages, completion)
    return {
        "split": sample["source"]["split"],
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "outcome": _derive_outcome(selection, len(full_ids), sequence_cap),
        "selected_prompt_tokens": len(selection["prompt_ids"]),
        "completion_tokens_full": len(full_ids) - len(selection["prompt_ids"]),
        "selected_full_sequence_tokens": len(full_ids),
        "removed_history_message_count": selection["selected_start_index"],
        "retained_history_message_count": len(sample["history"]) - selection["selected_start_index"],
    }


def _build(sample, caps, context, d042_row=None):
    tokenizer, prompt, schema, validator, _ = context
    prompt_cap, sequence_cap = caps
    d042_row = d042_row or _d042_row(sample, prompt, prompt_cap, sequence_cap, tokenizer, schema)
    return build_training_item(
        sample,
        sample["source"]["split"],
        prompt,
        prompt_cap,
        sequence_cap,
        tokenizer,
        schema,
        validator,
        d042_row,
    )


def test_s01_standard_sample_has_exact_prefix_and_completion_only_labels(sft_context):
    tokenizer, prompt, _, _, _ = sft_context
    sample = _sample()
    item, excluded, diagnostic = _build(sample, (1280, 2048), sft_context)
    assert excluded is None
    assert item is not None
    expected_prompt = encode_inference_prompt(
        tokenizer,
        build_messages(sample["history"], prompt),
    )
    assert item["input_ids"][:item["prompt_length"]] == expected_prompt
    assert item["labels"][:item["prompt_length"]] == [-100] * item["prompt_length"]
    assert item["labels"][item["prompt_length"]:] == item["input_ids"][item["prompt_length"]:]
    assert item["completion_length"] == len(item["input_ids"]) - item["prompt_length"]
    assert diagnostic["full_length"] == len(item["input_ids"])
    assert len(item["input_ids"]) <= 2048


def test_s02_t1_trims_only_the_older_contiguous_history(sft_context):
    _, prompt, _, _, _ = sft_context
    history = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "早期回答 " * 700},
        {"role": "user", "content": "当前问题"},
    ]
    sample = _sample(history=history)
    tokenizer = sft_context[0]
    floor = len(encode_inference_prompt(tokenizer, build_messages([history[-1]], prompt)))
    item, excluded, diagnostic = _build(sample, (floor, 2048), sft_context)
    assert excluded is None and item is not None
    assert diagnostic["selected_start_index"] == 2
    assert diagnostic["removed_history_message_count"] == 2
    assert item["input_ids"][:floor] == encode_inference_prompt(tokenizer, build_messages([history[-1]], prompt))


def test_s03_completion_length_does_not_change_selected_prompt(sft_context):
    first = _sample(response="短回复")
    second = _sample(response="长回复。" * 80)
    item_a, excluded_a, _ = _build(first, (1280, 2048), sft_context)
    item_b, excluded_b, _ = _build(second, (1280, 2048), sft_context)
    assert excluded_a is None and excluded_b is None
    assert item_a is not None and item_b is not None
    assert item_a["prompt_length"] == item_b["prompt_length"]
    assert item_a["input_ids"][:item_a["prompt_length"]] == item_b["input_ids"][:item_b["prompt_length"]]


def test_s04_prompt_floor_excess_is_excluded_without_truncating_current_user(sft_context):
    tokenizer, prompt, _, _, _ = sft_context
    history = [{"role": "user", "content": "当前消息必须完整保留🙂。" * 600}]
    sample = _sample(history=history)
    floor = len(encode_inference_prompt(tokenizer, build_messages(history, prompt)))
    item, excluded, diagnostic = _build(sample, (floor - 1, 2048), sft_context)
    assert item is None
    assert excluded is not None and excluded["reason"] == "prompt_floor_exceeds_cap"
    assert excluded["prompt_tokens"] == floor
    assert diagnostic["selected_start_index"] == 0
    assert len(sample["history"][0]["content"]) == len(history[0]["content"])


def test_s05_complete_json_over_s_is_excluded_and_target_unchanged(sft_context):
    tokenizer, prompt, _, _, _ = sft_context
    sample = _sample(history=[{"role": "user", "content": "当前消息"}], response="完整回复。" * 300)
    floor = len(encode_inference_prompt(tokenizer, build_messages(sample["history"], prompt)))
    item, excluded, _ = _build(sample, (floor + 8, floor + 64), sft_context)
    assert item is None
    assert excluded is not None and excluded["reason"] == "completion_overflows_train_budget"
    assert excluded["full_tokens"] > floor + 64
    assert sample["target"]["response_text"] == "完整回复。" * 300


def test_s06_a0_profile_sorting_and_empty_memory_do_not_mutate_target(sft_context):
    _, _, schema, _, _ = sft_context
    sample = _sample(
        profile={
            "personality_traits": ["sensitive", "extroverted"],
            "interests": ["music", "study_exam"],
            "style": ["rational", "brief"],
        },
        memories=["mem_000001"],
    )
    original = copy.deepcopy(sample["target"])
    serialized = serialize_completion(sample["target"], schema)
    target = json.loads(serialized)
    assert target["user_profile"]["personality_traits"] == ["extroverted", "sensitive"]
    assert target["user_profile"]["interests"] == ["study_exam", "music"]
    assert target["user_profile"]["style"] == ["brief", "rational"]
    assert target["memory_refs"] == []
    assert sample["target"] == original
    item, excluded, _ = _build(sample, (1280, 2048), sft_context)
    assert excluded is None and item is not None


def test_s07_unicode_and_literal_think_text_round_trip(sft_context):
    tokenizer, prompt, schema, _, _ = sft_context
    response = '第一行\n第二行含有"引号"、反斜线 \\、🙂 与字面 </think>。'
    sample = _sample(history=[
        {"role": "user", "content": "换行\nUnicode 🙂"},
        {"role": "assistant", "content": "历史包含字面 </think>，保持原文"},
        {"role": "user", "content": "当前问题"},
    ], response=response)
    completion = serialize_completion(sample["target"], schema)
    selection = select_recent_complete_suffix(sample["history"], prompt, 1280, tokenizer)
    full_ids = encode_full_training_sequence(
        tokenizer,
        build_messages(selection["selected_history"], prompt),
        completion,
    )
    completion_ids = full_ids[len(selection["prompt_ids"]):]
    decoded = tokenizer.decode(completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert decoded.startswith(completion)
    assert json.loads(completion)["response_text"] == response
    assert "</think>" in decoded
    item, excluded, _ = _build(sample, (1280, 2048), sft_context)
    assert excluded is None and item is not None


def test_s08_first_completion_token_and_terminal_eos_are_supervised(sft_context):
    tokenizer, _, _, _, _ = sft_context
    sample = _sample()
    item, excluded, _ = _build(sample, (1280, 2048), sft_context)
    assert excluded is None and item is not None
    completion_ids = item["input_ids"][item["prompt_length"]:]
    eos_id = tokenizer.eos_token_id
    terminal_eos = max(index for index, token in enumerate(completion_ids) if token == eos_id)
    assert item["labels"][item["prompt_length"]] == completion_ids[0]
    assert item["labels"][item["prompt_length"] + terminal_eos] == eos_id
    assert any(label != -100 for label in item["labels"])


def test_s12_any_d042_per_sample_length_or_outcome_drift_fails(sft_context):
    sample = _sample()
    row = _d042_row(sample, sft_context[1], 1280, 2048, sft_context[0], sft_context[2])
    row["selected_full_sequence_tokens"] += 1
    with pytest.raises(ValueError, match="selected_full_sequence_tokens"):
        _build(sample, (1280, 2048), sft_context, row)


def test_s16_invalid_s_p_overrides_are_rejected_before_build():
    config = load_config(ROOT / "configs/sft_data.yaml")
    args = parse_args([
        "--max-train-seq-length", "2048",
        "--max-prompt-tokens", "2048",
        "--dataset-dir", "data/processed/sft/invalid",
        "--artifacts-dir", "outputs/sft_data/invalid",
    ])
    with pytest.raises(ValueError, match="smaller than S"):
        _apply_overrides(config, args)


def test_s09_pad_id_equal_to_eos_does_not_mask_real_eos():
    eos = 9
    collator = SFTDataCollator(pad_token_id=eos, max_sequence_length=8)
    short = {
        "sample_id": "short",
        "prompt_length": 2,
        "completion_length": 2,
        "input_ids": [1, 2, 3, eos],
        "labels": [-100, -100, 3, eos],
    }
    long = {
        "sample_id": "long",
        "prompt_length": 2,
        "completion_length": 4,
        "input_ids": [4, 5, 6, 7, 8, eos],
        "labels": [-100, -100, 6, 7, 8, eos],
    }
    batch = collator([short, long])
    assert batch["input_ids"][0].tolist() == [1, 2, 3, eos, eos, eos]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]
    assert batch["labels"][0].tolist() == [-100, -100, 3, eos, -100, -100]
    assert batch["labels"][0, 3].item() == eos


def test_s10_real_tokenizer_batch_shapes_and_prefixes(tmp_path, sft_context):
    tokenizer, _, _, _, _ = sft_context
    rows = []
    for index, response in enumerate(("短。", "稍长一些。" * 5, "不同长度回复。" * 20, "最长回复。" * 55)):
        sample = _sample(response=response)
        sample["sample_id"] = f"train:train_conv_{index + 1:06d}:pp0"
        sample["conversation_id"] = f"train_conv_{index + 1:06d}"
        item, excluded, _ = _build(sample, (1280, 2048), sft_context)
        assert excluded is None and item is not None
        rows.append(item)

    collator = SFTDataCollator(
        int(tokenizer.pad_token_id),
        max_sequence_length=2048,
    )
    for size in (1, 2, 4):
        batch = collator(rows[:size])
        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
        assert int(batch["attention_mask"].sum().item()) == sum(len(row["input_ids"]) for row in rows[:size])
        for index, row in enumerate(rows[:size]):
            n = len(row["input_ids"])
            assert batch["input_ids"][index, :n].tolist() == row["input_ids"]
            assert batch["labels"][index, :n].tolist() == row["labels"]
            assert batch["attention_mask"][index, :n].tolist() == [1] * n
            assert batch["labels"][index, n:].tolist() == [-100] * (batch["labels"].shape[1] - n)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "dev.jsonl").write_text(train_path.read_text(encoding="utf-8"), encoding="utf-8")
    summary = validate_dataset_batches(
        tmp_path,
        split_names=("train", "dev"),
        batch_sizes=(1, 2, 4),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        max_sequence_length=2048,
    )
    assert summary["all_batch_invariants_pass"] is True
    assert all(len(value["checks"]) == 3 for value in summary["splits"].values())


def test_s10_padding_rounding_refuses_to_truncate_or_exceed_s():
    feature = {
        "sample_id": "x",
        "prompt_length": 1,
        "completion_length": 4,
        "input_ids": [1, 2, 3, 4, 5],
        "labels": [-100, 2, 3, 4, 5],
    }
    collator = SFTDataCollator(0, pad_to_multiple_of=4, max_sequence_length=8)
    assert collator([feature])["input_ids"].shape == (1, 8)
    too_small = SFTDataCollator(0, pad_to_multiple_of=8, max_sequence_length=7)
    with pytest.raises(ValueError, match="would exceed max_sequence_length"):
        too_small([feature])


def test_s15_pinned_trl_trainer_dataloader_preserves_prebuilt_labels(tmp_path, sft_context):
    tokenizer, _, _, _, _ = sft_context
    for split in ("train", "dev"):
        rows = []
        for index, response in enumerate(("短回复。", "不同长度的回复。" * 12)):
            sample = _sample(split, response=response)
            sample["sample_id"] = f"{'val' if split == 'val' else 'train'}:train_conv_{index + 20:06d}:pp0"
            sample["conversation_id"] = f"train_conv_{index + 20:06d}"
            item, excluded, _ = _build(sample, (1280, 2048), sft_context)
            assert excluded is None and item is not None
            rows.append(item)
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    result = validate_trainer_data_path(
        tmp_path,
        tokenizer=tokenizer,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        max_sequence_length=2048,
        batch_size=2,
    )
    assert result["status"] == "passed"
    assert result["trainer"] == "trl.SFTTrainer"
    assert result["versions"] == {
        "transformers": "5.20.2",
        "trl": "1.14.2",
        "datasets": "5.1.0",
        "torch": "2.10.0+cu128",
    }
    assert result["dataset_processing_skipped"] is True
    assert result["max_length"] is None
    assert result["packing"] is False and result["padding_free"] is False
    assert result["train_labels_equal_custom_collator"] is True
    assert result["dev_labels_equal_custom_collator"] is True
    assert result["no_sample_metadata_passed_to_model"] is True
    assert result["model"]["forward_calls"] == 0
    assert result["model"]["training_loop_called"] is False
    assert result["model"]["weights_sha256_before"] == result["model"]["weights_sha256_after"]

