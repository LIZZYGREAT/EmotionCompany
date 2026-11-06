"""Build one lossless, fully supervised F4 completion for A6 format ablations."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from src.shared.f4_protocol import parse_f4, serialize_f4
from src.shared.prompting import (
    build_messages,
    encode_full_training_sequence,
    encode_inference_prompt,
    validate_completion_suffix,
)


def build_f4_training_item(
    sample: dict[str, Any],
    f2_item: dict[str, Any],
    *,
    p0_prompt: str,
    f4_prompt: str,
    prompt_cap: int,
    sequence_cap: int,
    tokenizer: Any,
    official_schema: dict[str, Any],
    selected_start_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retokenize one frozen P0/T1 history with F4 fields and lossless labels."""
    from src.shared.f3_protocol import submission_enums

    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id or f2_item.get("sample_id") != sample_id:
        raise ValueError("normalized and F2 rows must have the same non-empty sample_id")
    history = sample.get("history")
    target = sample.get("target")
    if not isinstance(history, list) or not history or not isinstance(target, dict):
        raise ValueError(f"{sample_id}: normalized history/target contract is invalid")
    definitions = official_schema.get("$defs", {})
    if "target" not in definitions:
        raise ValueError("official data schema does not define $defs.target")
    target_schema = {
        "$schema": official_schema.get(
            "$schema", "https://json-schema.org/draft/2020-12/schema"
        ),
        "$ref": "#/$defs/target",
        "$defs": definitions,
    }
    Draft202012Validator(target_schema).validate(target)

    if isinstance(selected_start_index, bool) or not isinstance(selected_start_index, int):
        raise ValueError(f"{sample_id}: selected_start_index must be an integer")
    if not 0 <= selected_start_index < len(history):
        raise ValueError(f"{sample_id}: selected_start_index is out of range")
    selected_history = history[selected_start_index:]
    p0_prompt_ids = encode_inference_prompt(
        tokenizer, build_messages(selected_history, p0_prompt)
    )
    old_prompt_length = f2_item.get("prompt_length")
    old_ids = f2_item.get("input_ids")
    if (
        not isinstance(old_prompt_length, int)
        or not isinstance(old_ids, list)
        or old_prompt_length != len(p0_prompt_ids)
        or old_ids[:old_prompt_length] != p0_prompt_ids
    ):
        raise ValueError(f"{sample_id}: recomputed P0/T1 Token IDs differ from frozen F2 dataset")
    if len(p0_prompt_ids) > prompt_cap:
        raise ValueError(f"{sample_id}: frozen P0/T1 prompt exceeds P={prompt_cap}")

    messages = build_messages(selected_history, f4_prompt)
    f4_prompt_ids = encode_inference_prompt(tokenizer, messages)
    if len(f4_prompt_ids) > prompt_cap:
        raise ValueError(
            f"{sample_id}: F4 prompt exceeds P={prompt_cap} using the exact P0/T1 history "
            f"({len(f4_prompt_ids)} tokens)"
        )
    completion = serialize_f4(target, official_schema)
    enums = submission_enums({
        "properties": {
            "emotion_label": {"enum": [v for v in definitions["emotion"]["enum"] if v]},
            "user_profile": {"properties": definitions["profile"]["properties"]},
        }
    })
    parsed = parse_f4(completion, enums)
    expected_profile = {
        name: sorted(target["user_profile"][name], key=enums[name].index)
        for name in ("personality_traits", "interests", "style")
    }
    if parsed["emotion_label"] != target["emotion_label"]:
        raise ValueError(f"{sample_id}: F4 round trip changed emotion_label")
    if parsed["user_profile"] != expected_profile:
        raise ValueError(f"{sample_id}: F4 round trip changed canonical user_profile")
    if parsed["response_text"] != target["response_text"] or parsed["memory_refs"] != []:
        raise ValueError(f"{sample_id}: F4 round trip changed response_text or memory_refs")

    full_ids = encode_full_training_sequence(tokenizer, messages, completion)
    completion_ids = validate_completion_suffix(
        tokenizer, completion, f4_prompt_ids, full_ids, sample_id=sample_id
    )
    if len(full_ids) > sequence_cap:
        raise ValueError(
            f"{sample_id}: F4 sequence exceeds S={sequence_cap} ({len(full_ids)} tokens)"
        )
    labels = [-100] * len(f4_prompt_ids) + completion_ids
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None or not completion_ids or int(eos_id) not in completion_ids:
        raise ValueError(f"{sample_id}: template EOS is missing from the supervised completion")
    if labels[:len(f4_prompt_ids)] != [-100] * len(f4_prompt_ids):
        raise RuntimeError(f"{sample_id}: F4 prompt tokens are not fully masked")
    if labels[len(f4_prompt_ids):] != full_ids[len(f4_prompt_ids):]:
        raise RuntimeError(f"{sample_id}: F4 completion tokens are not fully supervised")

    item = {
        "sample_id": sample_id,
        "conversation_id": sample["conversation_id"],
        "source_split": f2_item["source_split"],
        "source": sample["source"],
        "prompt_length": len(f4_prompt_ids),
        "completion_length": len(completion_ids),
        "input_ids": full_ids,
        "labels": labels,
    }
    diagnostic = {
        "sample_id": sample_id,
        "selected_start_index": selected_start_index,
        "p0_prompt_tokens": len(p0_prompt_ids),
        "f4_prompt_tokens": len(f4_prompt_ids),
        "f2_completion_tokens": int(f2_item["completion_length"]),
        "f4_completion_tokens": len(completion_ids),
        "f4_sequence_tokens": len(full_ids),
        "p0_selected_history_matches_f2": True,
        "completion_round_trip_lossless": True,
        "prompt_labels_masked": True,
        "terminal_eos_supervised": int(eos_id) in labels[len(f4_prompt_ids):],
    }
    return item, diagnostic
