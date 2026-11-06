"""Build paired, classification-only E03-A8 SFT rows from frozen Train sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from src.shared.e03_a8_protocol import (
    PROFILE_DIMS,
    serialize_specialist_completion,
    validate_training_target_round_trip,
)
from src.shared.prompting import (
    build_messages,
    encode_full_training_sequence,
    encode_inference_prompt,
    validate_completion_suffix,
)

ARMS = ("P", "EP")


class PromptFloorExceeded(ValueError):
    def __init__(self, sample_id: str, prompt_lengths: Mapping[str, int], prompt_cap: int):
        super().__init__(
            f"{sample_id}: current-user-only prompt exceeds P={prompt_cap}: {dict(prompt_lengths)}"
        )
        self.sample_id = sample_id
        self.prompt_lengths = dict(prompt_lengths)
        self.prompt_cap = prompt_cap


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_common_history(
    history: Sequence[Mapping[str, Any]],
    *,
    p_prompt: str,
    ep_prompt: str,
    tokenizer: Any,
    initial_start_index: int,
    prompt_cap: int,
) -> dict[str, Any]:
    if not isinstance(initial_start_index, int) or isinstance(initial_start_index, bool):
        raise ValueError("initial_start_index must be an integer")
    if not 0 <= initial_start_index < len(history):
        raise ValueError("initial_start_index is out of range")
    if not history or history[-1].get("role") != "user":
        raise ValueError("A8 history must end with the current user message")

    def encode(start: int) -> dict[str, list[int]]:
        suffix = list(history[start:])
        return {
            "P": encode_inference_prompt(tokenizer, build_messages(suffix, p_prompt)),
            "EP": encode_inference_prompt(tokenizer, build_messages(suffix, ep_prompt)),
        }

    start = initial_start_index
    prompt_ids = encode(start)
    while max(map(len, prompt_ids.values())) > prompt_cap:
        if start == len(history) - 1:
            raise PromptFloorExceeded(
                "<current-sample>", {arm: len(ids) for arm, ids in prompt_ids.items()}, prompt_cap
            )
        start += 1
        prompt_ids = encode(start)
    selected = list(history[start:])
    if not selected or selected[-1].get("role") != "user":
        raise RuntimeError("common history selection lost the current user message")
    fingerprint = canonical_sha256(selected)
    return {
        "selected_start_index": start,
        "selected_history": selected,
        "selected_history_sha256": fingerprint,
        "prompt_ids": prompt_ids,
        "prompt_lengths": {arm: len(ids) for arm, ids in prompt_ids.items()},
        "extra_removed_messages": start - initial_start_index,
    }


def build_a8_training_item(
    sample: Mapping[str, Any],
    f2_item: Mapping[str, Any],
    budget_audit: Mapping[str, Any],
    *,
    p0_prompt: str,
    p_prompt: str,
    ep_prompt: str,
    prompt_cap: int,
    sequence_cap: int,
    tokenizer: Any,
    official_schema: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id or f2_item.get("sample_id") != sample_id:
        raise ValueError("normalized and frozen F2 rows must have the same non-empty sample_id")
    if f2_item.get("source_split") != "train":
        raise ValueError(f"{sample_id}: A8 training rows must come from the frozen Train split")
    history = sample.get("history")
    target = sample.get("target")
    if not isinstance(history, list) or not history or not isinstance(target, Mapping):
        raise ValueError(f"{sample_id}: normalized history/target contract is invalid")
    if sample.get("conversation_id") != f2_item.get("conversation_id"):
        raise ValueError(f"{sample_id}: normalized/F2 conversation identity differs")
    start = budget_audit.get("removed_history_message_count")
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError(f"{sample_id}: D0.4.2 selected history index is missing")
    if budget_audit.get("outcome") not in {"full_fit", "trimmed_fit"}:
        raise ValueError(f"{sample_id}: D0.4.2 did not accept the frozen P0/T1 sample")

    old_history = history[start:]
    p0_ids = encode_inference_prompt(tokenizer, build_messages(old_history, p0_prompt))
    old_prompt_length = f2_item.get("prompt_length")
    old_ids = f2_item.get("input_ids")
    if (
        not isinstance(old_prompt_length, int)
        or not isinstance(old_ids, list)
        or old_prompt_length != len(p0_ids)
        or old_ids[:old_prompt_length] != p0_ids
    ):
        raise ValueError(f"{sample_id}: reconstructed P0/T1 Prompt IDs differ from frozen F2")
    if len(p0_ids) > prompt_cap:
        raise ValueError(f"{sample_id}: frozen P0/T1 prompt exceeds P={prompt_cap}")

    try:
        common = select_common_history(
            history,
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            tokenizer=tokenizer,
            initial_start_index=start,
            prompt_cap=prompt_cap,
        )
    except PromptFloorExceeded as error:
        error.sample_id = sample_id
        error.args = (
            f"{sample_id}: current-user-only prompt exceeds P={prompt_cap}: "
            f"{error.prompt_lengths}",
        )
        raise

    rows: dict[str, dict[str, Any]] = {}
    completion_lengths: dict[str, int] = {}
    sequence_lengths: dict[str, int] = {}
    prompt_texts = {"P": p_prompt, "EP": ep_prompt}
    for arm in ARMS:
        validate_training_target_round_trip(target, official_schema, arm)
        completion = serialize_specialist_completion(target, official_schema, arm)
        prompt_ids = list(common["prompt_ids"][arm])
        if len(prompt_ids) > prompt_cap:
            raise ValueError(f"{sample_id}/{arm}: common history still exceeds P={prompt_cap}")
        messages = build_messages(common["selected_history"], prompt_texts[arm])
        full_ids = encode_full_training_sequence(tokenizer, messages, completion)
        completion_ids = validate_completion_suffix(
            tokenizer, completion, prompt_ids, full_ids, sample_id=f"{sample_id}/{arm}"
        )
        if len(full_ids) > sequence_cap:
            raise ValueError(
                f"{sample_id}/{arm}: complete classification sequence exceeds S={sequence_cap} ({len(full_ids)})"
            )
        labels = [-100] * len(prompt_ids) + list(completion_ids)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None or int(eos_id) not in completion_ids:
            raise ValueError(f"{sample_id}/{arm}: supervised completion lacks the template EOS token")
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{sample_id}/{arm}: encoded training row does not begin with inference Prompt IDs")
        if labels[:len(prompt_ids)] != [-100] * len(prompt_ids):
            raise ValueError(f"{sample_id}/{arm}: Prompt labels are not fully masked")
        if labels[len(prompt_ids):] != full_ids[len(prompt_ids):]:
            raise ValueError(f"{sample_id}/{arm}: supervised Completion labels do not match input IDs")
        rows[arm] = {
            "sample_id": sample_id,
            "conversation_id": sample["conversation_id"],
            "source_split": "train",
            "source": sample.get("source"),
            "prompt_length": len(prompt_ids),
            "completion_length": len(completion_ids),
            "input_ids": list(full_ids),
            "labels": labels,
            "common_history_sha256": common["selected_history_sha256"],
        }
        completion_lengths[arm] = len(completion_ids)
        sequence_lengths[arm] = len(full_ids)

    if rows["P"]["common_history_sha256"] != rows["EP"]["common_history_sha256"]:
        raise RuntimeError(f"{sample_id}: P and EP common history diverged")
    audit = {
        "sample_id": sample_id,
        "conversation_id": sample["conversation_id"],
        "selected_start_index": common["selected_start_index"],
        "d042_selected_start_index": start,
        "extra_removed_messages": common["extra_removed_messages"],
        "selected_history_sha256": common["selected_history_sha256"],
        "p0_prompt_tokens": len(p0_ids),
        "p_prompt_tokens": common["prompt_lengths"]["P"],
        "ep_prompt_tokens": common["prompt_lengths"]["EP"],
        "p_completion_tokens": completion_lengths["P"],
        "ep_completion_tokens": completion_lengths["EP"],
        "p_sequence_tokens": sequence_lengths["P"],
        "ep_sequence_tokens": sequence_lengths["EP"],
        "p_prompt_masked": True,
        "ep_prompt_masked": True,
        "p_terminal_eos_supervised": int(tokenizer.eos_token_id) in rows["P"]["labels"][rows["P"]["prompt_length"]:],
        "ep_terminal_eos_supervised": int(tokenizer.eos_token_id) in rows["EP"]["labels"][rows["EP"]["prompt_length"]:],
        "p_excludes_emotion_target": True,
        "both_exclude_response_and_memory_refs": True,
    }
    return rows, audit


def make_sample_order(sample_ids: Sequence[str], *, seed: int, accumulation_steps: int) -> list[dict[str, Any]]:
    import random

    ids = list(sample_ids)
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("sample order input must contain unique non-empty IDs")
    if accumulation_steps <= 0 or len(ids) % accumulation_steps:
        raise ValueError("sample count must be divisible by a positive accumulation step count")
    random.Random(seed).shuffle(ids)
    return [
        {"optimizer_step": index + 1, "sample_ids": ids[index * accumulation_steps:(index + 1) * accumulation_steps]}
        for index in range(len(ids) // accumulation_steps)
    ]
