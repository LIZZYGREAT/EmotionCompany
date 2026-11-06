"""Prompt-only classification data and deterministic paired schedule for E03-A7."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from src.data.e03_a5_labels import (
    DIMENSIONS,
    canonical_sha256,
    iter_jsonl,
    schema_label_order,
    sha256_file,
    validate_sft_row,
)


EMOTION_ORDER_VERSION = "official_train_schema_emotion_order_v1"
SCHEDULE_VERSION = "e03_a7_fit_epoch_then_second_pass_v1"
FIT_EPOCH_BOUNDARY_STEP = 2826
MAX_OPTIMIZER_STEPS = 4048
GRADIENT_ACCUMULATION_STEPS = 4


def emotion_label_order(train_schema: Mapping[str, Any], output_schema: Mapping[str, Any]) -> list[str]:
    """Read the 16 class labels from the protected Train schema and cross-check output order."""
    try:
        labels = list(train_schema["$defs"]["emotion"]["enum"])
        output_labels = list(output_schema["properties"]["emotion_label"]["enum"])
    except (KeyError, TypeError) as exc:
        raise ValueError("official schemas lack the frozen emotion enum arrays") from exc
    if len(labels) != 16 or len(labels) != len(set(labels)) or any(not isinstance(v, str) or not v for v in labels):
        raise ValueError("official Train emotion enum must contain 16 unique strings")
    if output_labels != ["", *labels]:
        raise ValueError("Train and output emotion enum order differs")
    return labels


def read_supervision(path: Path) -> dict[str, dict[str, Any]]:
    """Read A5's audited 33-label targets without exposing source prose to the model."""
    rows: dict[str, dict[str, Any]] = {}
    for line_number, row, _row_sha in iter_jsonl(path):
        sample_id = row.get("sample_id")
        target = row.get("y")
        mask = row.get("mask")
        if not isinstance(sample_id, str) or not sample_id or sample_id in rows:
            raise ValueError(f"A5 supervision row {line_number} has a missing or duplicate sample_id")
        if (
            not isinstance(target, list) or len(target) != 33
            or not isinstance(mask, list) or len(mask) != 33
            or any(value not in (0, 1) for value in target)
            or mask != [1] * 33
        ):
            raise ValueError(f"{sample_id}: A5 profile target must be 33 closed-world bits")
        rows[sample_id] = row
    return rows


def read_splits(path: Path, expected_fit_ids_sha256: str | None = None) -> dict[str, list[str]]:
    """Load frozen A5 conversation partitions and require exact, disjoint coverage."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != "e03_a5_conversation_group_stratified_v1":
        raise ValueError("A7 requires the frozen A5 v1 conversation split")
    source = value.get("splits")
    if not isinstance(source, Mapping):
        raise ValueError("A5 split artifact lacks split rows")
    result = {name: list(source[name]["sample_ids"]) for name in ("fit", "calibration", "internal_holdout")}
    sets = {name: set(ids) for name, ids in result.items()}
    if any(len(ids) != len(sets[name]) for name, ids in result.items()):
        raise ValueError("A5 split has duplicate sample IDs")
    if sets["fit"] & sets["calibration"] or sets["fit"] & sets["internal_holdout"] or sets["calibration"] & sets["internal_holdout"]:
        raise ValueError("A5 sample partitions overlap")
    if expected_fit_ids_sha256 and canonical_sha256(sorted(result["fit"])) != expected_fit_ids_sha256:
        raise ValueError("A5 Fit sample ID SHA differs from the expected frozen value")
    return result


def build_supervision(
    normalized_train_path: Path,
    a5_supervision_path: Path,
    train_schema: Mapping[str, Any],
    output_schema: Mapping[str, Any],
    *,
    expected_rows: int = 16_312,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Join A5 profile targets to Train emotion targets by exact sample ID."""
    profile_order, _enums = schema_label_order(train_schema, output_schema)
    if len(profile_order) != 33:
        raise AssertionError("official profile schema must yield 33 labels")
    emotions = emotion_label_order(train_schema, output_schema)
    emotion_to_index = {label: index for index, label in enumerate(emotions)}
    profile_targets = read_supervision(a5_supervision_path)
    normalized: dict[str, Mapping[str, Any]] = {}
    conversation_by_id: dict[str, str] = {}
    history_lengths: dict[str, int] = {}
    total = 0
    for line_number, row, _row_sha in iter_jsonl(normalized_train_path):
        total += 1
        sample_id = row.get("sample_id")
        conversation_id = row.get("conversation_id")
        source = row.get("source")
        target = row.get("target")
        history = row.get("history")
        if not isinstance(sample_id, str) or not sample_id or sample_id in normalized:
            raise ValueError(f"normalized Train row {line_number} has missing or duplicate sample_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"{sample_id}: normalized Train has no conversation_id")
        if not isinstance(source, Mapping) or source.get("split") != "train":
            raise ValueError(f"{sample_id}: D0.3 source is outside Train")
        if not isinstance(history, list) or not history:
            raise ValueError(f"{sample_id}: normalized history must be a non-empty list")
        if not isinstance(target, Mapping):
            raise ValueError(f"{sample_id}: normalized Train target is missing")
        emotion = target.get("emotion_label")
        if emotion not in emotion_to_index:
            raise ValueError(f"{sample_id}: emotion target is outside the official 16-class schema")
        normalized[sample_id] = row
        conversation_by_id[sample_id] = conversation_id
        history_lengths[sample_id] = len(history)
    if total != expected_rows or not set(profile_targets) <= set(normalized):
        raise ValueError("A5 profile supervision IDs are not a subset of the D0.3 Train source")

    result: dict[str, dict[str, Any]] = {}
    for sample_id, profile_row in profile_targets.items():
        source_row = normalized[sample_id]
        target = source_row["target"]
        result[sample_id] = {
            "sample_id": sample_id,
            "conversation_id": conversation_by_id[sample_id],
            "profile_target": list(profile_targets[sample_id]["y"]),
            "profile_mask": list(profile_targets[sample_id]["mask"]),
            "emotion_label": target["emotion_label"],
            "emotion_target": emotion_to_index[target["emotion_label"]],
            "history_message_count": history_lengths[sample_id],
        }
    summary = {
        "normalized_train_rows": total,
        "supervision_rows": len(profile_targets),
        "sft_excluded_train_rows": total - len(profile_targets),
        "profile_label_order_version": "official_train_schema_profile_order_v1",
        "profile_label_count": len(profile_order),
        "emotion_label_order_version": EMOTION_ORDER_VERSION,
        "emotion_label_order": emotions,
        "emotion_label_count": len(emotions),
        "sample_id_sha256": canonical_sha256(sorted(result)),
        "conversation_count": len({row["conversation_id"] for row in result.values()}),
        "source_hashes": {
            "normalized_train": sha256_file(normalized_train_path),
            "a5_supervision": sha256_file(a5_supervision_path),
        },
    }
    return result, summary


def validate_conversation_splits(
    split_ids: Mapping[str, Sequence[str]], supervision: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require ID-complete Fit/Calibration/Holdout partitions with no shared sessions."""
    expected = set(supervision)
    actual: set[str] = set()
    conversations: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}
    for name in ("fit", "calibration", "internal_holdout"):
        ids = list(split_ids[name])
        if len(ids) != len(set(ids)) or not set(ids) <= expected:
            raise ValueError(f"A5 {name} partition has duplicate or unknown IDs")
        if actual & set(ids):
            raise ValueError("A5 sample partitions overlap")
        actual.update(ids)
        groups = {str(supervision[sample_id]["conversation_id"]) for sample_id in ids}
        conversations[name] = groups
        summary[name] = {
            "sample_count": len(ids),
            "conversation_count": len(groups),
            "sample_ids_sha256": canonical_sha256(sorted(ids)),
            "conversation_ids_sha256": canonical_sha256(sorted(groups)),
        }
    if actual != expected:
        raise ValueError("A5 partitions do not cover all supervised D0.3 Train samples")
    if any(conversations[left] & conversations[right] for left, right in (
        ("fit", "calibration"), ("fit", "internal_holdout"), ("calibration", "internal_holdout")
    )):
        raise ValueError("A5 conversation partitions overlap")
    return summary


def build_schedule(
    fit_sample_ids: Sequence[str],
    *,
    optimizer_steps: int = MAX_OPTIMIZER_STEPS,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    seed: int = 42,
) -> dict[str, Any]:
    """Freeze one full Fit pass followed by the required prefix of a second permutation."""
    ids = list(fit_sample_ids)
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Fit IDs must be non-empty and unique")
    if micro_batch_size != 1 or gradient_accumulation_steps != 4:
        raise ValueError("A7 v1 freezes microbatch=1 and gradient accumulation=4")
    if optimizer_steps != MAX_OPTIMIZER_STEPS or len(ids) != 11_304:
        raise ValueError("A7 v1 freezes 11,304 Fit rows and 4,048 cumulative optimizer steps")
    exposures = optimizer_steps * micro_batch_size * gradient_accumulation_steps
    if exposures <= len(ids) or exposures - len(ids) >= len(ids):
        raise ValueError("A7 v1 schedule must contain one complete Fit pass and a second-pass prefix")
    ordered = sorted(ids)
    first = list(ordered)
    second = list(ordered)
    random.Random(seed).shuffle(first)
    random.Random(seed + 1).shuffle(second)
    draws = first + second[: exposures - len(first)]
    steps = []
    for index in range(optimizer_steps):
        start = index * gradient_accumulation_steps
        step_draws = draws[start : start + gradient_accumulation_steps]
        steps.append({
            "optimizer_step": index + 1,
            "epoch": 1 if index < FIT_EPOCH_BOUNDARY_STEP else 2,
            "sample_ids": step_draws,
        })
    schedule_sha = canonical_sha256(steps)
    return {
        "version": SCHEDULE_VERSION,
        "seed": seed,
        "source_split": "train_fit",
        "fit_sample_count": len(ids),
        "optimizer_steps": optimizer_steps,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "draw_count": exposures,
        "unique_fit_ids": len(ids),
        "fit_epoch_boundary_step": FIT_EPOCH_BOUNDARY_STEP,
        "first_epoch_ids_sha256": canonical_sha256(first),
        "second_epoch_ids_sha256": canonical_sha256(second),
        "ordered_draw_ids_sha256": canonical_sha256(draws),
        "draw_ids_set_sha256": canonical_sha256(sorted(set(draws))),
        "schedule_sha256": schedule_sha,
        "steps": steps,
    }


def verify_schedule(schedule: Mapping[str, Any], fit_sample_ids: Sequence[str]) -> dict[str, Any]:
    """Recheck step coverage, Fit membership, epoch boundary, and immutable schedule digest."""
    steps = schedule.get("steps")
    fit_ids = set(fit_sample_ids)
    if not isinstance(steps, list) or len(steps) != MAX_OPTIMIZER_STEPS:
        raise ValueError("A7 schedule must cover all 4,048 optimizer steps")
    draws: list[str] = []
    for expected_step, row in enumerate(steps, 1):
        values = row.get("sample_ids")
        expected_epoch = 1 if expected_step <= FIT_EPOCH_BOUNDARY_STEP else 2
        if (
            row.get("optimizer_step") != expected_step
            or row.get("epoch") != expected_epoch
            or not isinstance(values, list)
            or len(values) != GRADIENT_ACCUMULATION_STEPS
            or any(sample_id not in fit_ids for sample_id in values)
        ):
            raise ValueError(f"A7 schedule row {expected_step} violates the frozen Fit contract")
        draws.extend(values)
    first_pass = draws[: len(fit_ids)]
    if len(first_pass) != len(fit_ids) or set(first_pass) != fit_ids or len(first_pass) != len(set(first_pass)):
        raise ValueError("A7 first pass must consume every Fit ID exactly once")
    if schedule.get("fit_epoch_boundary_step") != FIT_EPOCH_BOUNDARY_STEP:
        raise ValueError("A7 schedule epoch boundary differs from 2,826 steps")
    if schedule.get("schedule_sha256") != canonical_sha256(steps):
        raise ValueError("A7 schedule SHA differs from its contents")
    return {
        "status": "schedule_validated",
        "optimizer_steps": len(steps),
        "draw_count": len(draws),
        "unique_fit_ids": len(fit_ids),
        "first_pass_exactly_once": True,
        "first_epoch_boundary_step": FIT_EPOCH_BOUNDARY_STEP,
        "schedule_sha256": schedule["schedule_sha256"],
        "ordered_draw_ids_sha256": canonical_sha256(draws),
    }


class SFTPromptIndex:
    """Seekable index of validated SFT rows that returns Prompt IDs only."""

    def __init__(self, path: Path, expected_ids: Sequence[str], *, max_sequence_length: int = 2048, max_prompt_tokens: int = 1280):
        self.path = path
        self.offsets: dict[str, int] = {}
        self.prompt_lengths: dict[str, int] = {}
        expected = set(expected_ids)
        with path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    raise ValueError(f"blank SFT JSONL row at line {line_number}")
                row = json.loads(line)
                facts = validate_sft_row(row, max_sequence_length=max_sequence_length)
                sample_id = str(facts["sample_id"])
                if sample_id in self.offsets:
                    raise ValueError(f"duplicate SFT sample_id: {sample_id}")
                prompt_length = int(facts["prompt_length"])
                if prompt_length > max_prompt_tokens:
                    raise ValueError(f"{sample_id}: Prompt exceeds the frozen P=1280 budget")
                self.offsets[sample_id] = offset
                self.prompt_lengths[sample_id] = prompt_length
        if set(self.offsets) != expected:
            raise ValueError("SFT Train IDs differ from the frozen D0.3/A5 supervision IDs")
        self.row_count = line_number
        self.max_prompt_length = max(self.prompt_lengths.values())

    def prompt_ids(self, sample_id: str) -> list[int]:
        if sample_id not in self.offsets:
            raise KeyError(f"unknown SFT sample_id: {sample_id}")
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[sample_id])
            row = json.loads(handle.readline())
        length = self.prompt_lengths[sample_id]
        values = row["input_ids"][:length]
        if len(values) != length:
            raise ValueError(f"{sample_id}: Prompt token count differs from prompt_length")
        return [int(value) for value in values]

    def iter_prompt_rows(self, sample_ids: Sequence[str]) -> Iterator[dict[str, Any]]:
        for sample_id in sample_ids:
            yield {
                "sample_id": sample_id,
                "input_ids": self.prompt_ids(sample_id),
                "prompt_length": self.prompt_lengths[sample_id],
            }


def verify_prompt_equivalence(
    normalized_train_path: Path,
    prompt_index: SFTPromptIndex,
    tokenizer: Any,
    prompt_text: str,
    *,
    prompt_cap: int = 1280,
) -> dict[str, Any]:
    """Compare every eligible saved SFT Prompt prefix with P0/F2/T1 inference token IDs."""
    from participant import run_inference as inference

    checked = 0
    over_budget = 0
    mismatch_ids: list[str] = []
    for _line, row, _sha in iter_jsonl(normalized_train_path):
        sample_id = row.get("sample_id")
        if sample_id not in prompt_index.offsets:
            continue
        sample = {"id": sample_id, "history": row.get("history")}
        selection, _messages = inference.prepare_sample_messages(sample, prompt_text, tokenizer, prompt_cap)
        if selection.get("floor_exceeds_cap"):
            over_budget += 1
            continue
        actual = [int(value) for value in selection.get("prompt_ids", [])]
        expected = prompt_index.prompt_ids(str(sample_id))
        checked += 1
        if actual != expected:
            mismatch_ids.append(str(sample_id))
            if len(mismatch_ids) >= 10:
                break
    if over_budget or checked != len(prompt_index.offsets) or mismatch_ids:
        raise ValueError(
            f"SFT Prompt equivalence failed: checked={checked}, rows={len(prompt_index.offsets)}, "
            f"over_budget={over_budget}, mismatches={mismatch_ids[:10]}"
        )
    return {
        "status": "all_prompt_ids_exact_match",
        "checked_rows": checked,
        "over_budget_rows": 0,
        "gold_completion_suffix_used_as_model_input": False,
        "prompt_protocol": "P0_F2_T1",
    }
