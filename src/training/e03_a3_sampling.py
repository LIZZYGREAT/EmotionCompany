"""Deterministic A3 short-history resampling over pre-encoded SFT rows."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from src.diagnostics.e03_a3_profile import length_bucket, validate_history


DIMENSIONS = ("personality_traits", "interests", "style")
NATURAL = "natural_full_train"
SHORT_EMPTY = "short_history_gold_empty"
SHORT_NONEMPTY = "short_history_gold_nonempty"


@dataclass(frozen=True)
class ScheduledDraw:
    row_index: int
    component: str


def _unique(rows: Iterable[dict[str, Any]], key: str, description: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{description} has a missing or duplicate {key}")
        result[value] = row
    return result


def _validate_profile(profile: Any, enums: dict[str, list[str]], sample_id: str) -> dict[str, list[str]]:
    if not isinstance(profile, dict) or set(profile) != set(DIMENSIONS):
        raise ValueError(f"{sample_id}: Gold profile dimensions differ from the locked schema")
    result: dict[str, list[str]] = {}
    for dimension in DIMENSIONS:
        values = profile[dimension]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or not set(values) <= set(enums[dimension])
        ):
            raise ValueError(f"{sample_id}: invalid or duplicate Gold labels in {dimension}")
        result[dimension] = list(values)
    return result


def prepare_training_records(
    sft_rows: list[dict[str, Any]],
    normalized_train_rows: list[dict[str, Any]],
    enums: dict[str, list[str]],
    *,
    expected_rows: int,
    eos_token_id: int,
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    """Join D0.4.3 token rows to verified D0.3 sources by sample ID."""
    from src.training.training_sft import validate_train_row

    sft_by_id = _unique(sft_rows, "sample_id", "D0.4.3 SFT Train")
    source_by_id = _unique(normalized_train_rows, "sample_id", "D0.3 Train")
    if len(sft_by_id) != expected_rows:
        raise ValueError(f"D0.4.3 SFT Train count differs: expected={expected_rows}, actual={len(sft_by_id)}")
    result: list[dict[str, Any]] = []
    for sample_id, encoded in sft_by_id.items():
        validate_train_row(encoded, expected_eos_token_id=eos_token_id, expected_split="train")
        if len(encoded["input_ids"]) > max_sequence_length:
            raise ValueError(f"{sample_id}: pre-encoded sample exceeds S={max_sequence_length}")
        if eos_token_id not in encoded["labels"][encoded["prompt_length"]:]:
            raise ValueError(f"{sample_id}: completion EOS must remain supervised")
        source = source_by_id.get(sample_id)
        if source is None:
            raise ValueError(f"{sample_id}: no D0.3 Train source row")
        if (
            encoded.get("conversation_id") != source.get("conversation_id")
            or encoded.get("source_split") != "train"
        ):
            raise ValueError(f"{sample_id}: D0.3/D0.4.3 identity mismatch")
        history_length = validate_history(source.get("history"))
        gold = _validate_profile(source.get("target", {}).get("user_profile"), enums, sample_id)
        result.append({
            "sample_id": sample_id,
            "conversation_id": source["conversation_id"],
            "raw_history_length": history_length,
            "raw_history_bucket": length_bucket(history_length),
            "gold_profile_empty": all(not gold[dimension] for dimension in DIMENSIONS),
            "gold_label_count": sum(len(gold[dimension]) for dimension in DIMENSIONS),
            "encoded": encoded,
        })
    if not set(sft_by_id) <= set(source_by_id):
        raise ValueError("D0.4.3 IDs are not a subset of D0.3 Train")
    return result


def build_schedule(
    records: list[dict[str, Any]],
    *,
    group: str,
    seed: int,
    per_device_micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> list[ScheduledDraw]:
    """Build one deterministic 16,192-draw schedule for a paired A3 run.

    Control sees every effective Train row once in seeded random order.
    Candidate assigns half of draws to that natural permutation and half to
    raw-history<=4 rows, balanced between all-empty and non-empty Gold. Each
    four-microbatch optimizer window contains two natural draws and one draw
    from each targeted stratum.
    """
    if group not in {"control", "candidate"}:
        raise ValueError("group must be control or candidate")
    if not records:
        raise ValueError("training records must not be empty")
    block_size = per_device_micro_batch_size * gradient_accumulation_steps
    if block_size != 4:
        raise ValueError("the frozen A3 paired schedule requires micro batch 1 x gradient accumulation 4")
    if len(records) % block_size:
        raise ValueError("effective Train row count must divide evenly into optimizer windows")

    rng = random.Random(seed)
    natural_indices = list(range(len(records)))
    rng.shuffle(natural_indices)
    if group == "control":
        return [ScheduledDraw(index, NATURAL) for index in natural_indices]

    empty_short = [
        index for index, row in enumerate(records)
        if row["raw_history_length"] <= 4 and row["gold_profile_empty"]
    ]
    nonempty_short = [
        index for index, row in enumerate(records)
        if row["raw_history_length"] <= 4 and not row["gold_profile_empty"]
    ]
    if not empty_short or not nonempty_short:
        raise ValueError("candidate sampling requires both short-history Gold empty and non-empty pools")

    optimizer_windows = len(records) // block_size
    draws: list[ScheduledDraw] = []
    natural_cursor = empty_cursor = nonempty_cursor = 0
    for _ in range(optimizer_windows):
        draws.append(ScheduledDraw(natural_indices[natural_cursor], NATURAL))
        natural_cursor += 1
        empty_index = rng.choice(empty_short)
        draws.append(ScheduledDraw(empty_index, SHORT_EMPTY))
        empty_cursor += 1
        draws.append(ScheduledDraw(natural_indices[natural_cursor], NATURAL))
        natural_cursor += 1
        nonempty_index = rng.choice(nonempty_short)
        draws.append(ScheduledDraw(nonempty_index, SHORT_NONEMPTY))
        nonempty_cursor += 1

    if natural_cursor != len(records) // 2 or empty_cursor != len(records) // 4 or nonempty_cursor != len(records) // 4:
        raise RuntimeError("candidate schedule failed its natural/short and empty/non-empty draw conservation")
    return draws


def materialize_schedule(
    records: list[dict[str, Any]], schedule: list[ScheduledDraw]
) -> list[dict[str, Any]]:
    """Prepare a Trainer dataset whose record order is the frozen draw order."""
    output: list[dict[str, Any]] = []
    for draw_number, draw in enumerate(schedule):
        record = records[draw.row_index]
        encoded = record["encoded"]
        output.append({
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
            "prompt_length": encoded["prompt_length"],
            "completion_length": encoded["completion_length"],
            "sample_id": record["sample_id"],
            "conversation_id": record["conversation_id"],
            "raw_history_length": record["raw_history_length"],
            "gold_profile_empty": record["gold_profile_empty"],
            "sampling_component": draw.component,
            "schedule_position": draw_number,
        })
    return output


def schedule_summary(rows: list[dict[str, Any]], optimizer_steps: int, accumulation_steps: int) -> dict[str, Any]:
    """Summarize a schedule without serializing sample identities."""
    consumed = rows[: optimizer_steps * accumulation_steps]
    counts: dict[str, int] = {}
    for row in consumed:
        component = row["sampling_component"]
        counts[component] = counts.get(component, 0) + 1
    ids = [row["sample_id"] for row in consumed]
    unique_ids = len(set(ids))
    return {
        "rows_in_schedule": len(rows),
        "optimizer_steps": optimizer_steps,
        "consumed_draws": len(consumed),
        "consumed_component_counts": counts,
        "consumed_unique_sample_ids": unique_ids,
        "consumed_repeated_draws": len(ids) - unique_ids,
        "consumed_unique_conversations": len({row["conversation_id"] for row in consumed}),
    }
