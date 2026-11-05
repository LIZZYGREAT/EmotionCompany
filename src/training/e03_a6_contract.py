"""Small, dependency-light identity checks for the isolated E03-A6 F3 trainer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


MICROFIT_EXPERIMENT_ID = "E03-A6-F3-MICROFIT"
PILOT_EXPERIMENT_ID = "E03-A6-F3-PILOT"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_microfit_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "E03-A6-F3-microfit-v1":
        raise ValueError("unsupported F3 microfit config version")
    if config.get("experiment_id") != MICROFIT_EXPERIMENT_ID:
        raise ValueError("microfit config must use the isolated A6 experiment ID")
    if config.get("training_scope") != "bounded_microfit":
        raise ValueError("microfit config must declare bounded_microfit scope")
    if config.get("protocol", {}).get("format_version") != "F3":
        raise ValueError("microfit must select F3 explicitly")
    training = config.get("training", {})
    if training.get("max_optimizer_steps") != 32 or training.get("sample_count") != 4:
        raise ValueError("microfit budget is frozen at four samples and 32 optimizer steps")
    if training.get("max_training_seconds") != 1200:
        raise ValueError("microfit time budget must remain fixed at 1200 seconds")
    if config.get("base_model", {}).get("expected_weight_sha256") != (
        "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
    ):
        raise ValueError("microfit Base model identity differs from the frozen Qwen3-1.7B")
    if config.get("lora", {}).get("r") != 8 or config.get("lora", {}).get("alpha") != 16:
        raise ValueError("microfit LoRA identity must remain r=8/alpha=16")


def validate_pilot_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "E03-A6-F3-pilot-v1":
        raise ValueError("unsupported F3 Pilot config version")
    if config.get("experiment_id") != PILOT_EXPERIMENT_ID:
        raise ValueError("Pilot config must use the isolated A6 experiment ID")
    if config.get("training_scope") != "pilot":
        raise ValueError("Pilot config must declare pilot scope")
    if config.get("protocol", {}).get("format_version") != "F3":
        raise ValueError("Pilot must select F3 explicitly")
    training = config.get("training", {})
    locked = {
        "seed": 42,
        "max_optimizer_steps": 128,
        "per_device_micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "save_interval_steps": 32,
        "save_total_limit": 2,
        "max_training_seconds": 7200,
    }
    changed = {name: (training.get(name), value) for name, value in locked.items() if training.get(name) != value}
    if changed:
        raise ValueError(f"Pilot training budget or optimizer identity changed: {changed}")
    if config.get("base_model", {}).get("expected_weight_sha256") != (
        "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
    ):
        raise ValueError("Pilot Base model identity differs from the frozen Qwen3-1.7B")
    if config.get("lora", {}).get("r") != 8 or config.get("lora", {}).get("alpha") != 16:
        raise ValueError("Pilot LoRA identity must remain r=8/alpha=16")


def validate_historical_pilot_selection(selection: Mapping[str, Any]) -> list[str]:
    rows = selection.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 512:
        raise ValueError("historical Pilot selection must contain exactly 512 rows")
    expected = selection.get("sha256")
    if expected != canonical_sha256(rows):
        raise ValueError("historical Pilot selected_samples.json SHA does not match its ordered rows")
    ids = [row.get("sample_id") if isinstance(row, Mapping) else None for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError("historical Pilot selection contains an invalid sample ID")
    if len(set(ids)) != 512:
        raise ValueError("historical Pilot selection contains duplicate sample IDs")
    if selection.get("selection_rule") != (
        "seeded_length_deciles_v1; train-only; deterministic seed hash within deciles; "
        "all Train emotion labels, both profile-empty states, and min/max sequence lengths"
    ):
        raise ValueError("historical Pilot selection rule differs from the frozen E01-B artifact")
    return ids


def f3_prefix_through_r_marker(
    token_ids: Sequence[int], decode_prefix: Any,
) -> tuple[list[int], str]:
    """Return IDs through the token that completes the generated ``\nR=`` marker."""
    for end in range(1, len(token_ids) + 1):
        decoded = decode_prefix(list(token_ids[:end]))
        marker = decoded.find("\nR=")
        if marker >= 0:
            boundary = marker + len("\nR=")
            return list(token_ids[:end]), decoded[:boundary]
    raise ValueError("generated token IDs did not complete the F3 R= boundary")
