"""Right-padding collator and exact Batch invariants for pre-tokenized SFT rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch


class SFTDataCollator:
    """Pad only the right edge; preserve caller-supplied Completion-only labels."""

    def __init__(
        self,
        pad_token_id: int,
        *,
        pad_to_multiple_of: int | None = None,
        max_sequence_length: int | None = None,
    ) -> None:
        if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        if pad_to_multiple_of is not None and (
            not isinstance(pad_to_multiple_of, int)
            or isinstance(pad_to_multiple_of, bool)
            or pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be null or a positive integer")
        if max_sequence_length is not None and (
            not isinstance(max_sequence_length, int)
            or isinstance(max_sequence_length, bool)
            or max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be null or a positive integer")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_sequence_length = max_sequence_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        for index, feature in enumerate(features):
            input_ids = feature.get("input_ids")
            target_ids = feature.get("labels")
            prompt_length = feature.get("prompt_length")
            completion_length = feature.get("completion_length")
            if not isinstance(input_ids, (list, tuple)) or not input_ids:
                raise ValueError(f"feature {index}: input_ids must be a non-empty sequence")
            if not isinstance(target_ids, (list, tuple)) or len(target_ids) != len(input_ids):
                raise ValueError(f"feature {index}: labels must have the same length as input_ids")
            if not isinstance(prompt_length, int) or isinstance(prompt_length, bool) or not 0 < prompt_length < len(input_ids):
                raise ValueError(f"feature {index}: invalid prompt_length")
            if completion_length != len(input_ids) - prompt_length:
                raise ValueError(f"feature {index}: completion_length disagrees with input_ids")
            ids = [int(value) for value in input_ids]
            row_labels = [int(value) for value in target_ids]
            if row_labels[:prompt_length] != [-100] * prompt_length:
                raise ValueError(f"feature {index}: Prompt labels are not fully masked")
            if row_labels[prompt_length:] != ids[prompt_length:]:
                raise ValueError(f"feature {index}: Completion labels differ from the input suffix")
            if all(value == -100 for value in row_labels):
                raise ValueError(f"feature {index}: all labels are masked")
            if self.max_sequence_length is not None and len(ids) > self.max_sequence_length:
                raise ValueError(f"feature {index}: sequence exceeds max_sequence_length")
            sequences.append(ids)
            labels.append(row_labels)

        padded_length = max(map(len, sequences))
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            padded_length = ((padded_length + multiple - 1) // multiple) * multiple
        if self.max_sequence_length is not None and padded_length > self.max_sequence_length:
            raise ValueError("dynamic padding would exceed max_sequence_length; no truncation is allowed")

        input_batch = torch.full(
            (len(features), padded_length), self.pad_token_id, dtype=torch.long
        )
        label_batch = torch.full((len(features), padded_length), -100, dtype=torch.long)
        attention_batch = torch.zeros((len(features), padded_length), dtype=torch.long)
        for index, (ids, row_labels) in enumerate(zip(sequences, labels)):
            length = len(ids)
            input_batch[index, :length] = torch.tensor(ids, dtype=torch.long)
            label_batch[index, :length] = torch.tensor(row_labels, dtype=torch.long)
            attention_batch[index, :length] = 1
        return {
            "input_ids": input_batch,
            "labels": label_batch,
            "attention_mask": attention_batch,
        }


def _select_examples_by_length(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if len(rows) < size:
        raise ValueError(f"need at least {size} usable examples for batch validation")
    ordered = sorted(rows, key=lambda row: (len(row["input_ids"]), row["sample_id"]))
    if size == 1:
        return [ordered[len(ordered) // 2]]
    positions = [round(index * (len(ordered) - 1) / (size - 1)) for index in range(size)]
    return [ordered[index] for index in positions]


def validate_dataset_batches(
    dataset_dir: Path,
    *,
    split_names: Sequence[str],
    batch_sizes: Sequence[int],
    pad_token_id: int,
    eos_token_id: int,
    max_sequence_length: int,
    pad_to_multiple_of: int | None = None,
) -> dict[str, Any]:
    collator = SFTDataCollator(
        pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
        max_sequence_length=max_sequence_length,
    )
    split_results: dict[str, Any] = {}
    for split in split_names:
        path = dataset_dir / f"{split}.jsonl"
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
                if len(rows) >= 128:
                    break
        if not rows:
            raise ValueError(f"{split}: no examples available for Batch validation")
        batch_checks = []
        for size in batch_sizes:
            features = _select_examples_by_length(rows, size)
            batch = collator(features)
            shape = tuple(batch["input_ids"].shape)
            if batch["labels"].shape != batch["input_ids"].shape or batch["attention_mask"].shape != batch["input_ids"].shape:
                raise RuntimeError(f"{split} batch {size}: input/labels/mask shapes differ")
            if int(batch["attention_mask"].sum().item()) != sum(len(row["input_ids"]) for row in features):
                raise RuntimeError(f"{split} batch {size}: attention sum does not match unpadded token count")
            pad_mask = batch["attention_mask"] == 0
            if bool((batch["labels"][pad_mask] != -100).any().item()):
                raise RuntimeError(f"{split} batch {size}: padding position has an active label")
            if bool((batch["input_ids"][pad_mask] != pad_token_id).any().item()):
                raise RuntimeError(f"{split} batch {size}: padding position does not contain pad_token_id")
            if bool((batch["attention_mask"][:, 1:] > batch["attention_mask"][:, :-1]).any().item()):
                raise RuntimeError(f"{split} batch {size}: padding is not exclusively on the right")
            for row_index, feature in enumerate(features):
                length = len(feature["input_ids"])
                if batch["input_ids"][row_index, :length].tolist() != feature["input_ids"]:
                    raise RuntimeError(f"{split} batch {size}: input prefix was changed")
                if batch["labels"][row_index, :length].tolist() != feature["labels"]:
                    raise RuntimeError(f"{split} batch {size}: prebuilt labels were changed")
                if batch["labels"][row_index, :feature["prompt_length"]].tolist() != [-100] * feature["prompt_length"]:
                    raise RuntimeError(f"{split} batch {size}: prompt region is not fully masked")
                if batch["labels"][row_index, feature["prompt_length"]:length].tolist() != feature["input_ids"][feature["prompt_length"]:length]:
                    raise RuntimeError(f"{split} batch {size}: completion region is not supervised verbatim")
                completion_ids = feature["input_ids"][feature["prompt_length"]:]
                eos_positions = [index for index, token in enumerate(completion_ids) if token == eos_token_id]
                if not eos_positions:
                    raise RuntimeError(f"{split} batch {size}: completion has no EOS token")
                terminal_eos = feature["prompt_length"] + eos_positions[-1]
                if int(batch["labels"][row_index, terminal_eos].item()) != eos_token_id:
                    raise RuntimeError(f"{split} batch {size}: terminal EOS is not supervised")
                if not any(label != -100 for label in feature["labels"]):
                    raise RuntimeError(f"{split} batch {size}: usable sample has all labels masked")
            batch_checks.append({
                "batch_size": size,
                "tensor_shape": list(shape),
                "sample_lengths": [len(row["input_ids"]) for row in features],
                "padding_tokens": int((batch["attention_mask"] == 0).sum().item()),
                "right_padding": True,
                "prompt_fully_masked": True,
                "completion_labels_preserved": True,
                "terminal_eos_supervised": True,
                "no_all_masked_samples": True,
            })
        split_results[split] = {
            "real_examples_sampled_for_validation": len(rows),
            "checks": batch_checks,
        }
    return {
        "batch_sizes": list(batch_sizes),
        "padding_side": "right",
        "packing": False,
        "pad_to_multiple_of": pad_to_multiple_of,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_equals_eos": pad_token_id == eos_token_id,
        "max_sequence_length": max_sequence_length,
        "trainer_data_path": "pending_D_validation",
        "splits": split_results,
        "all_batch_invariants_pass": True,
    }
