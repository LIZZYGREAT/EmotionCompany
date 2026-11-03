"""Loss weighting and verified token-region mapping for E03-A1."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from src.data.sft_collator import SFTDataCollator


VARIANTS = ("token_mean", "profile_x2", "label_value_x2")
PRIMARY_REGION_IDS = {
    "other": 0,
    "user_profile": 1,
    "emotion_label": 2,
    "response_text": 3,
}
LOSS_MAPPING_VERSION = "E03-A1-token-regions-v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_loss_weights(
    labels: list[int],
    primary_regions: list[str],
    fine_regions: list[str | None],
    variant: str,
    *,
    weight_factor: float = 2.0,
) -> list[float]:
    """Assign weights by target token position; masked prompt tokens get zero."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown E03-A1 loss variant: {variant}")
    if not math.isfinite(float(weight_factor)) or weight_factor < 1:
        raise ValueError("weight_factor must be finite and at least 1")
    if not (len(labels) == len(primary_regions) == len(fine_regions)):
        raise ValueError("label and region mapping lengths differ")
    weights = []
    for label, primary, fine in zip(labels, primary_regions, fine_regions):
        if label == -100:
            weights.append(0.0)
        elif variant == "profile_x2" and primary == "user_profile":
            weights.append(float(weight_factor))
        elif variant == "label_value_x2" and fine == "profile_label_value":
            weights.append(float(weight_factor))
        else:
            weights.append(1.0)
    if not any(weight > 0 for weight in weights):
        raise ValueError("loss mapping contains no active supervised tokens")
    return weights


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ignore_index: int = -100,
    token_chunk_size: int = 256,
) -> torch.Tensor:
    """Compute weighted next-token CE without materializing full FP32 logits."""
    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and labels/weights [batch, sequence]")
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError("logits, labels, and loss_weights sequence shapes differ")
    if logits.shape[1] < 2:
        raise ValueError("causal loss requires at least two sequence positions")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")

    shifted_labels = labels[:, 1:].reshape(-1)
    shifted_weights = loss_weights[:, 1:].reshape(-1).to(dtype=torch.float32)
    active = shifted_labels.ne(ignore_index)
    if not bool(active.any().item()):
        raise ValueError("causal loss has no active target tokens after shifting")
    if not bool(torch.isfinite(shifted_weights).all().item()):
        raise ValueError("loss_weights contain NaN or Inf")
    if bool((shifted_weights[active] <= 0).any().item()):
        raise ValueError("every active target token must have a positive loss weight")
    if bool((shifted_weights[~active] != 0).any().item()):
        raise ValueError("ignored target positions must have zero loss weight")
    denominator = shifted_weights[active].sum()

    shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    numerator: torch.Tensor | None = None
    for start in range(0, shifted_labels.numel(), token_chunk_size):
        end = min(start + token_chunk_size, shifted_labels.numel())
        local_active = active[start:end]
        if not bool(local_active.any().item()):
            continue
        local_logits = shifted_logits[start:end][local_active].float()
        local_labels = shifted_labels[start:end][local_active]
        local_weights = shifted_weights[start:end][local_active]
        local_ce = F.cross_entropy(local_logits, local_labels, reduction="none")
        local_numerator = (local_ce * local_weights).sum()
        numerator = local_numerator if numerator is None else numerator + local_numerator
    if numerator is None:
        raise ValueError("causal loss has no active target tokens")
    return numerator / denominator


@torch.no_grad()
def independently_recomputed_weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ignore_index: int = -100,
    token_chunk_size: int = 256,
) -> torch.Tensor:
    """Audit the objective with explicit log-softmax/gather arithmetic."""
    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and labels/weights [batch, sequence]")
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape or logits.shape[1] < 2:
        raise ValueError("logits, labels, and loss_weights sequence shapes differ")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    targets = labels[:, 1:].reshape(-1)
    weights = loss_weights[:, 1:].reshape(-1).float()
    keep = targets.ne(ignore_index)
    if not bool(keep.any().item()) or not bool(torch.isfinite(weights).all().item()):
        raise ValueError("independent loss audit has no valid targets or non-finite weights")
    if bool((weights[keep] <= 0).any().item()) or bool((weights[~keep] != 0).any().item()):
        raise ValueError("independent loss audit found invalid active/masked weights")
    predictions = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    weighted_nll = torch.zeros((), dtype=torch.float32, device=logits.device)
    weight_sum = weights[keep].sum()
    for start in range(0, targets.numel(), token_chunk_size):
        end = min(start + token_chunk_size, targets.numel())
        local_keep = keep[start:end]
        if not bool(local_keep.any().item()):
            continue
        local_logits = predictions[start:end][local_keep].float()
        local_targets = targets[start:end][local_keep]
        local_weights = weights[start:end][local_keep]
        log_probabilities = torch.log_softmax(local_logits, dim=-1)
        negative_log_likelihood = -log_probabilities.gather(1, local_targets.unsqueeze(1)).squeeze(1)
        weighted_nll += (negative_log_likelihood * local_weights).sum()
    return weighted_nll / weight_sum


def _region_id(primary: str) -> int:
    return PRIMARY_REGION_IDS.get(primary, PRIMARY_REGION_IDS["other"])


def build_alignment_cache(
    *,
    config: dict[str, Any],
    tokenizer: Any,
    prompt_text: str,
    schema: dict[str, Any],
    train_samples: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    normalized_train_path: Path,
    normalized_dev_path: Path,
    cache_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, list[int]]], dict[str, Any]]:
    """Build or load A0/A0.1-aligned region IDs without storing dialogue text."""
    from src.diagnostics import e03_a0 as a0
    from src.diagnostics import e03_a0_1_profile as a01
    from src.training import training_sft

    train_id_set = {str(item["encoded"]["sample_id"]) for item in train_samples}
    dev_ids = [str(row["sample_id"]) for row in dev_rows]
    if len(train_id_set) != len(train_samples) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("duplicate sample IDs in E03-A1 mapping inputs")
    source_by_id: dict[str, dict[str, Any]] = {}
    dev_id_set = set(dev_ids)
    for source in training_sft._iter_jsonl(normalized_dev_path):
        sample_id = str(source.get("sample_id", ""))
        if sample_id in dev_id_set:
            source_by_id[sample_id] = source
    if set(source_by_id) != dev_id_set:
        raise ValueError("one or more selected SFT Dev IDs are absent from normalized D0.3 Dev")

    module_paths = (
        Path(__file__),
        Path(a0.__file__),
        Path(a01.__file__),
        Path(training_sft.__file__),
    )
    identity = {
        "version": LOSS_MAPPING_VERSION,
        "train_sha256": training_sft.sha256_file(training_sft._resolve(config["data"]["train_path"])),
        "dev_sha256": training_sft.sha256_file(training_sft._resolve(config["data"]["dev_path"])),
        "normalized_train_sha256": training_sft.sha256_file(normalized_train_path),
        "normalized_dev_sha256": training_sft.sha256_file(normalized_dev_path),
        "p0_sha256": config["protocol"]["p0_sha256"],
        "tokenizer_signature_sha256": config["protocol"]["tokenizer_signature_sha256"],
        "chat_template_sha256": config["protocol"]["chat_template_sha256"],
        "schema_sha256": canonical_sha256(schema),
        "dev_sample_ids_sha256": canonical_sha256(dev_ids),
        "mapping_code_sha256": canonical_sha256({path.name: training_sft.sha256_file(path) for path in module_paths}),
    }
    cache_key = canonical_sha256(identity)
    cache_path = cache_root / f"{cache_key}.json"
    cache_root.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        rows = payload.get("mappings")
        if payload.get("identity") != identity or not isinstance(rows, list):
            raise ValueError("E03-A1 mapping cache identity is invalid")
        if canonical_sha256(rows) != payload.get("mappings_sha256"):
            raise ValueError("E03-A1 mapping cache content hash is invalid")
    else:
        label_enums = a01._profile_schema(schema)
        rows = []

        def map_one(split: str, source: dict[str, Any], encoded: dict[str, Any]) -> dict[str, Any]:
            completion, _rendered, offsets, primary, profile_regions, _counts, _profile_counts, metadata = a0._build_sample_mapping(
                source=source,
                sft_row=encoded,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                schema=schema,
                max_prompt_tokens=int(config["protocol"]["max_prompt_tokens"]),
                expected_sft_split=split,
            )
            fine = a01.assign_profile_fine_regions(
                completion=completion,
                completion_start=int(metadata["completion_start"]),
                offsets=offsets,
                labels=list(map(int, encoded["labels"])),
                top_regions=primary,
                profile_regions=profile_regions,
                label_enums=label_enums,
            )
            labels = list(map(int, encoded["labels"]))
            primary_ids = [
                _region_id(region) if label != -100 else 0
                for region, label in zip(primary, labels)
            ]
            label_value_mask = [
                int(region == "profile_label_value" and label != -100)
                for region, label in zip(fine["token_regions"], labels)
            ]
            return {
                "split": split,
                "sample_id": str(encoded["sample_id"]),
                "primary_region_ids": primary_ids,
                "label_value_mask": label_value_mask,
            }

        for item in train_samples:
            rows.append(map_one("train", item["source"], item["encoded"]))
        for encoded in dev_rows:
            rows.append(map_one("dev", source_by_id[str(encoded["sample_id"])], encoded))
        payload = {"identity": identity, "mappings": rows, "mappings_sha256": canonical_sha256(rows)}
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, cache_path)

    expected = {("train", sample_id) for sample_id in train_id_set} | {("dev", sample_id) for sample_id in dev_ids}
    encoded_by_key = {
        **{("train", str(item["encoded"]["sample_id"])): item["encoded"] for item in train_samples},
        **{("dev", str(row["sample_id"])): row for row in dev_rows},
    }
    result: dict[tuple[str, str], dict[str, list[int]]] = {}
    for row in rows:
        key = (str(row.get("split")), str(row.get("sample_id")))
        if key in result or key not in expected:
            raise ValueError("E03-A1 mapping cache contains a duplicate or unexpected sample")
        primary_ids = row.get("primary_region_ids")
        label_value_mask = row.get("label_value_mask")
        encoded = encoded_by_key.get(key)
        if encoded is None or not isinstance(primary_ids, list) or not isinstance(label_value_mask, list):
            raise ValueError("E03-A1 mapping cache row is malformed")
        if len(primary_ids) != len(encoded["labels"]) or len(label_value_mask) != len(encoded["labels"]):
            raise ValueError(f"E03-A1 mapping length differs from labels: {key[1]}")
        if any(not isinstance(value, int) or value < 0 or value > 3 for value in primary_ids):
            raise ValueError(f"E03-A1 primary-region mapping contains an invalid ID: {key[1]}")
        if any(not isinstance(value, int) or value not in {0, 1} for value in label_value_mask):
            raise ValueError(f"E03-A1 label-value mapping contains an invalid flag: {key[1]}")
        result[key] = {"primary_region_ids": primary_ids, "label_value_mask": label_value_mask}
    if set(result) != expected:
        raise ValueError("E03-A1 mapping cache does not cover exact Train and Dev IDs")

    train_active = profile_active = label_active = 0
    for item in train_samples:
        key = ("train", str(item["encoded"]["sample_id"]))
        encoded, mapped = item["encoded"], result[key]
        for label, region, fine in zip(encoded["labels"], mapped["primary_region_ids"], mapped["label_value_mask"]):
            if label != -100:
                train_active += 1
                profile_active += int(region == PRIMARY_REGION_IDS["user_profile"])
                label_active += int(fine == 1)
    facts = {
        "mapping_version": LOSS_MAPPING_VERSION,
        "mapping_sha256": canonical_sha256(rows),
        "cache_path": str(cache_path),
        "train_rows": len(train_samples),
        "dev_rows": len(dev_rows),
        "normalized_dev_sha256": identity["normalized_dev_sha256"],
        "train_active_tokens": train_active,
        "train_user_profile_tokens": profile_active,
        "train_exact_label_value_tokens": label_active,
        "dev_sample_ids_sha256": canonical_sha256(dev_ids),
    }
    return result, facts


def attach_loss_fields(
    *,
    train_samples: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    mappings: dict[tuple[str, str], dict[str, list[int]]],
    variant: str,
    weight_factor: float = 2.0,
) -> None:
    """Attach numerical loss fields to temporary in-memory pre-encoded rows."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown E03-A1 loss variant: {variant}")

    def attach(split: str, row: dict[str, Any], *, row_variant: str) -> None:
        key = (split, str(row["sample_id"]))
        mapped = mappings.get(key)
        if mapped is None:
            raise ValueError(f"E03-A1 token mapping is missing: {key}")
        region_ids = mapped["primary_region_ids"]
        label_mask = mapped["label_value_mask"]
        weights = []
        for label, region, is_label in zip(row["labels"], region_ids, label_mask):
            if label == -100:
                weights.append(0.0)
            elif row_variant == "profile_x2" and region == PRIMARY_REGION_IDS["user_profile"]:
                weights.append(float(weight_factor))
            elif row_variant == "label_value_x2" and is_label == 1:
                weights.append(float(weight_factor))
            else:
                weights.append(1.0)
        row["loss_weights"] = weights
        row["primary_region_ids"] = region_ids
        row["label_value_mask"] = label_mask

    for item in train_samples:
        attach("train", item["encoded"], row_variant=variant)
    # Dev loss is always ordinary, unweighted token-average CE.
    for row in dev_rows:
        attach("dev", row, row_variant="token_mean")


class E03A1DataCollator:
    """Keep A1 loss weights and region masks aligned through right padding."""

    def __init__(self, base: SFTDataCollator) -> None:
        self.base = base

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.base(features)
        padded_length = int(batch["input_ids"].shape[1])
        weights_batch = torch.zeros((len(features), padded_length), dtype=torch.float32)
        primary_batch = torch.zeros((len(features), padded_length), dtype=torch.long)
        label_batch = torch.zeros((len(features), padded_length), dtype=torch.long)
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            values = {
                "loss_weights": feature.get("loss_weights"),
                "primary_region_ids": feature.get("primary_region_ids"),
                "label_value_mask": feature.get("label_value_mask"),
            }
            if any(not isinstance(value, (list, tuple)) or len(value) != length for value in values.values()):
                raise ValueError(f"feature {index}: A1 loss fields must match input_ids length")
            weights = torch.tensor(values["loss_weights"], dtype=torch.float32)
            primary = torch.tensor(values["primary_region_ids"], dtype=torch.long)
            labels = torch.tensor(values["label_value_mask"], dtype=torch.long)
            active = torch.tensor(feature["labels"], dtype=torch.long).ne(-100)
            if not bool(torch.isfinite(weights).all().item()) or bool((weights[active] <= 0).any().item()):
                raise ValueError(f"feature {index}: active A1 weights must be finite and positive")
            if bool((weights[~active] != 0).any().item()):
                raise ValueError(f"feature {index}: masked tokens must have zero A1 weight")
            if bool(((primary < 0) | (primary > 3)).any().item()) or bool(((labels < 0) | (labels > 1)).any().item()):
                raise ValueError(f"feature {index}: invalid A1 region id")
            weights_batch[index, :length] = weights
            primary_batch[index, :length] = primary
            label_batch[index, :length] = labels
        batch["loss_weights"] = weights_batch
        batch["primary_region_ids"] = primary_batch
        batch["label_value_mask"] = label_batch
        return batch


@torch.no_grad()
def loss_breakdown_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    primary_region_ids: torch.Tensor,
    label_value_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
    token_chunk_size: int = 256,
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate unweighted token CE for the E03-A1 evaluation regions."""
    if logits.ndim != 3 or labels.ndim != 2 or primary_region_ids.shape != labels.shape or label_value_mask.shape != labels.shape:
        raise ValueError("loss-breakdown tensor shapes differ")
    shifted_labels = labels[:, 1:].reshape(-1)
    shifted_regions = primary_region_ids[:, 1:].reshape(-1)
    shifted_label_mask = label_value_mask[:, 1:].reshape(-1).bool()
    active = shifted_labels.ne(ignore_index)
    if not bool(active.any().item()):
        raise ValueError("loss-breakdown has no active target tokens")
    shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    names = ("all_tokens", "user_profile", "label_value", "emotion_label", "response_text")
    sums = {name: 0.0 for name in names}
    counts = {name: 0 for name in names}
    for start in range(0, shifted_labels.numel(), token_chunk_size):
        end = min(start + token_chunk_size, shifted_labels.numel())
        local_active = active[start:end]
        if not bool(local_active.any().item()):
            continue
        token_losses = F.cross_entropy(
            shifted_logits[start:end][local_active].float(),
            shifted_labels[start:end][local_active],
            reduction="none",
        ).detach()
        regions = shifted_regions[start:end][local_active]
        label_values = shifted_label_mask[start:end][local_active]
        local_groups = {
            "all_tokens": torch.ones_like(regions, dtype=torch.bool),
            "user_profile": regions.eq(PRIMARY_REGION_IDS["user_profile"]),
            "label_value": label_values,
            "emotion_label": regions.eq(PRIMARY_REGION_IDS["emotion_label"]),
            "response_text": regions.eq(PRIMARY_REGION_IDS["response_text"]),
        }
        for name, mask in local_groups.items():
            if bool(mask.any().item()):
                sums[name] += float(token_losses[mask].sum().item())
                counts[name] += int(mask.sum().item())
    result: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        mean_ce = sums[name] / counts[name] if counts[name] else None
        result[name] = {"tokens": counts[name], "mean_ce": mean_ce}
    return result


def evaluate_loss_breakdown(trainer: Any, rows: list[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    """Run a deterministic, unweighted teacher-forced region evaluation."""
    from torch.utils.data import DataLoader

    if not rows or batch_size <= 0:
        raise ValueError("loss breakdown requires non-empty rows and positive batch size")
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=trainer.data_collator)
    model = trainer.model
    device = model.get_input_embeddings().weight.device
    was_training = model.training
    model.eval()
    totals = {name: {"loss_sum": 0.0, "tokens": 0} for name in ("all_tokens", "user_profile", "label_value", "emotion_label", "response_text")}
    try:
        with torch.inference_mode():
            for batch in loader:
                model_batch = {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask", "labels"}}
                labels = model_batch.pop("labels")
                outputs = model(**model_batch)
                breakdown = loss_breakdown_from_logits(
                    outputs.logits,
                    labels,
                    batch["primary_region_ids"].to(device),
                    batch["label_value_mask"].to(device),
                )
                for name, item in breakdown.items():
                    token_count = int(item["tokens"])
                    if token_count:
                        mean_ce = item["mean_ce"]
                        if mean_ce is None:
                            raise ValueError(f"loss-breakdown region is missing its mean CE: {name}")
                        totals[name]["loss_sum"] += float(mean_ce) * token_count
                        totals[name]["tokens"] += token_count
                del outputs, breakdown, model_batch, labels
    finally:
        model.train(was_training)
    empty_regions = [name for name, value in totals.items() if value["tokens"] == 0]
    if empty_regions:
        raise ValueError(
            "loss-breakdown full Dev subset has no supervised tokens for regions: "
            + ", ".join(empty_regions)
        )
    return {
        name: {"tokens": value["tokens"], "mean_ce": value["loss_sum"] / value["tokens"]}
        for name, value in totals.items()
    }
