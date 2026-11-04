"""Validation-only contract for new isolated E03 experiment configurations."""
from __future__ import annotations

import re
from typing import Any, Mapping


EXPERIMENT_ID_PATTERN = re.compile(r"E03-[A-Za-z0-9][A-Za-z0-9._-]{1,63}\Z")
OBJECTIVES = {"sft", "sft_plus_bce"}
SAMPLING_STRATEGIES = {"uniform_without_replacement", "train_replay"}
LEGACY_EXPERIMENT_IDS = {"E03-A0", "E03-A0.1", "E03-A1", "E03-A2", "E03-A3"}


def validate_experiment_spec(
    spec: Mapping[str, Any],
    *,
    observed_adapter: Mapping[str, Any] | None = None,
    expected_label_order: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a future spec before any Trainer/model load is attempted."""
    if spec.get("version") != "E03-experiment-spec-v1":
        raise ValueError("unsupported experiment spec version")
    experiment_id = spec.get("experiment_id")
    if not isinstance(experiment_id, str) or not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("new E03 experiment requires a new versioned experiment_id")
    if experiment_id in LEGACY_EXPERIMENT_IDS:
        raise ValueError("legacy E01/E03 experiment identity cannot be reused; choose a new versioned experiment_id")
    if spec.get("execution", {}).get("enabled") is not False:
        raise ValueError("this contract is validation-only and cannot enable a training run")
    if spec.get("authorization", {}).get("status") != "requires_separate_approval":
        raise ValueError("future training still requires separate explicit authorization")

    objective = spec.get("objective")
    if objective not in OBJECTIVES:
        raise ValueError("objective must be sft or sft_plus_bce")
    sampling = spec.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling config is required")
    strategy = sampling.get("strategy")
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError("unsupported sampling strategy")
    if sampling.get("source_split") != "train":
        raise ValueError("sampling source_split must be train")
    seed, draw_count = sampling.get("seed"), sampling.get("draw_count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("sampling seed must be an integer")
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 1:
        raise ValueError("sampling draw_count must be a positive integer")
    source_sha256 = sampling.get("source_train_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256.lower())
    ):
        raise ValueError("sampling.source_train_sha256 must bind the original Train source")
    replay_fraction = sampling.get("replay_fraction", 0.0)
    if strategy == "train_replay":
        if not isinstance(replay_fraction, (int, float)) or isinstance(replay_fraction, bool) or not 0 < replay_fraction <= 0.5:
            raise ValueError("train_replay requires an explicit replay_fraction in (0, 0.5]")
        if not isinstance(sampling.get("replay_sample_ids"), list) or not sampling["replay_sample_ids"]:
            raise ValueError("train_replay requires an explicit Train replay_sample_ids pool")
    elif replay_fraction not in (0, 0.0):
        raise ValueError("uniform sampling cannot set a non-zero replay fraction")

    training = spec.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("training budget config is required")
    for field in ("max_optimizer_steps", "max_training_seconds"):
        value = training.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{field} must be explicitly positive")
    micro_batch = training.get("per_device_micro_batch_size")
    accumulation = training.get("gradient_accumulation_steps")
    if (
        isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1
        or isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1
    ):
        raise ValueError("micro-batch size and gradient accumulation must be explicit positive integers")
    if draw_count != training["max_optimizer_steps"] * micro_batch * accumulation:
        raise ValueError("sampling draw_count must equal optimizer steps times effective batch size")
    lora = spec.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("LoRA config is required")
    for field in ("r", "alpha"):
        value = lora.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"lora.{field} must be a positive integer")
    target_modules = lora.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or any(not isinstance(name, str) or not name for name in target_modules)
        or len(target_modules) != len(set(target_modules))
    ):
        raise ValueError("lora.target_modules must be an explicit unique non-empty list")

    resource_gate = spec.get("resource_gate")
    if not isinstance(resource_gate, Mapping):
        raise ValueError("a GPU memory/resource preflight gate is required")
    memory_budget = resource_gate.get("max_gpu_memory_bytes")
    if isinstance(memory_budget, bool) or not isinstance(memory_budget, int) or memory_budget < 1:
        raise ValueError("resource_gate.max_gpu_memory_bytes must be explicitly positive")

    initialization = spec.get("initialization")
    if not isinstance(initialization, Mapping) or initialization.get("source") != "base":
        raise ValueError("new LoRA comparison Runs must initialize from the Base model")
    if initialization.get("adapter_path") not in (None, ""):
        raise ValueError("new experiment specs cannot initialize from an existing Adapter")
    if initialization.get("resume_from_checkpoint") not in (None, ""):
        raise ValueError("new experiment specs cannot resume another Run")
    base_sha = initialization.get("base_weight_sha256")
    if not isinstance(base_sha, str) or len(base_sha) != 64 or any(ch not in "0123456789abcdef" for ch in base_sha.lower()):
        raise ValueError("initialization.base_weight_sha256 must identify the frozen Base")
    if observed_adapter is not None:
        configured_base_sha = initialization.get("base_weight_sha256")
        observed_base_sha = observed_adapter.get("base_weight_sha256")
        if configured_base_sha and observed_base_sha and configured_base_sha != observed_base_sha:
            raise ValueError("configured Base weight identity differs from the observed Adapter Base")
        observed_rank = observed_adapter.get("r")
        configured_rank = lora["r"]
        if observed_rank is not None and configured_rank != observed_rank:
            if initialization.get("source") != "base" or initialization.get("adapter_path") not in (None, ""):
                raise ValueError("a changed LoRA rank cannot load or resume the existing Adapter")

    bce = spec.get("bce")
    if objective == "sft_plus_bce":
        if not isinstance(bce, Mapping):
            raise ValueError("sft_plus_bce requires an explicit BCE contract")
        label_order = bce.get("label_order")
        if not isinstance(label_order, list) or len(label_order) != 33 or len(set(label_order)) != 33:
            raise ValueError("BCE label_order must contain 33 unique labels")
        if expected_label_order is not None and label_order != expected_label_order:
            raise ValueError("BCE label_order differs from the official schema order")
        if bce.get("missing_label_policy") != "masked_unknown":
            raise ValueError("BCE missing labels must remain masked unknowns")
        if not isinstance(bce.get("lambda"), (int, float)) or isinstance(bce.get("lambda"), bool) or bce["lambda"] <= 0:
            raise ValueError("BCE lambda must be explicitly positive")
    elif bce is not None:
        raise ValueError("plain SFT spec must not carry an unused BCE head config")

    return {
        "status": "validation_passed",
        "experiment_id": experiment_id,
        "objective": objective,
        "sampling_strategy": strategy,
        "source_split": "train",
        "sampling_seed": seed,
        "draw_count": draw_count,
        "source_train_sha256": source_sha256.lower(),
        "lora_r": lora["r"],
        "lora_alpha": lora["alpha"],
        "target_modules": list(target_modules),
        "max_gpu_memory_bytes": memory_budget,
        "initialization_source": "base",
        "training_enabled": False,
        "authorization_status": "requires_separate_approval",
    }
