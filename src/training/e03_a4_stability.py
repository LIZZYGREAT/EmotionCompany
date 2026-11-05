"""CPU contracts for the isolated E03-A4 low-learning-rate continuation."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.training.experiment_spec import validate_experiment_spec
from src.training.sampling import SamplingDraw, build_sampling_plan


EXPERIMENT_ID = "E03-A4-LR2E-05"
RUN_ID = "E03-A4-LR2E-05-64-20261105-01"
CONFIG_VERSION = "E03-A4-lr-stability-v1"
RUNNER_VERSION = "e03_a4_stability_runner_v1"
EXPECTED_LR = 2e-5
EXPECTED_STEPS = 64
EXPECTED_MICRO_BATCH = 1
EXPECTED_ACCUMULATION = 4
EXPECTED_DRAW_COUNT = 256


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_a4_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any run that differs from the single pre-authorized A4 candidate."""
    if config.get("version") != CONFIG_VERSION or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("A4 requires the versioned E03-A4-LR2E-05 identity")
    if config.get("run_id") != RUN_ID:
        raise ValueError("A4 run_id differs from the frozen one-run budget")
    if config.get("training_scope") != "low_lr_continuation":
        raise ValueError("A4 training scope must be low_lr_continuation")
    authorization = config.get("authorization", {})
    if authorization.get("authorized") is not True or authorization.get("scope") != "E03-A4-2-64-step-low-lr-continuation":
        raise ValueError("A4 low-LR training lacks the explicit task authorization")

    optimizer = config.get("optimizer", {})
    if (
        optimizer.get("name") != "AdamW"
        or optimizer.get("learning_rate") != EXPECTED_LR
        or optimizer.get("lr_scheduler") != "constant"
        or optimizer.get("warmup_steps") != 0
        or optimizer.get("weight_decay") != 0.0
        or optimizer.get("max_grad_norm") != 1.0
    ):
        raise ValueError("A4 optimizer must be AdamW at exactly 2e-5 with constant schedule and zero warmup/decay")

    training = config.get("training", {})
    if (
        training.get("max_optimizer_steps") != EXPECTED_STEPS
        or training.get("per_device_micro_batch_size") != EXPECTED_MICRO_BATCH
        or training.get("gradient_accumulation_steps") != EXPECTED_ACCUMULATION
        or training.get("seed") != 42
        or training.get("max_training_seconds") != 3600
        or training.get("checkpoint_steps") != [EXPECTED_STEPS]
        or training.get("optimizer_state_resume_enabled") is not False
        or training.get("gradient_checkpointing") is not True
    ):
        raise ValueError("A4 training budget, seed, checkpoint, or fresh-state contract changed")

    sampler = config.get("sampler", {})
    if (
        sampler.get("strategy") != "uniform_without_replacement"
        or sampler.get("source_split") != "train"
        or sampler.get("seed") != 42
        or sampler.get("draw_count") != EXPECTED_DRAW_COUNT
        or sampler.get("replay_fraction") != 0.0
    ):
        raise ValueError("A4 requires exactly 256 uniform, no-replacement Train draws without replay")
    if EXPECTED_STEPS * EXPECTED_MICRO_BATCH * EXPECTED_ACCUMULATION != EXPECTED_DRAW_COUNT:
        raise RuntimeError("internal A4 effective-batch invariant is inconsistent")

    protocol = config.get("protocol", {})
    if (
        protocol.get("format_version") != "F2"
        or protocol.get("history_policy") != "T1_recent_contiguous_complete_message_suffix"
        or protocol.get("prompt_sha256") != config.get("identity", {}).get("expected_prompt_sha256")
        or protocol.get("max_train_seq_length") != 2048
        or protocol.get("max_prompt_tokens") != 1280
        or protocol.get("max_new_tokens") != 1200
        or protocol.get("thinking") is not False
        or protocol.get("packing") is not False
        or protocol.get("padding_free") is not False
        or protocol.get("generation", {}).get("do_sample") is not False
    ):
        raise ValueError("A4 requires the frozen P0/F2/T1, S=2048/P=1280, Greedy G=1200 protocol")

    lora = config.get("lora", {})
    if (
        lora.get("r") != 8
        or lora.get("alpha") != 16
        or lora.get("target_modules") != ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        or lora.get("bias") != "none"
    ):
        raise ValueError("A4 must preserve the frozen E01-B LoRA rank, alpha, and target modules")

    initialization = config.get("adapter_init", {})
    if (
        initialization.get("mode") != "new_run_from_existing_adapter"
        or initialization.get("source_type") != "existing_adapter"
        or initialization.get("optimizer_state") != "fresh"
        or initialization.get("resume_from_checkpoint") is not None
        or initialization.get("autocast_adapter_dtype") is not False
    ):
        raise ValueError("A4 must load only the frozen E01-B Adapter weights with a fresh optimizer and dtype preservation")

    if not isinstance(config.get("evaluation", {}).get("frozen_eval_input_sha256"), str):
        raise ValueError("A4 must bind the frozen 224-point evaluation cohort")
    spec = config.get("experiment_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("A4 must provide the validation-only existing-Adapter experiment spec")
    if (
        spec.get("experiment_id") != EXPERIMENT_ID
        or spec.get("objective") != "sft"
        or spec.get("execution", {}).get("enabled") is not False
        or spec.get("authorization", {}).get("status") != "requires_separate_approval"
        or spec.get("sampling", {}).get("draw_count") != EXPECTED_DRAW_COUNT
        or spec.get("training", {}).get("max_optimizer_steps") != EXPECTED_STEPS
        or spec.get("training", {}).get("max_training_seconds") != 3600
        or spec.get("sampling", {}).get("source_split") != "train"
        or spec.get("sampling", {}).get("strategy") != "uniform_without_replacement"
        or spec.get("lora", {}).get("r") != lora.get("r")
        or spec.get("lora", {}).get("alpha") != lora.get("alpha")
        or spec.get("initialization", {}).get("mode") != "new_run_from_existing_adapter"
    ):
        raise ValueError("A4 validation-only experiment spec differs from the authorized run contract")
    return {
        "status": "config_validation_passed",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "optimizer_steps": EXPECTED_STEPS,
        "draw_count": EXPECTED_DRAW_COUNT,
        "learning_rate": EXPECTED_LR,
        "initialization_mode": "new_run_from_existing_adapter",
        "execution_authorized": True,
    }


def validate_experiment_initialization_spec(
    config: Mapping[str, Any], observed_adapter: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the repository's validation-only new-run-from-Adapter contract."""
    validate_a4_config(config)
    result = validate_experiment_spec(
        config["experiment_spec"], observed_adapter=observed_adapter,
    )
    if result.get("training_enabled") is not False:
        raise ValueError("the shared experiment spec must remain validation-only")
    return result


def validate_preencoded_row(
    row: Mapping[str, Any], *, eos_token_id: int, max_sequence_length: int,
) -> dict[str, Any]:
    """Validate pre-tokenized completion-only SFT fields without retokenizing."""
    sample_id = row.get("sample_id")
    ids, labels = row.get("input_ids"), row.get("labels")
    prompt_length, completion_length = row.get("prompt_length"), row.get("completion_length")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("pre-encoded row requires a non-empty sample_id")
    if not isinstance(ids, list) or not ids or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
        raise ValueError(f"{sample_id}: input_ids must be a non-empty sequence of non-negative integers")
    if len(ids) > max_sequence_length:
        raise ValueError(f"{sample_id}: sequence exceeds S={max_sequence_length}")
    if not isinstance(labels, list) or len(labels) != len(ids) or any(isinstance(token, bool) or not isinstance(token, int) for token in labels):
        raise ValueError(f"{sample_id}: labels must align with input_ids")
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or not 0 < prompt_length < len(ids):
        raise ValueError(f"{sample_id}: prompt_length is invalid")
    if isinstance(completion_length, bool) or not isinstance(completion_length, int) or completion_length != len(ids) - prompt_length:
        raise ValueError(f"{sample_id}: completion_length does not align with prompt_length")
    if labels[:prompt_length] != [-100] * prompt_length:
        raise ValueError(f"{sample_id}: Prompt Mask is incomplete")
    if labels[prompt_length:] != ids[prompt_length:]:
        raise ValueError(f"{sample_id}: completion labels differ from the pre-encoded completion")
    if eos_token_id not in labels[prompt_length:]:
        raise ValueError(f"{sample_id}: terminal completion EOS is not supervised")
    return {
        "sample_id": sample_id,
        "sequence_length": len(ids),
        "prompt_length": prompt_length,
        "completion_length": completion_length,
    }


def materialize_a4_schedule(
    records: Sequence[Mapping[str, Any]],
    draws: Sequence[SamplingDraw],
    *,
    eos_token_id: int,
    max_sequence_length: int,
    gradient_accumulation_steps: int = EXPECTED_ACCUMULATION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join the frozen sample order to pre-encoded rows and bind their token values."""
    if len(draws) != EXPECTED_DRAW_COUNT:
        raise ValueError(f"A4 sampling plan must contain exactly {EXPECTED_DRAW_COUNT} draws")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in by_id:
            raise ValueError("A4 source records have a missing or duplicate sample_id")
        by_id[sample_id] = record
    ids = [draw.sample_id for draw in draws]
    if len(set(ids)) != len(ids):
        raise ValueError("uniform A4 sample schedule contains a repeated sample_id")
    rows: list[dict[str, Any]] = []
    schedule_identity: list[dict[str, Any]] = []
    encoded_identity: list[dict[str, Any]] = []
    for position, draw in enumerate(draws):
        record = by_id.get(draw.sample_id)
        if record is None or record.get("conversation_id") != draw.conversation_id:
            raise ValueError(f"A4 draw {position} does not match its Train source record")
        encoded = record.get("encoded")
        if not isinstance(encoded, Mapping):
            raise ValueError(f"{draw.sample_id}: pre-encoded D0.4.3 sample is missing")
        row = {
            "sample_id": draw.sample_id,
            "conversation_id": draw.conversation_id,
            "source_split": "train",
            "input_ids": list(encoded.get("input_ids", [])),
            "labels": list(encoded.get("labels", [])),
            "prompt_length": encoded.get("prompt_length"),
            "completion_length": encoded.get("completion_length"),
            "component": draw.component,
            "schedule_position": position,
            "optimizer_step": position // gradient_accumulation_steps + 1,
            "microbatch_index": position % gradient_accumulation_steps,
        }
        fact = validate_preencoded_row(
            row, eos_token_id=eos_token_id, max_sequence_length=max_sequence_length,
        )
        row["sequence_length"] = fact["sequence_length"]
        if row["source_split"] != "train" or draw.component != "natural":
            raise ValueError("A4 only accepts natural, no-replacement Train draws")
        rows.append(row)
        schedule_identity.append({
            key: row[key] for key in (
                "sample_id", "conversation_id", "source_split", "component",
                "schedule_position", "optimizer_step", "microbatch_index",
            )
        })
        encoded_identity.append({
            **fact,
            "input_ids": row["input_ids"],
            "labels": row["labels"],
        })
    if [row["optimizer_step"] for row in rows] != [
        step for step in range(1, EXPECTED_STEPS + 1) for _ in range(gradient_accumulation_steps)
    ]:
        raise ValueError("A4 schedule is not aligned to complete optimizer steps")
    return rows, {
        "draw_count": len(rows),
        "unique_sample_count": len(ids),
        "repeated_sample_draw_count": len(ids) - len(set(ids)),
        "optimizer_steps": EXPECTED_STEPS,
        "microbatches_per_step": gradient_accumulation_steps,
        "source_split": "train",
        "schedule_sha256": canonical_sha256(schedule_identity),
        "encoded_rows_sha256": canonical_sha256(encoded_identity),
        "max_sequence_length": max(row["sequence_length"] for row in rows),
        "minimum_sequence_length": min(row["sequence_length"] for row in rows),
        "dev_rows_read": 0,
        "val_rows_read": 0,
    }


def schedule_json_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return an ID-only ledger: never serialize tokenized content or source text."""
    return [{
        key: row[key]
        for key in (
            "sample_id", "conversation_id", "source_split", "component",
            "schedule_position", "optimizer_step", "microbatch_index",
        )
    } | {
        "sequence_length": len(row["input_ids"]),
        "prompt_length": row["prompt_length"],
        "completion_length": row["completion_length"],
    } for row in rows]


def ticket_binding(ticket: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable inputs covered by a GPU preflight ticket digest."""
    fields = (
        "status", "run_id", "config_sha256", "cpu_preflight_sha256", "schedule_sha256",
        "encoded_rows_sha256", "source_hashes", "implementation_sha256",
        "gpu", "dry_step",
    )
    return {key: ticket[key] for key in fields}


def verify_gpu_ticket(
    ticket: Mapping[str, Any], *, run_id: str, config_sha256: str,
    cpu_preflight_sha256: str, schedule_sha256: str, encoded_rows_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if ticket.get("status") != "passed" or ticket.get("run_id") != run_id:
        raise ValueError("A4 GPU preflight ticket is absent, failed, or belongs to another Run")
    expected = {
        "config_sha256": config_sha256,
        "cpu_preflight_sha256": cpu_preflight_sha256,
        "schedule_sha256": schedule_sha256,
        "encoded_rows_sha256": encoded_rows_sha256,
        "implementation_sha256": dict(implementation_sha256),
    }
    for key, value in expected.items():
        if ticket.get(key) != value:
            raise ValueError(f"A4 GPU preflight ticket binding changed: {key}")
    if ticket.get("ticket_sha256") != canonical_sha256(ticket_binding(ticket)):
        raise ValueError("A4 GPU preflight ticket digest is invalid")


def finite_nonzero_gradient_facts(gradients: Sequence[Any]) -> dict[str, Any]:
    """Check LoRA gradients through a small tensor-like protocol used by the runner."""
    present = [gradient for gradient in gradients if gradient is not None]
    if not present:
        raise ValueError("A4 backward pass produced no LoRA gradients")
    finite = True
    nonzero = False
    for gradient in present:
        if hasattr(gradient, "detach"):
            gradient = gradient.detach()
        if hasattr(gradient, "isfinite"):
            finite = finite and bool(gradient.isfinite().all().item())
        if hasattr(gradient, "ne"):
            nonzero = nonzero or bool(gradient.ne(0).any().item())
        elif isinstance(gradient, (int, float)):
            finite = finite and math.isfinite(float(gradient))
            nonzero = nonzero or float(gradient) != 0.0
    if not finite or not nonzero:
        raise ValueError("A4 backward pass gradients must be finite and at least one LoRA gradient nonzero")
    return {"gradient_tensor_count": len(present), "finite": True, "nonzero": True}


def validate_trainable_lora_names(names: Sequence[str], target_modules: Sequence[str]) -> list[str]:
    """Require the optimizer parameter list to contain every, and only, frozen LoRA targets."""
    values = list(names)
    if not values or any(
        not isinstance(name, str) or (".lora_A." not in name and ".lora_B." not in name)
        for name in values
    ):
        raise ValueError("A4 optimizer may update only LoRA A/B parameters")
    observed: set[str] = set()
    for name in values:
        prefix = name.split(".lora_A.", 1)[0].split(".lora_B.", 1)[0]
        observed.add(prefix.rsplit(".", 1)[-1])
    expected = set(target_modules)
    if observed != expected:
        raise ValueError(
            f"A4 LoRA target modules changed: expected={sorted(expected)}, observed={sorted(observed)}"
        )
    return sorted(observed)


def validate_fresh_optimizer_state(state: Mapping[str, Any]) -> None:
    """Prevent loading optimizer moments from E01-B/A3 or a previous A4 attempt."""
    if state:
        raise ValueError("A4 AdamW optimizer must start with empty state")
