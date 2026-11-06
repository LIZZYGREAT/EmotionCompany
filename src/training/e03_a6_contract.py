"""Small, dependency-light identity checks for the isolated E03-A6 F3 trainer."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any


MICROFIT_EXPERIMENT_ID = "E03-A6-F3-MICROFIT"
PILOT_EXPERIMENT_ID = "E03-A6-F3-PILOT"
EXPANDED_PILOT_EXPERIMENT_ID = "E03-A6-F3-PILOT-EXPANDED"
EXPANDED_F4_PILOT_EXPERIMENT_ID = "E03-A6-F4-PILOT-EXPANDED"
HISTORICAL_SELECTION_ROWS_SHA256 = "b53336224071914500aa91a4ad82c6f9e6334b11eb5a69231a6b4e20ba09314e"
EXPANDED_SELECTION_ROWS_SHA256 = "37cf91d4ced6d7a286cf4b74faf8a65294fa63e3a3ca6b7ef1663095ec1848a4"


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
    if config.get("evaluation", {}).get("autocast_adapter_dtype") is not True:
        raise ValueError("microfit reload must restore the verified FP32 LoRA checkpoint dtype")
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
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "epochs": 1,
        "save_interval_steps": 32,
        "save_total_limit": 2,
        "evaluation_interval_steps": 32,
        "parameter_check_interval_steps": 16,
        "max_training_seconds": 7200,
        "min_free_disk_gb": 2,
        "gradient_checkpointing": True,
        "checkpointing": True,
        "pause_after_step": 32,
    }
    changed = {name: (training.get(name), value) for name, value in locked.items() if training.get(name) != value}
    if changed:
        raise ValueError(f"Pilot training budget or optimizer identity changed: {changed}")
    expected_target_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    if set(config.get("lora", {}).get("target_modules", [])) != expected_target_modules:
        raise ValueError("Pilot LoRA target modules differ from the frozen E01-B set")
    if config.get("quantization", {}).get("method") != "bitsandbytes_4bit" or config.get("quantization", {}).get("quant_type") != "nf4":
        raise ValueError("Pilot quantization must remain NF4")
    if config.get("quantization", {}).get("compute_dtype") != "bfloat16" or config.get("quantization", {}).get("use_double_quant") is not True:
        raise ValueError("Pilot compute dtype and double quantization differ from the frozen profile")
    if config.get("quantization", {}).get("max_cuda_memory_fraction") != 0.92:
        raise ValueError("Pilot CUDA allocator budget must remain fixed at 92 percent")
    optimizer = config.get("optimizer", {})
    expected_optimizer = {
        "name": "AdamW", "learning_rate": 0.0001, "lr_scheduler": "constant",
        "warmup_steps": 0, "weight_decay": 0.0,
    }
    if any(optimizer.get(name) != value for name, value in expected_optimizer.items()):
        raise ValueError("Pilot optimizer settings differ from the frozen E01-B Pilot profile")
    sample_selection = training.get("sample_selection", {})
    expected_selection = {
        "method": "seeded_length_deciles_v1", "count": 512, "seed": 42,
        "expected_rows_sha256": "b53336224071914500aa91a4ad82c6f9e6334b11eb5a69231a6b4e20ba09314e",
    }
    if any(sample_selection.get(name) != value for name, value in expected_selection.items()):
        raise ValueError("Pilot Train selection identity differs from the historical 512-row artifact")
    if training.get("train_sampling_strategy") != "random" or training.get("checkpointing") is not True:
        raise ValueError("Pilot must retain the E01-B random sampler and checkpointing behavior")
    if training.get("pause_after_step") != 32 or training.get("evaluation_interval_steps") != 32:
        raise ValueError("Pilot must pause and evaluate at optimizer step 32")
    if config.get("evaluation", {}).get("autocast_adapter_dtype") is not True:
        raise ValueError("Pilot reload must preserve its verified FP32 LoRA dtype")
    if config.get("base_model", {}).get("expected_weight_sha256") != (
        "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
    ):
        raise ValueError("Pilot Base model identity differs from the frozen Qwen3-1.7B")
    if config.get("base_model", {}).get("model_id") != "Qwen/Qwen3-1.7B" or config.get("base_model", {}).get("revision") != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
        raise ValueError("Pilot Base model revision differs from the frozen E01-B identity")
    if config.get("lora", {}).get("r") != 8 or config.get("lora", {}).get("alpha") != 16:
        raise ValueError("Pilot LoRA identity must remain r=8/alpha=16")
    if config.get("lora", {}).get("dropout") != 0.0 or config.get("lora", {}).get("bias") != "none":
        raise ValueError("Pilot LoRA dropout/bias differs from the frozen E01-B profile")
    if config.get("lora", {}).get("checkpoint_dtype") != "float32":
        raise ValueError("Pilot LoRA checkpoint dtype must match the verified FP32 training adapter")
    if config.get("data", {}).get("microfit_run_id") != "E03-A6-F3-MICROFIT-20261105-03":
        raise ValueError("Pilot must depend on the passed, independently reloaded Microfit Run")
    if config.get("data", {}).get("expected_historical_selection_sha256") != HISTORICAL_SELECTION_ROWS_SHA256:
        raise ValueError("Pilot historical ordered sample hash differs from the frozen E01-B artifact")
    if config.get("protocol", {}).get("max_prompt_tokens") != 1280 or config.get("protocol", {}).get("max_train_seq_length") != 2048:
        raise ValueError("Pilot prompt or sequence token budget differs from the frozen D0.4.2 profile")
    if config.get("protocol", {}).get("packing") is not False or config.get("protocol", {}).get("padding_free") is not False:
        raise ValueError("Pilot requires packing=false and padding_free=false")
    selection = config.get("training", {}).get("sample_selection", {})
    if selection.get("count") != 512 or selection.get("seed") != 42 or selection.get("method") != "seeded_length_deciles_v1":
        raise ValueError("Pilot sample selection method/count/seed differs from the historical Pilot")
    evaluation = config.get("evaluation", {})
    if evaluation.get("sft_dev_loss_rows") != 256 or evaluation.get("sft_dev_loss_seed") != 43:
        raise ValueError("Pilot teacher-forced SFT Dev Loss selection differs from the historical Pilot")
    if evaluation.get("pilot_dev_generation_rows") != 160 or evaluation.get("probe_max_new_tokens") != 64:
        raise ValueError("Pilot Dev cohort or reload probe budget differs from the frozen plan")
    if evaluation.get("autocast_adapter_dtype") is not True:
        raise ValueError("Pilot reload must restore its verified FP32 LoRA checkpoint dtype")
    expected_gates = {
        "fast": {
            "min_strict_f3_parse_rate": 0.90,
            "min_cases_with_nonempty_profile_prediction": 16,
            "min_profile_true_positives": 5,
            "max_profile_false_positives": 30,
            "min_profile_micro_precision": 0.20,
            "min_emotion_accuracy": 0.30,
            "max_single_emotion_prediction_share": 0.80,
        },
        "full_generation_smoke": {
            "min_strict_f3_parse_rate": 0.95,
            "min_official_schema_valid_rate": 1.0,
            "min_prefix_full_equivalence_rate": 1.0,
            "min_nonempty_response_rate": 0.95,
            "min_eos_rate": 0.90,
        },
    }
    if config.get("pilot_quality_gates") != expected_gates:
        raise ValueError("Pilot quality review thresholds must remain fixed and versioned before training")


def validate_expanded_pilot_config(config: Mapping[str, Any]) -> None:
    """Validate the separately identified, nested 1,024-row / 256-step follow-up Pilot."""
    if config.get("version") != "E03-A6-F3-expanded-pilot-v1":
        raise ValueError("unsupported F3 expanded Pilot config version")
    if config.get("experiment_id") != EXPANDED_PILOT_EXPERIMENT_ID:
        raise ValueError("expanded Pilot must use its isolated experiment ID")
    if config.get("training_scope") != "supplemental_pilot":
        raise ValueError("expanded Pilot must declare supplemental_pilot scope")
    training = config.get("training", {})
    if (
        training.get("max_optimizer_steps") != 256
        or training.get("pause_after_step") != 64
        or training.get("epochs") != 1
    ):
        raise ValueError("expanded Pilot budget is frozen at one 1,024-row pass and a step-64 checkpoint pause")
    selection = training.get("sample_selection", {})
    if selection != {
        "method": "seeded_length_deciles_v1",
        "count": 1024,
        "seed": 42,
        "length_bins": 10,
        "base_count": 512,
        "base_rows_sha256": HISTORICAL_SELECTION_ROWS_SHA256,
        "expected_rows_sha256": EXPANDED_SELECTION_ROWS_SHA256,
    }:
        raise ValueError("expanded Pilot selection must be the frozen nested 1,024-row cohort")

    # Reuse the original Pilot contract to ensure every non-sample-size training
    # setting, model identity, protocol gate, and quality threshold stays fixed.
    baseline = deepcopy(dict(config))
    baseline["version"] = "E03-A6-F3-pilot-v1"
    baseline["experiment_id"] = PILOT_EXPERIMENT_ID
    baseline["training_scope"] = "pilot"
    baseline["training"]["max_optimizer_steps"] = 128
    baseline["training"]["pause_after_step"] = 32
    baseline["training"]["sample_selection"] = {
        "method": "seeded_length_deciles_v1",
        "count": 512,
        "seed": 42,
        "expected_rows_sha256": HISTORICAL_SELECTION_ROWS_SHA256,
    }
    validate_pilot_config(baseline)


def validate_expanded_f4_pilot_config(config: Mapping[str, Any]) -> None:
    """Validate a full-field F4 follow-up with the same 1,024-row Pilot budget."""
    if config.get("version") != "E03-A6-F4-full-field-pilot-v1":
        raise ValueError("unsupported F4 full-field Pilot config version")
    if config.get("experiment_id") != EXPANDED_F4_PILOT_EXPERIMENT_ID:
        raise ValueError("F4 full-field Pilot must use its isolated experiment ID")
    if config.get("training_scope") != "supplemental_pilot":
        raise ValueError("F4 full-field Pilot must declare supplemental_pilot scope")
    if config.get("protocol", {}).get("format_version") != "F4":
        raise ValueError("F4 full-field Pilot must explicitly select format_version=F4")
    if config.get("protocol", {}).get("inference_config") != (
        "participant/configs/e03_a6_f4_inference_config.json"
    ):
        raise ValueError("F4 full-field Pilot must use its isolated F4 inference config")
    if config.get("data", {}).get("historical_f2_dev") != "data/processed/sft/v1/dev.jsonl":
        raise ValueError("F4 full-field Pilot must use the frozen F2 Dev source")

    baseline = deepcopy(dict(config))
    baseline["version"] = "E03-A6-F3-expanded-pilot-v1"
    baseline["experiment_id"] = EXPANDED_PILOT_EXPERIMENT_ID
    baseline["protocol"]["format_version"] = "F3"
    baseline["protocol"]["inference_config"] = "participant/configs/e03_a6_f3_inference_config.json"
    validate_expanded_pilot_config(baseline)
def validate_pilot_resume_manifest(
    manifest: Mapping[str, Any], *, run_id: str, checkpoint_name: str,
) -> None:
    if not run_id.startswith("E03-A6-F3-PILOT-") or manifest.get("run_id") != run_id:
        raise ValueError("resume Run ID does not match an isolated A6 Pilot Run")
    if manifest.get("experiment_id") != PILOT_EXPERIMENT_ID or manifest.get("training_scope") != "pilot":
        raise ValueError("resume manifest is not an A6 Pilot Run")
    if manifest.get("status") != "interrupted" or manifest.get("stage") != "checkpoint_saved":
        raise ValueError("resume requires the same Run to be paused at its verified checkpoint")
    if manifest.get("optimizer_steps") != 32 or manifest.get("max_optimizer_steps") != 128:
        raise ValueError("resume progress must be exactly checkpoint-32 of the 128-step Pilot")
    if manifest.get("stop_reason") != "requested_checkpoint_pause":
        raise ValueError("resume checkpoint was not produced by the requested step-32 pause")
    last_checkpoint = manifest.get("last_checkpoint")
    if not isinstance(last_checkpoint, str) or last_checkpoint.rsplit("/", 1)[-1] != checkpoint_name:
        raise ValueError("resume checkpoint path does not match the same Run manifest")
    checkpoint_parts = last_checkpoint.replace("\\", "/").split("/")
    if len(checkpoint_parts) < 3 or checkpoint_parts[-3:-1] != [run_id, "trainer"]:
        raise ValueError("resume checkpoint must be stored inside this Pilot Run's trainer directory")
    if checkpoint_name != "checkpoint-32":
        raise ValueError("A6 Pilot resume is limited to its same-Run checkpoint-32")
    if manifest.get("full_training_status") != "not_authorized_not_run":
        raise ValueError("Pilot resume cannot alter or authorize the Full training boundary")


def validate_expanded_pilot_resume_manifest(
    manifest: Mapping[str, Any], *, run_id: str, checkpoint_name: str,
    pause_step: int = 64, max_steps: int = 256,
) -> None:
    if not run_id.startswith("E03-A6-F3-PILOT-EXPANDED-") or manifest.get("run_id") != run_id:
        raise ValueError("resume Run ID does not match the supplemental A6 Pilot")
    if (
        manifest.get("experiment_id") != EXPANDED_PILOT_EXPERIMENT_ID
        or manifest.get("training_scope") != "supplemental_pilot"
    ):
        raise ValueError("resume manifest is not the 1,024-row supplemental Pilot")
    if manifest.get("status") != "interrupted" or manifest.get("stage") != "checkpoint_saved":
        raise ValueError("resume requires the same supplemental Run paused at its verified checkpoint")
    if (
        manifest.get("optimizer_steps") != pause_step
        or manifest.get("max_optimizer_steps") != max_steps
        or manifest.get("stop_reason") != "requested_checkpoint_pause"
    ):
        raise ValueError("supplemental Pilot resume progress differs from its frozen step boundary")
    expected_name = f"checkpoint-{pause_step}"
    if checkpoint_name != expected_name:
        raise ValueError("supplemental Pilot resume must use its configured same-Run pause checkpoint")
    last_checkpoint = manifest.get("last_checkpoint")
    if not isinstance(last_checkpoint, str) or not last_checkpoint.replace("\\", "/").endswith(
        f"/{run_id}/trainer/{expected_name}"
    ):
        raise ValueError("supplemental Pilot checkpoint path does not belong to this Run")
    if manifest.get("full_training_status") != "not_authorized_not_run":
        raise ValueError("supplemental Pilot resume cannot alter or authorize Full training")


def validate_expanded_f4_pilot_resume_manifest(
    manifest: Mapping[str, Any], *, run_id: str, checkpoint_name: str,
    pause_step: int = 64, max_steps: int = 256,
) -> None:
    if not run_id.startswith("E03-A6-F4-PILOT-EXPANDED-") or manifest.get("run_id") != run_id:
        raise ValueError("resume Run ID does not match the F4 full-field Pilot")
    if (
        manifest.get("experiment_id") != EXPANDED_F4_PILOT_EXPERIMENT_ID
        or manifest.get("training_scope") != "supplemental_pilot"
        or manifest.get("format_version") != "F4"
    ):
        raise ValueError("resume manifest is not the F4 full-field supplemental Pilot")
    if manifest.get("status") != "interrupted" or manifest.get("stage") != "checkpoint_saved":
        raise ValueError("F4 resume requires the same Pilot paused at its verified checkpoint")
    if (
        manifest.get("optimizer_steps") != pause_step
        or manifest.get("max_optimizer_steps") != max_steps
        or manifest.get("stop_reason") != "requested_checkpoint_pause"
    ):
        raise ValueError("F4 Pilot resume progress differs from its frozen step boundary")
    expected_name = f"checkpoint-{pause_step}"
    if checkpoint_name != expected_name:
        raise ValueError("F4 Pilot resume must use its configured same-Run pause checkpoint")
    last_checkpoint = manifest.get("last_checkpoint")
    if not isinstance(last_checkpoint, str) or not last_checkpoint.replace("\\", "/").endswith(
        f"/{run_id}/trainer/{expected_name}"
    ):
        raise ValueError("F4 Pilot checkpoint path does not belong to this Run")
    if manifest.get("full_training_status") != "not_authorized_not_run":
        raise ValueError("F4 supplemental resume cannot alter or authorize Full training")


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
        "all Train emotion labels, both profile-empty states, and min/max sequence lengths included"
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


def f4_prefix_through_response_marker(
    token_ids: Sequence[int], decode_prefix: Any,
) -> tuple[list[int], str]:
    """Return generated IDs through the token that completes ``\nResponse=``."""
    marker_text = "\nResponse="
    for end in range(1, len(token_ids) + 1):
        decoded = decode_prefix(list(token_ids[:end]))
        marker = decoded.find(marker_text)
        if marker >= 0:
            boundary = marker + len(marker_text)
            return list(token_ids[:end]), decoded[:boundary]
    raise ValueError("generated token IDs did not complete the F4 Response= boundary")
