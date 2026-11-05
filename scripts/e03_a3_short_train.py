#!/usr/bin/env python3
"""Isolated CPU audit, GPU dry-step check, and bounded E03-A3 QLoRA continuation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from scripts import e01_b_diagnose
from src.data.sft_collator import SFTDataCollator
from src.diagnostics.e03_a3_profile import validate_history
from src.evaluation import read_jsonl, schema_enums, sha256_file
from src.training.e03_a3_sampling import (
    NATURAL,
    SHORT_EMPTY,
    SHORT_NONEMPTY,
    build_schedule,
    materialize_schedule,
    prepare_training_records,
    schedule_summary,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "E03-A3-short-training-v1":
        raise ValueError("A3 short training requires version E03-A3-short-training-v1")
    if value.get("experiment_id") != "E03-A3" or value.get("training_scope") != "short_contrast":
        raise ValueError("A3 config must be limited to the short contrast")
    approval = value.get("authorization", {})
    if approval.get("scope") != "E03-A3-stage-3-short-contrast" or approval.get("authorized") is not True:
        raise ValueError("A3 short contrast has no explicit stage-3 authorization receipt")
    if value.get("adapter_init", {}).get("optimizer_state") != "fresh" or value.get("adapter_init", {}).get("resume_from_checkpoint") is not None:
        raise ValueError("A3 must initialize only Adapter weights and create a fresh optimizer")
    return value


def _verified_hash(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    observed = sha256_file(path)
    if not expected or observed != expected:
        raise ValueError(f"{description} SHA-256 differs from the frozen configuration")
    return observed


def _official_enums(schema: dict[str, Any]) -> dict[str, list[str]]:
    definitions = schema.get("$defs", {})
    profile = definitions.get("profile", {}).get("properties", {})
    result = {"emotion_label": list(definitions.get("emotion", {}).get("enum", []))}
    for dimension in ("personality_traits", "interests", "style"):
        result[dimension] = list(profile.get(dimension, {}).get("items", {}).get("enum", []))
    if any(not labels for labels in result.values()):
        raise ValueError("official Train schema lacks a fixed emotion/profile enum")
    return result


def _validate_frozen_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    identity = config["identity"]
    root = _resolve(identity["diagnosis_run_root"])
    manifest_path = root / "manifest.json"
    aggregate_path = root / "aggregate.json"
    selection_path = root / "frozen_eval_selection.json"
    expected = {
        "aggregate.json": identity["expected_diagnosis_aggregate_sha256"],
        "frozen_eval_selection.json": identity["expected_frozen_eval_selection_sha256"],
    }
    _verified_hash(aggregate_path, expected["aggregate.json"], "A3 diagnosis aggregate")
    _verified_hash(selection_path, expected["frozen_eval_selection.json"], "frozen evaluation selection")
    manifest = _json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("run_id") != root.name:
        raise ValueError("A3 diagnosis Run is not complete or its Run ID differs")
    selection = _json(selection_path)
    if (
        selection.get("status") != "frozen"
        or selection.get("selected_points") != int(config["evaluation"]["frozen_eval_points"])
        or selection.get("selected_conversations") != int(config["evaluation"]["frozen_eval_conversations"])
        or selection.get("ids_sha256") != identity["expected_frozen_eval_ids_sha256"]
        or selection.get("selection_uses_predictions") is not False
        or selection.get("conversation_groups_split") is not False
    ):
        raise ValueError("frozen evaluation membership differs from the authorized selection")
    records = {
        "frozen_eval_input.jsonl": (config["evaluation"]["frozen_eval_input_sha256"], "input"),
        "frozen_eval_references.jsonl": (config["evaluation"]["frozen_eval_references_sha256"], "references"),
        "frozen_eval_id_map.jsonl": (config["evaluation"]["frozen_eval_id_map_sha256"], "id_map"),
    }
    outputs = manifest.get("outputs", {})
    checked: dict[str, str] = {}
    for filename, (digest, _role) in records.items():
        path = root / filename
        fact = outputs.get(filename, {})
        _verified_hash(path, digest, filename)
        if fact.get("sha256") != digest:
            raise ValueError(f"A3 diagnosis Manifest does not bind {filename}")
        checked[filename] = digest
    inputs = read_jsonl(root / "frozen_eval_input.jsonl")
    references = read_jsonl(root / "frozen_eval_references.jsonl")
    id_map = read_jsonl(root / "frozen_eval_id_map.jsonl")
    input_by_id = {row.get("id"): row for row in inputs}
    ref_by_id = {row.get("eval_id"): row for row in references}
    map_by_id = {row.get("eval_id"): row for row in id_map}
    expected_ids = set(ref_by_id)
    if (
        len(input_by_id) != len(inputs) or len(ref_by_id) != len(references)
        or len(map_by_id) != len(id_map) or set(input_by_id) != expected_ids
        or set(map_by_id) != expected_ids or len(expected_ids) != selection["selected_points"]
    ):
        raise ValueError("frozen input, References, and ID map do not have the same unique ID set")
    conversations: set[str] = set()
    for eval_id in expected_ids:
        row = input_by_id[eval_id]
        reference = ref_by_id[eval_id]
        mapping = map_by_id[eval_id]
        if set(row) != {"id", "history"}:
            raise ValueError("frozen inference input is not Gold-free")
        history_length = validate_history(row.get("history"))
        if reference.get("sample_id") != mapping.get("sample_id") or not mapping.get("conversation_id"):
            raise ValueError("frozen References and ID map do not match")
        conversations.add(mapping["conversation_id"])
        if history_length < 1:
            raise ValueError("frozen history is empty")
    if len(conversations) != selection["selected_conversations"]:
        raise ValueError("frozen ID map conversation count differs")
    return {
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "aggregate_sha256": expected["aggregate.json"],
        "selection_sha256": expected["frozen_eval_selection.json"],
        "selection_ids_sha256": selection["ids_sha256"],
        "sample_count": len(expected_ids),
        "conversation_count": len(conversations),
        "outputs": checked,
        "input_target_free": True,
        "conversation_groups_split": False,
    }


def _verify_static_assets(config: dict[str, Any]) -> dict[str, Any]:
    identity = config["identity"]
    data = config["data"]
    protocol = config["protocol"]
    base_config = _resolve("configs/training/e01_a.yaml")
    base_reference = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    e01_run = _resolve(identity["e01_b_training_run_root"])
    e01_manifest_path = e01_run / "manifest.json"
    e01_snapshot_path = e01_run / "config_snapshot.yaml"
    _verified_hash(e01_manifest_path, identity["expected_e01_b_manifest_sha256"], "E01-B Full Manifest")
    e01_manifest = _json(e01_manifest_path)
    if (
        e01_manifest.get("status") != "passed"
        or e01_manifest.get("experiment_id") != "E01-B"
        or e01_manifest.get("training_scope") != "full_train"
        or e01_manifest.get("optimizer_state_resume_enabled") is not True
        or e01_manifest.get("base_weights_unchanged") is not True
    ):
        raise ValueError("E01-B source Run is not the completed frozen Full baseline")
    _verified_hash(e01_snapshot_path, e01_manifest.get("config_sha256"), "E01-B config snapshot")
    e01_config = yaml.safe_load(e01_snapshot_path.read_text(encoding="utf-8"))
    for key in ("base_model", "quantization", "lora"):
        if config[key] != e01_config[key] or config[key] != base_reference[key]:
            raise ValueError(f"A3 {key} differs from the E01-B/E01-A frozen training configuration")
    protocol_keys = {
        "prompt_path": "prompt_path",
        "prompt_sha256": "p0_sha256",
        "runner_config": "runner_config",
        "tokenizer_id": "tokenizer_id",
        "tokenizer_revision": "tokenizer_revision",
        "tokenizer_signature_sha256": "tokenizer_signature_sha256",
        "chat_template_sha256": "chat_template_sha256",
        "eos_token_id": "eos_token_id",
        "max_train_seq_length": "max_train_seq_length",
        "max_prompt_tokens": "max_prompt_tokens",
        "history_policy": "history_policy",
        "format_version": "format_version",
        "thinking": "thinking",
        "packing": "packing",
        "padding_free": "padding_free",
    }
    for a3_key, baseline_key in protocol_keys.items():
        expected = e01_config["protocol"].get(baseline_key)
        if config["protocol"].get(a3_key) != expected or expected != base_reference["protocol"].get(baseline_key):
            raise ValueError(f"A3 protocol.{a3_key} differs from the frozen E01-B/E01-A value")
    for key in ("name", "learning_rate", "lr_scheduler", "weight_decay"):
        if config["optimizer"].get(key) != e01_config["optimizer"].get(key):
            raise ValueError(f"A3 optimizer.{key} differs from E01-B")
    if config["optimizer"].get("warmup_steps", 0) != e01_config["optimizer"].get("warmup_steps", 0):
        raise ValueError("A3 optimizer.warmup_steps differs from E01-B")
    if e01_manifest.get("train_sha256") != identity["expected_sft_train_sha256"]:
        raise ValueError("E01-B Full did not use the locked SFT Train data")

    adapter_path = _resolve(config["adapter_init"]["source"])
    adapter_identity, adapter_config = inference.adapter_weight_identity(adapter_path)
    if (
        adapter_identity.get("weight_sha256") != config["adapter_init"]["expected_weight_sha256"]
        or adapter_identity.get("config_sha256") != config["adapter_init"]["expected_config_sha256"]
        or adapter_identity.get("weight_sha256") != (e01_manifest.get("adapter_identity") or {}).get("weight_sha256")
        or adapter_identity.get("config_sha256") != (e01_manifest.get("adapter_identity") or {}).get("config_sha256")
    ):
        raise ValueError("read-only E01-B Adapter identity changed")
    base_path = _resolve(config["base_model"]["local_path"])
    inference.validate_adapter_base_reference(adapter_config, config["base_model"]["model_id"], base_path)
    base_identity = inference._weight_identity(base_path, {
        "model_id": config["base_model"]["model_id"],
        "model_revision": config["base_model"]["revision"],
    })
    if base_identity.get("observed_weight_hash") != identity["expected_base_weight_sha256"]:
        raise ValueError("local Base model weight identity differs from E01-B")

    train_path = _resolve(data["train_path"])
    normalized_path = _resolve(data["normalized_train_path"])
    sft_manifest_path = _resolve(data["sft_manifest"])
    d03_manifest_path = _resolve(data["d03_manifest"])
    d042_manifest_path = _resolve(data["d042_manifest"])
    train_schema_path = _resolve(data["official_data_schema"])
    output_schema_path = _resolve(protocol["output_schema"])
    for path, expected, description in (
        (train_path, identity["expected_sft_train_sha256"], "SFT Train"),
        (normalized_path, identity["expected_normalized_train_sha256"], "D0.3 Train"),
        (d03_manifest_path, identity["expected_d03_manifest_sha256"], "D0.3 Manifest"),
        (sft_manifest_path, identity["expected_sft_manifest_sha256"], "D0.4.3 Manifest"),
        (d042_manifest_path, identity["expected_d042_manifest_sha256"], "D0.4.2 Manifest"),
        (train_schema_path, identity["expected_official_data_schema_sha256"], "official Train schema"),
        (output_schema_path, identity["expected_output_schema_sha256"], "output schema"),
    ):
        _verified_hash(path, expected, description)
    sft_manifest = _json(sft_manifest_path)
    if (
        sft_manifest.get("d03_manifest_sha256") != identity["expected_d03_manifest_sha256"]
        or sft_manifest.get("d042_manifest_sha256") != identity["expected_d042_manifest_sha256"]
        or sft_manifest.get("outputs", {}).get("files", {}).get("train.jsonl", {}).get("sha256") != identity["expected_sft_train_sha256"]
        or sft_manifest.get("p0_sha256") != protocol["prompt_sha256"]
    ):
        raise ValueError("D0.4.3 Manifest does not bind the locked D0.3/D0.4.2/SFT/P0 inputs")
    d03_manifest = _json(d03_manifest_path)
    train_fact = next((fact for fact in d03_manifest.get("outputs", []) if fact.get("path") == data["normalized_train_path"]), None)
    if not train_fact or train_fact.get("sha256") != identity["expected_normalized_train_sha256"] or train_fact.get("records") != data["expected_d03_train_rows"]:
        raise ValueError("D0.3 Manifest does not bind the locked normalized Train rows")

    prompt_path = _resolve(protocol["prompt_path"])
    _verified_hash(prompt_path, protocol["prompt_sha256"], "P0 prompt")
    official_schema = _json(train_schema_path)
    output_schema = _json(output_schema_path)
    enums = schema_enums(output_schema)
    if _official_enums(official_schema) != enums:
        raise ValueError("official Train and inference output schema enums/order differ")
    runner_config = _json(_resolve(protocol["runner_config"]))
    if (
        runner_config.get("prompt", {}).get("sha256") != protocol["prompt_sha256"]
        or runner_config.get("context", {}).get("max_prompt_tokens") != protocol["max_prompt_tokens"]
        or runner_config.get("generation", {}).get("max_new_tokens") != protocol["max_new_tokens"]
        or runner_config.get("generation", {}).get("do_sample") is not False
        or runner_config.get("tokenizer", {}).get("expected_signature_sha256") != protocol["tokenizer_signature_sha256"]
        or runner_config.get("tokenizer", {}).get("expected_template_sha256") != protocol["chat_template_sha256"]
        or runner_config.get("tokenizer", {}).get("eos_token_id") != protocol["eos_token_id"]
    ):
        raise ValueError("P0/F2/T1/Tokenizer/G generation protocol differs from the frozen inference config")
    tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runner_config, base_path)
    if (
        tokenizer_facts.get("signature_sha256") != protocol["tokenizer_signature_sha256"]
        or tokenizer_facts.get("template_sha256") != protocol["chat_template_sha256"]
        or tokenizer.eos_token_id != protocol["eos_token_id"]
    ):
        raise ValueError("local tokenizer signature, chat template, or EOS differs from D0.4.2")

    sft_rows = read_jsonl(train_path)
    normalized_rows = read_jsonl(normalized_path)
    records = prepare_training_records(
        sft_rows,
        normalized_rows,
        enums,
        expected_rows=int(data["expected_train_rows"]),
        eos_token_id=int(protocol["eos_token_id"]),
        max_sequence_length=int(protocol["max_train_seq_length"]),
    )
    if len(normalized_rows) != int(data["expected_d03_train_rows"]):
        raise ValueError("D0.3 normalized Train count differs from the locked 16,312 rows")
    del tokenizer
    eval_facts = _validate_frozen_evaluation(config)
    free_disk = shutil.disk_usage(_resolve(data["output_root"]).parent).free
    min_disk = int(config["training"]["min_free_disk_gb"] * 1024 ** 3)
    if free_disk < min_disk:
        raise ValueError(f"insufficient free disk for paired short runs: free={free_disk}, required={min_disk}")

    schedule_facts = {}
    materialized: dict[str, list[dict[str, Any]]] = {}
    for group in ("control", "candidate"):
        schedule = build_schedule(
            records,
            group=group,
            seed=int(config["sampler"]["seed"]),
            per_device_micro_batch_size=int(config["training"]["per_device_micro_batch_size"]),
            gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        )
        rows = materialize_schedule(records, schedule)
        materialized[group] = rows
        summary = schedule_summary(
            rows,
            int(config["training"]["max_optimizer_steps"]),
            int(config["training"]["gradient_accumulation_steps"]),
        )
        if summary["consumed_draws"] != 512:
            raise ValueError("128-step run does not consume the frozen 512 training draws")
        if group == "control" and (summary["consumed_unique_sample_ids"] != 512 or summary["consumed_repeated_draws"] != 0):
            raise ValueError("Control must draw unique SFT samples without replacement")
        if group == "candidate" and summary["consumed_component_counts"] != {
            NATURAL: 256, SHORT_EMPTY: 128, SHORT_NONEMPTY: 128,
        }:
            raise ValueError("Candidate first 128 steps do not meet the locked 50/50 and 25/25 mix")
        schedule_facts[group] = summary

    return {
        "status": "passed_cpu_identity_and_sampler_audit",
        "created_at_utc": _now(),
        "experiment_id": "E03-A3",
        "stage": "G2_cpu_preflight",
        "training_rows": len(records),
        "unique_training_conversations": len({row["conversation_id"] for row in records}),
        "raw_short_training_rows": sum(row["raw_history_length"] <= 4 for row in records),
        "raw_short_gold_empty_rows": sum(row["raw_history_length"] <= 4 and row["gold_profile_empty"] for row in records),
        "raw_short_gold_nonempty_rows": sum(row["raw_history_length"] <= 4 and not row["gold_profile_empty"] for row in records),
        "source_hashes": {
            "sft_train": identity["expected_sft_train_sha256"],
            "normalized_train": identity["expected_normalized_train_sha256"],
            "d03_manifest": identity["expected_d03_manifest_sha256"],
            "sft_manifest": identity["expected_sft_manifest_sha256"],
            "d042_manifest": identity["expected_d042_manifest_sha256"],
            "base_weight": base_identity["observed_weight_hash"],
            "initial_adapter_weight": adapter_identity["weight_sha256"],
            "initial_adapter_config": adapter_identity["config_sha256"],
            "p0_prompt": protocol["prompt_sha256"],
            "official_data_schema": identity["expected_official_data_schema_sha256"],
            "output_schema": identity["expected_output_schema_sha256"],
        },
        "training_config_parity": {
            "base_model": True,
            "P0_F2_T1_tokenizer_EOS": True,
            "NF4_BF16_quantization": True,
            "LoRA": True,
            "AdamW_LR_scheduler_weight_decay": True,
            "completion_only_preencoded_input_labels": True,
        "completion_EOS_supervised": True,
        },
        "optimizer_state_resume_enabled": False,
        "frozen_evaluation": eval_facts,
        "free_disk_bytes": free_disk,
        "minimum_free_disk_bytes": min_disk,
        "sampling_schedules": schedule_facts,
        "gpu_dry_step": "not_run_by_codex",
    }


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_trainable_adapter(config: dict[str, Any], torch: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    base_path = _resolve(config["base_model"]["local_path"])
    adapter_path = _resolve(config["adapter_init"]["source"])
    before_adapter, _adapter_config = inference.adapter_weight_identity(adapter_path)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bool(config["quantization"]["use_double_quant"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        device_map=config["quantization"]["device_map"],
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
    )
    model.config.use_cache = False
    prepare_kwargs: dict[str, Any] = {
        "use_gradient_checkpointing": bool(config["training"]["gradient_checkpointing"]),
    }
    try:
        import inspect
        if "gradient_checkpointing_kwargs" in inspect.signature(prepare_model_for_kbit_training).parameters:
            prepare_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    except (TypeError, ValueError):
        pass
    model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=True,
        local_files_only=True,
        autocast_adapter_dtype=False,
    )
    if not getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(model, "_require_grads_hook", None) is None:
        model.enable_input_require_grads()
    model.train()

    source_weights = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    loaded_weights = get_peft_model_state_dict(model, adapter_name="default")
    if set(source_weights) != set(loaded_weights):
        raise RuntimeError("loaded Adapter tensor keys differ from the frozen E01-B weights")
    for name, source_tensor in source_weights.items():
        observed = loaded_weights[name].detach().to(device="cpu", dtype=source_tensor.dtype)
        if not torch.equal(observed, source_tensor):
            raise RuntimeError(f"loaded Adapter tensor changed during trainable initialization: {name}")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".lora_A." not in name and ".lora_B." not in name for name, _ in trainable):
        raise RuntimeError("A3 trainable parameters must be limited to LoRA A/B tensors")
    quant_facts = inference.inspect_quantization_runtime(
        model,
        {"quantization": {"enabled": True, "use_double_quant": bool(config["quantization"]["use_double_quant"])}},
        torch,
    )
    if not quant_facts.get("real_nf4_verified") or quant_facts.get("offload_devices"):
        raise RuntimeError("A3 requires verified CUDA NF4/BF16 placement without CPU/disk offload")
    quant_facts.update(
        trainable_parameter_count=sum(parameter.numel() for _name, parameter in trainable),
        all_trainable_parameters_are_lora=True,
        loaded_adapter_values_exact_match=True,
        adapter_dtype_autocast=False,
    )
    after_adapter, _ = inference.adapter_weight_identity(adapter_path)
    if any(before_adapter.get(key) != after_adapter.get(key) for key in ("weight_sha256", "config_sha256")):
        raise RuntimeError("read-only E01-B Adapter changed while loading A3")
    return model, quant_facts, after_adapter


def _gpu_gate(config: dict[str, Any], torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("blocked_gpu_unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_bf16_unavailable")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    minimum_total = int(config["training"]["min_total_vram_gb"] * 1024 ** 3)
    minimum_free = int(config["training"]["min_free_vram_gb"] * 1024 ** 3)
    if props.total_memory < minimum_total:
        raise RuntimeError(f"blocked_total_vram_below_gate: {props.total_memory} < {minimum_total}")
    if free < minimum_free:
        raise RuntimeError(f"blocked_free_vram_below_gate: {free} < {minimum_free}")
    return {
        "device": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "total_vram_bytes": int(total),
        "free_vram_bytes_before_load": int(free),
        "minimum_total_vram_bytes": minimum_total,
        "minimum_free_vram_bytes": minimum_free,
        "bf16_supported": True,
        "cuda_version": torch.version.cuda,
    }


def _load_training_protocol(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], Any, str, dict[str, Any], dict[str, list[str]]]:
    data, protocol = config["data"], config["protocol"]
    official_schema = _json(_resolve(data["official_data_schema"]))
    enums = _official_enums(official_schema)
    records = prepare_training_records(
        read_jsonl(_resolve(data["train_path"])),
        read_jsonl(_resolve(data["normalized_train_path"])),
        enums,
        expected_rows=int(data["expected_train_rows"]),
        eos_token_id=int(protocol["eos_token_id"]),
        max_sequence_length=int(protocol["max_train_seq_length"]),
    )
    runner_config = _json(_resolve(protocol["runner_config"]))
    tokenizer, facts = inference.load_runtime_tokenizer(runner_config, _resolve(config["base_model"]["local_path"]))
    if (
        facts.get("signature_sha256") != protocol["tokenizer_signature_sha256"]
        or facts.get("template_sha256") != protocol["chat_template_sha256"]
        or tokenizer.eos_token_id != int(protocol["eos_token_id"])
    ):
        raise ValueError("runtime tokenizer signature, template, or EOS differs from the frozen protocol")
    prompt = _resolve(protocol["prompt_path"]).read_text(encoding="utf-8")
    return records, tokenizer, prompt, inference.load_submission_schema(), enums


def _run_gpu_preflight(config: dict[str, Any], cpu_facts: dict[str, Any]) -> dict[str, Any]:
    import torch

    gate = _gpu_gate(config, torch)
    base_path = _resolve(config["base_model"]["local_path"])
    base_before = inference._weight_identity(base_path, {
        "model_id": config["base_model"]["model_id"],
        "model_revision": config["base_model"]["revision"],
    })
    model = None
    try:
        model, model_facts, adapter_identity = _load_trainable_adapter(config, torch)
        records, tokenizer, _prompt, _schema, _enums = _load_training_protocol(config)
        longest = max(records, key=lambda row: len(row["encoded"]["input_ids"]))
        collator = SFTDataCollator(
            int(tokenizer.pad_token_id),
            max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
        )
        batch = collator([longest["encoded"]])
        device = model.get_input_embeddings().weight.device
        batch = {key: value.to(device) for key, value in batch.items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        before = {name: parameter.detach().to(device="cpu", copy=True) for name, parameter in trainable}
        torch.cuda.reset_peak_memory_stats(device)
        output = model(**batch)
        loss = output.loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("max-length dry-step loss is non-finite")
        loss.backward()
        gradients = [parameter.grad for _name, parameter in trainable if parameter.grad is not None]
        if (
            not gradients
            or not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
            or not any(bool(gradient.ne(0).any().item()) for gradient in gradients)
        ):
            raise FloatingPointError("max-length dry-step lacks finite, nonzero LoRA gradients")
        unchanged = all(
            torch.equal(before[name], parameter.detach().to(device="cpu"))
            for name, parameter in trainable
        )
        if not unchanged:
            raise RuntimeError("dry backward changed Adapter weights without an optimizer step")
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        del batch, output
        model.zero_grad(set_to_none=True)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        free_after, total_after = torch.cuda.mem_get_info(gate["device"])
        base_after = inference._weight_identity(base_path, {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"]["revision"],
        })
        if base_before.get("observed_weight_hash") != base_after.get("observed_weight_hash"):
            raise RuntimeError("Base weights changed during the no-optimizer dry-step")
        return {
            "status": "passed_gpu_dry_step_no_optimizer_update",
            "gpu": gate,
            "model_facts": model_facts,
            "dry_step": {
                "sample_id": longest["sample_id"],
                "sequence_tokens": len(longest["encoded"]["input_ids"]),
                "loss_finite": True,
                "gradient_finite_nonzero": True,
                "optimizer_created": False,
                "optimizer_step_count": 0,
                "adapter_values_unchanged_after_backward": unchanged,
                "adapter_identity": adapter_identity,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "free_vram_bytes_after_unload": int(free_after),
                "total_vram_bytes_after": int(total_after),
                "base_weights_unchanged": True,
                "device": str(device),
            },
            "cpu_identity_and_sampler_audit": cpu_facts,
        }
    finally:
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _verify_gpu_preflight(config: dict[str, Any], config_path: Path, run_id: str) -> dict[str, Any]:
    if not run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}", run_id):
        raise ValueError("train mode requires a valid --gpu-preflight-run-id")
    path = _resolve(config["data"]["output_root"]).parent / "E03-A3-preflight" / run_id
    manifest_path = path / "manifest.json"
    record = _json(manifest_path)
    expected_implementation = {
        "runner": sha256_file(Path(__file__).resolve()),
        "sampler": sha256_file(ROOT / "src/training/e03_a3_sampling.py"),
    }
    if (
        record.get("run_id") != run_id
        or record.get("status") != "passed_gpu_dry_step_no_optimizer_update"
        or record.get("stage") != "G2_gpu_dry_step"
        or record.get("config_sha256") != sha256_file(config_path)
        or record.get("implementation_sha256") != expected_implementation
        or record.get("gpu_preflight_sha256") != sha256_file(path / "gpu_preflight.json")
    ):
        raise ValueError("training requires a successful GPU dry-step with this exact frozen config")
    facts = _json(path / "gpu_preflight.json")
    static_hashes = facts.get("cpu_identity_and_sampler_audit", {}).get("source_hashes", {})
    if (
        static_hashes.get("base_weight") != config["base_model"]["expected_weight_sha256"]
        or static_hashes.get("initial_adapter_weight") != config["adapter_init"]["expected_weight_sha256"]
        or static_hashes.get("initial_adapter_config") != config["adapter_init"]["expected_config_sha256"]
    ):
        raise ValueError("GPU preflight did not verify the configured Base/Adapter identities")
    dry = facts.get("dry_step", {})
    if (
        facts.get("status") != "passed_gpu_dry_step_no_optimizer_update"
        or dry.get("optimizer_created") is not False
        or dry.get("optimizer_step_count") != 0
        or dry.get("loss_finite") is not True
        or dry.get("gradient_finite_nonzero") is not True
        or dry.get("adapter_values_unchanged_after_backward") is not True
        or dry.get("base_weights_unchanged") is not True
    ):
        raise ValueError("GPU dry-step manifest failed its no-update evidence")
    return {"root": str(path), "manifest_sha256": sha256_file(manifest_path), "preflight_sha256": record["gpu_preflight_sha256"]}


def _write_schedule(run_root: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    path = run_root / "sample_schedule.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                key: row[key] for key in (
                    "schedule_position", "sample_id", "conversation_id", "raw_history_length",
                    "gold_profile_empty", "sampling_component",
                )
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sha256_file(path), schedule_summary(
        rows,
        int(config["training"]["max_optimizer_steps"]),
        int(config["training"]["gradient_accumulation_steps"]),
    )


def _make_trainer(
    config: dict[str, Any], run_root: Path, rows: list[dict[str, Any]], tokenizer: Any, model: Any,
) -> tuple[Any, Any]:
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainerCallback, TrainingArguments

    training = config["training"]
    accumulation = int(training["gradient_accumulation_steps"])
    microbatch_size = int(training["per_device_micro_batch_size"])
    collator_base = SFTDataCollator(
        int(tokenizer.pad_token_id),
        max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
    )

    class DrawCollator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            batch = collator_base(features)
            batch.update({
                "a3_sample_id": [row["sample_id"] for row in features],
                "a3_conversation_id": [row["conversation_id"] for row in features],
                "a3_raw_history_length": [row["raw_history_length"] for row in features],
                "a3_gold_profile_empty": [row["gold_profile_empty"] for row in features],
                "a3_sampling_component": [row["sampling_component"] for row in features],
                "a3_completion_tokens": [int(row["completion_length"]) for row in features],
            })
            return batch

    class A3Trainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.a3_pending_draws: list[dict[str, Any]] = []
            self.a3_last_loss: float | None = None
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,
            )
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError("A3 completion-only loss is non-finite")
            self.a3_last_loss = float(loss.detach().float().item())
            return result

        def training_step(self, model, inputs, num_items_in_batch=None):
            metadata = {
                "sample_ids": list(inputs.pop("a3_sample_id")),
                "conversation_ids": list(inputs.pop("a3_conversation_id")),
                "raw_history_lengths": [int(value) for value in inputs.pop("a3_raw_history_length")],
                "gold_profile_empty": [bool(value) for value in inputs.pop("a3_gold_profile_empty")],
                "sampling_components": list(inputs.pop("a3_sampling_component")),
                "completion_tokens": [int(value) for value in inputs.pop("a3_completion_tokens")],
            }
            loss = super().training_step(model, inputs, num_items_in_batch)
            if self.a3_last_loss is None or not math.isfinite(self.a3_last_loss):
                raise FloatingPointError("A3 microbatch loss evidence is missing or non-finite")
            metadata["loss"] = self.a3_last_loss
            metadata["optimizer_step"] = int(self.state.global_step) + 1
            self.a3_pending_draws.append(metadata)
            return loss

    class EvidenceCallback(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.trainable: list[tuple[str, Any]] = []
            self.before: dict[str, Any] = {}
            self.started = 0.0
            self.stop_reason: str | None = None
            self.checkpoints: list[int] = []
            self.fresh_optimizer_state_verified = False
            self.steps_path = run_root / "optimizer_steps.jsonl"
            self.logs_path = run_root / "trainer_logs.jsonl"

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            self.trainable = [(name, parameter) for name, parameter in self.trainer.model.named_parameters() if parameter.requires_grad]
            if not self.trainable:
                raise RuntimeError("A3 Trainer has no trainable LoRA parameters")
            optimizer = self.trainer.optimizer
            if optimizer is None or optimizer.state:
                raise RuntimeError("A3 AdamW optimizer must have a fresh, empty state at training start")
            self.fresh_optimizer_state_verified = True
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gradients = [parameter.grad for _name, parameter in self.trainable if parameter.grad is not None]
            if (
                not gradients
                or not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
                or not any(bool(gradient.ne(0).any().item()) for gradient in gradients)
            ):
                raise FloatingPointError("A3 optimizer step has missing, non-finite, or zero LoRA gradients")
            upcoming = int(state.global_step) + 1
            self.before = ({
                name: parameter.detach().to(device="cpu", copy=True)
                for name, parameter in self.trainable
            } if upcoming in {1, 64, int(training["max_optimizer_steps"])} else {})
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            upcoming = int(state.global_step) + 1
            if upcoming in {1, 64, int(training["max_optimizer_steps"])}:
                if not self.before or not any(
                    not torch.equal(self.before[name], parameter.detach().to(device="cpu"))
                    for name, parameter in self.trainable
                ):
                    raise RuntimeError(f"A3 AdamW update did not change LoRA parameters at step {upcoming}")
            self.before = {}
            return control

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            draws = self.trainer.a3_pending_draws
            if len(draws) != accumulation or any(row["optimizer_step"] != step for row in draws):
                raise RuntimeError(f"A3 optimizer step {step} has an incomplete or misaligned draw ledger")
            flat = lambda name: [value for row in draws for value in row[name]]
            record = {
                "optimizer_step": step,
                "microbatch_count": len(draws),
                "sample_ids": flat("sample_ids"),
                "conversation_ids": flat("conversation_ids"),
                "raw_history_lengths": flat("raw_history_lengths"),
                "gold_profile_empty": flat("gold_profile_empty"),
                "sampling_components": flat("sampling_components"),
                "completion_tokens": sum(sum(row["completion_tokens"]) for row in draws),
                "microbatch_losses": [row["loss"] for row in draws],
                "microbatch_loss_mean": sum(row["loss"] for row in draws) / len(draws),
                "finite_gradient_nonzero_checked": True,
                "parameter_update_checked": step in {1, 64, int(training["max_optimizer_steps"])},
            }
            with self.steps_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            draws.clear()
            if time.monotonic() - self.started >= float(training["max_training_seconds"]):
                self.stop_reason = "training_time_budget_exhausted"
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                record = {"optimizer_step": int(state.global_step), **logs}
                if not math.isfinite(float(record["loss"])):
                    raise FloatingPointError("Trainer log contains a non-finite loss")
                with self.logs_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            return control

        def on_save(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if not (run_root / "trainer" / f"checkpoint-{step}").is_dir():
                raise RuntimeError(f"Trainer reported a missing checkpoint at step {step}")
            self.checkpoints.append(step)
            return control

    args = TrainingArguments(
        output_dir=str(run_root / "trainer"),
        do_train=True,
        do_eval=False,
        use_cpu=False,
        per_device_train_batch_size=microbatch_size,
        gradient_accumulation_steps=accumulation,
        max_steps=int(training["max_optimizer_steps"]),
        num_train_epochs=1,
        learning_rate=float(config["optimizer"]["learning_rate"]),
        lr_scheduler_type=config["optimizer"]["lr_scheduler"],
        weight_decay=float(config["optimizer"]["weight_decay"]),
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(training["checkpoint_steps"][0]),
        save_total_limit=int(training["save_total_limit"]),
        save_only_model=False,
        warmup_steps=int(config["optimizer"]["warmup_steps"]),
        report_to="none",
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=1.0,
        include_num_input_tokens_seen=True,
    )
    evidence_callback = EvidenceCallback()
    trainer = A3Trainer(
        model=model,
        args=args,
        train_dataset=rows,
        data_collator=DrawCollator(),
        processing_class=tokenizer,
        callbacks=[evidence_callback],
    )
    evidence_callback.trainer = trainer
    return trainer, evidence_callback


def _training_run(
    config: dict[str, Any], config_path: Path, group: str, run_id: str,
    facts: dict[str, Any], preflight_identity: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from src.training import training_sft

    output_root = _resolve(config["data"]["output_root"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}", run_id):
        raise ValueError("--run-id must contain 3-96 safe ASCII characters")
    run_root = output_root / run_id
    if run_root.exists():
        raise FileExistsError(f"A3 Run ID exists; refusing overwrite: {run_root}")
    if not run_root.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("A3 Run path escaped its configured output root")
    run_root.mkdir(parents=True)
    snapshot_path = run_root / "config_snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    records, tokenizer, prompt, _schema, _enums = _load_training_protocol(config)
    schedule = build_schedule(
        records,
        group=group,
        seed=int(config["sampler"]["seed"]),
        per_device_micro_batch_size=int(config["training"]["per_device_micro_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
    )
    scheduled_rows = materialize_schedule(records, schedule)
    schedule_sha, schedule_facts = _write_schedule(run_root, scheduled_rows, config)
    expected_draws = int(config["training"]["max_optimizer_steps"]) * int(config["training"]["gradient_accumulation_steps"])
    if schedule_facts.get("consumed_draws") != expected_draws:
        raise ValueError("A3 128-step schedule does not contain exactly 512 consumed draws")
    if group == "control" and schedule_facts.get("consumed_repeated_draws") != 0:
        raise ValueError("A3 Control must not repeat a training sample in the first 128 steps")
    if group == "candidate" and schedule_facts.get("consumed_component_counts") != {
        NATURAL: 256, SHORT_EMPTY: 128, SHORT_NONEMPTY: 128,
    }:
        raise ValueError("A3 Candidate first 128 steps differ from the frozen 50/25/25 draw policy")
    initial_adapter_path = _resolve(config["adapter_init"]["source"])
    initial_adapter_identity, _ = inference.adapter_weight_identity(initial_adapter_path)
    base_path = _resolve(config["base_model"]["local_path"])
    base_before = inference._weight_identity(base_path, {
        "model_id": config["base_model"]["model_id"],
        "model_revision": config["base_model"]["revision"],
    })
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": "E03-A3",
        "training_scope": "short_contrast",
        "group": group,
        "status": "running",
        "stage": "loading_model",
        "created_at_utc": _now(),
        "config_sha256": sha256_file(snapshot_path),
        "authorization_receipt": config["authorization"],
        "gpu_preflight": preflight_identity,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "sampler": sha256_file(ROOT / "src/training/e03_a3_sampling.py"),
        },
        "initial_adapter_identity": initial_adapter_identity,
        "initial_adapter_values_exact_match": True,
        "optimizer_state": "fresh_adamw",
        "optimizer_state_resume_enabled": False,
        "resume_from_checkpoint": None,
        "training_rows": int(config["data"]["expected_train_rows"]),
        "schedule_sha256": schedule_sha,
        "schedule_summary": schedule_facts,
        "source_hashes": facts["source_hashes"],
        "frozen_evaluation": facts["frozen_evaluation"],
    }
    _write_json(run_root / "manifest.json", manifest)
    model = trainer = None
    started = time.monotonic()
    try:
        gate = _gpu_gate(config, torch)
        _seed_everything(int(config["training"]["seed"]), torch)
        model, model_facts, _loaded_identity = _load_trainable_adapter(config, torch)
        trainer, callback = _make_trainer(config, run_root, scheduled_rows, tokenizer, model)
        training_result = trainer.train(resume_from_checkpoint=None)
        steps = int(trainer.state.global_step)
        step_rows = read_jsonl(run_root / "optimizer_steps.jsonl") if (run_root / "optimizer_steps.jsonl").is_file() else []
        if [row.get("optimizer_step") for row in step_rows] != list(range(1, steps + 1)):
            raise RuntimeError("A3 optimizer-step evidence is incomplete, duplicated, or out of order")
        if not all(row.get("finite_gradient_nonzero_checked") for row in step_rows):
            raise RuntimeError("A3 gradient finite/nonzero evidence is incomplete")
        if not callback.fresh_optimizer_state_verified:
            raise RuntimeError("A3 did not verify a fresh empty AdamW state before the first step")
        if trainer.a3_pending_draws:
            raise RuntimeError("A3 ended with unlogged microbatch draws")
        optimizer_state_steps = []
        if trainer.optimizer is not None:
            for state in trainer.optimizer.state.values():
                value = state.get("step")
                if value is not None:
                    optimizer_state_steps.append(int(value.item()) if hasattr(value, "item") else int(value))
        if any(value < 1 or value > steps for value in optimizer_state_steps):
            raise RuntimeError("A3 AdamW state has an impossible step counter")
        if steps == int(config["training"]["max_optimizer_steps"]) and not optimizer_state_steps:
            raise RuntimeError("A3 completed without any fresh AdamW parameter state")
        if steps == int(config["training"]["max_optimizer_steps"]) and callback.checkpoints != [64, 128]:
            raise RuntimeError(f"A3 checkpoints differ from [64,128]: {callback.checkpoints}")
        adapter_dir = run_root / "adapter"
        adapter_facts = training_sft._save_adapter(model, adapter_dir, config)
        base_during = inference._weight_identity(base_path, {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"]["revision"],
        })
        final_initial_adapter_identity, _ = inference.adapter_weight_identity(initial_adapter_path)
        if any(
            initial_adapter_identity.get(key) != final_initial_adapter_identity.get(key)
            for key in ("weight_sha256", "config_sha256")
        ):
            raise RuntimeError("read-only E01-B source Adapter changed during A3 training")
        if base_before.get("observed_weight_hash") != base_during.get("observed_weight_hash"):
            raise RuntimeError("Base model weights changed during A3 training")
        if steps == int(config["training"]["max_optimizer_steps"]):
            probe_row = next(
                row for row in read_jsonl(_resolve(config["data"]["normalized_train_path"]))
                if row["sample_id"] == records[0]["sample_id"]
            )
            probe_prompt = training_sft._make_probe_prompt(
                [{"source": {"sample_id": probe_row["sample_id"], "history": probe_row["history"]}}],
                tokenizer, prompt, int(config["protocol"]["max_prompt_tokens"]),
            )
            probe_before = training_sft._capture_probe(
                model, tokenizer, probe_prompt, max_new_tokens=int(config["evaluation"]["probe_max_new_tokens"]),
            )
            del trainer, model
            trainer = model = None
            gc.collect()
            torch.cuda.empty_cache()
            reloaded, reload_torch, reload_model_facts = training_sft._load_train_model(
                config, base_path, adapter_path=adapter_dir,
            )
            probe_after = training_sft._capture_probe(
                reloaded, tokenizer, probe_prompt, max_new_tokens=int(config["evaluation"]["probe_max_new_tokens"]),
            )
            base_after = inference._weight_identity(base_path, {
                "model_id": config["base_model"]["model_id"],
                "model_revision": config["base_model"]["revision"],
            })
            base_unchanged = (
                base_before.get("observed_weight_hash")
                == base_during.get("observed_weight_hash")
                == base_after.get("observed_weight_hash")
                == config["base_model"]["expected_weight_sha256"]
            )
            token_match = probe_before["generated_token_ids"] == probe_after["generated_token_ids"]
            reload_record = {
                "status": "passed" if token_match and base_unchanged else "failed",
                "token_ids_exact_match": token_match,
                "base_weights_unchanged": base_unchanged,
                "base_weight_sha256_before": base_before.get("observed_weight_hash"),
                "base_weight_sha256_after": base_after.get("observed_weight_hash"),
                "adapter_identity": adapter_facts,
                "reload_model_facts": reload_model_facts,
                "before_release": probe_before,
                "after_reload": probe_after,
            }
            _write_json(run_root / "reload_check.json", reload_record)
            del reloaded
            gc.collect()
            reload_torch.cuda.empty_cache()
            if not token_match or not base_unchanged:
                raise RuntimeError("A3 standalone Adapter reload check failed")
        else:
            base_after = base_during
            reload_record = {"status": "not_run_incomplete_budget", "token_ids_exact_match": None}
            _write_json(run_root / "reload_check.json", reload_record)

        status = "trained_pending_eval" if steps == int(config["training"]["max_optimizer_steps"]) else "stopped_budget"
        manifest.update(
            status=status,
            stage="training_and_reload_complete" if steps == 128 else "training_stopped_before_budget",
            ended_at_utc=_now(),
            optimizer_steps=steps,
            training_seconds=time.monotonic() - started,
            time_budget_stop_reason=callback.stop_reason,
            trainer_metrics={
                key: value for key, value in training_result.metrics.items()
                if isinstance(value, (str, int, float, bool))
            },
            gpu=gate,
            model_facts=model_facts,
            adapter_identity=adapter_facts,
            reload_check_status=reload_record["status"],
            base_weight_sha256=base_after.get("observed_weight_hash"),
            base_weights_unchanged=base_before.get("observed_weight_hash") == base_after.get("observed_weight_hash"),
            optimizer_state_resume_enabled=False,
            fresh_optimizer_state_verified=callback.fresh_optimizer_state_verified,
            optimizer_parameter_state_count=len(optimizer_state_steps),
            optimizer_state_step_min=min(optimizer_state_steps) if optimizer_state_steps else None,
            optimizer_state_step_max=max(optimizer_state_steps) if optimizer_state_steps else None,
            next_stage="frozen_224_point_generation" if status == "trained_pending_eval" else "stop_no_evaluation",
        )
        _write_json(run_root / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            stage="training_or_reload_failed",
            ended_at_utc=_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(run_root / "manifest.json", manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/e03_a3_short.yaml"))
    parser.add_argument("--mode", choices=("preflight", "gpu-preflight", "train"), required=True)
    parser.add_argument("--group", choices=("control", "candidate"))
    parser.add_argument("--run-id")
    parser.add_argument("--gpu-preflight-run-id")
    parser.add_argument("--allow-e03-a3-short-training", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve(args.config)
    config = _config(config_path)
    cpu_facts = _verify_static_assets(config)
    if args.mode == "preflight":
        print(json.dumps(cpu_facts, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "gpu-preflight":
        if not args.run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}", args.run_id):
            raise ValueError("gpu-preflight mode requires a valid unique --run-id")
        root = _resolve(config["data"]["output_root"]).parent / "E03-A3-preflight" / args.run_id
        root.mkdir(parents=True, exist_ok=False)
        try:
            result = _run_gpu_preflight(config, cpu_facts)
            _write_json(root / "gpu_preflight.json", result)
            _write_json(root / "manifest.json", {
                "run_id": args.run_id,
                "experiment_id": "E03-A3",
                "stage": "G2_gpu_dry_step",
                "status": result["status"],
                "created_at_utc": _now(),
                "config_sha256": sha256_file(config_path),
                "implementation_sha256": {
                    "runner": sha256_file(Path(__file__).resolve()),
                    "sampler": sha256_file(ROOT / "src/training/e03_a3_sampling.py"),
                },
                "gpu_preflight_sha256": sha256_file(root / "gpu_preflight.json"),
            })
            print(json.dumps({
                "status": result["status"], "output_dir": str(root),
                "gpu": result["gpu"], "dry_step": result["dry_step"],
            }, ensure_ascii=False, indent=2))
            return 0
        except BaseException as exc:
            _write_json(root / "manifest.json", {
                "run_id": args.run_id,
                "experiment_id": "E03-A3",
                "stage": "G2_gpu_dry_step",
                "status": "failed",
                "created_at_utc": _now(),
                "config_sha256": sha256_file(config_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
    if args.group is None or not args.run_id:
        raise ValueError("train mode requires --group and --run-id")
    if not args.allow_e03_a3_short_training:
        raise ValueError("train mode requires --allow-e03-a3-short-training")
    preflight_identity = _verify_gpu_preflight(config, config_path, args.gpu_preflight_run_id or "")
    result = _training_run(config, config_path, args.group, args.run_id, cpu_facts, preflight_identity)
    print(json.dumps({
        "status": result["status"], "run_id": result["run_id"], "group": result["group"],
        "optimizer_steps": result.get("optimizer_steps"),
        "output_dir": str(_resolve(config["data"]["output_root"]) / result["run_id"]),
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "trained_pending_eval" else 2


if __name__ == "__main__":
    raise SystemExit(main())
