#!/usr/bin/env python3
"""CPU preflight, 2048-token GPU dry backward, and one isolated A4 QLoRA run."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e03_a3_short_train as a3
from src.evaluation import read_jsonl, schema_enums, sha256_file
from src.training.e03_a4_stability import (
    EXPECTED_DRAW_COUNT,
    EXPECTED_LR,
    EXPECTED_STEPS,
    canonical_sha256,
    RUN_ID,
    finite_nonzero_gradient_facts,
    materialize_a4_schedule,
    schedule_json_rows,
    validate_a4_config,
    validate_experiment_initialization_spec,
    validate_fresh_optimizer_state,
    validate_trainable_lora_names,
    verify_gpu_ticket,
)
from src.training.sampling import build_sampling_plan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("A4 config must be a YAML object")
    validate_a4_config(value)
    return value


def _verified_hash(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    observed = sha256_file(path)
    if not expected or observed != expected:
        raise ValueError(f"{description} SHA-256 differs from the frozen A4 config")
    return observed


def _implementation_hashes() -> dict[str, str]:
    return {
        "runner": sha256_file(Path(__file__).resolve()),
        "a4_contracts": sha256_file(ROOT / "src/training/e03_a4_stability.py"),
        "a3_adapter_loader": sha256_file(ROOT / "scripts/e03_a3_short_train.py"),
        "a3_sampling_source": sha256_file(ROOT / "src/training/e03_a3_sampling.py"),
        "sampling": sha256_file(ROOT / "src/training/sampling.py"),
        "experiment_spec": sha256_file(ROOT / "src/training/experiment_spec.py"),
        "sft_collator": sha256_file(ROOT / "src/data/sft_collator.py"),
        "inference": sha256_file(ROOT / "participant/run_inference.py"),
    }


def _validate_baseline(config: dict[str, Any]) -> dict[str, Any]:
    from participant import run_inference as inference

    baseline = config["fast_baseline"]
    report_path = _resolve(baseline["report_path"])
    prediction_path = _resolve(baseline["predictions_path"])
    alignment_path = _resolve(baseline["alignment_path"])
    _verified_hash(report_path, baseline["report_sha256"], "A4-1 Fast-only baseline report")
    _verified_hash(prediction_path, baseline["predictions_sha256"], "A4-1 Fast-only baseline predictions")
    _verified_hash(alignment_path, baseline["alignment_sha256"], "A4-1 baseline reconciliation")
    report = _json(report_path)
    alignment = _json(alignment_path)
    artifact = report.get("profile_predictions", {})
    identity = report.get("run_identity", {})
    generation = report.get("generation", {})
    if (
        report.get("status") != "passed"
        or report.get("sample_count") != baseline["expected_rows"]
        or report.get("valid_count") != baseline["expected_rows"]
        or report.get("attempted_count") != baseline["expected_rows"]
        or artifact.get("rows") != baseline["expected_rows"]
        or artifact.get("expected_rows") != baseline["expected_rows"]
        or artifact.get("sha256") != baseline["predictions_sha256"]
        or _resolve(artifact.get("path", "")) != prediction_path.resolve()
        or artifact.get("ids_sha256") != baseline["expected_ids_sha256"]
        or artifact.get("strict_quality_comparison_ready") is not True
        or generation.get("model_generate_call_count") != baseline["expected_rows"]
        or generation.get("full_generation_performed") is not False
        or generation.get("response_text_generated_or_saved") is not False
        or report.get("response_text_generated_or_saved") is not False
        or report.get("selection", {}).get("selected_count") != baseline["expected_rows"]
        or report.get("selection", {}).get("selected_conversation_count") != baseline["expected_conversations"]
        or report.get("selection", {}).get("selected_ids_sha256") != baseline["expected_ids_sha256"]
        or identity.get("base_weight_sha256") != baseline["expected_base_weight_sha256"]
        or identity.get("adapter_weight_sha256") != baseline["expected_adapter_weight_sha256"]
        or identity.get("adapter_config_sha256") != baseline["expected_adapter_config_sha256"]
    ):
        raise ValueError("A4-1 Fast-only baseline report does not match its frozen identity and 224-point contract")
    pairwise = alignment.get("pairwise_comparisons", {}).get("R_raw_prefix_vs_F_fast_only", {})
    emotion = alignment.get("emotion_label_alignment", {})
    if (
        alignment.get("status") != "reconciled"
        or alignment.get("sample_count") != baseline["expected_rows"]
        or pairwise.get("different_sample_count") != 0
        or emotion.get("raw_prefix_vs_fast_only_different_ids") != []
    ):
        raise ValueError("A4-1 did not reconcile E01-B Fast-only predictions to the historical Raw prefix")
    rows = read_jsonl(prediction_path)
    expected_ids = set()
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or sample_id in expected_ids:
            raise ValueError("A4-1 Fast-only baseline predictions have missing or duplicate IDs")
        expected_ids.add(sample_id)
    from src.evaluation.profile_fast_eval import selected_id_sha256
    if len(expected_ids) != baseline["expected_rows"] or selected_id_sha256([row["id"] for row in rows]) != baseline["expected_ids_sha256"]:
        raise ValueError("A4-1 Fast-only baseline ID set differs from the frozen cohort")
    eval_manifest_path = _resolve(config["evaluation"]["e01_b_full_eval_run_root"]) / "run_manifest.json"
    eval_manifest = _json(eval_manifest_path)
    if identity.get("inference_config_sha256") != eval_manifest.get("inference_config_sha256"):
        raise ValueError("A4-1 Fast-only baseline used a different E01-B inference protocol")
    return {
        "run_id": baseline["run_id"],
        "report_sha256": baseline["report_sha256"],
        "predictions_sha256": baseline["predictions_sha256"],
        "alignment_sha256": baseline["alignment_sha256"],
        "point_count": len(rows),
        "conversation_count": baseline["expected_conversations"],
        "ids_sha256": baseline["expected_ids_sha256"],
        "raw_prefix_profile_and_emotion_aligned": True,
        "response_text_generated_or_saved": False,
    }


def _verify_source_assets(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    from participant import run_inference as inference

    identity, data, protocol = config["identity"], config["data"], config["protocol"]
    e01_root = _resolve(identity["e01_b_training_run_root"])
    e01_manifest_path = e01_root / "manifest.json"
    e01_snapshot_path = e01_root / "config_snapshot.yaml"
    _verified_hash(e01_manifest_path, identity["expected_e01_b_manifest_sha256"], "E01-B Full Manifest")
    e01_manifest = _json(e01_manifest_path)
    if (
        e01_manifest.get("status") != "passed"
        or e01_manifest.get("experiment_id") != "E01-B"
        or e01_manifest.get("training_scope") != "full_train"
        or e01_manifest.get("base_weights_unchanged") is not True
        or e01_manifest.get("train_sha256") != identity["expected_sft_train_sha256"]
    ):
        raise ValueError("A4 source is not the completed, frozen E01-B Full Adapter run")
    _verified_hash(e01_snapshot_path, e01_manifest.get("config_sha256"), "E01-B config snapshot")
    e01_config = yaml.safe_load(e01_snapshot_path.read_text(encoding="utf-8"))
    base_config = yaml.safe_load(_resolve("configs/training/e01_a.yaml").read_text(encoding="utf-8"))
    for key in ("base_model", "quantization", "lora"):
        if config[key] != e01_config[key] or config[key] != base_config[key]:
            raise ValueError(f"A4 {key} differs from the frozen E01-B/E01-A configuration")
    if e01_config.get("adapter_init", {}).get("optimizer_state") != "fresh" and e01_manifest.get("optimizer_state_resume_enabled") is not True:
        raise ValueError("E01-B source Adapter does not have the expected Full training provenance")
    protocol_keys = {
        "prompt_path": "prompt_path", "prompt_sha256": "p0_sha256",
        "runner_config": "runner_config", "tokenizer_id": "tokenizer_id",
        "tokenizer_revision": "tokenizer_revision",
        "tokenizer_signature_sha256": "tokenizer_signature_sha256",
        "chat_template_sha256": "chat_template_sha256", "eos_token_id": "eos_token_id",
        "max_train_seq_length": "max_train_seq_length", "max_prompt_tokens": "max_prompt_tokens",
        "history_policy": "history_policy", "format_version": "format_version",
        "thinking": "thinking", "packing": "packing", "padding_free": "padding_free",
    }
    for a4_key, e01_key in protocol_keys.items():
        expected = e01_config["protocol"].get(e01_key)
        if protocol.get(a4_key) != expected or expected != base_config["protocol"].get(e01_key):
            raise ValueError(f"A4 protocol.{a4_key} differs from the frozen E01-B/E01-A value")

    asset_pairs = (
        (data["train_path"], identity["expected_sft_train_sha256"], "SFT Train"),
        (data["normalized_train_path"], identity["expected_normalized_train_sha256"], "D0.3 normalized Train"),
        (data["d03_manifest"], identity["expected_d03_manifest_sha256"], "D0.3 Manifest"),
        (data["sft_manifest"], identity["expected_sft_manifest_sha256"], "D0.4.3 Manifest"),
        (data["d042_manifest"], identity["expected_d042_manifest_sha256"], "D0.4.2 Manifest"),
        (data["official_data_schema"], identity["expected_official_data_schema_sha256"], "official Train schema"),
        (protocol["output_schema"], identity["expected_output_schema_sha256"], "official output schema"),
        (protocol["prompt_path"], protocol["prompt_sha256"], "P0 prompt"),
    )
    source_hashes: dict[str, str] = {"config": sha256_file(config_path), "e01_b_manifest": sha256_file(e01_manifest_path), "e01_b_config": sha256_file(e01_snapshot_path)}
    for relative, expected, description in asset_pairs:
        source_hashes[description.lower().replace(" ", "_")] = _verified_hash(_resolve(relative), expected, description)
    d03_manifest = _json(_resolve(data["d03_manifest"]))
    sft_manifest = _json(_resolve(data["sft_manifest"]))
    d042_manifest = _json(_resolve(data["d042_manifest"]))
    if (
        sft_manifest.get("d03_manifest_sha256") != identity["expected_d03_manifest_sha256"]
        or sft_manifest.get("d042_manifest_sha256") != identity["expected_d042_manifest_sha256"]
        or sft_manifest.get("outputs", {}).get("files", {}).get("train.jsonl", {}).get("sha256") != identity["expected_sft_train_sha256"]
        or sft_manifest.get("p0_sha256") != protocol["prompt_sha256"]
        or d042_manifest.get("d03_manifest", {}).get("sha256") != identity["expected_d03_manifest_sha256"]
        or d042_manifest.get("p0", {}).get("sha256") != protocol["prompt_sha256"]
        or d042_manifest.get("normalized_inputs", {}).get("train", {}).get("sha256_manifest") != identity["expected_normalized_train_sha256"]
    ):
        raise ValueError("A4 D0.3/D0.4 manifests do not bind the frozen P0/SFT Train sources")
    train_fact = next((fact for fact in d03_manifest.get("outputs", []) if fact.get("path") == data["normalized_train_path"]), None)
    if not train_fact or train_fact.get("sha256") != identity["expected_normalized_train_sha256"] or train_fact.get("records") != data["expected_d03_train_rows"]:
        raise ValueError("D0.3 Manifest does not bind the frozen normalized Train rows")

    adapter_path = _resolve(config["adapter_init"]["source"])
    adapter_identity, adapter_config = inference.adapter_weight_identity(adapter_path)
    if (
        adapter_identity.get("weight_sha256") != config["adapter_init"]["expected_weight_sha256"]
        or adapter_identity.get("config_sha256") != config["adapter_init"]["expected_config_sha256"]
        or adapter_identity.get("weight_sha256") != identity["expected_initial_adapter_weight_sha256"]
        or adapter_identity.get("config_sha256") != identity["expected_initial_adapter_config_sha256"]
    ):
        raise ValueError("read-only E01-B Adapter identity changed")
    if (
        adapter_config.get("r") != config["lora"]["r"]
        or adapter_config.get("lora_alpha") != config["lora"]["alpha"]
        or set(adapter_config.get("target_modules", [])) != set(config["lora"]["target_modules"])
    ):
        raise ValueError("E01-B source Adapter rank, alpha, or targets differ from the A4 lock")
    from safetensors.torch import load_file
    adapter_tensors = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    observed_dtypes = {str(tensor.dtype).removeprefix("torch.") for tensor in adapter_tensors.values()}
    if observed_dtypes != {config["adapter_init"]["parameter_dtype"]}:
        raise ValueError(f"E01-B Adapter tensor dtype differs from the A4 lock: {sorted(observed_dtypes)}")
    base_path = _resolve(config["base_model"]["local_path"])
    inference.validate_adapter_base_reference(adapter_config, config["base_model"]["model_id"], base_path)
    base_identity = inference._weight_identity(base_path, {
        "model_id": config["base_model"]["model_id"],
        "model_revision": config["base_model"]["revision"],
    })
    if base_identity.get("observed_weight_hash") != identity["expected_base_weight_sha256"]:
        raise ValueError("local Base model weights differ from the frozen E01-B Base")
    source_hashes.update({
        "base_weights": base_identity["observed_weight_hash"],
        "initial_adapter_weights": adapter_identity["weight_sha256"],
        "initial_adapter_config": adapter_identity["config_sha256"],
    })

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
        raise ValueError("runtime inference config differs from the frozen P0/F2/T1/Greedy contract")

    spec_quantization = config["adapter_init"]["quantization"]
    observed_adapter = {
        "base_weight_sha256": base_identity["observed_weight_hash"],
        "weight_sha256": adapter_identity["weight_sha256"],
        "config_sha256": adapter_identity["config_sha256"],
        "r": adapter_config.get("r"), "alpha": adapter_config.get("lora_alpha"),
        "target_modules": list(adapter_config.get("target_modules", [])),
        "quantization": dict(spec_quantization),
        "parameter_dtype": config["adapter_init"]["parameter_dtype"],
    }
    initialization_facts = validate_experiment_initialization_spec(config, observed_adapter)
    baseline_facts = _validate_baseline(config)
    frozen_evaluation = a3._validate_frozen_evaluation(config)
    source_hashes.update({
        "frozen_eval_input": frozen_evaluation["outputs"]["frozen_eval_input.jsonl"],
        "frozen_eval_references": frozen_evaluation["outputs"]["frozen_eval_references.jsonl"],
        "frozen_eval_id_map": frozen_evaluation["outputs"]["frozen_eval_id_map.jsonl"],
        "fast_baseline_report": baseline_facts["report_sha256"],
        "fast_baseline_predictions": baseline_facts["predictions_sha256"],
        "fast_baseline_alignment": baseline_facts["alignment_sha256"],
    })
    output_parent = _resolve(config["data"]["output_root"]).parent
    free_disk = shutil.disk_usage(output_parent).free
    min_disk = int(config["training"]["min_free_disk_gb"] * 1024**3)
    if free_disk < min_disk:
        raise RuntimeError(f"insufficient disk for A4: free={free_disk}, required={min_disk}")
    official_schema = _json(_resolve(data["official_data_schema"]))
    output_schema = _json(_resolve(protocol["output_schema"]))
    if a3._official_enums(official_schema) != schema_enums(output_schema):
        raise ValueError("official Train and inference output schema label enums/order differ")
    return {
        "source_hashes": source_hashes,
        "base_identity": base_identity,
        "adapter_identity": adapter_identity,
        "adapter_config": adapter_config,
        "initialization_spec": initialization_facts,
        "baseline": baseline_facts,
        "frozen_evaluation": frozen_evaluation,
        "free_disk_bytes": free_disk,
        "minimum_free_disk_bytes": min_disk,
    }


def _prepare_cpu_facts(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    static = _verify_source_assets(config, config_path)
    records, tokenizer, _prompt, _schema, _enums = a3._load_training_protocol(config)
    normalized = read_jsonl(_resolve(config["data"]["normalized_train_path"]))
    if len(records) != config["data"]["expected_train_rows"] or len(normalized) != config["data"]["expected_d03_train_rows"]:
        raise ValueError("A4 Train/normalized row count changed")
    normalized_by_id: dict[str, dict[str, Any]] = {}
    for row in normalized:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in normalized_by_id:
            raise ValueError("normalized Train has a missing or duplicate sample_id")
        normalized_by_id[sample_id] = row
    sampling_rows: list[dict[str, Any]] = []
    for record in records:
        source = normalized_by_id.get(record["sample_id"])
        if source is None:
            raise ValueError("pre-encoded SFT Train row is missing from normalized Train")
        sampling_rows.append({
            "sample_id": record["sample_id"],
            "conversation_id": record["conversation_id"],
            "source_split": "train",
            "target": source.get("target"),
        })
    draws, sampling_summary = build_sampling_plan(
        sampling_rows,
        draw_count=EXPECTED_DRAW_COUNT,
        seed=int(config["sampler"]["seed"]),
        source_train_sha256=static["source_hashes"]["sft_train"],
        replay_fraction=0.0,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    scheduled_rows, schedule_summary = materialize_a4_schedule(
        records,
        draws,
        eos_token_id=int(config["protocol"]["eos_token_id"]),
        max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
    )
    if (
        sampling_summary["draw_count"] != EXPECTED_DRAW_COUNT
        or sampling_summary["unique_sample_count"] != EXPECTED_DRAW_COUNT
        or sampling_summary["repeated_sample_draw_count"] != 0
        or sampling_summary["dev_rows_read"] != 0
        or sampling_summary["val_rows_read"] != 0
    ):
        raise ValueError("A4 sampling plan is not exactly 256 unique Train-only draws")
    if tokenizer.eos_token_id != int(config["protocol"]["eos_token_id"]):
        raise ValueError("A4 runtime tokenizer EOS differs from the frozen protocol")
    implementation_sha256 = _implementation_hashes()
    return {
        **static,
        "status": "passed",
        "created_at_utc": _now(),
        "runner_version": "e03_a4_stability_runner_v1",
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "config_sha256": sha256_file(config_path),
        "training_rows": len(records),
        "normalized_train_rows": len(normalized),
        "sampled_draws": schedule_summary,
        "sampling_summary": {
            key: sampling_summary[key]
            for key in (
                "source_split", "sampling_seed", "draw_count", "micro_batch_size",
                "gradient_accumulation_steps", "optimizer_step_count", "unique_sample_count",
                "repeated_sample_draw_count", "conversation_exposures",
                "sampling_schedule_sha256", "dev_rows_read", "val_rows_read",
            )
        },
        "source_hashes": static["source_hashes"],
        "implementation_sha256": implementation_sha256,
        "runtime_tokenizer": {
            "signature_sha256": config["protocol"]["tokenizer_signature_sha256"],
            "chat_template_sha256": config["protocol"]["chat_template_sha256"],
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "evaluation_protocol": {
            "points": config["evaluation"]["frozen_eval_points"],
            "conversations": config["evaluation"]["frozen_eval_conversations"],
            "ids_sha256": config["identity"]["expected_frozen_eval_ids_sha256"],
            "baseline_reconciled": True,
            "fast_only_max_seconds": config["evaluation"]["fast_budget_seconds"],
        },
        "optimizer_state": "fresh_adamw",
        "optimizer_state_resume_enabled": False,
        "gpu_dry_step": "not_run_by_cpu_preflight",
        "_schedule_rows": scheduled_rows,
    }


def _cpu_output_root(config: dict[str, Any]) -> Path:
    return _resolve("outputs/diagnostics/e03_a4/stability_preflight") / config["run_id"]


def preflight(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root = _cpu_output_root(config)
    if output_root.exists():
        raise FileExistsError(f"A4 CPU preflight is immutable and already exists: {output_root}")
    output_root.mkdir(parents=True)
    try:
        facts = _prepare_cpu_facts(config, config_path)
        rows = facts.pop("_schedule_rows")
        schedule_path = output_root / "sampling_schedule.jsonl"
        with schedule_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in schedule_json_rows(rows):
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        facts["sampling_schedule_artifact"] = {
            "path": str(schedule_path),
            "sha256": sha256_file(schedule_path),
            "rows": len(rows),
            "contains_token_ids_or_source_text": False,
        }
        snapshot = output_root / "config_snapshot.yaml"
        with snapshot.open("xb") as handle:
            handle.write(config_path.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        facts["config_snapshot_sha256"] = sha256_file(snapshot)
        _write_once(output_root / "cpu_preflight.json", facts)
        return {"status": facts["status"], "output_dir": str(output_root), "schedule_sha256": facts["sampled_draws"]["schedule_sha256"]}
    except BaseException as exc:
        _write_once(output_root / "cpu_preflight.json", {
            "status": "failed", "created_at_utc": _now(),
            "run_id": config["run_id"], "config_sha256": sha256_file(config_path),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def _verify_cpu_preflight(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cpu_root = _cpu_output_root(config)
    report_path = cpu_root / "cpu_preflight.json"
    report = _json(report_path)
    if report.get("status") != "passed" or report.get("run_id") != config["run_id"]:
        raise ValueError("A4 CPU preflight did not pass for the configured Run")
    if report.get("config_sha256") != sha256_file(config_path):
        raise ValueError("A4 config changed after CPU preflight; the receipt cannot be reused")
    snapshot_path = cpu_root / "config_snapshot.yaml"
    if report.get("config_snapshot_sha256") != sha256_file(snapshot_path):
        raise ValueError("A4 CPU preflight config snapshot changed")
    facts = _prepare_cpu_facts(config, config_path)
    current_rows = facts.pop("_schedule_rows")
    if (
        report.get("sampled_draws", {}).get("schedule_sha256") != facts["sampled_draws"]["schedule_sha256"]
        or report.get("sampled_draws", {}).get("encoded_rows_sha256") != facts["sampled_draws"]["encoded_rows_sha256"]
        or report.get("source_hashes") != facts["source_hashes"]
        or report.get("implementation_sha256") != facts["implementation_sha256"]
    ):
        raise ValueError("A4 sources, tokenized rows, schedule, or implementation changed after CPU preflight")
    schedule_path = cpu_root / "sampling_schedule.jsonl"
    artifact = report.get("sampling_schedule_artifact", {})
    if artifact.get("sha256") != sha256_file(schedule_path) or artifact.get("rows") != EXPECTED_DRAW_COUNT:
        raise ValueError("A4 schedule artifact changed after CPU preflight")
    return report, facts | {"_schedule_rows": current_rows}, schedule_path


def _make_max_length_batch(row: dict[str, Any], *, eos_token_id: int, length: int = 2048) -> tuple[list[int], list[int]]:
    original = row["input_ids"]
    ids = list(original)
    while len(ids) < length:
        ids.extend(original)
    ids = ids[:length]
    ids[-1] = eos_token_id
    prompt_length = min(1280, length - 1)
    labels = [-100] * prompt_length + ids[prompt_length:]
    return ids, labels


def _dry_backward(model: Any, torch: Any, row: dict[str, Any], eos_token_id: int) -> dict[str, Any]:
    device = model.get_input_embeddings().weight.device
    if device.type != "cuda":
        raise RuntimeError(f"A4 NF4 model is not fully CUDA-resident: {device}")
    ids, labels = _make_max_length_batch(row, eos_token_id=eos_token_id)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    label_ids = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    validate_trainable_lora_names(
        [name for name, _ in trainable],
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
        loss = output.loss
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError("A4 2048-token dry-step loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for _, parameter in trainable]
    gradient_facts = finite_nonzero_gradient_facts(gradients)
    unchanged = all(torch.equal(before[name], parameter.detach().cpu()) for name, parameter in trainable)
    if not unchanged:
        raise RuntimeError("A4 dry backward changed LoRA weights without an optimizer update")
    model.zero_grad(set_to_none=True)
    return {
        "status": "passed",
        "sequence_length": len(ids),
        "prompt_length": min(1280, len(ids) - 1),
        "loss": float(loss.detach().float().item()),
        "gradient": gradient_facts,
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "all_trainable_parameters_are_lora": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "parameters_unchanged": unchanged,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def gpu_preflight(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    cpu_report, facts, _schedule_path = _verify_cpu_preflight(config, config_path)
    cpu_root = _cpu_output_root(config)
    ticket_path = cpu_root / "gpu_preflight.json"
    if ticket_path.exists():
        raise FileExistsError(f"A4 GPU preflight ticket already exists: {ticket_path}")
    result: dict[str, Any] = {
        "status": "failed",
        "created_at_utc": _now(),
        "run_id": config["run_id"],
        "config_sha256": sha256_file(config_path),
        "cpu_preflight_sha256": sha256_file(cpu_root / "cpu_preflight.json"),
        "schedule_sha256": facts["sampled_draws"]["schedule_sha256"],
        "encoded_rows_sha256": facts["sampled_draws"]["encoded_rows_sha256"],
        "source_hashes": facts["source_hashes"],
        "implementation_sha256": facts["implementation_sha256"],
        "gpu": None,
        "dry_step": None,
    }
    model = None
    try:
        import torch

        if (_resolve(config["data"]["output_root"]) / config["run_id"]).exists():
            raise FileExistsError("A4 training output Run root already exists; refusing to reuse or overwrite")
        gate = a3._gpu_gate(config, torch)
        output_parent = _resolve(config["data"]["output_root"]).parent
        free_disk = shutil.disk_usage(output_parent).free
        minimum_disk = int(config["training"]["min_free_disk_gb"] * 1024**3)
        if free_disk < minimum_disk:
            raise RuntimeError(f"blocked_free_disk_below_gate: {free_disk} < {minimum_disk}")
        gate.update(free_disk_bytes=free_disk, minimum_free_disk_bytes=minimum_disk)
        model, model_facts, adapter_identity = a3._load_trainable_adapter(config, torch)
        if not model_facts.get("loaded_adapter_values_exact_match") or model_facts.get("adapter_dtype_autocast") is not False:
            raise RuntimeError("E01-B Adapter weights/dtype changed while loading A4 QLoRA")
        result["gpu"] = gate | {"model_facts": model_facts, "initial_adapter_identity": adapter_identity}
        result["dry_step"] = _dry_backward(
            model, torch, facts["_schedule_rows"][0], int(config["protocol"]["eos_token_id"]),
        )
        result["status"] = "passed"
        result["ended_at_utc"] = _now()
        result["ticket_sha256"] = canonical_sha256({
            key: result[key] for key in (
                "status", "run_id", "config_sha256", "cpu_preflight_sha256", "schedule_sha256",
                "encoded_rows_sha256", "source_hashes", "implementation_sha256", "gpu", "dry_step",
            )
        })
        _write_once(ticket_path, result)
        return {"status": result["status"], "ticket": str(ticket_path), "dry_step": result["dry_step"]}
    except BaseException as exc:
        result["ended_at_utc"] = _now()
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write_once(ticket_path, result)
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _verify_train_ticket(config: dict[str, Any], config_path: Path, facts: dict[str, Any]) -> dict[str, Any]:
    cpu_root = _cpu_output_root(config)
    ticket = _json(cpu_root / "gpu_preflight.json")
    verify_gpu_ticket(
        ticket,
        run_id=config["run_id"],
        config_sha256=sha256_file(config_path),
        cpu_preflight_sha256=sha256_file(cpu_root / "cpu_preflight.json"),
        schedule_sha256=facts["sampled_draws"]["schedule_sha256"],
        encoded_rows_sha256=facts["sampled_draws"]["encoded_rows_sha256"],
        implementation_sha256=facts["implementation_sha256"],
    )
    if ticket.get("source_hashes") != facts["source_hashes"]:
        raise ValueError("A4 source hashes changed after the successful GPU preflight")
    if ticket.get("dry_step", {}).get("status") != "passed" or ticket.get("dry_step", {}).get("optimizer_updates") != 0:
        raise ValueError("A4 training requires a passed no-update 2048-token GPU dry backward")
    return ticket


def _save_adapter_and_checkpoint(
    model: Any, optimizer: Any, scheduler: Any,
    run_root: Path, torch: Any, step: int, step_seconds: float,
    tokenizer_signature_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    from participant import run_inference as inference

    if step != EXPECTED_STEPS:
        raise RuntimeError("A4 only saves a scoreable Adapter at the exact final step 64")
    adapter_dir = run_root / "adapter"
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    identity, adapter_config = inference.adapter_weight_identity(adapter_dir)
    if (
        adapter_config.get("r") != 8
        or adapter_config.get("lora_alpha") != 16
        or set(adapter_config.get("target_modules", []))
        != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ):
        raise RuntimeError("saved A4 Adapter changed the frozen LoRA rank, alpha, or target modules")
    checkpoint = run_root / "checkpoint-64"
    checkpoint.mkdir()
    shutil.copytree(adapter_dir, checkpoint / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, checkpoint / "optimizer_scheduler.pt")
    _write_once(checkpoint / "checkpoint.json", {
        "status": "saved",
        "optimizer_step": step,
        "optimizer_state_resume_enabled": False,
        "optimizer_state_sha256": sha256_file(checkpoint / "optimizer_scheduler.pt"),
        "adapter_identity": identity,
        "training_seconds": step_seconds,
        "tokenizer_signature_sha256": tokenizer_signature_sha256,
    })
    return adapter_dir, identity


def train(config_path: Path, *, allow_training: bool) -> dict[str, Any]:
    config = _load_config(config_path)
    if not allow_training:
        raise ValueError("train mode requires --allow-e03-a4-lr-stability-training")
    if config["authorization"].get("authorized") is not True:
        raise ValueError("A4 training requires the explicit authorization recorded in its config")
    cpu_report, facts, _schedule_path = _verify_cpu_preflight(config, config_path)
    ticket = _verify_train_ticket(config, config_path, facts)
    run_root = _resolve(config["data"]["output_root"]) / config["run_id"]
    if run_root.exists():
        raise FileExistsError(f"A4 Run already exists; refusing any overwrite or retry: {run_root}")
    import torch
    from src.data.sft_collator import SFTDataCollator
    from src.training import training_sft
    from participant import run_inference as inference

    run_root.mkdir(parents=True)
    initial_adapter_identity = dict(facts["adapter_identity"])
    run_identity = {
        "created_at_utc": _now(),
        "status": "training_started",
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "run_version": "E03-A4-low-lr-64-step-v1",
        "config_sha256": sha256_file(config_path),
        "authorization_receipt": config["authorization"],
        "optimizer": config["optimizer"],
        "sampling": facts["sampling_summary"],
        "sampled_draws": facts["sampled_draws"],
        "source_hashes": facts["source_hashes"],
        "implementation_sha256": facts["implementation_sha256"],
        "initial_adapter_identity": initial_adapter_identity,
        "base_weight_sha256": facts["base_identity"].get("observed_weight_hash"),
        "frozen_evaluation": facts["evaluation_protocol"],
        "cpu_preflight_sha256": sha256_file(_cpu_output_root(config) / "cpu_preflight.json"),
        "gpu_preflight_sha256": sha256_file(_cpu_output_root(config) / "gpu_preflight.json"),
        "gpu_preflight_ticket_sha256": ticket["ticket_sha256"],
    }
    _write_once(run_root / "run_identity.json", run_identity)
    _append_jsonl(run_root / "run_events.jsonl", {"time": _now(), "event": "run_identity_frozen"})
    model = optimizer = scheduler = None
    status = "failed"
    steps_completed = 0
    adapter_identity: dict[str, Any] | None = None
    reload_facts: dict[str, Any] | None = None
    runtime_start: dict[str, Any] | None = None
    base_during: dict[str, Any] | None = None
    optimizer_state_proof: dict[str, Any] | None = None
    step_start = 0.0
    try:
        gate = a3._gpu_gate(config, torch)
        a3._seed_everything(int(config["training"]["seed"]), torch)
        model, model_facts, loaded_identity = a3._load_trainable_adapter(config, torch)
        if loaded_identity.get("weight_sha256") != initial_adapter_identity.get("weight_sha256"):
            raise RuntimeError("A4 loaded a different source Adapter than the CPU preflight")
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        observed_targets = validate_trainable_lora_names(
            [name for name, _ in trainable], config["lora"]["target_modules"],
        )
        records, tokenizer, _prompt, _schema, _enums = a3._load_training_protocol(config)
        rows = facts["_schedule_rows"]
        collator = SFTDataCollator(
            int(tokenizer.pad_token_id),
            max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
        )
        device = model.get_input_embeddings().weight.device
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable],
            lr=EXPECTED_LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        validate_fresh_optimizer_state(optimizer.state)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
        if float(optimizer.param_groups[0]["lr"]) != EXPECTED_LR:
            raise RuntimeError("A4 optimizer LR differs from the fixed 2e-5 candidate")
        run_identity["optimizer_implementation"] = {
            "class": type(optimizer).__name__,
            "lr": EXPECTED_LR,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "scheduler": "constant_lambda_1.0",
            "warmup_steps": 0,
            "fresh_state_count_at_start": len(optimizer.state),
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "trainable_parameter_names_sha256": hashlib.sha256(
                json.dumps([name for name, _ in trainable], separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "all_trainable_parameters_are_lora_ab": True,
            "target_modules": observed_targets,
        }
        run_identity["gpu"] = gate
        run_identity["model_facts"] = model_facts
        # The frozen identity file is written before GPU model construction; record runtime facts separately.
        runtime_start = {
            "created_at_utc": _now(),
            "gpu": gate,
            "model_facts": model_facts,
            "optimizer": run_identity["optimizer_implementation"],
        }
        _write_once(run_root / "runtime_start.json", runtime_start)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        step_start = time.monotonic()
        step_rows: list[dict[str, Any]] = []
        accum = int(config["training"]["gradient_accumulation_steps"])
        max_seconds = float(config["training"]["max_training_seconds"])
        for step in range(1, EXPECTED_STEPS + 1):
            if time.monotonic() - step_start >= max_seconds:
                raise TimeoutError("A4 training exceeded the 3600-second wall-time budget")
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            ids_for_step: list[str] = []
            convs_for_step: list[str] = []
            for micro_index in range(accum):
                row = rows[(step - 1) * accum + micro_index]
                ids_for_step.append(row["sample_id"])
                convs_for_step.append(row["conversation_id"])
                batch = collator([row])
                model_inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(**model_inputs)
                    loss = output.loss
                if not bool(torch.isfinite(loss.detach()).all().item()):
                    raise FloatingPointError(f"A4 non-finite SFT loss at optimizer step {step}")
                micro_losses.append(float(loss.detach().float().item()))
                (loss / accum).backward()
            gradients = [parameter.grad for _, parameter in trainable]
            gradient_facts = finite_nonzero_gradient_facts(gradients)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable],
                float(config["optimizer"]["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(gradient_norm).all().item()) or float(gradient_norm.detach().float().item()) <= 0:
                raise FloatingPointError(f"A4 invalid LoRA gradient norm at optimizer step {step}")
            check_update = step in {1, 32, EXPECTED_STEPS}
            before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in trainable
            } if check_update else {}
            optimizer.step()
            scheduler.step()
            if float(optimizer.param_groups[0]["lr"]) != EXPECTED_LR:
                raise RuntimeError("A4 constant learning-rate scheduler changed the approved LR")
            if check_update and not any(not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in trainable):
                raise RuntimeError(f"A4 AdamW did not update any LoRA tensor at step {step}")
            steps_completed = step
            elapsed = time.monotonic() - step_start
            record = {
                "optimizer_step": step,
                "microbatch_count": accum,
                "sample_ids": ids_for_step,
                "conversation_ids": convs_for_step,
                "microbatch_losses": micro_losses,
                "mean_microbatch_loss": sum(micro_losses) / len(micro_losses),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "gradient": gradient_facts,
                "gradient_norm_preclip": float(gradient_norm.detach().float().item()),
                "parameter_update_checked": check_update,
                "parameters_changed": check_update,
                "elapsed_training_seconds": elapsed,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
            _append_jsonl(run_root / "optimizer_steps.jsonl", record)
            step_rows.append(record)
            if elapsed >= max_seconds:
                raise TimeoutError("A4 training exceeded the 3600-second wall-time budget")

        training_seconds = time.monotonic() - step_start
        if steps_completed != EXPECTED_STEPS or training_seconds > max_seconds:
            raise RuntimeError("A4 exact step count or 3600-second training budget was not met")
        if len(read_jsonl(run_root / "optimizer_steps.jsonl")) != EXPECTED_STEPS:
            raise RuntimeError("A4 optimizer step ledger is incomplete")
        optimizer_state_steps = []
        for state in optimizer.state.values():
            state_step = state.get("step")
            if state_step is None:
                continue
            optimizer_state_steps.append(int(state_step.item()) if hasattr(state_step, "item") else int(state_step))
        optimizer_state_proof = {
            "parameter_state_count": len(optimizer.state),
            "parameter_count": len(trainable),
            "state_step_min": min(optimizer_state_steps) if optimizer_state_steps else None,
            "state_step_max": max(optimizer_state_steps) if optimizer_state_steps else None,
            "fresh_state_at_start": True,
        }
        if (
            optimizer_state_proof["parameter_state_count"] != len(trainable)
            or optimizer_state_proof["state_step_min"] != EXPECTED_STEPS
            or optimizer_state_proof["state_step_max"] != EXPECTED_STEPS
        ):
            raise RuntimeError("A4 AdamW moments do not prove exactly 64 fresh optimizer updates")
        initial_adapter_after = inference.adapter_weight_identity(_resolve(config["adapter_init"]["source"]))[0]
        if initial_adapter_after.get("weight_sha256") != initial_adapter_identity.get("weight_sha256") or initial_adapter_after.get("config_sha256") != initial_adapter_identity.get("config_sha256"):
            raise RuntimeError("A4 modified the read-only E01-B Adapter source")
        base_during = inference._weight_identity(_resolve(config["base_model"]["local_path"]), {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"]["revision"],
        })
        if base_during.get("observed_weight_hash") != facts["base_identity"].get("observed_weight_hash"):
            raise RuntimeError("A4 modified frozen Base weights")

        adapter_dir, adapter_identity = _save_adapter_and_checkpoint(
            model, optimizer, scheduler, run_root, torch, steps_completed,
            training_seconds, config["protocol"]["tokenizer_signature_sha256"],
        )
        del model, optimizer, scheduler
        model = optimizer = scheduler = None
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, reload_torch, reload_model_facts = training_sft._load_train_model(
            config,
            _resolve(config["base_model"]["local_path"]),
            adapter_path=adapter_dir,
            autocast_adapter_dtype=False,
        )
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        saved_tensors = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
        reloaded_tensors = get_peft_model_state_dict(reloaded, adapter_name="default")
        exact_weights = set(saved_tensors) == set(reloaded_tensors) and all(
            torch.equal(saved_tensors[name], reloaded_tensors[name].detach().cpu().to(dtype=saved_tensors[name].dtype))
            for name in saved_tensors
        )
        if not exact_weights:
            raise RuntimeError("A4 standalone Adapter reload changed saved LoRA tensor values")
        base_after = inference._weight_identity(_resolve(config["base_model"]["local_path"]), {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"]["revision"],
        })
        base_unchanged = (
            facts["base_identity"].get("observed_weight_hash")
            == base_during.get("observed_weight_hash")
            == base_after.get("observed_weight_hash")
            == config["base_model"]["expected_weight_sha256"]
        )
        if not base_unchanged:
            raise RuntimeError("A4 Base identity changed before/after standalone Adapter reload")
        reload_facts = {
            "status": "passed" if exact_weights and base_unchanged else "failed",
            "adapter_weight_sha256": adapter_identity["weight_sha256"],
            "adapter_config_sha256": adapter_identity["config_sha256"],
            "saved_reloaded_tensor_values_exact_match": exact_weights,
            "autocast_adapter_dtype": False,
            "reload_model_facts": reload_model_facts,
            "base_weight_sha256_before": facts["base_identity"]["observed_weight_hash"],
            "base_weight_sha256_after": base_after["observed_weight_hash"],
            "base_weights_unchanged": base_unchanged,
        }
        if not exact_weights:
            raise RuntimeError("A4 standalone Adapter reload check failed")
        del reloaded
        gc.collect()
        reload_torch.cuda.empty_cache()
        status = "trained_pending_eval"
        manifest = {
            "status": status,
            "created_at_utc": run_identity["created_at_utc"],
            "ended_at_utc": _now(),
            "experiment_id": config["experiment_id"],
            "run_id": config["run_id"],
            "training_scope": config["training_scope"],
            "optimizer_steps": steps_completed,
            "draw_count": EXPECTED_DRAW_COUNT,
            "training_seconds": training_seconds,
            "max_training_seconds": max_seconds,
            "config_sha256": run_identity["config_sha256"],
            "run_identity_sha256": sha256_file(run_root / "run_identity.json"),
            "runtime_start_sha256": sha256_file(run_root / "runtime_start.json"),
            "optimizer_state_resume_enabled": False,
            "fresh_optimizer_state_verified": True,
            "optimizer_state_proof": optimizer_state_proof,
            "optimizer_step_ledger_sha256": sha256_file(run_root / "optimizer_steps.jsonl"),
            "optimizer": runtime_start["optimizer"] if runtime_start else None,
            "gpu": runtime_start["gpu"] if runtime_start else None,
            "protocol": {
                "prompt_sha256": config["protocol"]["prompt_sha256"],
                "format_version": config["protocol"]["format_version"],
                "history_policy": config["protocol"]["history_policy"],
                "max_train_seq_length": config["protocol"]["max_train_seq_length"],
                "max_prompt_tokens": config["protocol"]["max_prompt_tokens"],
                "max_new_tokens": config["protocol"]["max_new_tokens"],
                "tokenizer_signature_sha256": config["protocol"]["tokenizer_signature_sha256"],
                "chat_template_sha256": config["protocol"]["chat_template_sha256"],
                "eos_token_id": config["protocol"]["eos_token_id"],
            },
            "train_sha256": facts["source_hashes"]["sft_train"],
            "normalized_train_sha256": facts["source_hashes"]["d0.3_normalized_train"],
            "source_hashes": facts["source_hashes"],
            "implementation_sha256": facts["implementation_sha256"],
            "sampling": facts["sampled_draws"],
            "gpu_preflight_ticket_sha256": ticket["ticket_sha256"],
            "base_weights_unchanged": base_unchanged,
            "base_weight_sha256_before": facts["base_identity"]["observed_weight_hash"],
            "base_weight_sha256_during": base_during["observed_weight_hash"],
            "base_weight_sha256_after": base_after["observed_weight_hash"],
            "initial_source_adapter_unchanged": True,
            "initial_source_adapter_identity": initial_adapter_identity,
            "adapter_identity": adapter_identity,
            "reload_check": reload_facts,
            "model_facts": model_facts,
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in step_rows),
            "next_stage": "frozen_224_point_fast_only_evaluation",
        }
        _write_once(run_root / "run_manifest.json", manifest)
        _append_jsonl(run_root / "run_events.jsonl", {"time": _now(), "event": status, "optimizer_steps": steps_completed})
        return {"status": status, "run_id": config["run_id"], "output_dir": str(run_root), "optimizer_steps": steps_completed, "adapter_identity": adapter_identity}
    except BaseException as exc:
        _append_jsonl(run_root / "run_events.jsonl", {
            "time": _now(),
            "event": "training_failed_or_interrupted",
            "steps_completed": steps_completed,
            "error": f"{type(exc).__name__}: {exc}",
        })
        failure = {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "budget_exhausted" if isinstance(exc, TimeoutError) else "failed",
            "created_at_utc": run_identity["created_at_utc"],
            "ended_at_utc": _now(),
            "experiment_id": config["experiment_id"],
            "run_id": config["run_id"],
            "optimizer_steps": steps_completed,
            "max_optimizer_steps": EXPECTED_STEPS,
            "training_seconds": time.monotonic() - step_start if step_start else None,
            "source_hashes": facts["source_hashes"],
            "gpu_preflight_ticket_sha256": ticket["ticket_sha256"],
            "error": f"{type(exc).__name__}: {exc}",
            "candidate_scoring_allowed": False,
        }
        _write_once(run_root / "run_manifest.json", failure)
        raise
    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/e03_a4_lr_stability.yaml")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("preflight", help="validate frozen assets and write the CPU sampling receipt")
    subparsers.add_parser("gpu-preflight", help="run a 2048-token backward pass without an optimizer update")
    train_parser = subparsers.add_parser("train", help="run the single authorized 64-step low-LR continuation")
    train_parser.add_argument("--allow-e03-a4-lr-stability-training", action="store_true")
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    if args.mode == "preflight":
        result = preflight(config_path)
    elif args.mode == "gpu-preflight":
        result = gpu_preflight(config_path)
    else:
        result = train(config_path, allow_training=args.allow_e03_a4_lr_stability_training)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("status") in {"passed", "trained_pending_eval"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
