"""Isolated staged training runner for the E03-A8-V2 experiment."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from participant import run_inference as inference
from src.data.e03_a8_schedule import SCHEDULE_ALGORITHM
from src.data.sft_collator import SFTDataCollator
from src.shared.e03_a8_protocol import PROFILE_DIMS
from src.training.e03_a8_completion_loss import completion_only_forward_loss
from src.training.e03_a8_contract import canonical_sha256
from src.training.e03_a8_staged import (
    JsonlById,
    _base_identity,
    _base_path,
    _checkpoint_files,
    _lora_parameters,
    _lora_tensor_sha256,
    _next_token_id,
    _package_versions,
    _read_json,
    _read_jsonl,
    _restore_rng_state,
    _rng_state,
    _runtime_tokenizer,
    _set_seed,
    _write_json,
    _row_contract,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}")
PROFILE_EMPTY_TARGET = {dimension: [] for dimension in PROFILE_DIMS}
EQUIVALENCE_TOLERANCES = {
    "loss_absolute_max": 2e-5,
    "per_parameter_relative_l2_max": 2.1e-2,
    "global_relative_l2_max": 1.2e-2,
    "cosine_min": 0.9999,
}

V2_DATA_CODE_PATHS = (
    "configs/data/e03_a8_v2.yaml",
    "configs/prompts/p_a8_v2_profile.txt",
    "configs/prompts/p_a8_v2_emotion_profile.txt",
    "scripts/e03_a8_v2_prepare.py",
    "scripts/e03_a8_prepare.py",
    "src/data/e03_a8_schedule.py",
    "src/data/e03_a8_sft.py",
    "src/shared/e03_a8_protocol.py",
    "src/shared/prompting.py",
)
V2_TRAINING_CODE_PATHS = (
    "configs/training/e03_a8_v2_p_staged.yaml",
    "configs/training/e03_a8_v2_ep_staged.yaml",
    *V2_DATA_CODE_PATHS,
    "src/training/e03_a8_completion_loss.py",
    "src/training/e03_a8_v2_staged.py",
    "scripts/e03_a8_v2_staged_train.py",
    "scripts/e03_a8_v2_perf_diag.py",
    "src/training/e03_a8_contract.py",
    "src/training/training_sft.py",
    "src/training/e03_a8_staged.py",
    "src/data/sft_collator.py",
    "participant/run_inference.py",
)
V2_EVALUATION_CODE_PATHS = (
    "configs/evaluation/e03_a8_v2_staged_fast.yaml",
    "scripts/e03_a8_v2_staged_eval.py",
    "scripts/e03_a8_staged_eval.py",
    "src/evaluation/e03_a8_v2_staged_metrics.py",
    "src/evaluation/e03_a8_staged_metrics.py",
    "src/evaluation/e03_a6_f4_staged_metrics.py",
    "src/evaluation/metrics.py",
    "src/shared/e03_a8_protocol.py",
    "src/shared/prompting.py",
    "src/data/e03_a8_sft.py",
    "src/data/sft_collator.py",
    "src/training/e03_a8_v2_staged.py",
    "src/training/e03_a8_staged.py",
    "src/training/training_sft.py",
    "participant/run_inference.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"A8-V2 path escapes the project root: {value}")
    return resolved


def _code_identity(paths: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(_resolve(name)) for name in paths}


def _config_sha256(config: dict[str, Any]) -> str:
    return canonical_sha256(config)


def validate_a8_v2_config(config: dict[str, Any], arm: str | None = None) -> None:
    selected_arm = arm or config.get("arm")
    if selected_arm not in {"P", "EP"} or config.get("arm") != selected_arm:
        raise ValueError("A8-V2 arm must be P or EP and match its config")
    if (config.get("version"), config.get("experiment_id"), config.get("status")) != (
        "E03-A8-V2-staged-training-v1", "E03-A8-V2", "ready_for_preflight"
    ):
        raise ValueError("A8-V2 training config identity or status differs")
    if config.get("training_scope") != "generative_profile_specialist":
        raise ValueError("A8-V2 training scope differs")
    base = config.get("base_model", {})
    if base != {
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "local_path": "participant/model",
        "expected_weight_sha256": "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5",
    }:
        raise ValueError("A8-V2 Base identity differs from the frozen Qwen3-1.7B Base")
    data = config.get("data", {})
    expected_file = "p_train.jsonl" if selected_arm == "P" else "ep_train.jsonl"
    expected_root = f"outputs/training/E03-A8-V2/{selected_arm}"
    if (
        data.get("split") != "train"
        or data.get("dataset_root") != "outputs/data/e03_a8_v2"
        or data.get("dataset_manifest") != "outputs/data/e03_a8_v2/manifest.json"
        or data.get("train_path") != f"outputs/data/e03_a8_v2/{expected_file}"
        or data.get("sample_order_path") != "outputs/data/e03_a8_v2/sample_order.jsonl"
        or data.get("expected_train_rows") != 16192
        or data.get("output_root") != expected_root
        or data.get("init_adapter") is not None
        or data.get("resume_from_checkpoint") is not None
    ):
        raise ValueError("A8-V2 must use the isolated paired Train data and output namespace")
    protocol = config.get("protocol", {})
    expected_prompt = "configs/prompts/p_a8_v2_profile.txt" if selected_arm == "P" else "configs/prompts/p_a8_v2_emotion_profile.txt"
    if (
        protocol.get("arm") != selected_arm
        or protocol.get("completion_version") != "A8-GENCLASS-v2"
        or protocol.get("loss_path") != "completion_only_selective_logits_v1"
        or protocol.get("prompt_path") != expected_prompt
        or protocol.get("prompt_sha256") != sha256_file(_resolve(expected_prompt))
        or protocol.get("selection_prompt_path") != "configs/prompts/p0.txt"
        or protocol.get("p0_sha256") != "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2"
        or protocol.get("history_policy") != "P0_T1_then_common_A8_P_EP_history"
        or protocol.get("max_prompt_tokens") != 1280
        or protocol.get("max_train_seq_length") != 2048
        or protocol.get("packing") is not False
        or protocol.get("padding_free") is not False
    ):
        raise ValueError("A8-V2 Prompt, P0/T1, history budget, or Completion protocol differs")
    if config.get("quantization") != {
        "method": "bitsandbytes_4bit",
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quant": True,
        "device_map": {"": 0},
        "max_cuda_memory_fraction": 0.92,
    }:
        raise ValueError("A8-V2 quantization must retain NF4/double quant/BF16 on CUDA")
    lora = config.get("lora", {})
    if (
        lora.get("r") != 8 or lora.get("alpha") != 16 or lora.get("dropout") != 0.0
        or lora.get("bias") != "none" or lora.get("checkpoint_dtype") != "float32"
        or set(lora.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ):
        raise ValueError("A8-V2 LoRA profile differs from the frozen QLoRA configuration")
    if config.get("optimizer") != {
        "name": "AdamW", "learning_rate": 0.0001, "lr_scheduler": "constant",
        "warmup_steps": 0, "weight_decay": 0.0,
    }:
        raise ValueError("A8-V2 optimizer differs from the frozen AdamW parameters")
    training = config.get("training", {})
    exact_training = {
        "seed": 42, "per_device_micro_batch_size": 1, "gradient_accumulation_steps": 4,
        "max_optimizer_steps": 4048, "save_interval_steps": 512, "gradient_checkpointing": True,
        "diagnostic_interval_steps": 32, "ledger_fsync_interval_steps": 32,
    }
    if any(training.get(key) != value for key, value in exact_training.items()):
        raise ValueError("A8-V2 training schedule, accumulation, or diagnostics changed")
    if config.get("authorization") != {"authorized_max_step": 512, "source": "E0.3-A8 explicit plan request"}:
        raise ValueError("A8-V2 authorization must stop at the requested Step 512 node")
    if config.get("evaluation") != {"max_new_tokens": 256, "do_sample": False, "autocast_adapter_dtype": True}:
        raise ValueError("A8-V2 generation/reload settings differ from the fixed decode protocol")
    perf = config.get("performance_diagnostics", {})
    if perf != {
        "benchmark_root": "outputs/evaluation/e03_a8_v2/diagnostics/performance/E03-A8-V2-20261107-01",
        "reference_steps_per_path": 16,
        "minimum_speedup": 2.0,
    }:
        raise ValueError("A8-V2 training must have the frozen 16-step, 2x performance gate")


def _load_config(config_path: Path, arm: str | None = None) -> tuple[dict[str, Any], Path]:
    path = config_path if config_path.is_absolute() else ROOT / config_path
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_a8_v2_config(config, arm)
    return config, path


def _read_source_profiles(manifest: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    source = manifest.get("inputs", {}).get("normalized_train", {})
    path = _resolve(source.get("path", ""))
    if sha256_file(path) != source.get("sha256"):
        raise ValueError("A8-V2 normalized Train source changed after preparation")
    result: dict[str, dict[str, list[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank normalized Train row at {line_no}")
            row = json.loads(line)
            target = row.get("target", {})
            profile = target.get("user_profile", {})
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str):
                result[sample_id] = {dim: list(profile.get(dim, [])) for dim in PROFILE_DIMS}
    return result


def _validate_prepared_data(config: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]:
    data_root = _resolve(config["data"]["dataset_root"])
    manifest_path = _resolve(config["data"]["dataset_manifest"])
    manifest = _read_json(manifest_path)
    if manifest.get("version") != "E03-A8-V2-data-v1" or manifest.get("status") != "cpu_preparation_passed":
        raise ValueError("A8-V2 data Manifest is not in its prepared state")
    if manifest.get("counts", {}).get("effective_train") != 16192 or manifest.get("schedule", {}).get("steps") != 4048:
        raise ValueError("A8-V2 Train count or schedule length differs from the frozen contract")
    for key, record in manifest.get("inputs", {}).items():
        if sha256_file(_resolve(record["path"])) != record.get("sha256"):
            raise ValueError(f"A8-V2 source input changed after preparation: {key}")
    for name, expected in manifest.get("code_sha256", {}).items():
        if sha256_file(_resolve(name)) != expected:
            raise ValueError(f"A8-V2 data builder identity changed after preparation: {name}")
    for name, prompt_record in manifest.get("prompts", {}).items():
        if sha256_file(_resolve(prompt_record["path"])) != prompt_record.get("sha256"):
            raise ValueError(f"A8-V2 {name} prompt changed after data preparation")
    p_path, ep_path = data_root / "p_train.jsonl", data_root / "ep_train.jsonl"
    audit_path, order_path = data_root / "common_history_audit.jsonl", _resolve(config["data"]["sample_order_path"])
    for name, path in (("p_train.jsonl", p_path), ("ep_train.jsonl", ep_path), ("common_history_audit.jsonl", audit_path), ("sample_order.jsonl", order_path)):
        if sha256_file(path) != manifest.get("files", {}).get(name):
            raise ValueError(f"A8-V2 prepared file SHA differs: {name}")
    selected_path = _resolve(config["data"]["train_path"])
    if selected_path not in {p_path.resolve(), ep_path.resolve()}:
        raise ValueError("A8-V2 config does not point to the matching P/EP dataset")

    source_profiles = _read_source_profiles(manifest)
    ids: list[str] = []
    history_hashes: list[str] = []
    prompt_min = prompt_max = None
    sequence_max = 0
    completion_tokens = 0
    row_facts: list[dict[str, Any]] = []
    with p_path.open("r", encoding="utf-8") as p_handle, ep_path.open("r", encoding="utf-8") as ep_handle:
        for line_no, (p_line, ep_line) in enumerate(zip(p_handle, ep_handle), 1):
            p_row, ep_row = json.loads(p_line), json.loads(ep_line)
            if p_row.get("sample_id") != ep_row.get("sample_id") or p_row.get("conversation_id") != ep_row.get("conversation_id"):
                raise ValueError(f"A8-V2 P/EP ID or conversation differs at row {line_no}")
            if p_row.get("common_history_sha256") != ep_row.get("common_history_sha256"):
                raise ValueError(f"A8-V2 P/EP common history differs at row {line_no}")
            for row in (p_row, ep_row):
                _row_contract(row, eos_token_id=int(config["protocol"]["eos_token_id"]), prompt_cap=1280, sequence_cap=2048)
            sample_id = p_row["sample_id"]
            if sample_id in ids or sample_id not in source_profiles:
                raise ValueError("A8-V2 data repeats a Train ID or includes an ID outside normalized Train")
            ids.append(sample_id)
            history_hashes.append(p_row["common_history_sha256"])
            prompt_min = min(prompt_min, p_row["prompt_length"]) if prompt_min is not None else p_row["prompt_length"]
            prompt_max = max(prompt_max or 0, p_row["prompt_length"], ep_row["prompt_length"])
            sequence_max = max(sequence_max, len(p_row["input_ids"]), len(ep_row["input_ids"]))
            completion_tokens += p_row["completion_length"] + ep_row["completion_length"]
            if tokenizer is not None:
                p_text = tokenizer.decode(p_row["input_ids"][p_row["prompt_length"]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                ep_text = tokenizer.decode(ep_row["input_ids"][ep_row["prompt_length"]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if "Emotion=" in p_text or "Response=" in p_text or "memory_refs" in p_text:
                    raise ValueError(f"{sample_id}: P Completion contains a forbidden field")
                if "Response=" in ep_text or "memory_refs" in ep_text:
                    raise ValueError(f"{sample_id}: EP Completion contains a forbidden field")
            profile = source_profiles[sample_id]
            row_facts.append({
                "sample_id": sample_id,
                "profile_nonempty": any(profile[dim] for dim in PROFILE_DIMS),
                "sequence_tokens_p": len(p_row["input_ids"]),
                "sequence_tokens_ep": len(ep_row["input_ids"]),
            })
        if p_handle.readline() or ep_handle.readline():
            raise ValueError("A8-V2 P and EP have different row counts")
    if len(ids) != 16192 or canonical_sha256(ids) != manifest.get("sample_ids_ordered_sha256"):
        raise ValueError("A8-V2 ordered Train ID hash/count differs from its Manifest")
    if canonical_sha256(history_hashes) != manifest.get("counts", {}).get("common_history_sha256_ordered"):
        raise ValueError("A8-V2 ordered common-history hash differs from its Manifest")

    schedule = _read_jsonl(order_path)
    if len(schedule) != 4048 or any(row.get("optimizer_step") != index + 1 or len(row.get("sample_ids", [])) != 4 for index, row in enumerate(schedule)):
        raise ValueError("A8-V2 schedule does not contain 4,048 groups of four")
    flattened = [sample_id for row in schedule for sample_id in row["sample_ids"]]
    if len(flattened) != 16192 or len(set(flattened)) != 16192 or set(flattened) != set(ids) or flattened != ids:
        raise ValueError("A8-V2 schedule and paired data rows repeat, omit, or reorder IDs inconsistently")
    if canonical_sha256(schedule) != manifest.get("schedule", {}).get("schedule_sha256"):
        raise ValueError("A8-V2 sample order does not match its Manifest SHA")
    row_facts_by_id = {row["sample_id"]: row for row in row_facts}
    first_ids = flattened[:2048]
    first_nonempty = sum(row_facts_by_id[sample_id]["profile_nonempty"] for sample_id in first_ids)
    if first_nonempty != 1536 or len(first_ids) - first_nonempty != 512:
        raise ValueError("A8-V2 Step 512 does not contain exactly 1,536 positive and 512 empty Gold profiles")
    for step_index, step in enumerate(schedule[:512]):
        positives = sum(row_facts_by_id[sample_id]["profile_nonempty"] for sample_id in step["sample_ids"])
        if positives != 3:
            raise ValueError(f"A8-V2 optimizer step {step_index + 1} is not a 3:1 positive/empty mix")
    first_nonempty_rows = [row_facts_by_id[sample_id] for sample_id in first_ids if row_facts_by_id[sample_id]["profile_nonempty"]]
    first_empty_rows = [row_facts_by_id[sample_id] for sample_id in first_ids if not row_facts_by_id[sample_id]["profile_nonempty"]]
    representatives = {
        "positive_short": min(first_nonempty_rows, key=lambda row: row["sequence_tokens_p"])["sample_id"],
        "positive_medium": sorted(first_nonempty_rows, key=lambda row: row["sequence_tokens_p"])[len(first_nonempty_rows) // 2]["sample_id"],
        "positive_long": max(first_nonempty_rows, key=lambda row: row["sequence_tokens_p"])["sample_id"],
        "empty_long": max(first_empty_rows, key=lambda row: row["sequence_tokens_p"])["sample_id"],
    }
    return {
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "data_manifest": manifest,
        "train_rows": len(ids),
        "train_ids_ordered_sha256": canonical_sha256(ids),
        "common_history_sha256_ordered": canonical_sha256(history_hashes),
        "schedule_sha256": canonical_sha256(schedule),
        "sample_order_ids_sha256": canonical_sha256(flattened),
        "prompt_length_min": prompt_min,
        "prompt_length_max": prompt_max,
        "sequence_length_max": sequence_max,
        "completion_tokens_both_arms": completion_tokens,
        "schedule_rows": len(schedule),
        "schedule": schedule,
        "representative_ids": representatives,
    }


def _run_paths(config: dict[str, Any], run_id: str) -> tuple[Path, Path, Path]:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("A8-V2 run ID must use 3-96 ASCII letters, digits, dots, underscores, or hyphens")
    root = _resolve(config["data"]["output_root"])
    return root / f"{run_id}-preflight", root / f"{run_id}-dry", root / run_id


def prepare_preflight(config_path: Path, run_id: str) -> Path:
    config, config_file = _load_config(config_path)
    preflight_root, _dry_root, _run_root = _run_paths(config, run_id)
    if preflight_root.exists():
        raise FileExistsError(f"refusing to overwrite A8-V2 preflight: {preflight_root}")
    tokenizer, tokenizer_facts = _runtime_tokenizer(config)
    data_facts = _validate_prepared_data(config, tokenizer)
    prompt_path = _resolve(config["protocol"]["prompt_path"])
    p0_path = _resolve(config["protocol"]["selection_prompt_path"])
    if sha256_file(prompt_path) != config["protocol"]["prompt_sha256"] or sha256_file(p0_path) != config["protocol"]["p0_sha256"]:
        raise ValueError("A8-V2 prompt or protected P0 identity changed")
    schema_path = ROOT / "data/train/train_public.schema.json"
    schema_hash = sha256_file(schema_path)
    if schema_hash != "a9467e520b06a96859acaa2ff022e395d08ff6ad5e8ae3532a22cc24c630cada":
        raise ValueError("official Train Schema differs from the frozen repository baseline")
    base = _base_identity(config)
    output_root = _resolve(config["data"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_root)
    minimum_free = int(config["training"]["min_free_disk_gb"]) * 1024**3
    if disk.free < minimum_free:
        raise OSError(f"A8-V2 output root has {disk.free} free bytes; {minimum_free} required")
    result = {
        "status": "passed",
        "experiment_id": "E03-A8-V2",
        "training_scope": "generative_profile_specialist",
        "arm": config["arm"],
        "run_id": run_id,
        "created_at_utc": _now(),
        "config_path": str(config_file.relative_to(ROOT)),
        "config_sha256": _config_sha256(config),
        "data": {
            "dataset_manifest_sha256": data_facts["dataset_manifest_sha256"],
            "train_file_sha256": sha256_file(_resolve(config["data"]["train_path"])),
            "train_rows": data_facts["train_rows"],
            "train_ids_ordered_sha256": data_facts["train_ids_ordered_sha256"],
            "common_history_sha256_ordered": data_facts["common_history_sha256_ordered"],
            "schedule_sha256": data_facts["schedule_sha256"],
            "sample_order_ids_sha256": data_facts["sample_order_ids_sha256"],
            "schedule_rows": data_facts["schedule_rows"],
            "prompt_length_min": data_facts["prompt_length_min"],
            "prompt_length_max_both_arms": data_facts["prompt_length_max"],
            "sequence_length_max_both_arms": data_facts["sequence_length_max"],
            "completion_tokens_both_arms": data_facts["completion_tokens_both_arms"],
        },
        "tokenizer": tokenizer_facts,
        "base_model": base,
        "official_schema_sha256": schema_hash,
        "prompt_sha256": config["protocol"]["prompt_sha256"],
        "p0_sha256": config["protocol"]["p0_sha256"],
        "training_code_sha256": _code_identity(V2_TRAINING_CODE_PATHS),
        "evaluation_code_sha256": _code_identity(V2_EVALUATION_CODE_PATHS),
        "runtime_packages": _package_versions(),
        "output": {"free_bytes": disk.free, "required_minimum_free_bytes": minimum_free},
        "official_val_read": False,
    }
    preflight_root.mkdir(parents=True, exist_ok=False)
    _write_json(preflight_root / "preflight.json", result, exclusive=True)
    _write_json(preflight_root / "manifest.json", {
        "status": "passed", "run_id": run_id, "arm": config["arm"],
        "config_sha256": _config_sha256(config),
        "preflight_sha256": sha256_file(preflight_root / "preflight.json"),
    }, exclusive=True)
    _write_json(preflight_root / "config_snapshot.json", config, exclusive=True)
    return preflight_root


def _validate_preflight(config: dict[str, Any], run_id: str, path: Path) -> dict[str, Any]:
    root = path if path.is_absolute() else ROOT / path
    root = root.resolve()
    record = _read_json(root / "preflight.json")
    manifest = _read_json(root / "manifest.json")
    if (
        record.get("status") != "passed" or manifest.get("status") != "passed"
        or record.get("run_id") != run_id or record.get("arm") != config["arm"]
        or manifest.get("run_id") != run_id or manifest.get("arm") != config["arm"]
        or record.get("config_sha256") != _config_sha256(config)
        or manifest.get("config_sha256") != _config_sha256(config)
        or manifest.get("preflight_sha256") != sha256_file(root / "preflight.json")
    ):
        raise ValueError("A8-V2 CPU preflight identity/hash is incomplete")
    if record.get("training_code_sha256") != _code_identity(V2_TRAINING_CODE_PATHS):
        raise ValueError("A8-V2 training code changed after preflight")
    if record.get("evaluation_code_sha256") != _code_identity(V2_EVALUATION_CODE_PATHS):
        raise ValueError("A8-V2 evaluation code changed after preflight")
    current_data = _validate_prepared_data(config)
    if record.get("data", {}).get("dataset_manifest_sha256") != current_data["dataset_manifest_sha256"]:
        raise ValueError("A8-V2 dataset changed after preflight")
    return record


def _require_performance_gate(config: dict[str, Any], arm: str) -> dict[str, Any]:
    benchmark_root = _resolve(config["performance_diagnostics"]["benchmark_root"])
    path = benchmark_root / f"{arm}.json"
    record = _read_json(path)
    if (
        record.get("arm") != arm
        or record.get("run_id") != "E03-A8-V2-20261107-01"
        or record.get("benchmark_steps_per_path") != 16
        or record.get("samples_per_path") != 64
        or not record.get("hardware_match", {}).get("same_gpu_name")
    ):
        raise ValueError("A8-V2 performance benchmark identity or hardware comparison is incomplete")
    speedup = float(record.get("selective_speedup", 0.0))
    configured_minimum = float(config["performance_diagnostics"]["minimum_speedup"])
    if record.get("status") == "passed" and speedup >= configured_minimum:
        return {
            "status": "passed_original_gate",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "speedup": speedup,
            "minimum_speedup": configured_minimum,
        }

    override_path = benchmark_root / "user_approved_override.json"
    override = _read_json(override_path)
    expected_arms = {"P", "EP"}
    observed_reports = {name: _read_json(benchmark_root / f"{name}.json") for name in expected_arms}
    observed_hashes = {name: sha256_file(benchmark_root / f"{name}.json") for name in expected_arms}
    observed_speeds = {name: float(observed_reports[name].get("selective_speedup", 0.0)) for name in expected_arms}
    accepted_minimum = float(override.get("accepted_minimum_speedup", 0.0))
    if (
        override.get("version") != "E03-A8-V2-performance-override-v1"
        or override.get("status") != "user_approved"
        or override.get("benchmark_run_id") != "E03-A8-V2-20261107-01"
        or override.get("original_minimum_speedup") != configured_minimum
        or accepted_minimum < 1.3
        or override.get("benchmark_report_sha256") != observed_hashes
        or override.get("measured_speedups") != observed_speeds
        or override.get("sample_ids_sha256") != observed_reports["P"].get("sample_ids_sha256")
        or observed_reports["EP"].get("sample_ids_sha256") != observed_reports["P"].get("sample_ids_sha256")
        or speedup < accepted_minimum
    ):
        raise ValueError("A8-V2 performance gate is below 2x and lacks a valid paired user-approved override")
    return {
        "status": "user_approved_below_original_gate",
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "speedup": speedup,
        "original_minimum_speedup": configured_minimum,
        "accepted_minimum_speedup": accepted_minimum,
        "override_path": str(override_path.relative_to(ROOT)),
        "override_sha256": sha256_file(override_path),
    }


def _gradient_summary(
    parameters: list[tuple[str, Any]],
    torch: Any,
    *,
    full: bool,
    loss_tensors: list[Any] | None = None,
) -> dict[str, Any]:
    named = [(name, parameter.grad) for name, parameter in parameters if parameter.grad is not None]
    if not named:
        raise RuntimeError("A8-V2 accumulated no LoRA gradients")
    grads = [gradient for _name, gradient in named]
    norms = torch._foreach_norm(grads, 2.0)
    global_norm = torch.linalg.vector_norm(torch.stack([value.float() for value in norms]), ord=2)
    if full:
        checks = torch.stack([
            torch.stack([
                torch.isfinite(gradient).all().to(dtype=torch.float32),
                (torch.count_nonzero(gradient) > 0).to(dtype=torch.float32),
            ])
            for gradient in grads
        ])
        loss_finite = torch.isfinite(torch.stack(loss_tensors)).all().float().reshape(1) if loss_tensors else torch.ones(1, device=global_norm.device)
        facts = torch.cat([checks.reshape(-1), global_norm.reshape(1), loss_finite]).detach().cpu().tolist()
        finite_count = sum(int(facts[index * 2]) for index in range(len(grads)))
        nonzero_count = sum(int(facts[index * 2 + 1]) for index in range(len(grads)))
        norm_value = float(facts[-2])
        finite = finite_count == len(grads) and math.isfinite(norm_value) and bool(facts[-1])
        if not finite or nonzero_count == 0 or norm_value <= 0:
            raise FloatingPointError("A8-V2 full LoRA gradient diagnostic found non-finite or zero gradients")
        return {
            "finite": True,
            "checked_parameter_gradients": len(grads),
            "nonzero_parameter_gradients": nonzero_count,
            "global_norm": norm_value,
            "check_mode": "all_parameter_tensors_batched",
        }
    loss_finite = torch.isfinite(torch.stack(loss_tensors)).all().float() if loss_tensors else global_norm.new_ones(())
    facts = torch.stack([torch.isfinite(global_norm).float(), (global_norm > 0).float(), global_norm.float(), loss_finite]).detach().cpu().tolist()
    finite, nonzero, norm_value = bool(facts[0] and facts[3]), bool(facts[1]), float(facts[2])
    if not finite or not nonzero or not math.isfinite(norm_value):
        raise FloatingPointError("A8-V2 optimizer step has non-finite or zero LoRA gradients")
    return {"finite": finite, "global_norm": norm_value, "check_mode": "batched_global_norm"}


def _gradient_equivalence(old_grads: list[Any], new_grads: list[Any], torch: Any) -> dict[str, Any]:
    if len(old_grads) != len(new_grads) or not old_grads:
        raise RuntimeError("A8-V2 old/selective LoRA gradient lists differ")
    max_abs = torch.stack([(old - new).abs().max() for old, new in zip(old_grads, new_grads)]).max()
    per_tensor_rel = torch.stack([
        (old - new).norm() / old.norm().clamp_min(1e-12)
        for old, new in zip(old_grads, new_grads)
    ])
    old_norm_sq = torch.stack([old.float().pow(2).sum() for old in old_grads]).sum()
    new_norm_sq = torch.stack([new.float().pow(2).sum() for new in new_grads]).sum()
    diff_norm_sq = torch.stack([(old.float() - new.float()).pow(2).sum() for old, new in zip(old_grads, new_grads)]).sum()
    dot = torch.stack([(old.float() * new.float()).sum() for old, new in zip(old_grads, new_grads)]).sum()
    cosine = dot / (old_norm_sq.sqrt() * new_norm_sq.sqrt()).clamp_min(1e-12)
    facts = torch.stack([max_abs, per_tensor_rel.max(), diff_norm_sq.sqrt() / old_norm_sq.sqrt().clamp_min(1e-12), cosine]).detach().cpu().tolist()
    return {
        "max_absolute_error": float(facts[0]),
        "maximum_per_parameter_relative_l2_error": float(facts[1]),
        "global_relative_l2_error": float(facts[2]),
        "gradient_cosine_similarity": float(facts[3]),
        "compared_trainable_lora_tensors": len(old_grads),
    }


def execute_gpu_dry_step(config_path: Path, run_id: str, preflight_root: Path, *, allow_gpu_dry_run: bool) -> Path:
    if not allow_gpu_dry_run:
        raise ValueError("A8-V2 GPU dry-run requires --allow-gpu-dry-run")
    config, _ = _load_config(config_path)
    preflight = _validate_preflight(config, run_id, preflight_root)
    preflight_path, dry_root, _run_root = _run_paths(config, run_id)
    if dry_root.exists():
        raise FileExistsError(f"refusing to overwrite A8-V2 dry-run: {dry_root}")
    data_facts = _validate_prepared_data(config)
    tokenizer = model = None
    torch = None
    comparisons: list[dict[str, Any]] = []
    failed_comparisons: list[str] = []
    dry_root.mkdir(parents=True, exist_ok=False)
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(config)
        torch = __import__("torch")
        _set_seed(int(config["training"]["seed"]), torch)
        from src.training.training_sft import _load_train_model

        model, torch, model_facts = _load_train_model(config, _base_path(config))
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A8-V2 requires CUDA and BF16")
        initial_sha = _lora_tensor_sha256(model, torch)
        parameters = _lora_parameters(model)
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        id_index = JsonlById(_resolve(config["data"]["train_path"]))
        for name, sample_id in data_facts["representative_ids"].items():
            row = id_index.get(sample_id)
            _row_contract(row, eos_token_id=int(config["protocol"]["eos_token_id"]), prompt_cap=1280, sequence_cap=2048)
            batch = {key: value.to(model.get_input_embeddings().weight.device) for key, value in collator([row]).items()}
            model.zero_grad(set_to_none=True)
            full_loss = model(**batch).loss
            if not bool(torch.isfinite(full_loss).item()):
                raise FloatingPointError("A8-V2 full-logit reference loss is non-finite")
            full_loss.backward()
            old_grads = [parameter.grad.detach().clone() for _name, parameter in parameters]
            full_value = float(full_loss.detach().float().cpu())
            model.zero_grad(set_to_none=True)
            selective_loss = completion_only_forward_loss(
                model, batch, prompt_length=int(row["prompt_length"]), sequence_length=len(row["input_ids"])
            )
            if not bool(torch.isfinite(selective_loss).item()):
                raise FloatingPointError("A8-V2 selective-logit loss is non-finite")
            selective_loss.backward()
            new_grads = [parameter.grad.detach() for _name, parameter in parameters]
            gradient_facts = _gradient_equivalence(old_grads, new_grads, torch)
            selective_value = float(selective_loss.detach().float().cpu())
            loss_abs = abs(full_value - selective_value)
            loss_rel = loss_abs / max(abs(full_value), 1e-12)
            passed = (
                loss_abs <= EQUIVALENCE_TOLERANCES["loss_absolute_max"]
                and gradient_facts["maximum_per_parameter_relative_l2_error"] <= EQUIVALENCE_TOLERANCES["per_parameter_relative_l2_max"]
                and gradient_facts["global_relative_l2_error"] <= EQUIVALENCE_TOLERANCES["global_relative_l2_max"]
                and gradient_facts["gradient_cosine_similarity"] >= EQUIVALENCE_TOLERANCES["cosine_min"]
            )
            comparisons.append({
                "sample_id": sample_id,
                "length_group": name,
                "full_loss": full_value,
                "selective_loss": selective_value,
                "loss_absolute_error": loss_abs,
                "loss_relative_error": loss_rel,
                "gradient": gradient_facts,
                "tolerance": EQUIVALENCE_TOLERANCES,
                "passed": passed,
            })
            if not passed:
                failed_comparisons.append(name)
            model.zero_grad(set_to_none=True)
            del old_grads, new_grads, batch
        id_index.close()
        if failed_comparisons:
            raise RuntimeError(f"A8-V2 selective loss/gradient equivalence failed for {failed_comparisons}")
        final_lora_sha = _lora_tensor_sha256(model, torch)
        if initial_sha != final_lora_sha:
            raise RuntimeError("A8-V2 dry-run modified LoRA parameters without an optimizer step")
        gradient_check = {
            "status": "passed",
            "representative_samples_compared": len(comparisons),
            "trainable_lora_tensors_per_sample": comparisons[0]["gradient"]["compared_trainable_lora_tensors"],
            "optimizer_update_performed": False,
        }
        memory = inference.cuda_memory_facts(torch)
        if int(memory.get("allocator_peak_allocated_bytes", 0)) > int(memory.get("total_memory_bytes", 0)) * float(config["quantization"]["max_cuda_memory_fraction"]):
            raise RuntimeError("A8-V2 equivalence gate exceeded the frozen 92% CUDA memory budget")
        if not model_facts.get("real_nf4_verified") or model_facts.get("offload_devices"):
            raise RuntimeError("A8-V2 dry-run did not use verified NF4/BF16 CUDA placement")
        result = {
            "status": "passed",
            "experiment_id": "E03-A8-V2",
            "arm": config["arm"],
            "run_id": run_id,
            "config_sha256": _config_sha256(config),
            "preflight_sha256": sha256_file(preflight_path / "preflight.json"),
            "data_manifest_sha256": preflight["data"]["dataset_manifest_sha256"],
            "dry_step_only": True,
            "optimizer_update_performed": False,
            "initial_lora_tensor_sha256": initial_sha,
            "post_backward_lora_tensor_sha256": final_lora_sha,
            "equivalence_tolerance_rationale": "Full and selective target logits matched exactly; maximum observed loss difference over eight P/EP representatives was 9.06e-6 and isolated final-hidden gradient relative difference was 8.8e-7. Shape-dependent backward accumulation through the NF4/BF16 LoRA stack yielded up to 1.041% global gradient relative error, 1.904% maximum per-parameter relative error, and 0.999946 minimum cosine. The frozen bounds (2e-5 loss, 1.2% global, 2.1% per-parameter, cosine 0.9999) include roughly 10-15% margin over the observed maxima while still rejecting shifts or omitted supervised tokens.",
            "equivalence_comparisons": comparisons,
            "gradient_check": gradient_check,
            "tokenizer": tokenizer_facts,
            "model": model_facts,
            "cuda_memory": memory,
            "created_at_utc": _now(),
        }
        _write_json(dry_root / "dry_step.json", result, exclusive=True)
        _write_json(dry_root / "manifest.json", {
            "status": "passed", "run_id": run_id, "arm": config["arm"],
            "config_sha256": _config_sha256(config),
            "preflight_sha256": result["preflight_sha256"],
            "dry_step_sha256": sha256_file(dry_root / "dry_step.json"),
        }, exclusive=True)
        return dry_root
    except BaseException as exc:
        _write_json(dry_root / "dry_step.json", {
            "status": "failed", "run_id": run_id, "arm": config["arm"],
            "error": f"{type(exc).__name__}: {exc}", "created_at_utc": _now(),
            "equivalence_comparisons": comparisons,
            "failed_comparison_groups": failed_comparisons,
        })
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_dry(config: dict[str, Any], run_id: str, preflight_root: Path, dry_root: Path) -> dict[str, Any]:
    preflight = _validate_preflight(config, run_id, preflight_root)
    dry_root = dry_root if dry_root.is_absolute() else ROOT / dry_root
    dry_root = dry_root.resolve()
    dry = _read_json(dry_root / "dry_step.json")
    manifest = _read_json(dry_root / "manifest.json")
    if (
        dry.get("status") != "passed" or manifest.get("status") != "passed"
        or dry.get("run_id") != run_id or dry.get("arm") != config["arm"]
        or dry.get("config_sha256") != _config_sha256(config)
        or dry.get("preflight_sha256") != sha256_file((preflight_root if preflight_root.is_absolute() else ROOT / preflight_root) / "preflight.json")
        or manifest.get("dry_step_sha256") != sha256_file(dry_root / "dry_step.json")
        or dry.get("optimizer_update_performed") is not False
        or dry.get("initial_lora_tensor_sha256") != dry.get("post_backward_lora_tensor_sha256")
        or len(dry.get("equivalence_comparisons", [])) != 4
    ):
        raise ValueError("A8-V2 selective loss/gradient equivalence dry-run is incomplete")
    if dry.get("data_manifest_sha256") != preflight.get("data", {}).get("dataset_manifest_sha256"):
        raise ValueError("A8-V2 dry-run data identity differs from preflight")
    return dry


def _paired_initialization_check(config: dict[str, Any], run_id: str, own_dry: dict[str, Any]) -> None:
    other_arm = "EP" if config["arm"] == "P" else "P"
    other_root = ROOT / "outputs/training/E03-A8-V2" / other_arm / f"{run_id}-dry"
    other = _read_json(other_root / "dry_step.json")
    if other.get("status") != "passed" or other.get("initial_lora_tensor_sha256") != own_dry.get("initial_lora_tensor_sha256"):
        raise ValueError("A8-V2 P and EP did not start from identical fresh LoRA tensors")
    if other.get("equivalence_comparisons") is None:
        raise ValueError("A8-V2 paired Arm has no passed selective-loss equivalence gate")


def save_recovery_bundle(
    checkpoint: Path,
    *,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    run_id: str,
    arm: str,
    step: int,
    accumulation_steps: int = 4,
    last_loss: float = 0.0,
    training_seconds: float = 0.0,
) -> dict[str, Any]:
    """Persist optimizer, scheduler, RNG and exact sample cursor in a checkpoint."""
    if arm not in {"P", "EP"} or step < 0 or accumulation_steps <= 0:
        raise ValueError("invalid A8-V2 recovery bundle identity")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(_rng_state(torch), checkpoint / "rng_state.pth")
    state = {
        "global_step": step,
        "sample_cursor": step * accumulation_steps,
        "run_id": run_id,
        "arm": arm,
        "last_loss": float(last_loss),
        "training_seconds": float(training_seconds),
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }
    _write_json(checkpoint / "trainer_state.json", state, exclusive=True)
    return state


def restore_recovery_bundle(
    checkpoint: Path,
    *,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    expected_run_id: str,
    expected_arm: str,
    accumulation_steps: int = 4,
) -> dict[str, Any]:
    """Restore training state only after validating the frozen step cursor."""
    state = _read_json(checkpoint / "trainer_state.json")
    step = state.get("global_step")
    if (
        state.get("run_id") != expected_run_id
        or state.get("arm") != expected_arm
        or not isinstance(step, int)
        or isinstance(step, bool)
        or step < 0
        or state.get("sample_cursor") != step * accumulation_steps
    ):
        raise ValueError("A8-V2 recovery Run/Arm/step/sample cursor differs")
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
    _restore_rng_state(torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False), torch)
    return state


class _LedgerBuffer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("xb")
        self.pending: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.pending.append(row)

    def flush(self) -> float:
        if not self.pending:
            return 0.0
        started = time.monotonic()
        payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for row in self.pending)
        self.handle.write(payload)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.pending.clear()
        return time.monotonic() - started

    def close(self) -> None:
        self.handle.close()


def _save_checkpoint(
    *, model: Any, optimizer: Any, scheduler: Any, torch: Any, run_root: Path,
    config: dict[str, Any], run_id: str, step: int, data_facts: dict[str, Any],
    preflight: dict[str, Any], initial_lora_sha: str, last_loss: float,
    training_seconds: float, cuda_memory: dict[str, Any], code_identity: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    checkpoint = run_root / "trainer" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    adapter_config_path = checkpoint / "adapter_config.json"
    adapter_config = _read_json(adapter_config_path)
    adapter_config["base_model_name_or_path"] = config["base_model"]["model_id"]
    _write_json(adapter_config_path, adapter_config)
    state = save_recovery_bundle(
        checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
        torch=torch,
        run_id=run_id,
        arm=config["arm"],
        step=step,
        accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        last_loss=last_loss,
        training_seconds=training_seconds,
    )
    adapter_identity, restored_config = inference.adapter_weight_identity(checkpoint)
    if restored_config.get("r") != 8 or restored_config.get("lora_alpha") != 16:
        raise RuntimeError("A8-V2 checkpoint adapter config differs from its frozen LoRA identity")
    files = _checkpoint_files(checkpoint)
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "adapter_config.json"}
    if not required <= {record["path"] for record in files} or not any(record["path"].startswith("adapter_model") for record in files):
        raise RuntimeError("A8-V2 checkpoint is missing adapter or recovery state")
    record = {
        "status": "complete",
        "experiment_id": "E03-A8-V2",
        "training_scope": "generative_profile_specialist",
        "arm": config["arm"],
        "run_id": run_id,
        "optimizer_step": step,
        "sample_cursor": state["sample_cursor"],
        "config_sha256": _config_sha256(config),
        "dataset_manifest_sha256": data_facts["dataset_manifest_sha256"],
        "train_file_sha256": sha256_file(_resolve(config["data"]["train_path"])),
        "sample_order_sha256": data_facts["schedule_sha256"],
        "consumed_sample_ids_sha256": canonical_sha256([sid for row in data_facts["schedule"][:step] for sid in row["sample_ids"]]),
        "prompt_sha256": config["protocol"]["prompt_sha256"],
        "p0_sha256": config["protocol"]["p0_sha256"],
        "training_code_sha256": code_identity,
        "initial_lora_tensor_sha256": initial_lora_sha,
        "adapter_identity": adapter_identity,
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "cuda_memory": cuda_memory,
        "last_loss": last_loss,
        "cumulative_training_seconds": training_seconds,
        "files": files,
        "files_sha256": canonical_sha256(files),
        "created_at_utc": _now(),
    }
    _write_json(checkpoint / "checkpoint_manifest.json", record, exclusive=True)
    return checkpoint, record


def _verify_v2_checkpoint(checkpoint: Path, expected: dict[str, Any]) -> dict[str, Any]:
    record = _read_json(checkpoint / "checkpoint_manifest.json")
    if record.get("status") != "complete":
        raise ValueError("A8-V2 checkpoint is incomplete")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"A8-V2 checkpoint {key} differs from its Run")
    files = _checkpoint_files(checkpoint)
    if record.get("files_sha256") != canonical_sha256(record.get("files")) or files != record.get("files"):
        raise ValueError("A8-V2 checkpoint file SHA/size differs")
    state = _read_json(checkpoint / "trainer_state.json")
    step = int(checkpoint.name.removeprefix("checkpoint-"))
    if state.get("global_step") != step or state.get("sample_cursor") != step * 4 or record.get("optimizer_step") != step:
        raise ValueError("A8-V2 checkpoint step and sample cursor disagree")
    return record


def _execute_training(
    config: dict[str, Any],
    run_id: str,
    run_root: Path,
    *,
    preflight: dict[str, Any],
    data_facts: dict[str, Any],
    dry: dict[str, Any],
    continuation: dict[str, Any] | None = None,
) -> Path:
    torch = __import__("torch")
    tokenizer = model = None
    ledger = None
    started = time.monotonic()
    start_step = int(continuation["start_step"]) if continuation else 0
    stop_step = int(continuation["stop_step"]) if continuation else 512
    prior_training_seconds = 0.0
    code_identity = _code_identity(V2_TRAINING_CODE_PATHS) if continuation else preflight["training_code_sha256"]
    benchmark_gate = _require_performance_gate(config, config["arm"])
    try:
        tokenizer, _tokenizer_facts = _runtime_tokenizer(config)
        from src.training.training_sft import _load_train_model

        _set_seed(int(config["training"]["seed"]), torch)
        if continuation:
            model, torch, model_facts = _load_train_model(
                config,
                _base_path(config),
                adapter_path=continuation["parent_checkpoint"],
                is_trainable_adapter=True,
                autocast_adapter_dtype=True,
            )
        else:
            model, torch, model_facts = _load_train_model(config, _base_path(config))
        trainable = _lora_parameters(model)
        loaded_lora_sha = _lora_tensor_sha256(model, torch)
        initial_lora_sha = dry["initial_lora_tensor_sha256"]
        if continuation:
            if loaded_lora_sha != continuation["parent_adapter_tensor_sha256"]:
                raise ValueError("A8-V2 continuation Adapter tensors differ from the parent checkpoint")
        elif loaded_lora_sha != initial_lora_sha:
            raise ValueError("A8-V2 fresh training LoRA differs from its dry-run initialization")
        optimizer = torch.optim.AdamW(
            [parameter for _name, parameter in trainable],
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        if continuation:
            restored = restore_recovery_bundle(
                continuation["parent_checkpoint"],
                optimizer=optimizer,
                scheduler=scheduler,
                torch=torch,
                expected_run_id=continuation["parent_run_id"],
                expected_arm=config["arm"],
                accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
            )
            if (
                restored.get("global_step") != start_step
                or restored.get("sample_cursor") != start_step * int(config["training"]["gradient_accumulation_steps"])
                or scheduler.last_epoch != start_step
                or not optimizer.state
            ):
                raise ValueError("A8-V2 continuation parent recovery state is not at the requested boundary")
            prior_training_seconds = float(restored.get("training_seconds", 0.0))
        base_before = _base_identity(config)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        schedule = data_facts["schedule"]
        train_path = _resolve(config["data"]["train_path"])
        ledger = _LedgerBuffer(run_root / "step_ledger.jsonl")
        pending: list[dict[str, Any]] = []
        recent_step_events = []
        pending_loss_tensors = []
        with JsonlById(train_path) as train_rows:
            for step_index in range(start_step, stop_step):
                step = step_index + 1
                step_host_started = time.monotonic()
                scheduled = schedule[step_index]
                optimizer.zero_grad(set_to_none=True)
                micro_loss_tensors = []
                completion_tokens = 0
                sequence_tokens = 0
                prompt_tokens = 0
                data_fetch_seconds = collate_seconds = h2d_seconds = 0.0
                forward_enqueue_seconds = backward_enqueue_seconds = 0.0
                step_event_start = torch.cuda.Event(enable_timing=True)
                step_event_end = torch.cuda.Event(enable_timing=True)
                diagnostic = step <= 4 or step % int(config["training"]["diagnostic_interval_steps"]) == 0 or step == stop_step
                micro_events = []
                for sample_id in scheduled["sample_ids"]:
                    timer = time.monotonic()
                    row = train_rows.get(sample_id)
                    _row_contract(row, eos_token_id=int(config["protocol"]["eos_token_id"]), prompt_cap=1280, sequence_cap=2048)
                    data_fetch_seconds += time.monotonic() - timer
                    timer = time.monotonic()
                    batch = collator([row])
                    collate_seconds += time.monotonic() - timer
                    timer = time.monotonic()
                    device = model.get_input_embeddings().weight.device
                    batch = {key: value.to(device) for key, value in batch.items()}
                    h2d_seconds += time.monotonic() - timer
                    if len(micro_loss_tensors) == 0:
                        step_event_start.record()
                    event_start = torch.cuda.Event(enable_timing=True) if diagnostic else None
                    event_forward_end = torch.cuda.Event(enable_timing=True) if diagnostic else None
                    event_backward_end = torch.cuda.Event(enable_timing=True) if diagnostic else None
                    if event_start is not None:
                        event_start.record()
                    timer = time.monotonic()
                    loss = completion_only_forward_loss(
                        model,
                        batch,
                        prompt_length=int(row["prompt_length"]),
                        sequence_length=len(row["input_ids"]),
                    )
                    forward_enqueue_seconds += time.monotonic() - timer
                    if event_forward_end is not None:
                        event_forward_end.record()
                    micro_loss_tensors.append(loss.detach().float())
                    completion_tokens += int(row["completion_length"])
                    sequence_tokens += len(row["input_ids"])
                    prompt_tokens += int(row["prompt_length"])
                    timer = time.monotonic()
                    (loss / 4).backward()
                    backward_enqueue_seconds += time.monotonic() - timer
                    if event_backward_end is not None:
                        event_backward_end.record()
                        micro_events.append((event_start, event_forward_end, event_backward_end))

                timer = time.monotonic()
                gradient = _gradient_summary(trainable, torch, full=diagnostic, loss_tensors=micro_loss_tensors)
                gradient_check_seconds = time.monotonic() - timer
                timer = time.monotonic()
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                optimizer_enqueue_seconds = time.monotonic() - timer
                step_event_end.record()
                step_end_host = time.monotonic()
                recent_step_events.append((step, step_event_start, step_event_end, micro_events))
                pending.append({
                    "optimizer_step": step,
                    "sample_ids": list(scheduled["sample_ids"]),
                    "completion_tokens": completion_tokens,
                    "sequence_tokens": sequence_tokens,
                    "prompt_tokens": prompt_tokens,
                    "loss_tensors": micro_loss_tensors,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient": gradient,
                    "data_fetch_seconds": data_fetch_seconds,
                    "collate_seconds": collate_seconds,
                    "host_to_device_seconds": h2d_seconds,
                    "forward_enqueue_seconds": forward_enqueue_seconds,
                    "backward_enqueue_seconds": backward_enqueue_seconds,
                    "gradient_check_seconds": gradient_check_seconds,
                    "optimizer_enqueue_seconds": optimizer_enqueue_seconds,
                    "host_step_end_monotonic": step_end_host,
                    "host_elapsed_segment_seconds": step_end_host - started,
                    "step_wall_host_seconds": step_end_host - step_host_started,
                })

                if step % int(config["training"]["diagnostic_interval_steps"]) == 0 or step == stop_step:
                    torch.cuda.synchronize()
                    flat_losses = torch.stack([loss for item in pending for loss in item["loss_tensors"]]).detach().cpu().tolist()
                    memory = inference.cuda_memory_facts(torch)
                    if memory.get("cuda_available") and int(memory["allocator_peak_allocated_bytes"]) > int(memory["total_memory_bytes"]) * float(config["quantization"]["max_cuda_memory_fraction"]):
                        raise RuntimeError("A8-V2 training exceeded the frozen 92% CUDA allocator budget")
                    loss_cursor = 0
                    elapsed_now = time.monotonic() - started
                    for offset, item in enumerate(pending):
                        item_losses = [float(value) for value in flat_losses[loss_cursor:loss_cursor + 4]]
                        loss_cursor += 4
                        event_item = recent_step_events[offset]
                        gpu_step_seconds = event_item[1].elapsed_time(event_item[2]) / 1000.0
                        timing = {
                            "gpu_step_seconds": gpu_step_seconds,
                            "data_fetch_seconds": item["data_fetch_seconds"],
                            "collate_seconds": item["collate_seconds"],
                            "host_to_device_seconds": item["host_to_device_seconds"],
                            "forward_enqueue_seconds": item["forward_enqueue_seconds"],
                            "backward_enqueue_seconds": item["backward_enqueue_seconds"],
                            "gradient_check_seconds": item["gradient_check_seconds"],
                            "optimizer_enqueue_seconds": item["optimizer_enqueue_seconds"],
                            "step_wall_host_seconds": item["step_wall_host_seconds"],
                        }
                        if item["optimizer_step"] % int(config["training"]["diagnostic_interval_steps"]) == 0:
                            event_components = []
                            for _fetch_start, forward_end, backward_end in event_item[3]:
                                event_components.append({
                                    "forward_gpu_seconds": _fetch_start.elapsed_time(forward_end) / 1000.0,
                                    "backward_gpu_seconds": forward_end.elapsed_time(backward_end) / 1000.0,
                                })
                            timing["sampled_microbatch_cuda_events"] = event_components
                            timing["cuda_memory_sample"] = memory
                        row_ledger = {
                            "run_id": run_id,
                            "arm": config["arm"],
                            "optimizer_step": item["optimizer_step"],
                            "sample_ids": item["sample_ids"],
                            "sample_ids_sha256": canonical_sha256(item["sample_ids"]),
                            "completion_tokens": item["completion_tokens"],
                            "sequence_tokens": item["sequence_tokens"],
                            "prompt_tokens": item["prompt_tokens"],
                            "micro_batch_losses": item_losses,
                            "mean_completion_loss": sum(item_losses) / len(item_losses),
                            "learning_rate": item["learning_rate"],
                            "gradient": item["gradient"],
                            "timing": timing,
                            "elapsed_segment_seconds": item["host_elapsed_segment_seconds"],
                            "created_at_utc": _now(),
                        }
                        ledger.append(row_ledger)
                    persist_seconds = ledger.flush()
                    pending.clear()
                    recent_step_events.clear()
                    if prior_training_seconds + elapsed_now > float(config["training"]["max_training_seconds"]):
                        raise TimeoutError("A8-V2 Run exceeded its frozen cumulative training-time budget")
                    print(f"[E03-A8-V2 {config['arm']}] optimizer_step={step}/{stop_step} loss={row_ledger['mean_completion_loss']:.5f} supervised_tokens={row_ledger['completion_tokens']} ledger_flush_s={persist_seconds:.2f}", flush=True)

        torch.cuda.synchronize()
        if pending or ledger.pending:
            raise RuntimeError("A8-V2 final ledger was not flushed before checkpointing")
        segment_training_seconds = time.monotonic() - started
        training_seconds = prior_training_seconds + segment_training_seconds
        base_after = _base_identity(config)
        if base_before.get("observed_weight_hash") != base_after.get("observed_weight_hash"):
            raise RuntimeError("A8-V2 Base weights changed during training")
        all_ledger = _read_jsonl(run_root / "step_ledger.jsonl")
        expected_steps = list(range(start_step + 1, stop_step + 1))
        if len(all_ledger) != len(expected_steps) or [row["optimizer_step"] for row in all_ledger] != expected_steps:
            raise RuntimeError("A8-V2 stage ledger is not a complete, ordered record for its step range")
        checkpoint, checkpoint_record = _save_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, torch=torch, run_root=run_root,
            config=config, run_id=run_id, step=stop_step, data_facts=data_facts, preflight=preflight,
            initial_lora_sha=initial_lora_sha, last_loss=float(all_ledger[-1]["mean_completion_loss"]),
            training_seconds=training_seconds, cuda_memory=inference.cuda_memory_facts(torch),
            code_identity=code_identity,
        )
        probe_id = schedule[start_step]["sample_ids"][0]
        with JsonlById(train_path) as train_rows:
            probe_row = train_rows.get(probe_id)
        probe_ids = list(probe_row["input_ids"][:probe_row["prompt_length"]])
        before_token = _next_token_id(model, torch, probe_ids)
        before_lora = _lora_tensor_sha256(model, torch)
        expected_adapter_sha = checkpoint_record["adapter_identity"].get("weight_sha256")
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        reload_check = _independent_reload_check(checkpoint, config, probe_ids, before_token, before_lora, expected_adapter_sha)
        gpu_times = [float(row["timing"]["gpu_step_seconds"]) for row in all_ledger]
        wall_times = [float(row["timing"]["step_wall_host_seconds"]) for row in all_ledger]
        parent_training_node = None
        if continuation:
            parent_training_node = continuation["parent_training_node"]
        segment_loss_mean = sum(float(row["mean_completion_loss"]) for row in all_ledger) / len(all_ledger)
        segment_completion_tokens = sum(int(row["completion_tokens"]) for row in all_ledger)
        cumulative_completion_tokens = segment_completion_tokens + int(parent_training_node["completion_tokens_seen"] if parent_training_node else 0)
        cumulative_loss_mean = (
            (float(parent_training_node["loss_mean"]) * start_step + segment_loss_mean * len(all_ledger)) / stop_step
            if parent_training_node else segment_loss_mean
        )
        training_node = {
            "status": "training_checkpoint_passed",
            "experiment_id": "E03-A8-V2",
            "run_id": run_id,
            "arm": config["arm"],
            "optimizer_step": stop_step,
            "steps_in_segment": stop_step - start_step,
            "segment_start_step_exclusive": start_step,
            "samples_seen": stop_step * int(config["training"]["gradient_accumulation_steps"]),
            "continuation_parent": {
                "run_id": continuation["parent_run_id"],
                "step": start_step,
                "checkpoint_manifest_sha256": continuation["parent_checkpoint_manifest_sha256"],
                "adapter_weight_sha256": continuation["parent_adapter_weight_sha256"],
                "adapter_tensor_sha256": continuation["parent_adapter_tensor_sha256"],
            } if continuation else None,
            "sample_order_sha256": data_facts["schedule_sha256"],
            "consumed_sample_ids_sha256": checkpoint_record["consumed_sample_ids_sha256"],
            "completion_tokens_seen": cumulative_completion_tokens,
            "checkpoint_path": str(checkpoint.relative_to(run_root)),
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
            "checkpoint": checkpoint_record,
            "independent_reload_check": reload_check,
            "performance_gate": benchmark_gate,
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "base_weights_unchanged": True,
            "cumulative_training_seconds": training_seconds,
            "loss_mean": cumulative_loss_mean,
            "segment_loss_mean": segment_loss_mean,
            "last_loss": float(all_ledger[-1]["mean_completion_loss"]),
            "step_gpu_seconds": {
                "mean": sum(gpu_times) / len(gpu_times),
                "p50": sorted(gpu_times)[len(gpu_times) // 2],
                "p90": sorted(gpu_times)[math.ceil(0.90 * len(gpu_times)) - 1],
                "max": max(gpu_times),
            },
            "step_host_elapsed_seconds": {
                "mean": sum(wall_times) / len(wall_times),
                "p50": sorted(wall_times)[len(wall_times) // 2],
                "p90": sorted(wall_times)[math.ceil(0.90 * len(wall_times)) - 1],
                "max": max(wall_times),
                "note": "host step time includes the one batched loss/gradient check sync; GPU event time separately measures device execution.",
            },
            "cuda_memory": inference.cuda_memory_facts(torch),
            "model_facts": model_facts,
            "created_at_utc": _now(),
        }
        node_dir = run_root / f"step-{stop_step:04d}"
        node_dir.mkdir(parents=False, exist_ok=False)
        _write_json(node_dir / "training.json", training_node, exclusive=True)
        _write_json(run_root / "manifest.json", {
            **_read_json(run_root / "manifest.json"),
            "status": "awaiting_evaluation",
            "stage": "checkpoint_saved_awaiting_evaluation",
            "optimizer_steps": stop_step,
            "last_checkpoint": str(checkpoint.relative_to(run_root)),
            "checkpoint_manifest_sha256": training_node["checkpoint_manifest_sha256"],
            "cumulative_training_seconds": training_seconds,
            "base_weight_sha256_after_training": base_after.get("observed_weight_hash"),
            "nodes": {str(stop_step): {
                "training_status": "passed",
                "training_summary_path": str((node_dir / "training.json").relative_to(run_root)),
                "checkpoint_manifest_sha256": training_node["checkpoint_manifest_sha256"],
                "evaluation_status": "not_run",
                "authorization_status": "awaiting_evaluation",
            }},
            "ended_at_utc": _now(),
        })
        return run_root
    except BaseException as exc:
        if run_root.exists():
            manifest = _read_json(run_root / "manifest.json")
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", stage="training_interrupted" if isinstance(exc, KeyboardInterrupt) else "training_failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
            _write_json(run_root / "manifest.json", manifest)
        raise
    finally:
        if ledger is not None:
            ledger.close()
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _independent_reload_check(checkpoint: Path, config: dict[str, Any], prompt_ids: list[int], before_token: int, before_lora_sha: str, expected_adapter_sha: str) -> dict[str, Any]:
    torch = __import__("torch")
    base_before = _base_identity(config)
    from src.training.training_sft import _load_train_model

    reloaded, torch, facts = _load_train_model(config, _base_path(config), adapter_path=checkpoint, autocast_adapter_dtype=True)
    after_token = _next_token_id(reloaded, torch, prompt_ids)
    after_lora_sha = _lora_tensor_sha256(reloaded, torch)
    dtypes = sorted({str(parameter.dtype) for name, parameter in reloaded.named_parameters() if ".lora_A." in name or ".lora_B." in name})
    adapter_identity, _adapter_config = inference.adapter_weight_identity(checkpoint)
    base_after = _base_identity(config)
    result = {
        "status": "passed" if before_token == after_token and before_lora_sha == after_lora_sha and adapter_identity.get("weight_sha256") == expected_adapter_sha and dtypes == ["torch.float32"] and base_before.get("observed_weight_hash") == base_after.get("observed_weight_hash") else "failed",
        "next_token_exact_match": before_token == after_token,
        "next_token_before": before_token,
        "next_token_after": after_token,
        "adapter_tensor_sha256_before_save": before_lora_sha,
        "adapter_tensor_sha256_after_reload": after_lora_sha,
        "adapter_file_identity": adapter_identity,
        "adapter_dtypes_after_reload": dtypes,
        "model_facts": facts,
        "base_weight_sha256_before": base_before.get("observed_weight_hash"),
        "base_weight_sha256_after": base_after.get("observed_weight_hash"),
        "base_weights_unchanged": base_before.get("observed_weight_hash") == base_after.get("observed_weight_hash"),
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    if result["status"] != "passed":
        raise RuntimeError(f"A8-V2 independent checkpoint reload failed: {result}")
    return result


def execute_start(config_path: Path, run_id: str, preflight_root: Path, dry_root: Path, *, stop_at_step: int, allow_first_stage_training: bool) -> Path:
    if not allow_first_stage_training:
        raise ValueError("A8-V2 Step 0→512 requires --allow-first-stage-training")
    if stop_at_step != 512:
        raise ValueError("A8-V2 plan authorizes the first Step 512 node only")
    config, _ = _load_config(config_path)
    preflight = _validate_preflight(config, run_id, preflight_root)
    dry = _validate_dry(config, run_id, preflight_root, dry_root)
    _paired_initialization_check(config, run_id, dry)
    benchmark_gate = _require_performance_gate(config, config["arm"])
    data_facts = _validate_prepared_data(config)
    run_root = _run_paths(config, run_id)[2]
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite A8-V2 Run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "trainer").mkdir()
    shutil.copy2(preflight_root / "preflight.json", run_root / "preflight.json")
    shutil.copy2(dry_root / "dry_step.json", run_root / "dry_step_gate.json")
    shutil.copy2(_resolve(config["data"]["sample_order_path"]), run_root / "sample_order.jsonl")
    _write_json(run_root / "config_snapshot.json", config, exclusive=True)
    _write_json(run_root / "manifest.json", {
        "status": "running",
        "stage": "training_initialized",
        "experiment_id": "E03-A8-V2",
        "training_scope": "generative_profile_specialist",
        "arm": config["arm"],
        "run_id": run_id,
        "config_sha256": _config_sha256(config),
        "preflight_sha256": sha256_file(run_root / "preflight.json"),
        "dry_step_sha256": sha256_file(run_root / "dry_step_gate.json"),
        "training_code_sha256": preflight["training_code_sha256"],
        "evaluation_code_sha256": preflight["evaluation_code_sha256"],
        "dataset_manifest_sha256": data_facts["dataset_manifest_sha256"],
        "train_file_sha256": sha256_file(_resolve(config["data"]["train_path"])),
        "sample_order_sha256": data_facts["schedule_sha256"],
        "initial_lora_tensor_sha256": dry["initial_lora_tensor_sha256"],
        "performance_gate": benchmark_gate,
        "max_optimizer_steps": 4048,
        "authorized_max_step": 512,
        "recovery_state_persisted": True,
        "optimizer_steps": 0,
        "nodes": {},
        "created_at_utc": _now(),
    }, exclusive=True)
    return _execute_training(config, run_id, run_root, preflight=preflight, data_facts=data_facts, dry=dry)


def execute_continue(
    config_path: Path,
    run_id: str,
    parent_run_id: str,
    preflight_root: Path,
    dry_root: Path,
    *,
    stop_at_step: int,
    allow_authorized_continuation: bool,
) -> Path:
    """Continue the P Arm from a reviewed, verified checkpoint boundary."""
    if not allow_authorized_continuation or stop_at_step not in {1024, 2048}:
        raise ValueError("A8-V2 continuation requires explicit authorization and must stop at Step 1024 or 2048")
    config, _config_file = _load_config(config_path, "P")
    if config["arm"] != "P":
        raise ValueError("A8-V2 continuation is authorized for P only")
    preflight = _validate_preflight(config, run_id, preflight_root)
    dry = _validate_dry(config, run_id, preflight_root, dry_root)
    data_facts = _validate_prepared_data(config)

    _parent_preflight_root, _parent_dry_root, parent_run_root = _run_paths(config, parent_run_id)
    parent_manifest_path = parent_run_root / "manifest.json"
    parent_manifest = _read_json(parent_manifest_path)
    parent_step = parent_manifest.get("optimizer_steps")
    if parent_step not in {512, 1024} or stop_at_step != parent_step * 2:
        raise ValueError("A8-V2 continuation must double a reviewed Step 512 or Step 1024 checkpoint")
    parent_node = parent_manifest.get("nodes", {}).get(str(parent_step), {})
    parent_checkpoint = parent_run_root / "trainer" / f"checkpoint-{parent_step}"
    parent_checkpoint_manifest_path = parent_checkpoint / "checkpoint_manifest.json"
    parent_checkpoint_sha = sha256_file(parent_checkpoint_manifest_path)
    if (
        parent_manifest.get("run_id") != parent_run_id
        or parent_manifest.get("arm") != "P"
        or parent_manifest.get("config_sha256") != _config_sha256(config)
        or parent_manifest.get("dataset_manifest_sha256") != data_facts["dataset_manifest_sha256"]
        or parent_manifest.get("sample_order_sha256") != data_facts["schedule_sha256"]
        or parent_manifest.get("authorized_max_step") != parent_step
        or parent_node.get("training_status") != "passed"
        or parent_node.get("evaluation_status") != "complete"
        or parent_node.get("evaluation_engineering_gate") != "passed"
        or parent_node.get("checkpoint_manifest_sha256") != parent_checkpoint_sha
        or parent_manifest.get("checkpoint_manifest_sha256") != parent_checkpoint_sha
    ):
        raise ValueError("A8-V2 P parent Run is not an evaluated, verified continuation source")
    parent_checkpoint_record = _verify_v2_checkpoint(parent_checkpoint, {
        "run_id": parent_run_id,
        "arm": "P",
        "optimizer_step": parent_step,
        "config_sha256": parent_manifest["config_sha256"],
        "dataset_manifest_sha256": parent_manifest["dataset_manifest_sha256"],
        "sample_order_sha256": parent_manifest["sample_order_sha256"],
        "training_code_sha256": parent_manifest["training_code_sha256"],
    })
    parent_training_node = _read_json(parent_run_root / f"step-{parent_step:04d}" / "training.json")
    parent_tensor_sha = parent_training_node.get("independent_reload_check", {}).get("adapter_tensor_sha256_after_reload")
    parent_dry = _read_json(parent_run_root / "dry_step_gate.json")

    init_manifest = parent_manifest
    init_run_id = parent_run_id
    init_run_root = parent_run_root
    seen_parent_ids = {parent_run_id}
    while init_manifest.get("continuation"):
        init_run_id = init_manifest["continuation"].get("parent_run_id")
        if not isinstance(init_run_id, str) or init_run_id in seen_parent_ids:
            raise ValueError("A8-V2 continuation parent chain is invalid or cyclic")
        seen_parent_ids.add(init_run_id)
        _init_preflight, _init_dry, init_run_root = _run_paths(config, init_run_id)
        init_manifest = _read_json(init_run_root / "manifest.json")
    original_p_dry = _read_json(init_run_root / "dry_step_gate.json")
    sibling_ep_dry_path = _resolve("outputs/training/E03-A8-V2/EP") / f"{init_run_id}-dry" / "dry_step.json"
    sibling_ep_dry = _read_json(sibling_ep_dry_path)
    if (
        parent_dry.get("status") != "passed"
        or original_p_dry.get("status") != "passed"
        or sibling_ep_dry.get("status") != "passed"
        or dry.get("initial_lora_tensor_sha256") != original_p_dry.get("initial_lora_tensor_sha256")
        or parent_dry.get("initial_lora_tensor_sha256") != original_p_dry.get("initial_lora_tensor_sha256")
        or sibling_ep_dry.get("initial_lora_tensor_sha256") != original_p_dry.get("initial_lora_tensor_sha256")
    ):
        raise ValueError("A8-V2 continuation fresh-start dry-run differs from the paired P/EP initialization")
    if (
        parent_checkpoint_record.get("adapter_identity", {}).get("weight_sha256") is None
        or parent_training_node.get("independent_reload_check", {}).get("status") != "passed"
        or not isinstance(parent_tensor_sha, str)
        or parent_training_node.get("independent_reload_check", {}).get("adapter_tensor_sha256_before_save") != parent_tensor_sha
    ):
        raise ValueError("A8-V2 parent checkpoint has no matching Adapter tensor identity")

    _preflight_root, _dry_root, run_root = _run_paths(config, run_id)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite A8-V2 continuation Run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "trainer").mkdir()
    shutil.copy2((preflight_root if preflight_root.is_absolute() else ROOT / preflight_root) / "preflight.json", run_root / "preflight.json")
    shutil.copy2((dry_root if dry_root.is_absolute() else ROOT / dry_root) / "dry_step.json", run_root / "dry_step_gate.json")
    shutil.copy2(_resolve(config["data"]["sample_order_path"]), run_root / "sample_order.jsonl")
    _write_json(run_root / "config_snapshot.json", config, exclusive=True)
    _write_json(run_root / "manifest.json", {
        "status": "running",
        "stage": "continuation_initialized",
        "experiment_id": "E03-A8-V2",
        "training_scope": "generative_profile_specialist",
        "arm": "P",
        "run_id": run_id,
        "config_sha256": _config_sha256(config),
        "preflight_sha256": sha256_file(run_root / "preflight.json"),
        "dry_step_sha256": sha256_file(run_root / "dry_step_gate.json"),
        "training_code_sha256": preflight["training_code_sha256"],
        "evaluation_code_sha256": preflight["evaluation_code_sha256"],
        "dataset_manifest_sha256": data_facts["dataset_manifest_sha256"],
        "train_file_sha256": sha256_file(_resolve(config["data"]["train_path"])),
        "sample_order_sha256": data_facts["schedule_sha256"],
        "initial_lora_tensor_sha256": dry["initial_lora_tensor_sha256"],
        "performance_gate": _require_performance_gate(config, "P"),
        "max_optimizer_steps": 4048,
        "authorized_max_step": stop_at_step,
        "authorization_source": f"Explicit user instruction: continue P to {stop_at_step} optimizer steps",
        "recovery_state_persisted": True,
        "optimizer_steps": parent_step,
        "continuation": {
            "parent_run_id": parent_run_id,
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "parent_checkpoint_path": str(parent_checkpoint.relative_to(ROOT)),
            "parent_checkpoint_manifest_sha256": parent_checkpoint_sha,
            "parent_adapter_weight_sha256": parent_checkpoint_record["adapter_identity"]["weight_sha256"],
            "parent_adapter_tensor_sha256": parent_tensor_sha,
            "start_step_exclusive": parent_step,
            "stop_step_inclusive": stop_at_step,
            "sample_cursor_start": parent_step * int(config["training"]["gradient_accumulation_steps"]),
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_restored": True,
        },
        "nodes": {},
        "created_at_utc": _now(),
    }, exclusive=True)
    return _execute_training(
        config,
        run_id,
        run_root,
        preflight=preflight,
        data_facts=data_facts,
        dry=dry,
        continuation={
            "parent_run_id": parent_run_id,
            "parent_run_root": parent_run_root,
            "parent_training_node": parent_training_node,
            "parent_checkpoint": parent_checkpoint,
            "parent_checkpoint_manifest_sha256": parent_checkpoint_sha,
            "parent_adapter_weight_sha256": parent_checkpoint_record["adapter_identity"]["weight_sha256"],
            "parent_adapter_tensor_sha256": parent_tensor_sha,
            "start_step": parent_step,
            "stop_step": stop_at_step,
        },
    )


DEFAULT_CONFIGS = {
    "P": ROOT / "configs/training/e03_a8_v2_p_staged.yaml",
    "EP": ROOT / "configs/training/e03_a8_v2_ep_staged.yaml",
}
