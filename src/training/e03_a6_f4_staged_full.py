"""Staged, resumable E03-A6 F4 Full training built on the reviewed SFT Trainer."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from participant import run_inference as inference
from src.data.e03_a6_f4_full import prepare_f4_full_dataset, sha256_file
from src.data.sft_collator import SFTDataCollator
from src.training.e03_a6_f4_staged_contract import (
    EXPERIMENT_ID,
    MAX_STEPS,
    TRAINING_SCOPE,
    SampleOrderRecorder,
    canonical_sha256,
    validate_stage_authorization,
    validate_stage_transition,
    validate_staged_full_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/training/e03_a6_f4_staged_full.yaml"
DATA_CONFIG = PROJECT_ROOT / "configs/data/e03_a6_f4_full.yaml"
CODE_PATHS = (
    "configs/data/e03_a6_f4_full.yaml",
    "participant/configs/e03_a6_f4_inference_config.json",
    "configs/prompts/p_a6_f4.txt",
    "src/training/training_sft.py",
    "src/training/e03_a6_f4_staged_full.py",
    "src/training/e03_a6_f4_staged_contract.py",
    "src/data/e03_a6_f4_full.py",
    "src/data/e03_a6_f4_sft.py",
    "src/data/e03_a6_sft.py",
    "src/data/sft_collator.py",
    "participant/run_inference.py",
    "src/shared/f4_protocol.py",
    "src/shared/prompting.py",
    "scripts/e03_a6_f4_staged_train.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _load_config(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    path = _resolve(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("F4 staged-full config must be a YAML object")
    validate_staged_full_config(config)
    return config, path


def config_sha256(config: dict[str, Any]) -> str:
    return canonical_sha256(config)


def _code_identity() -> dict[str, str]:
    return {name: sha256_file(PROJECT_ROOT / name) for name in CODE_PATHS}


def _packages() -> dict[str, str | None]:
    output = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes", "tokenizers"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


def _train_ids(rows: list[dict[str, Any]]) -> list[str]:
    ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("F4 Train IDs must be present and unique")
    return ids


def _validate_dataset(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    from src.training.training_sft import validate_train_row

    dataset_path = _resolve(config["data"]["train_path"])
    manifest_path = _resolve(config["data"]["dataset_manifest"])
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("prepared F4 full dataset or Manifest is missing")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("version") != "E03-A6-F4-data-v1"
        or manifest.get("status") != "cpu_preparation_passed"
        or manifest.get("format_version") != "F4"
        or manifest.get("outputs", {}).get("train_count") != config["data"]["expected_train_rows"]
    ):
        raise ValueError("F4 dataset Manifest identity or row count changed")
    file_hashes = manifest.get("outputs", {}).get("files", {})
    expected_hash = file_hashes.get("train.jsonl")
    if expected_hash != sha256_file(dataset_path):
        raise ValueError("F4 Train file hash differs from its prepared Manifest")
    rows = _read_jsonl(dataset_path)
    if len(rows) != int(config["data"]["expected_train_rows"]):
        raise ValueError("F4 Train count differs from the frozen 16,192-row cohort")
    for row in rows:
        validate_train_row(row, expected_eos_token_id=int(config["protocol"]["eos_token_id"]), expected_split="train")
    ids = _train_ids(rows)
    if manifest.get("train_summary", {}).get("ids_sha256_ordered") != canonical_sha256(ids):
        # The preparation manifest uses the historical D0.3 lineage hash.
        from src.data.e03_a6_sft import _lineage_hash

        if manifest.get("train_summary", {}).get("ids_sha256_ordered") != _lineage_hash(ids):
            raise ValueError("F4 ordered Train ID hash differs from its Manifest")
    return dataset_path, rows, manifest


def _source_identity(config: dict[str, Any], rows: list[dict[str, Any]], data_manifest: dict[str, Any]) -> dict[str, Any]:
    ids = _train_ids(rows)
    source_paths = (
        "f2_train_path", "normalized_train_path", "f2_manifest", "d03_manifest", "d042_manifest",
    )
    inputs = {name: _resolve(config["data"][name]) for name in source_paths}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"F4 training source is missing: {name}={path}")
    current_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    prepared_inputs = data_manifest.get("inputs", {})
    alias = {
        "f2_train_path": "f2_train",
        "normalized_train_path": "normalized_train",
        "f2_manifest": "f2_manifest",
        "d03_manifest": "d03_manifest",
        "d042_manifest": "d042_manifest",
    }
    for name, source_key in alias.items():
        if prepared_inputs.get(source_key, {}).get("sha256") != current_hashes[name]:
            raise ValueError(f"F4 prepared data source changed after CPU encoding: {name}")
    for name in ("prompt_path", "selection_prompt_path", "runner_config"):
        path = _resolve(config["protocol"][name])
        if not path.is_file():
            raise FileNotFoundError(f"F4 protocol input is missing: {path}")
    if sha256_file(_resolve(config["protocol"]["prompt_path"])) != config["protocol"]["prompt_sha256"]:
        raise ValueError("P_A6_F4 prompt hash differs from the frozen config")
    if sha256_file(_resolve(config["protocol"]["selection_prompt_path"])) != config["protocol"]["p0_sha256"]:
        raise ValueError("protected P0 prompt hash differs from the frozen config")
    return {
        "train_sha256": sha256_file(_resolve(config["data"]["train_path"])),
        "train_rows": len(rows),
        "train_ids_ordered_sha256": canonical_sha256(ids),
        "source_hashes": current_hashes,
        "f4_data_manifest_sha256": sha256_file(_resolve(config["data"]["dataset_manifest"])),
        "d03_manifest_sha256": current_hashes["d03_manifest"],
        "d043_manifest_sha256": current_hashes["f2_manifest"],
        "sft_dev_sha256": "not_used",
        "selected_samples_sha256": canonical_sha256(ids),
        "p0_sha256": config["protocol"]["p0_sha256"],
        "prompt_sha256": config["protocol"]["prompt_sha256"],
        "tokenizer_signature_sha256": config["protocol"]["tokenizer_signature_sha256"],
        "chat_template_sha256": config["protocol"]["chat_template_sha256"],
    }


def _runtime_tokenizer(config: dict[str, Any]):
    runner_config = _read_json(_resolve(config["protocol"]["runner_config"]))
    tokenizer, facts = inference.load_runtime_tokenizer(runner_config, _resolve(config["base_model"]["local_path"]))
    if (
        facts.get("signature_sha256") != config["protocol"]["tokenizer_signature_sha256"]
        or facts.get("template_sha256") != config["protocol"]["chat_template_sha256"]
        or int(tokenizer.eos_token_id) != int(config["protocol"]["eos_token_id"])
    ):
        raise ValueError("runtime F4 Tokenizer or Chat Template differs from the locked D0.4.3 identity")
    return tokenizer, facts


def prepare_preflight(config_path: Path, run_id: str) -> Path:
    config, path = _load_config(config_path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}", run_id):
        raise ValueError("run-id must use 3-96 ASCII letters, digits, dots, underscores, or hyphens")
    dataset_dir = _resolve(config["data"]["train_path"]).parent
    if not dataset_dir.exists():
        data_manifest = prepare_f4_full_dataset(DATA_CONFIG)
        if data_manifest.get("status") != "cpu_preparation_passed":
            raise RuntimeError("F4 full-data CPU preparation did not pass")
    dataset_path, rows, data_manifest = _validate_dataset(config)
    source = _source_identity(config, rows, data_manifest)
    tokenizer, tokenizer_facts = _runtime_tokenizer(config)
    if source["train_rows"] // int(config["training"]["gradient_accumulation_steps"]) != MAX_STEPS:
        raise ValueError("the frozen DataLoader schedule does not equal the 4,048-step one-epoch plan")
    del tokenizer
    output_root = _resolve(config["data"]["output_root"])
    run_root = output_root / f"{run_id}-preflight"
    config_hash = config_sha256(config)
    base_identity = inference._weight_identity(
        _resolve(config["base_model"]["local_path"]),
        {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"]["revision"]},
    )
    if base_identity.get("observed_weight_hash") != config["base_model"]["expected_weight_sha256"]:
        raise ValueError("F4 Full Base weight SHA differs from the frozen E01-B Base")
    device = shutil.disk_usage(output_root)
    required_free = int(config["training"]["min_free_disk_gb"]) * 1024**3
    if device.free < required_free:
        raise OSError(f"insufficient free disk for retained F4 Full checkpoints: free={device.free}, required={required_free}")
    checks = {
        "data": {**source, "dataset_manifest_status": data_manifest["status"], "dataset_manifest_sha256": sha256_file(_resolve(config["data"]["dataset_manifest"])), "train_file_sha256": sha256_file(dataset_path)},
        "tokenizer": tokenizer_facts,
        "base_model": base_identity,
        "protocol": {"format_version": "F4", "prompt_sha256": config["protocol"]["prompt_sha256"], "p0_sha256": config["protocol"]["p0_sha256"], "history_policy": config["protocol"]["history_policy"], "P": 1280, "S": 2048, "packing": False, "padding_free": False},
        "schedule": {"train_rows": len(rows), "micro_batch": 1, "gradient_accumulation_steps": 4, "steps_per_epoch": len(rows) // 4, "planned_max_steps": MAX_STEPS, "epochs": 1, "sampler": "random", "seed": 42},
        "output": {"root": str(output_root), "free_bytes": device.free, "required_minimum_free_bytes": required_free, "checkpoint_limit": config["training"]["save_total_limit"]},
        "code": _code_identity(),
        "packages": _packages(),
        "official_val_read": False,
    }
    result = {
        "status": "passed",
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "training_scope": TRAINING_SCOPE,
        "created_at_utc": _now(),
        "config_path": str(path.relative_to(PROJECT_ROOT)),
        "config_sha256": config_hash,
        "checks": checks,
    }
    run_root.mkdir(parents=True, exist_ok=False)
    _write_json(run_root / "preflight.json", result, exclusive=True)
    _write_json(run_root / "manifest.json", {
        "status": "passed", "run_id": run_id, "experiment_id": EXPERIMENT_ID,
        "training_scope": TRAINING_SCOPE, "config_sha256": config_hash,
        "preflight_sha256": sha256_file(run_root / "preflight.json"),
    }, exclusive=True)
    _write_json(run_root / "config_snapshot.json", config, exclusive=True)
    return run_root


def _validate_gate_run(path: Path, run_id: str, config: dict[str, Any], stage: str) -> dict[str, Any]:
    root = path.resolve()
    output_root = _resolve(config["data"]["output_root"])
    if not root.is_relative_to(output_root):
        raise ValueError(f"{stage} Run must be under outputs/training")
    record = _read_json(root / ("preflight.json" if stage == "preflight" else "dry_step.json"))
    manifest = _read_json(root / "manifest.json")
    if record.get("status") != "passed" or manifest.get("status") != "passed":
        raise ValueError(f"F4 staged Full requires a passed {stage} gate")
    if manifest.get("run_id") != run_id or record.get("run_id") != run_id:
        raise ValueError(f"{stage} Run belongs to another run-id")
    if manifest.get("config_sha256") != config_sha256(config) or record.get("config_sha256") != config_sha256(config):
        raise ValueError(f"{stage} gate was produced from a different frozen config")
    expected_hash = manifest.get("preflight_sha256") if stage == "preflight" else manifest.get("dry_step_sha256")
    if expected_hash != sha256_file(root / ("preflight.json" if stage == "preflight" else "dry_step.json")):
        raise ValueError(f"{stage} result hash differs from its gate manifest")
    return record


def execute_dry_step(config_path: Path, run_id: str, preflight_root: Path, *, allow_staged_training: bool) -> Path:
    if not allow_staged_training:
        raise ValueError("GPU dry-run requires the explicit --allow-staged-training gate")
    config, path = _load_config(config_path)
    gate = _validate_gate_run(preflight_root, run_id, config, "preflight")
    output_root = _resolve(config["data"]["output_root"])
    dry_root = output_root / f"{run_id}-dry"
    dataset_path, rows, dataset_manifest = _validate_dataset(config)
    source = _source_identity(config, rows, dataset_manifest)
    if _code_identity() != gate["checks"].get("code"):
        raise ValueError("F4 staged training code changed after the CPU preflight")
    for key, value in gate["checks"]["data"].items():
        if key in source and source[key] != value:
            raise ValueError(f"F4 data changed after CPU preflight: {key}")
    base_path = _resolve(config["base_model"]["local_path"])
    base_before = inference._weight_identity(base_path, {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"]["revision"]})
    if base_before.get("observed_weight_hash") != config["base_model"]["expected_weight_sha256"]:
        raise ValueError("Base SHA changed before F4 GPU dry-step")
    dry_root.mkdir(parents=True, exist_ok=False)
    from src.training.training_sft import _load_train_model, _step_update, _trainable_parameters

    model = tokenizer = torch = None
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(config)
        model, torch, model_facts = _load_train_model(config, base_path)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("F4 dry-step requires CUDA and BF16 support")
        torch.cuda.reset_peak_memory_stats()
        trainable = _trainable_parameters(model)
        optimizer = torch.optim.AdamW(
            [parameter for _name, parameter in trainable],
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        evidence = _step_update(
            model, torch, optimizer, rows[:4], collator, accumulation_steps=4,
        )
        if not evidence.get("finite_gradients") or int(evidence.get("updated_element_count", 0)) <= 0:
            raise RuntimeError("F4 GPU dry-step did not produce finite gradients and LoRA updates")
        memory = evidence.get("cuda_memory", {})
        maximum_allocated = int(memory.get("allocator_peak_allocated_bytes", 0))
        device_total = int(memory.get("total_memory_bytes", 0))
        if not device_total or maximum_allocated > device_total * float(config["quantization"]["max_cuda_memory_fraction"]):
            raise RuntimeError("F4 GPU dry-step exceeded the frozen CUDA allocated-memory budget")
        base_after = inference._weight_identity(base_path, {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"]["revision"]})
        if base_after.get("observed_weight_hash") != base_before.get("observed_weight_hash"):
            raise RuntimeError("Base weight SHA changed during the isolated F4 dry-step")
        result = {
            "status": "passed",
            "run_id": run_id,
            "experiment_id": EXPERIMENT_ID,
            "training_scope": TRAINING_SCOPE,
            "created_at_utc": _now(),
            "config_sha256": config_sha256(config),
            "preflight_sha256": sha256_file(preflight_root / "preflight.json"),
            "dry_step_only": True,
            "formal_run_adapter_written": False,
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "base_weights_unchanged": True,
            "train_sample_ids": [row["sample_id"] for row in rows[:4]],
            "training_step": evidence,
            "model_facts": model_facts,
            "tokenizer_facts": tokenizer_facts,
            "cuda": {"name": torch.cuda.get_device_name(0), "runtime": torch.version.cuda, "bf16_supported": torch.cuda.is_bf16_supported(), "peak_memory": inference.cuda_memory_facts(torch)},
        }
        _write_json(dry_root / "dry_step.json", result, exclusive=True)
        _write_json(dry_root / "manifest.json", {
            "status": "passed", "run_id": run_id, "experiment_id": EXPERIMENT_ID,
            "training_scope": TRAINING_SCOPE, "config_sha256": config_sha256(config),
            "dry_step_sha256": sha256_file(dry_root / "dry_step.json"),
        }, exclusive=True)
    except BaseException as exc:
        _write_json(dry_root / "manifest.json", {
            "status": "failed", "run_id": run_id, "experiment_id": EXPERIMENT_ID,
            "training_scope": TRAINING_SCOPE, "config_sha256": config_sha256(config),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        try:
            import torch as torch_module

            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception:
            pass
    return dry_root


class _IdentitySFTCollator:
    def __init__(self, inner: SFTDataCollator):
        self.inner = inner

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.inner(features)
        batch["_e03_sample_ids"] = [str(row["sample_id"]) for row in features]
        return batch


def _cuda_identity(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("F4 staged training requires CUDA with BF16 support")
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "compute_capability": [int(props.major), int(props.minor)],
        "total_memory_bytes": int(props.total_memory),
        "torch_cuda_runtime": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _checkpoint_identity(config: dict[str, Any], config_hash: str, preflight: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    data = preflight["checks"]["data"]
    return {
        "experiment_id": EXPERIMENT_ID,
        "training_scope": TRAINING_SCOPE,
        "config_sha256": config_hash,
        "train_sha256": data["train_sha256"],
        "sft_dev_sha256": "not_used",
        "d03_manifest_sha256": data["d03_manifest_sha256"],
        "d043_manifest_sha256": data["d043_manifest_sha256"],
        "f4_data_manifest_sha256": data["f4_data_manifest_sha256"],
        "base_weight_sha256": config["base_model"]["expected_weight_sha256"],
        "p0_sha256": config["protocol"]["p0_sha256"],
        "prompt_sha256": config["protocol"]["prompt_sha256"],
        "tokenizer_signature_sha256": config["protocol"]["tokenizer_signature_sha256"],
        "chat_template_sha256": config["protocol"]["chat_template_sha256"],
        "sample_selection_sha256": data["train_ids_ordered_sha256"],
        "training_code_sha256": canonical_sha256(_code_identity()),
        "packages": preflight["checks"]["packages"],
        "cuda": cuda,
    }


def _verify_sample_order_checkpoint(
    checkpoint_record: dict[str, Any], order_path: Path, expected_train_ids: list[str], step: int,
) -> None:
    rows = _read_jsonl(order_path)
    prefix = rows[:step]
    ids = [sample_id for row in prefix for sample_id in row.get("sample_ids", [])]
    if len(prefix) != step or len(ids) != step * 4 or len(ids) != len(set(ids)):
        raise ValueError("sample-order evidence does not cover the saved checkpoint without repeats")
    if any(sample_id not in set(expected_train_ids) for sample_id in ids):
        raise ValueError("sample-order evidence contains a non-cohort ID")
    if (
        checkpoint_record.get("sample_order_steps") != step
        or checkpoint_record.get("sample_order_micro_batches") != step * 4
        or checkpoint_record.get("sample_order_ordered_ids_sha256") != canonical_sha256(ids)
    ):
        raise ValueError("checkpoint sample-order hashes do not match its persisted sample-order prefix")
    prefix_bytes = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in prefix).encode("utf-8")
    if hashlib.sha256(prefix_bytes).hexdigest() != checkpoint_record.get("sample_order_log_sha256"):
        raise ValueError("checkpoint sample-order log SHA differs from its persisted prefix")


def _verify_preflight_for_train(root: Path, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return _validate_gate_run(root, run_id, config, "preflight")


def _verify_dry_for_train(root: Path, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    record = _validate_gate_run(root, run_id, config, "dry-step")
    if record.get("base_weights_unchanged") is not True or record.get("formal_run_adapter_written") is not False:
        raise ValueError("F4 dry-step is not an isolated, Base-preserving gate")
    return record


def _prepare_resume_log(run_root: Path, step: int) -> None:
    from src.training.training_sft import _prepare_resume_logs

    _prepare_resume_logs(run_root, step)


def _run_training_segment(
    config: dict[str, Any], config_path: Path, run_root: Path, preflight: dict[str, Any],
    *, start_step: int, stop_at_step: int, resume_checkpoint: Path | None,
) -> Path:
    from src.training.training_sft import (
        _TrainingCounters,
        _build_sft_trainer,
        _load_train_model,
        _optimizer_evidence_callback,
        _read_training_log,
        _seed_everything,
        _trainable_parameters,
        _verify_checkpoint,
        _write_json as write_training_json,
        _write_run_manifest,
        _write_training_progress,
        _write_checkpoint_manifest,
        _save_adapter,
    )
    import torch

    config_hash = config_sha256(config)
    if _read_json(run_root / "config_snapshot.json") != config:
        raise ValueError("Run config snapshot differs from the frozen F4 staged config")
    dataset_path, rows, data_manifest = _validate_dataset(config)
    data_facts = _source_identity(config, rows, data_manifest)
    if _code_identity() != preflight["checks"].get("code"):
        raise ValueError("F4 staged training code changed after the CPU preflight")
    expected_preflight_data = preflight["checks"]["data"]
    for key in ("train_sha256", "train_rows", "train_ids_ordered_sha256", "f4_data_manifest_sha256", "d03_manifest_sha256", "d043_manifest_sha256"):
        if data_facts[key] != expected_preflight_data[key]:
            raise ValueError(f"F4 training inputs changed since CPU preflight: {key}")
    train_ids = _train_ids(rows)
    if resume_checkpoint is not None:
        if resume_checkpoint.name != f"checkpoint-{start_step}" or resume_checkpoint.parent != (run_root / "trainer").resolve():
            raise ValueError("resume requires an explicit checkpoint in this Run's trainer directory")
        if not (run_root / "train_log.jsonl").is_file():
            raise ValueError("resume checkpoint has no optimizer-step log")
        _prepare_resume_log(run_root, start_step)
        from src.training.e03_a6_f4_staged_contract import rewind_sample_order_log

        rewind_sample_order_log(run_root / "sample_order.jsonl", start_step, run_root)
    order_recorder = SampleOrderRecorder(run_root / "sample_order.jsonl", train_ids)
    if order_recorder.optimizer_steps != start_step:
        raise ValueError("sample-order log step differs from the requested checkpoint")
    base_path = _resolve(config["base_model"]["local_path"])
    base_before = inference._weight_identity(base_path, {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"]["revision"]})
    if base_before.get("observed_weight_hash") != config["base_model"]["expected_weight_sha256"]:
        raise ValueError("Base SHA changed before training segment")
    tokenizer, tokenizer_facts = _runtime_tokenizer(config)
    _seed_everything(int(config["training"]["seed"]), torch)
    model = None
    training_started = time.perf_counter()
    previous_training_seconds = float(_read_json(run_root / "manifest.json").get("training_seconds", 0.0))
    try:
        model, torch, model_facts = _load_train_model(config, base_path)
        cuda_identity = _cuda_identity(torch)
        run_manifest = _read_json(run_root / "manifest.json")
        frozen_cuda = run_manifest.get("cuda_identity")
        if frozen_cuda is not None and frozen_cuda != cuda_identity:
            raise ValueError("CUDA hardware identity changed within the staged F4 Run")
        checkpoint_facts = _checkpoint_identity(config, config_hash, preflight, cuda_identity)
        if resume_checkpoint is not None:
            checkpoint_record = _verify_checkpoint(
                resume_checkpoint,
                expected=checkpoint_facts,
                run_id=run_root.name,
            )
            _verify_sample_order_checkpoint(checkpoint_record, run_root / "sample_order.jsonl", train_ids, start_step)
        torch.cuda.reset_peak_memory_stats()
        base_collator = SFTDataCollator(
            tokenizer.pad_token_id,
            max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
        )
        collator = _IdentitySFTCollator(base_collator)
        counters = _TrainingCounters(_read_training_log(run_root))
        callback = _optimizer_evidence_callback(
            model,
            _trainable_parameters(model),
            config=config,
            samples=[],
            tokenizer=tokenizer,
            prompt="",
            schema=None,
            enums=None,
            run_root=run_root,
            snapshot_adapter=False,
            max_optimizer_steps=MAX_STEPS,
            collator=counters,
            checkpoint_facts=checkpoint_facts,
            stop_after_step=stop_at_step,
            previous_training_seconds=previous_training_seconds,
            sample_order_recorder=order_recorder,
        )
        trainer = _build_sft_trainer(
            model,
            tokenizer,
            rows,
            collator,
            config,
            max_steps=MAX_STEPS,
            callback=callback,
            run_root=run_root,
            eval_rows=None,
            training_counts=counters,
            sample_order_recorder=order_recorder,
        )
        data_path_facts = trainer.e01_data_path_facts
        if data_path_facts.get("train_micro_batches_per_epoch") != 16192 or data_path_facts.get("optimizer_steps_per_epoch") != MAX_STEPS:
            raise ValueError("actual Trainer DataLoader schedule differs from the frozen one-epoch plan")
        run_manifest.update(
            status="running",
            stage="training",
            optimizer_steps=start_step,
            training_scope=TRAINING_SCOPE,
            cuda_identity=cuda_identity,
            model_facts=model_facts,
            tokenizer_facts=tokenizer_facts,
            trainer_data_path=data_path_facts,
            checkpoint_identity=checkpoint_facts,
            current_stage={"from_step": start_step, "stop_at_step": stop_at_step},
        )
        _write_run_manifest(run_root, run_manifest)
        if stop_at_step <= start_step or stop_at_step not in (512, 1024, 2048, 4048):
            raise ValueError("staged training must stop at the next frozen F4 milestone")
        if previous_training_seconds >= float(config["training"]["max_training_seconds"]):
            raise ValueError("the frozen F4 Full training time budget has been exhausted")
        trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
        end_step = int(trainer.state.global_step)
        if end_step == MAX_STEPS:
            final_checkpoint = run_root / "trainer" / f"checkpoint-{MAX_STEPS}"
            if not final_checkpoint.is_dir():
                # 4,048 is the frozen one-epoch endpoint but does not align with
                # the 512-step save interval; persist the complete Trainer state.
                trainer._save_checkpoint(trainer.model, trial=None)
                _write_checkpoint_manifest(final_checkpoint, {
                    **checkpoint_facts,
                    **order_recorder.checkpoint_facts(MAX_STEPS),
                    "run_id": run_root.name,
                    "optimizer_step": MAX_STEPS,
                })
        attempt_seconds = time.perf_counter() - training_started
        run_manifest = _read_json(run_root / "manifest.json")
        run_manifest["training_seconds"] = previous_training_seconds + attempt_seconds
        run_manifest.setdefault("training_attempts", []).append({
            "starting_optimizer_step": start_step,
            "ending_optimizer_step": end_step,
            "stop_at_step": stop_at_step,
            "training_seconds": attempt_seconds,
            "stop_reason": callback.stop_reason,
            "cuda_memory": inference.cuda_memory_facts(torch),
        })
        log_rows = _read_training_log(run_root)
        if [int(row.get("optimizer_step", -1)) for row in log_rows] != list(range(1, end_step + 1)):
            raise ValueError("optimizer-step log is incomplete, duplicated, or out of order")
        if order_recorder.optimizer_steps != end_step:
            raise ValueError("actual DataLoader sample order and Trainer global_step differ")
        checkpoint = run_root / "trainer" / f"checkpoint-{end_step}"
        if end_step != stop_at_step or not checkpoint.is_dir():
            run_manifest.update(
                status="blocked", stage="checkpoint_not_at_requested_milestone",
                optimizer_steps=end_step, stop_reason=callback.stop_reason,
                ended_at_utc=_now(),
            )
            _write_run_manifest(run_root, run_manifest)
            _write_json(run_root / "blocked.json", {"status": "blocked", "expected_step": stop_at_step, "actual_step": end_step, "reason": callback.stop_reason})
            return run_root
        checkpoint_record = _verify_checkpoint(checkpoint, expected=checkpoint_facts, run_id=run_root.name)
        _verify_sample_order_checkpoint(checkpoint_record, run_root / "sample_order.jsonl", train_ids, end_step)
        retained = sorted(
            int(path.name.removeprefix("checkpoint-"))
            for path in (run_root / "trainer").glob("checkpoint-[0-9]*")
            if path.is_dir() and re.fullmatch(r"checkpoint-[0-9]+", path.name)
        )
        required_retained = [step for step in (512, 1024, 2048, 4048) if step <= end_step]
        if not set(required_retained) <= set(retained):
            raise RuntimeError(f"a required F4 milestone checkpoint was rotated: expected={required_retained}, retained={retained}")
        base_after = inference._weight_identity(base_path, {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"]["revision"]})
        if base_after.get("observed_weight_hash") != base_before.get("observed_weight_hash"):
            raise RuntimeError("Base weight SHA changed during the F4 staged training segment")
        stage_dir = run_root / f"step-{end_step:04d}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        training_log_at_node = [row for row in log_rows if int(row["optimizer_step"]) <= end_step]
        order_facts = order_recorder.checkpoint_facts(end_step)
        node = {
            "status": "training_checkpoint_passed",
            "run_id": run_root.name,
            "optimizer_step": end_step,
            "max_optimizer_steps": MAX_STEPS,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
            "adapter_checkpoint_weight_sha256": inference.adapter_weight_identity(checkpoint)[0]["weight_sha256"],
            "sample_order": order_facts,
            "training_samples_seen": counters.counts()["samples_seen"],
            "micro_batches_seen": counters.counts()["micro_batches_seen"],
            "supervised_tokens_seen": counters.counts()["completion_tokens_seen"],
            "input_tokens_seen": counters.counts()["input_tokens_seen"],
            "training_loss_start": training_log_at_node[0]["loss"],
            "training_loss_node": training_log_at_node[-1]["loss"],
            "segment_training_seconds": attempt_seconds,
            "cumulative_training_seconds": previous_training_seconds + attempt_seconds,
            "cuda_memory": inference.cuda_memory_facts(torch),
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "base_weights_unchanged": True,
            "required_checkpoints_retained": retained,
        }
        _write_json(stage_dir / "training.json", node, exclusive=True)
        final = end_step == MAX_STEPS
        if final:
            adapter_facts = _save_adapter(model, run_root / "adapter", config)
            node["adapter_export"] = adapter_facts
            _write_json(stage_dir / "training.json", node)
        run_manifest.update(
            status="completed" if final else "interrupted",
            stage="full_training_completed_awaiting_final_review" if final else "checkpoint_saved",
            optimizer_steps=end_step,
            stop_reason=callback.stop_reason or "requested_checkpoint_pause",
            last_checkpoint=str(checkpoint.resolve()),
            checkpoint_manifest_sha256=sha256_file(checkpoint / "checkpoint_manifest.json"),
            checkpoint_sample_order=order_facts,
            required_checkpoints_retained=retained,
            base_weight_sha256_after_training_segment=base_after.get("observed_weight_hash"),
            full_training_status="completed_awaiting_final_review" if final else "in_progress",
            ended_at_utc=_now(),
        )
        run_manifest.setdefault("nodes", {})[str(end_step)] = {
            "training_status": "passed",
            "training_summary": str((stage_dir / "training.json").relative_to(run_root)),
            "checkpoint_manifest_sha256": run_manifest["checkpoint_manifest_sha256"],
            "review_status": "awaiting_manual_review",
            "progression_authorization_status": "not_recorded",
        }
        _write_run_manifest(run_root, run_manifest)
        _write_training_progress(run_root, stage="training_checkpoint_saved", optimizer_step=end_step, max_optimizer_steps=MAX_STEPS)
        if final:
            order_recorder.verify_full_epoch()
        return run_root
    except BaseException as exc:
        run_manifest = _read_json(run_root / "manifest.json")
        run_manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            stage="training_interrupted" if isinstance(exc, KeyboardInterrupt) else "training_failed",
            error=f"{type(exc).__name__}: {exc}",
            ended_at_utc=_now(),
        )
        _write_run_manifest(run_root, run_manifest)
        raise
    finally:
        if model is not None:
            del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute_start(
    config_path: Path, run_id: str, preflight_root: Path, dry_root: Path,
    *, stop_at_step: int, allow_staged_training: bool,
) -> Path:
    if not allow_staged_training:
        raise ValueError("F4 staged GPU training requires the explicit --allow-staged-training flag")
    validate_stage_transition(None, stop_at_step)
    config, path = _load_config(config_path)
    preflight = _verify_preflight_for_train(preflight_root, run_id, config)
    dry = _verify_dry_for_train(dry_root, run_id, config)
    if dry.get("preflight_sha256") != sha256_file(preflight_root / "preflight.json"):
        raise ValueError("GPU dry-step was run against another CPU preflight")
    output_root = _resolve(config["data"]["output_root"])
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    config_hash = config_sha256(config)
    source_data = preflight["checks"]["data"]
    cuda = dry["cuda"]
    cuda_identity = {
        "available": True,
        "name": cuda["name"],
        "compute_capability": list(torch.cuda.get_device_capability(0)) if (torch := __import__("torch")).cuda.is_available() else None,
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None,
        "torch_cuda_runtime": cuda["runtime"],
        "bf16_supported": cuda["bf16_supported"],
    }
    _write_json(run_root / "config_snapshot.json", config, exclusive=True)
    _write_json(run_root / "preflight.json", preflight, exclusive=True)
    _write_json(run_root / "dry_step_gate.json", dry, exclusive=True)
    manifest = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "training_scope": TRAINING_SCOPE,
        "status": "initialized",
        "stage": "initialized",
        "created_at_utc": _now(),
        "config_path": str(path.relative_to(PROJECT_ROOT)),
        "config_sha256": config_hash,
        "preflight_sha256": sha256_file(run_root / "preflight.json"),
        "dry_step_sha256": sha256_file(run_root / "dry_step_gate.json"),
        "preflight_data": source_data,
        "max_optimizer_steps": MAX_STEPS,
        "optimizer_state_resume_enabled": True,
        "init_adapter": None,
        "resume_history": [],
        "training_seconds": 0.0,
        "cuda_identity": cuda_identity,
        "checkpoints": [],
        "nodes": {},
    }
    _write_json(run_root / "manifest.json", manifest, exclusive=True)
    (run_root / "trainer").mkdir()
    return _run_training_segment(
        config, path, run_root, preflight,
        start_step=0, stop_at_step=stop_at_step, resume_checkpoint=None,
    )


def execute_resume(
    config_path: Path,
    run_id: str,
    *,
    from_step: int,
    approved_from_step: int,
    stop_at_step: int,
    approval_file: Path,
    allow_staged_training: bool,
) -> Path:
    if not allow_staged_training:
        raise ValueError("F4 staged resume requires the explicit --allow-staged-training flag")
    if approved_from_step not in (512, 1024, 2048):
        raise ValueError("approved-from-step must be one of 512, 1024, or 2048")
    expected_target = {512: 1024, 1024: 2048, 2048: 4048}[approved_from_step]
    if stop_at_step != expected_target or not approved_from_step <= from_step < stop_at_step:
        raise ValueError("resume stage target or checkpoint interval differs from its approved milestone")
    config, _path = _load_config(config_path)
    run_root = _resolve(config["data"]["output_root"]) / run_id
    manifest = _read_json(run_root / "manifest.json")
    if manifest.get("run_id") != run_id or manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("training_scope") != TRAINING_SCOPE:
        raise ValueError("resume Run identity differs from the requested staged F4 Run")
    if manifest.get("config_sha256") != config_sha256(config):
        raise ValueError("resume config differs from the frozen same-Run config")
    if manifest.get("status") not in {"interrupted", "failed", "blocked", "awaiting_manual_review"}:
        raise ValueError("resume requires the same Run to be paused or interrupted at a checkpoint")
    checkpoint = run_root / "trainer" / f"checkpoint-{from_step}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"requested F4 checkpoint does not exist: {checkpoint}")
    approval_path = _resolve(approval_file)
    if not approval_path.is_file():
        raise FileNotFoundError("resume requires a node-specific approval receipt")
    approved_checkpoint = run_root / "trainer" / f"checkpoint-{approved_from_step}"
    if not approved_checkpoint.is_dir():
        raise FileNotFoundError("approved node checkpoint is missing")
    validate_stage_authorization(
        approval_path,
        run_id=run_id,
        from_step=approved_from_step,
        to_step=stop_at_step,
        checkpoint_manifest_sha256=sha256_file(approved_checkpoint / "checkpoint_manifest.json"),
    )
    newer = [
        path for path in (run_root / "trainer").glob("checkpoint-[0-9]*")
        if path.is_dir() and re.fullmatch(r"checkpoint-[0-9]+", path.name)
        and int(path.name.removeprefix("checkpoint-")) > from_step
    ]
    if newer:
        raise ValueError("resume must use the latest verified checkpoint; newer checkpoint directories exist")
    preflight = _read_json(run_root / "preflight.json")
    if preflight.get("status") != "passed" or manifest.get("preflight_sha256") != sha256_file(run_root / "preflight.json"):
        raise ValueError("same-Run CPU preflight is missing or changed")
    checkpoint_identity = manifest.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, dict):
        raise ValueError("same-Run checkpoint identity is missing")
    manifest.setdefault("resume_history", []).append({
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
        "optimizer_step": from_step,
        "approved_from_step": approved_from_step,
        "approved_to_step": stop_at_step,
        "approval_file": str(approval_path),
        "authorization_source": _read_json(approval_path).get("authorization_source"),
        "resumed_at_utc": _now(),
    })
    manifest.update(status="running", stage="resuming", resume_from_checkpoint=str(checkpoint.resolve()))
    if str(approved_from_step) in manifest.get("nodes", {}):
        manifest["nodes"][str(approved_from_step)]["progression_authorization_status"] = "preauthorized_by_user_instruction"
        manifest["nodes"][str(approved_from_step)]["approval_file"] = str(approval_path)
        manifest["nodes"][str(approved_from_step)]["approval_sha256"] = sha256_file(approval_path)
    _write_json(run_root / "manifest.json", manifest)
    return _run_training_segment(
        config, _resolve(config_path), run_root, preflight,
        start_step=from_step, stop_at_step=stop_at_step, resume_checkpoint=checkpoint.resolve(),
    )


def status_run(config_path: Path, run_id: str) -> dict[str, Any]:
    config, _ = _load_config(config_path)
    root = _resolve(config["data"]["output_root"]) / run_id
    manifest = _read_json(root / "manifest.json")
    if manifest.get("run_id") != run_id or manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("status Run ID does not match the requested F4 staged Full")
    checkpoints = []
    for path in sorted((root / "trainer").glob("checkpoint-[0-9]*"), key=lambda item: int(item.name.removeprefix("checkpoint-"))):
        if path.is_dir() and (path / "checkpoint_manifest.json").is_file():
            record = _read_json(path / "checkpoint_manifest.json")
            checkpoints.append({"step": int(path.name.removeprefix("checkpoint-")), "status": record.get("status"), "manifest_sha256": sha256_file(path / "checkpoint_manifest.json")})
    return {**manifest, "observed_checkpoints": checkpoints}
