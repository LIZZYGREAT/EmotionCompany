"""Reproducible E03-A8 QLoRA stages with complete same-Run recovery state."""

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
from src.data.sft_collator import SFTDataCollator
from src.training.e03_a8_contract import (
    ACCUMULATION_STEPS,
    AUTHORIZED_MAX_STEP,
    DATA_CODE_PATHS,
    EVALUATION_CODE_PATHS,
    EXPERIMENT_ID,
    TRAINING_CODE_PATHS,
    TRAINING_SCOPE,
    canonical_sha256,
    conditional_quality_decision,
    sha256_file,
    validate_a8_training_config,
    validate_conditional_authorization,
    validate_stage_transition,
    write_conditional_authorization,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = {
    "P": ROOT / "configs/training/e03_a8_p_staged.yaml",
    "EP": ROOT / "configs/training/e03_a8_ep_staged.yaml",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"A8 data/output path escapes the project root: {value}")
    return resolved


def _base_path(config: dict[str, Any]) -> Path:
    # Resolve through the repository's known local model symlink, which points to
    # the immutable pinned Hugging Face snapshot outside the checkout.
    return ROOT / config["base_model"]["local_path"]


def _load_config(config_path: Path, arm: str | None = None) -> tuple[dict[str, Any], Path]:
    path = config_path if config_path.is_absolute() else ROOT / config_path
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    selected_arm = arm or config.get("arm")
    validate_a8_training_config(config, selected_arm)
    return config, path


def config_sha256(config: dict[str, Any]) -> str:
    return canonical_sha256(config)


def _code_identity(paths: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"A8 code identity path is missing: {name}")
        result[name] = sha256_file(path)
    return result


def _package_versions() -> dict[str, str | None]:
    import importlib.metadata

    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class JsonlById:
    """Small byte-offset index for random access to ignored, pre-encoded rows."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("rb")
        self.offsets: dict[str, int] = {}
        while True:
            offset = self.handle.tell()
            line = self.handle.readline()
            if not line:
                break
            if not line.strip():
                self.close()
                raise ValueError(f"blank JSONL row in {path}")
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in self.offsets:
                self.close()
                raise ValueError(f"missing/duplicate sample_id in {path}")
            self.offsets[sample_id] = offset

    def get(self, sample_id: str) -> dict[str, Any]:
        if sample_id not in self.offsets:
            raise KeyError(f"unknown A8 sample_id: {sample_id}")
        self.handle.seek(self.offsets[sample_id])
        row = json.loads(self.handle.readline())
        if row.get("sample_id") != sample_id:
            raise RuntimeError("A8 byte-offset index points at a different sample")
        return row

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self) -> "JsonlById":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _row_contract(row: dict[str, Any], *, eos_token_id: int, prompt_cap: int, sequence_cap: int) -> None:
    required = {
        "sample_id", "conversation_id", "source_split", "source", "prompt_length",
        "completion_length", "input_ids", "labels", "common_history_sha256",
    }
    if set(row) != required:
        raise ValueError(f"A8 row fields differ from the locked contract: {sorted(row)}")
    if row.get("source_split") != "train":
        raise ValueError("A8 training data contains a non-Train row")
    ids, labels = row.get("input_ids"), row.get("labels")
    prompt_length, completion_length = row.get("prompt_length"), row.get("completion_length")
    if not isinstance(ids, list) or not ids or not isinstance(labels, list) or len(ids) != len(labels):
        raise ValueError(f"{row.get('sample_id')}: malformed pre-encoded input/labels")
    if (
        not isinstance(prompt_length, int) or isinstance(prompt_length, bool)
        or not isinstance(completion_length, int) or isinstance(completion_length, bool)
        or prompt_length <= 0 or completion_length <= 0 or prompt_length + completion_length != len(ids)
    ):
        raise ValueError(f"{row.get('sample_id')}: invalid A8 prompt/completion lengths")
    if prompt_length > prompt_cap or len(ids) > sequence_cap:
        raise ValueError(f"{row.get('sample_id')}: A8 row exceeds frozen P/S limits")
    if labels[:prompt_length] != [-100] * prompt_length or labels[prompt_length:] != ids[prompt_length:]:
        raise ValueError(f"{row.get('sample_id')}: Prompt mask or Completion labels changed")
    if eos_token_id not in labels[prompt_length:]:
        raise ValueError(f"{row.get('sample_id')}: supervised Completion lacks EOS")
    fingerprint = row.get("common_history_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError(f"{row.get('sample_id')}: invalid common-history fingerprint")


def _validate_prepared_data(config: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]:
    data_root = _resolve(config["data"]["dataset_root"])
    manifest_path = _resolve(config["data"]["dataset_manifest"])
    manifest = _read_json(manifest_path)
    if manifest.get("version") != "E03-A8-data-v1" or manifest.get("status") != "cpu_preparation_passed":
        raise ValueError("A8 prepared dataset manifest is not in passed state")
    if manifest.get("counts", {}).get("effective_train") != 16192 or manifest.get("schedule", {}).get("steps") != 4048:
        raise ValueError("A8 prepared dataset count or frozen schedule differs")
    for key, record in manifest.get("inputs", {}).items():
        path = _resolve(record["path"])
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"A8 source input changed after dataset construction: {key}")
    for name, expected in manifest.get("code_sha256", {}).items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"A8 data builder/protocol code changed after data freeze: {name}")
    p_path = data_root / "p_train.jsonl"
    ep_path = data_root / "ep_train.jsonl"
    audit_path = data_root / "common_history_audit.jsonl"
    order_path = _resolve(config["data"]["sample_order_path"])
    for name, path in (("p_train.jsonl", p_path), ("ep_train.jsonl", ep_path), ("common_history_audit.jsonl", audit_path), ("sample_order.jsonl", order_path)):
        if sha256_file(path) != manifest.get("files", {}).get(name):
            raise ValueError(f"A8 prepared data file hash differs: {name}")
    selected_path = _resolve(config["data"]["train_path"])
    if selected_path not in {p_path.resolve(), ep_path.resolve()}:
        raise ValueError("A8 arm config does not point at its matching prepared dataset")
    ids: list[str] = []
    history_hashes: list[str] = []
    prompt_min = prompt_max = None
    sequence_max = 0
    completion_tokens = 0
    with p_path.open("r", encoding="utf-8") as p_handle, ep_path.open("r", encoding="utf-8") as ep_handle:
        for line_no, (p_line, ep_line) in enumerate(zip(p_handle, ep_handle), 1):
            p_row, ep_row = json.loads(p_line), json.loads(ep_line)
            if p_row.get("sample_id") != ep_row.get("sample_id") or p_row.get("conversation_id") != ep_row.get("conversation_id"):
                raise ValueError(f"A8 P/EP ID or conversation mismatch at row {line_no}")
            if p_row.get("common_history_sha256") != ep_row.get("common_history_sha256"):
                raise ValueError(f"A8 P/EP common history mismatch at row {line_no}")
            for row in (p_row, ep_row):
                _row_contract(row, eos_token_id=int(config["protocol"]["eos_token_id"]), prompt_cap=1280, sequence_cap=2048)
            sample_id = p_row["sample_id"]
            if sample_id in ids:
                raise ValueError("A8 paired dataset repeats a Train ID")
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
        if p_handle.readline() or ep_handle.readline():
            raise ValueError("A8 P/EP datasets have different row counts")
    if len(ids) != 16192 or canonical_sha256(ids) != manifest.get("sample_ids_ordered_sha256"):
        raise ValueError("A8 ordered Train ID hash/count differs from the prepared data Manifest")
    if canonical_sha256(history_hashes) != manifest.get("counts", {}).get("common_history_sha256_ordered"):
        raise ValueError("A8 ordered common-history hash differs from the prepared data Manifest")
    order = _read_jsonl(order_path)
    if len(order) != 4048 or any(row.get("optimizer_step") != index + 1 or len(row.get("sample_ids", [])) != 4 for index, row in enumerate(order)):
        raise ValueError("A8 frozen schedule does not contain four IDs at each of 4,048 ordered steps")
    flattened = [sample_id for row in order for sample_id in row["sample_ids"]]
    if len(flattened) != 16192 or len(set(flattened)) != 16192 or set(flattened) != set(ids):
        raise ValueError("A8 frozen schedule repeats, omits, or adds a Train ID")
    if canonical_sha256(order) != manifest.get("schedule", {}).get("schedule_sha256"):
        raise ValueError("A8 frozen schedule hash differs from its Manifest")
    return {
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "data_manifest": manifest,
        "train_rows": len(ids),
        "train_ids_ordered_sha256": canonical_sha256(ids),
        "common_history_sha256_ordered": canonical_sha256(history_hashes),
        "schedule_sha256": canonical_sha256(order),
        "sample_order_ids_sha256": canonical_sha256(flattened),
        "prompt_length_min": prompt_min,
        "prompt_length_max": prompt_max,
        "sequence_length_max": sequence_max,
        "completion_tokens_both_arms": completion_tokens,
        "schedule_rows": len(order),
        "schedule": order,
    }


def _runtime_tokenizer(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    runner_config = _read_json(_resolve(config["protocol"]["runner_config"]))
    tokenizer, facts = inference.load_runtime_tokenizer(runner_config)
    expected = {
        "signature_sha256": config["protocol"]["tokenizer_signature_sha256"],
        "template_sha256": config["protocol"]["chat_template_sha256"],
        "revision": config["base_model"]["revision"],
    }
    observed = {
        "signature_sha256": facts.get("signature_sha256"),
        "template_sha256": facts.get("template_sha256"),
        "revision": facts.get("revision"),
    }
    if observed != expected or int(tokenizer.eos_token_id) != int(config["protocol"]["eos_token_id"]):
        raise ValueError(f"A8 runtime Tokenizer/Chat Template identity changed: {observed}")
    return tokenizer, facts


def _base_identity(config: dict[str, Any]) -> dict[str, Any]:
    facts = inference._weight_identity(
        _base_path(config),
        {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"]["revision"],
        },
    )
    if facts.get("observed_weight_hash") != config["base_model"]["expected_weight_sha256"]:
        raise ValueError("A8 Base weights differ from the pinned E01-B Base SHA")
    return facts


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lora_parameters(model: Any) -> list[tuple[str, Any]]:
    values = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not values or any(".lora_A." not in name and ".lora_B." not in name for name, _ in values):
        raise RuntimeError("only LoRA A/B parameters may be trainable in A8")
    return values


def _lora_tensor_sha256(model: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    entries = []
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        tensor = parameter.detach().cpu().contiguous()
        entries.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()})
    entries.sort(key=lambda value: value["name"])
    digest.update(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _gradient_facts(parameters: list[tuple[str, Any]], torch: Any) -> dict[str, Any]:
    gradients = [(name, parameter.grad) for name, parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("A8 generated no LoRA gradients")
    finite = all(bool(torch.isfinite(gradient).all().item()) for _name, gradient in gradients)
    nonzero = sum(bool(torch.count_nonzero(gradient).item()) for _name, gradient in gradients)
    norm_square = sum(torch.sum(gradient.detach().float() ** 2) for _name, gradient in gradients)
    norm = torch.sqrt(norm_square)
    if not finite or nonzero == 0 or not bool(torch.isfinite(norm).item()) or float(norm.item()) == 0.0:
        raise RuntimeError("A8 LoRA gradients are non-finite or zero")
    return {"finite": finite, "nonzero_parameter_gradients": nonzero, "global_norm": float(norm.detach().cpu().item())}


def _backward_scaled_completion_loss(loss: Any) -> None:
    (loss / ACCUMULATION_STEPS).backward()


def _run_paths(config: dict[str, Any], run_id: str) -> tuple[Path, Path, Path]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}", run_id) is None:
        raise ValueError("A8 run-id must use 3-96 ASCII letters, digits, dots, underscores, or hyphens")
    root = _resolve(config["data"]["output_root"])
    return root / f"{run_id}-preflight", root / f"{run_id}-dry", root / run_id


def prepare_preflight(config_path: Path, run_id: str) -> Path:
    config, config_file = _load_config(config_path)
    preflight_root, _dry_root, _run_root = _run_paths(config, run_id)
    if preflight_root.exists():
        raise FileExistsError(f"refusing to overwrite A8 preflight: {preflight_root}")
    tokenizer, tokenizer_facts = _runtime_tokenizer(config)
    data_facts = _validate_prepared_data(config, tokenizer)
    for name, expected in data_facts["data_manifest"].get("prompts", {}).items():
        if sha256_file(ROOT / expected["path"]) != expected["sha256"]:
            raise ValueError(f"A8 {name} prompt changed after data preparation")
    prompt_path = _resolve(config["protocol"]["prompt_path"])
    if sha256_file(prompt_path) != config["protocol"]["prompt_sha256"]:
        raise ValueError("A8 training Prompt hash differs from the frozen config")
    p0_path = _resolve(config["protocol"]["selection_prompt_path"])
    if sha256_file(p0_path) != config["protocol"]["p0_sha256"]:
        raise ValueError("protected P0 Prompt hash changed")
    schema_path = ROOT / "data/train/train_public.schema.json"
    schema_hash = sha256_file(schema_path)
    if schema_hash != "a9467e520b06a96859acaa2ff022e395d08ff6ad5e8ae3532a22cc24c630cada":
        raise ValueError("official Train Schema differs from the protected repository baseline")
    base = _base_identity(config)
    output_root = _resolve(config["data"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_root)
    minimum_free = int(config["training"]["min_free_disk_gb"]) * 1024**3
    if disk.free < minimum_free:
        raise OSError(f"A8 output root has {disk.free} free bytes; {minimum_free} required")
    result = {
        "status": "passed",
        "experiment_id": EXPERIMENT_ID,
        "training_scope": TRAINING_SCOPE,
        "arm": config["arm"],
        "run_id": run_id,
        "created_at_utc": _now(),
        "config_path": str(config_file.relative_to(ROOT)),
        "config_sha256": config_sha256(config),
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
        "training_code_sha256": _code_identity(TRAINING_CODE_PATHS),
        "evaluation_code_sha256": _code_identity(EVALUATION_CODE_PATHS),
        "runtime_packages": _package_versions(),
        "output": {"free_bytes": disk.free, "required_minimum_free_bytes": minimum_free},
        "official_val_read": False,
    }
    preflight_root.mkdir(parents=True, exist_ok=False)
    _write_json(preflight_root / "preflight.json", result, exclusive=True)
    _write_json(preflight_root / "manifest.json", {
        "status": "passed", "run_id": run_id, "arm": config["arm"],
        "config_sha256": config_sha256(config),
        "preflight_sha256": sha256_file(preflight_root / "preflight.json"),
    }, exclusive=True)
    _write_json(preflight_root / "config_snapshot.json", config, exclusive=True)
    return preflight_root


def _validate_preflight(config: dict[str, Any], run_id: str, root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"A8 CPU preflight is missing: {root}")
    record = _read_json(root / "preflight.json")
    manifest = _read_json(root / "manifest.json")
    if (
        record.get("status") != "passed"
        or manifest.get("status") != "passed"
        or record.get("run_id") != run_id
        or record.get("arm") != config["arm"]
        or manifest.get("run_id") != run_id
        or manifest.get("arm") != config["arm"]
        or record.get("config_sha256") != config_sha256(config)
        or manifest.get("config_sha256") != config_sha256(config)
        or manifest.get("preflight_sha256") != sha256_file(root / "preflight.json")
    ):
        raise ValueError("A8 CPU preflight identity/hash is incomplete or differs")
    if record.get("training_code_sha256") != _code_identity(TRAINING_CODE_PATHS):
        raise ValueError("A8 training code changed after preflight")
    if record.get("evaluation_code_sha256") != _code_identity(EVALUATION_CODE_PATHS):
        raise ValueError("A8 evaluation code changed after preflight")
    current_data = _validate_prepared_data(config)
    if record.get("data", {}).get("dataset_manifest_sha256") != current_data["dataset_manifest_sha256"]:
        raise ValueError("A8 prepared dataset changed after CPU preflight")
    return record


def _first_microbatch(config: dict[str, Any], schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with JsonlById(_resolve(config["data"]["train_path"])) as dataset:
        return [dataset.get(sample_id) for sample_id in schedule[0]["sample_ids"]]


def execute_gpu_dry_step(
    config_path: Path,
    run_id: str,
    preflight_root: Path,
    *,
    allow_gpu_dry_run: bool,
) -> Path:
    if not allow_gpu_dry_run:
        raise ValueError("A8 GPU dry-run requires --allow-gpu-dry-run")
    config, _config_file = _load_config(config_path)
    preflight = _validate_preflight(config, run_id, preflight_root)
    dry_root = _run_paths(config, run_id)[1]
    if dry_root.exists():
        raise FileExistsError(f"refusing to overwrite A8 GPU dry-run: {dry_root}")
    data_facts = _validate_prepared_data(config)
    rows = _first_microbatch(config, data_facts["schedule"])
    dry_root.mkdir(parents=True, exist_ok=False)
    model = tokenizer = torch = None
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(config)
        torch = __import__("torch")
        _set_seed(int(config["training"]["seed"]), torch)
        from src.training.training_sft import _load_train_model

        model, torch, model_facts = _load_train_model(config, _base_path(config))
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A8 GPU dry-run requires CUDA and BF16")
        initial_sha = _lora_tensor_sha256(model, torch)
        parameters = _lora_parameters(model)
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        torch.cuda.reset_peak_memory_stats()
        model.train()
        model.zero_grad(set_to_none=True)
        losses, token_count = [], 0
        for row in rows:
            batch = collator([row])
            device = model.get_input_embeddings().weight.device
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("A8 GPU dry-run produced non-finite Completion loss")
            losses.append(float(loss.detach().float().item()))
            token_count += row["completion_length"]
            (loss / 4).backward()
        gradients = _gradient_facts(parameters, torch)
        model.zero_grad(set_to_none=True)
        after_sha = _lora_tensor_sha256(model, torch)
        if initial_sha != after_sha:
            raise RuntimeError("A8 GPU dry-run changed LoRA parameters despite no Optimizer update")
        memory = inference.cuda_memory_facts(torch)
        if int(memory.get("allocator_peak_allocated_bytes", 0)) > int(memory.get("total_memory_bytes", 0)) * float(config["quantization"]["max_cuda_memory_fraction"]):
            raise RuntimeError("A8 GPU dry-run exceeded the frozen 92% CUDA allocator budget")
        if not model_facts.get("real_nf4_verified") or model_facts.get("offload_devices"):
            raise RuntimeError("A8 GPU dry-run did not use verified NF4/BF16 CUDA placement")
        result = {
            "status": "passed",
            "run_id": run_id,
            "arm": config["arm"],
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": config_sha256(config),
            "preflight_sha256": sha256_file(preflight_root / "preflight.json"),
            "data_manifest_sha256": preflight["data"]["dataset_manifest_sha256"],
            "dry_step_only": True,
            "optimizer_update_performed": False,
            "initial_lora_tensor_sha256": initial_sha,
            "post_backward_lora_tensor_sha256": after_sha,
            "sample_ids": [row["sample_id"] for row in rows],
            "completion_tokens": token_count,
            "mean_micro_loss": sum(losses) / len(losses),
            "gradient_check": gradients,
            "tokenizer": tokenizer_facts,
            "model": model_facts,
            "cuda_memory": memory,
            "created_at_utc": _now(),
        }
        _write_json(dry_root / "dry_step.json", result, exclusive=True)
        _write_json(dry_root / "manifest.json", {
            "status": "passed", "run_id": run_id, "arm": config["arm"],
            "config_sha256": config_sha256(config),
            "preflight_sha256": result["preflight_sha256"],
            "dry_step_sha256": sha256_file(dry_root / "dry_step.json"),
        }, exclusive=True)
        return dry_root
    except BaseException as exc:
        _write_json(dry_root / "dry_step.json", {
            "status": "failed", "run_id": run_id, "arm": config["arm"],
            "error": f"{type(exc).__name__}: {exc}", "created_at_utc": _now(),
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


def _validate_dry_step(config: dict[str, Any], run_id: str, preflight_root: Path, dry_root: Path) -> dict[str, Any]:
    record = _validate_preflight(config, run_id, preflight_root)
    dry = _read_json(dry_root / "dry_step.json")
    manifest = _read_json(dry_root / "manifest.json")
    if (
        dry.get("status") != "passed"
        or manifest.get("status") != "passed"
        or dry.get("arm") != config["arm"]
        or dry.get("run_id") != run_id
        or dry.get("config_sha256") != config_sha256(config)
        or dry.get("preflight_sha256") != sha256_file(preflight_root / "preflight.json")
        or manifest.get("run_id") != run_id
        or manifest.get("arm") != config["arm"]
        or manifest.get("config_sha256") != config_sha256(config)
        or manifest.get("preflight_sha256") != dry.get("preflight_sha256")
        or manifest.get("dry_step_sha256") != sha256_file(dry_root / "dry_step.json")
        or dry.get("optimizer_update_performed") is not False
        or dry.get("initial_lora_tensor_sha256") != dry.get("post_backward_lora_tensor_sha256")
    ):
        raise ValueError("A8 GPU dry-step gate is incomplete or failed its identity checks")
    expected_ids = _read_jsonl(_resolve(config["data"]["sample_order_path"]))[0]["sample_ids"]
    gradient = dry.get("gradient_check", {})
    memory = dry.get("cuda_memory", {})
    if (
        dry.get("sample_ids") != expected_ids
        or int(dry.get("completion_tokens", 0)) <= 0
        or not math.isfinite(float(dry.get("mean_micro_loss", float("nan"))))
        or gradient.get("finite") is not True
        or int(gradient.get("nonzero_parameter_gradients", 0)) <= 0
        or not math.isfinite(float(gradient.get("global_norm", float("nan"))))
        or float(gradient.get("global_norm", 0.0)) <= 0.0
        or dry.get("model", {}).get("real_nf4_verified") is not True
        or dry.get("model", {}).get("offload_devices")
        or int(memory.get("allocator_peak_allocated_bytes", 0))
        > int(memory.get("total_memory_bytes", 0)) * float(config["quantization"]["max_cuda_memory_fraction"])
    ):
        raise ValueError("A8 GPU dry-step failed a loss, gradient, sample-order, NF4, or memory invariant")
    other_arm = "EP" if config["arm"] == "P" else "P"
    other_config, _other_config_path = _load_config(DEFAULT_CONFIGS[other_arm], other_arm)
    other_preflight_root = _run_paths(other_config, run_id)[0]
    _validate_preflight(other_config, run_id, other_preflight_root)
    other_root = _run_paths(other_config, run_id)[1]
    other_path = other_root / "dry_step.json"
    if not other_path.is_file():
        raise FileNotFoundError(f"paired A8 {other_arm} GPU dry-run is required before training")
    other = _read_json(other_path)
    other_manifest = _read_json(other_root / "manifest.json")
    if (
        other.get("status") != "passed"
        or other.get("run_id") != run_id
        or other.get("arm") != other_arm
        or other.get("config_sha256") != config_sha256(other_config)
        or other.get("preflight_sha256") != sha256_file(other_preflight_root / "preflight.json")
        or other.get("data_manifest_sha256") != record["data"]["dataset_manifest_sha256"]
        or other.get("sample_ids") != expected_ids
        or other.get("optimizer_update_performed") is not False
        or other.get("initial_lora_tensor_sha256") != dry.get("initial_lora_tensor_sha256")
        or other_manifest.get("status") != "passed"
        or other_manifest.get("run_id") != run_id
        or other_manifest.get("arm") != other_arm
        or other_manifest.get("config_sha256") != config_sha256(other_config)
        or other_manifest.get("dry_step_sha256") != sha256_file(other_path)
    ):
        raise ValueError("P and EP do not share the verified fresh LoRA initialization")
    if record.get("data", {}).get("dataset_manifest_sha256") != dry.get("data_manifest_sha256"):
        raise ValueError("A8 dry-run dataset identity differs from its preflight")
    return dry


def _rng_state(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any], torch: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda_all"):
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _checkpoint_files(checkpoint: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"A8 checkpoint contains a symlink: {path}")
        if path.is_file() and path.name != "checkpoint_manifest.json":
            files.append({"path": path.relative_to(checkpoint).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    run_root: Path,
    config: dict[str, Any],
    run_id: str,
    step: int,
    data_facts: dict[str, Any],
    preflight: dict[str, Any],
    initial_lora_sha: str,
    last_loss: float,
    training_seconds: float,
    cuda_memory: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    checkpoint = run_root / "trainer" / f"checkpoint-{step}"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    adapter_config_path = checkpoint / "adapter_config.json"
    adapter_config = _read_json(adapter_config_path)
    adapter_config["base_model_name_or_path"] = config["base_model"]["model_id"]
    _write_json(adapter_config_path, adapter_config)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(_rng_state(torch), checkpoint / "rng_state.pth")
    state = {
        "global_step": step,
        "sample_cursor": step * ACCUMULATION_STEPS,
        "run_id": run_id,
        "arm": config["arm"],
        "last_loss": last_loss,
        "training_seconds": training_seconds,
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }
    _write_json(checkpoint / "trainer_state.json", state, exclusive=True)
    adapter_identity, restored_config = inference.adapter_weight_identity(checkpoint)
    if restored_config.get("r") != 8 or restored_config.get("lora_alpha") != 16:
        raise RuntimeError("A8 saved adapter config differs from the frozen LoRA contract")
    expected_files = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "adapter_config.json"}
    file_facts = _checkpoint_files(checkpoint)
    if not expected_files <= {record["path"] for record in file_facts} or not any(record["path"].startswith("adapter_model") for record in file_facts):
        raise RuntimeError("A8 checkpoint is missing required recovery or adapter files")
    record = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "training_scope": TRAINING_SCOPE,
        "arm": config["arm"],
        "run_id": run_id,
        "optimizer_step": step,
        "sample_cursor": step * ACCUMULATION_STEPS,
        "config_sha256": config_sha256(config),
        "dataset_manifest_sha256": data_facts["dataset_manifest_sha256"],
        "train_file_sha256": sha256_file(_resolve(config["data"]["train_path"])),
        "sample_order_sha256": data_facts["schedule_sha256"],
        "consumed_sample_ids_sha256": canonical_sha256([sample_id for row in data_facts["schedule"][:step] for sample_id in row["sample_ids"]]),
        "prompt_sha256": config["protocol"]["prompt_sha256"],
        "p0_sha256": config["protocol"]["p0_sha256"],
        "training_code_sha256": preflight["training_code_sha256"],
        "initial_lora_tensor_sha256": initial_lora_sha,
        "adapter_identity": adapter_identity,
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "cuda_memory": cuda_memory,
        "last_loss": last_loss,
        "cumulative_training_seconds": training_seconds,
        "files": file_facts,
        "files_sha256": canonical_sha256(file_facts),
        "created_at_utc": _now(),
    }
    _write_json(checkpoint / "checkpoint_manifest.json", record, exclusive=True)
    return checkpoint, record


def _verify_checkpoint(checkpoint: Path, expected: dict[str, Any], run_id: str, arm: str) -> dict[str, Any]:
    record = _read_json(checkpoint / "checkpoint_manifest.json")
    if record.get("status") != "complete" or record.get("run_id") != run_id or record.get("arm") != arm:
        raise ValueError("A8 checkpoint status, Run ID, or Arm differs")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"A8 checkpoint {key} differs from the current Run")
    if record.get("files_sha256") != canonical_sha256(record.get("files")) or _checkpoint_files(checkpoint) != record.get("files"):
        raise ValueError("A8 checkpoint file hash/size differs from its immutable manifest")
    state = _read_json(checkpoint / "trainer_state.json")
    step = int(checkpoint.name.removeprefix("checkpoint-"))
    if state.get("global_step") != step or state.get("sample_cursor") != step * ACCUMULATION_STEPS or record.get("optimizer_step") != step:
        raise ValueError("A8 checkpoint directory, Trainer state, and sample cursor disagree")
    return record


def _append_ledger(path: Path, row: dict[str, Any]) -> None:
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _next_token_id(model: Any, torch: Any, prompt_ids: list[int]) -> int:
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        next_id = int(torch.argmax(logits[0, -1, :]).item())
    return next_id


def _independent_reload_check(
    checkpoint: Path,
    config: dict[str, Any],
    prompt_ids: list[int],
    *,
    before_next_token: int,
    before_lora_sha: str,
    expected_adapter_sha: str,
) -> dict[str, Any]:
    torch = __import__("torch")
    base_before = _base_identity(config)
    from src.training.training_sft import _load_train_model

    reloaded, torch, model_facts = _load_train_model(
        config,
        _base_path(config),
        adapter_path=checkpoint,
        autocast_adapter_dtype=bool(config["evaluation"]["autocast_adapter_dtype"]),
    )
    after_next = _next_token_id(reloaded, torch, prompt_ids)
    after_lora_sha = _lora_tensor_sha256(reloaded, torch)
    dtypes = sorted({str(parameter.dtype) for name, parameter in reloaded.named_parameters() if ".lora_A." in name or ".lora_B." in name})
    adapter_identity, _adapter_config = inference.adapter_weight_identity(checkpoint)
    base_after = _base_identity(config)
    result = {
        "status": "passed" if before_next_token == after_next and before_lora_sha == after_lora_sha == expected_adapter_sha and dtypes == ["torch.float32"] and base_before.get("observed_weight_hash") == base_after.get("observed_weight_hash") else "failed",
        "next_token_exact_match": before_next_token == after_next,
        "next_token_before": before_next_token,
        "next_token_after": after_next,
        "adapter_tensor_sha256_before_save": before_lora_sha,
        "adapter_tensor_sha256_after_reload": after_lora_sha,
        "adapter_file_identity": adapter_identity,
        "adapter_dtypes_after_reload": dtypes,
        "model_facts": model_facts,
        "base_weight_sha256_before": base_before.get("observed_weight_hash"),
        "base_weight_sha256_after": base_after.get("observed_weight_hash"),
        "base_weights_unchanged": base_before.get("observed_weight_hash") == base_after.get("observed_weight_hash"),
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    if result["status"] != "passed":
        raise RuntimeError(f"A8 independent adapter reload check failed: {result}")
    return result


def _activate_resume_model(model: Any, config: dict[str, Any]) -> list[tuple[str, Any]]:
    parameters = []
    for _name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            parameter.requires_grad_(True)
            parameters.append((name, parameter))
    if not parameters:
        raise RuntimeError("A8 resume checkpoint did not load trainable LoRA parameters")
    if not getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(model, "_require_grads_hook", None) is None:
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    return parameters


def _paired_initialization_check(config: dict[str, Any], run_id: str, own_dry: dict[str, Any]) -> None:
    other_arm = "EP" if config["arm"] == "P" else "P"
    other_root = ROOT / "outputs/training/E03-A8" / other_arm / f"{run_id}-dry"
    other = _read_json(other_root / "dry_step.json")
    if other.get("status") != "passed" or other.get("initial_lora_tensor_sha256") != own_dry.get("initial_lora_tensor_sha256"):
        raise ValueError("paired A8 arms did not start from identical fresh LoRA tensors")


def _execute_segment(
    config: dict[str, Any],
    run_id: str,
    run_root: Path,
    *,
    preflight: dict[str, Any],
    data_facts: dict[str, Any],
    from_step: int,
    to_step: int,
    checkpoint: Path | None,
) -> Path:
    torch = __import__("torch")
    tokenizer = model = None
    segment_started = time.monotonic()
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(config)
        from src.training.training_sft import _load_train_model

        if checkpoint is None:
            _set_seed(int(config["training"]["seed"]), torch)
            model, torch, model_facts = _load_train_model(config, _base_path(config))
            trainable = _lora_parameters(model)
            initial_lora_sha = _lora_tensor_sha256(model, torch)
            if initial_lora_sha != run_manifest_value(run_root, "initial_lora_tensor_sha256"):
                raise ValueError("A8 fresh LoRA tensor hash differs from the paired dry-run")
            optimizer = torch.optim.AdamW(
                [parameter for _name, parameter in trainable],
                lr=float(config["optimizer"]["learning_rate"]),
                weight_decay=float(config["optimizer"]["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
            global_step = 0
            training_seconds_before = 0.0
        else:
            model, torch, model_facts = _load_train_model(
                config,
                _base_path(config),
                adapter_path=checkpoint,
                autocast_adapter_dtype=bool(config["evaluation"]["autocast_adapter_dtype"]),
            )
            trainable = _activate_resume_model(model, config)
            optimizer = torch.optim.AdamW(
                [parameter for _name, parameter in trainable],
                lr=float(config["optimizer"]["learning_rate"]),
                weight_decay=float(config["optimizer"]["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
            scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
            checkpoint_state = _read_json(checkpoint / "trainer_state.json")
            global_step = int(checkpoint_state["global_step"])
            training_seconds_before = float(checkpoint_state["training_seconds"])
            _restore_rng_state(torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False), torch)
            initial_lora_sha = run_manifest_value(run_root, "initial_lora_tensor_sha256")
            if global_step != from_step:
                raise ValueError("A8 resume checkpoint cursor differs from from_step")
        if global_step != from_step:
            raise ValueError("A8 segment started from the wrong global optimizer step")
        base_before = _base_identity(config)
        model_facts["all_non_lora_parameters_frozen"] = all(
            not parameter.requires_grad for name, parameter in model.named_parameters()
            if ".lora_A." not in name and ".lora_B." not in name
        )
        if not model_facts["all_non_lora_parameters_frozen"]:
            raise RuntimeError("A8 found a trainable non-LoRA parameter")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        schedule = data_facts["schedule"]
        train_path = _resolve(config["data"]["train_path"])
        with JsonlById(train_path) as train_rows:
            for step_index in range(from_step, to_step):
                scheduled = schedule[step_index]
                if int(scheduled["optimizer_step"]) != step_index + 1:
                    raise ValueError("A8 schedule index and optimizer step diverged")
                optimizer.zero_grad(set_to_none=True)
                losses: list[float] = []
                completion_tokens = 0
                for sample_id in scheduled["sample_ids"]:
                    row = train_rows.get(sample_id)
                    _row_contract(row, eos_token_id=int(config["protocol"]["eos_token_id"]), prompt_cap=1280, sequence_cap=2048)
                    batch = collator([row])
                    device = model.get_input_embeddings().weight.device
                    batch = {key: value.to(device) for key, value in batch.items()}
                    output = model(**batch)
                    loss = output.loss
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(f"A8 non-finite loss at optimizer step {step_index + 1}")
                    losses.append(float(loss.detach().float().item()))
                    completion_tokens += int(row["completion_length"])
                    _backward_scaled_completion_loss(loss)
                gradient = _gradient_facts(trainable, torch)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step = step_index + 1
                elapsed = time.monotonic() - segment_started
                total_seconds = training_seconds_before + elapsed
                if total_seconds > float(config["training"]["max_training_seconds"]):
                    raise TimeoutError("A8 Run exceeded its frozen cumulative training-time budget")
                memory = inference.cuda_memory_facts(torch)
                if memory.get("cuda_available") and int(memory["allocator_peak_allocated_bytes"]) > int(memory["total_memory_bytes"]) * float(config["quantization"]["max_cuda_memory_fraction"]):
                    raise RuntimeError("A8 training exceeded the frozen CUDA allocator budget")
                row_ledger = {
                    "run_id": run_id,
                    "arm": config["arm"],
                    "optimizer_step": global_step,
                    "sample_ids": list(scheduled["sample_ids"]),
                    "sample_ids_sha256": canonical_sha256(scheduled["sample_ids"]),
                    "completion_tokens": completion_tokens,
                    "micro_batch_losses": losses,
                    "mean_completion_loss": sum(losses) / len(losses),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient": gradient,
                    "cuda_memory": memory,
                    "elapsed_segment_seconds": elapsed,
                    "created_at_utc": _now(),
                }
                _append_ledger(run_root / "step_ledger.jsonl", row_ledger)
                if global_step % 32 == 0 or global_step == to_step:
                    print(f"[E03-A8 {config['arm']}] optimizer_step={global_step}/{to_step} loss={row_ledger['mean_completion_loss']:.5f} tokens={completion_tokens}", flush=True)

        training_seconds = training_seconds_before + time.monotonic() - segment_started
        base_after = _base_identity(config)
        if base_before.get("observed_weight_hash") != base_after.get("observed_weight_hash"):
            raise RuntimeError("A8 Base weights changed during a training segment")
        if global_step not in {512, 1024, 2048} or global_step != to_step:
            raise RuntimeError("A8 training segment did not stop on its requested frozen checkpoint")
        last_ledger = _read_jsonl(run_root / "step_ledger.jsonl")[-1]
        checkpoint, checkpoint_record = _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            torch=torch,
            run_root=run_root,
            config=config,
            run_id=run_id,
            step=global_step,
            data_facts=data_facts,
            preflight=preflight,
            initial_lora_sha=initial_lora_sha,
            last_loss=float(last_ledger["mean_completion_loss"]),
            training_seconds=training_seconds,
            cuda_memory=inference.cuda_memory_facts(torch),
        )
        expected_adapter_sha = _lora_tensor_sha256(model, torch)
        first_id = schedule[0]["sample_ids"][0]
        with JsonlById(train_path) as train_rows:
            probe_row = train_rows.get(first_id)
        probe_ids = list(probe_row["input_ids"][:probe_row["prompt_length"]])
        before_next_token = _next_token_id(model, torch, probe_ids)
        before_lora_sha = _lora_tensor_sha256(model, torch)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        reload_check = _independent_reload_check(
            checkpoint,
            config,
            probe_ids,
            before_next_token=before_next_token,
            before_lora_sha=before_lora_sha,
            expected_adapter_sha=expected_adapter_sha,
        )
        node_dir = run_root / f"step-{global_step:04d}"
        node_dir.mkdir(parents=False, exist_ok=False)
        training_node = {
            "status": "training_checkpoint_passed",
            "run_id": run_id,
            "arm": config["arm"],
            "optimizer_step": global_step,
            "samples_seen": global_step * ACCUMULATION_STEPS,
            "sample_order_sha256": data_facts["schedule_sha256"],
            "consumed_sample_ids_sha256": checkpoint_record["consumed_sample_ids_sha256"],
            "checkpoint_path": str(checkpoint.relative_to(run_root)),
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
            "checkpoint": checkpoint_record,
            "independent_reload_check": reload_check,
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "base_weights_unchanged": True,
            "cumulative_training_seconds": training_seconds,
            "last_loss": float(last_ledger["mean_completion_loss"]),
            "cuda_memory": inference.cuda_memory_facts(torch),
            "created_at_utc": _now(),
        }
        _write_json(node_dir / "training.json", training_node, exclusive=True)
        run_manifest = _read_json(run_root / "manifest.json")
        run_manifest.update(
            status="awaiting_evaluation" if global_step < AUTHORIZED_MAX_STEP else "awaiting_manual_review",
            stage="checkpoint_saved_awaiting_evaluation",
            optimizer_steps=global_step,
            last_checkpoint=str(checkpoint.relative_to(run_root)),
            checkpoint_manifest_sha256=training_node["checkpoint_manifest_sha256"],
            cumulative_training_seconds=training_seconds,
            base_weight_sha256_after_training=base_after.get("observed_weight_hash"),
            ended_at_utc=_now(),
        )
        run_manifest.setdefault("nodes", {})[str(global_step)] = {
            "training_status": "passed",
            "training_summary_path": str((node_dir / "training.json").relative_to(run_root)),
            "checkpoint_manifest_sha256": training_node["checkpoint_manifest_sha256"],
            "evaluation_status": "not_run",
            "authorization_status": "awaiting_evaluation",
        }
        _write_json(run_root / "manifest.json", run_manifest)
        return run_root
    except BaseException as exc:
        if run_root.exists():
            manifest = _read_json(run_root / "manifest.json")
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", stage="training_interrupted" if isinstance(exc, KeyboardInterrupt) else "training_failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
            _write_json(run_root / "manifest.json", manifest)
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_manifest_value(run_root: Path, key: str) -> Any:
    return _read_json(run_root / "manifest.json").get(key)


def execute_start(
    config_path: Path,
    run_id: str,
    preflight_root: Path,
    dry_root: Path,
    *,
    stop_at_step: int,
    allow_first_stage_training: bool,
) -> Path:
    if not allow_first_stage_training:
        raise ValueError("A8 Step 0→512 requires --allow-first-stage-training")
    validate_stage_transition(None, stop_at_step)
    config, _ = _load_config(config_path)
    preflight = _validate_preflight(config, run_id, preflight_root)
    dry = _validate_dry_step(config, run_id, preflight_root, dry_root)
    _paired_initialization_check(config, run_id, dry)
    data_facts = _validate_prepared_data(config)
    run_root = _run_paths(config, run_id)[2]
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite A8 Run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "trainer").mkdir()
    (run_root / "approvals").mkdir()
    shutil.copy2(preflight_root / "preflight.json", run_root / "preflight.json")
    shutil.copy2(dry_root / "dry_step.json", run_root / "dry_step_gate.json")
    shutil.copy2(_resolve(config["data"]["sample_order_path"]), run_root / "sample_order.jsonl")
    _write_json(run_root / "config_snapshot.json", config, exclusive=True)
    _write_json(run_root / "manifest.json", {
        "status": "running",
        "stage": "training_initialized",
        "experiment_id": EXPERIMENT_ID,
        "training_scope": TRAINING_SCOPE,
        "arm": config["arm"],
        "run_id": run_id,
        "config_sha256": config_sha256(config),
        "preflight_sha256": sha256_file(run_root / "preflight.json"),
        "dry_step_sha256": sha256_file(run_root / "dry_step_gate.json"),
        "training_code_sha256": preflight["training_code_sha256"],
        "evaluation_code_sha256": preflight["evaluation_code_sha256"],
        "dataset_manifest_sha256": data_facts["dataset_manifest_sha256"],
        "train_file_sha256": sha256_file(_resolve(config["data"]["train_path"])),
        "sample_order_sha256": data_facts["schedule_sha256"],
        "initial_lora_tensor_sha256": dry["initial_lora_tensor_sha256"],
        "max_optimizer_steps": int(config["training"]["max_optimizer_steps"]),
        "authorized_max_step": AUTHORIZED_MAX_STEP,
        "optimizer_state_resume_enabled": True,
        "resume_history": [],
        "cumulative_training_seconds": 0.0,
        "nodes": {},
        "created_at_utc": _now(),
    }, exclusive=True)
    return _execute_segment(
        config,
        run_id,
        run_root,
        preflight=preflight,
        data_facts=data_facts,
        from_step=0,
        to_step=stop_at_step,
        checkpoint=None,
    )


def execute_resume(
    config_path: Path,
    run_id: str,
    *,
    from_step: int,
    stop_at_step: int,
    approval_file: Path,
    allow_conditional_resume: bool,
) -> Path:
    if not allow_conditional_resume:
        raise ValueError("A8 continuation requires --allow-conditional-resume and a passing conditional receipt")
    validate_stage_transition(from_step, stop_at_step)
    config, _ = _load_config(config_path)
    run_root = _run_paths(config, run_id)[2]
    manifest = _read_json(run_root / "manifest.json")
    if manifest.get("status") != "conditionally_authorized" or manifest.get("optimizer_steps") != from_step:
        raise ValueError("A8 resume requires the same Run paused at an authorized step")
    if manifest.get("config_sha256") != config_sha256(config):
        raise ValueError("A8 resume config differs from the frozen Run")
    code_now = _code_identity(TRAINING_CODE_PATHS)
    if manifest.get("training_code_sha256") != code_now:
        raise ValueError("A8 training code changed after this Run was frozen")
    eval_code_now = _code_identity(EVALUATION_CODE_PATHS)
    if manifest.get("evaluation_code_sha256") != eval_code_now:
        raise ValueError("A8 evaluation code changed after this Run was frozen")
    checkpoint = run_root / "trainer" / f"checkpoint-{from_step}"
    checkpoint_sha = sha256_file(checkpoint / "checkpoint_manifest.json")
    if manifest.get("checkpoint_manifest_sha256") != checkpoint_sha:
        raise ValueError("A8 resume Checkpoint SHA differs from the current Run Manifest")
    evaluation_path = _resolve(manifest.get("last_evaluation_summary_path", ""))
    evaluation_sha = sha256_file(evaluation_path)
    if evaluation_sha != manifest.get("last_evaluation_summary_sha256"):
        raise ValueError("A8 latest evaluation summary changed after conditional authorization")
    previous_step = {1024: 512}.get(from_step)
    previous_evaluation_sha = None
    if previous_step is not None:
        previous_node = manifest.get("nodes", {}).get(str(previous_step), {})
        previous_path = _resolve(previous_node.get("evaluation_summary_path", ""))
        previous_evaluation_sha = sha256_file(previous_path)
        previous_stage = _read_json(previous_path.parent / "stage_manifest.json")
        if (
            previous_node.get("evaluation_status") != "complete"
            or previous_node.get("evaluation_summary_sha256") != previous_evaluation_sha
            or previous_stage.get("evaluation_summary_sha256") != previous_evaluation_sha
        ):
            raise ValueError("A8 previous evaluation node changed after its quality gate")
    data_manifest_sha = sha256_file(_resolve(config["data"]["dataset_manifest"]))
    receipt = approval_file if approval_file.is_absolute() else ROOT / approval_file
    receipt = receipt.resolve()
    if not receipt.is_relative_to(run_root.resolve()):
        raise ValueError("A8 conditional receipt must belong to the same Run approvals directory")
    receipt_record = validate_conditional_authorization(
        receipt,
        run_id=run_id,
        arm=config["arm"],
        from_step=from_step,
        to_step=stop_at_step,
        checkpoint_manifest_sha256=checkpoint_sha,
        evaluation_summary_sha256=evaluation_sha,
        previous_evaluation_summary_sha256=previous_evaluation_sha,
        config_sha256=config_sha256(config),
        data_manifest_sha256=data_manifest_sha,
        code_sha256=canonical_sha256({"training": code_now, "evaluation": eval_code_now}),
    )
    if manifest.get("authorization_receipts", {}).get(str(from_step), {}).get("sha256") != sha256_file(receipt):
        raise ValueError("A8 conditional receipt hash is not bound in the Run Manifest")
    preflight = _read_json(run_root / "preflight.json")
    data_facts = _validate_prepared_data(config)
    expected = {
        "config_sha256": config_sha256(config),
        "dataset_manifest_sha256": data_facts["dataset_manifest_sha256"],
        "sample_order_sha256": data_facts["schedule_sha256"],
        "training_code_sha256": preflight["training_code_sha256"],
    }
    checkpoint_record = _verify_checkpoint(checkpoint, expected, run_id, config["arm"])
    state = _read_json(checkpoint / "trainer_state.json")
    manifest.setdefault("resume_history", []).append({
        "from_step": from_step,
        "to_step": stop_at_step,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "authorization_sha256": sha256_file(receipt),
        "quality_decision": receipt_record["quality_decision"],
        "resumed_at_utc": _now(),
    })
    manifest.update(status="running", stage="resuming_from_full_checkpoint", last_resume_from_step=from_step)
    _write_json(run_root / "manifest.json", manifest)
    return _execute_segment(
        config,
        run_id,
        run_root,
        preflight=preflight,
        data_facts=data_facts,
        from_step=int(state["global_step"]),
        to_step=stop_at_step,
        checkpoint=checkpoint,
    )


def authorize_conditional_resume(
    config_path: Path,
    run_id: str,
    *,
    from_step: int,
    allow_conditional_authorization: bool,
) -> Path:
    """Create a hash-bound receipt using the user's pre-authorized loose gate."""
    if not allow_conditional_authorization:
        raise ValueError("A8 conditional authorization requires --allow-conditional-authorization")
    next_step = {512: 1024, 1024: 2048}.get(from_step)
    if next_step is None:
        raise ValueError("A8 has no pre-authorized continuation after this checkpoint")
    validate_stage_transition(from_step, next_step)
    config, _config_file = _load_config(config_path)
    run_root = _run_paths(config, run_id)[2]
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "awaiting_conditional_authorization"
        or manifest.get("optimizer_steps") != from_step
        or manifest.get("run_id") != run_id
        or manifest.get("arm") != config["arm"]
        or manifest.get("config_sha256") != config_sha256(config)
    ):
        raise ValueError("A8 conditional receipt requires the same Run paused at its current quality-gated step")

    training_code = _code_identity(TRAINING_CODE_PATHS)
    evaluation_code = _code_identity(EVALUATION_CODE_PATHS)
    if manifest.get("training_code_sha256") != training_code or manifest.get("evaluation_code_sha256") != evaluation_code:
        raise ValueError("A8 training/evaluation code changed after the Run was frozen")
    checkpoint = run_root / "trainer" / f"checkpoint-{from_step}"
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    checkpoint_sha = sha256_file(checkpoint_manifest_path)
    if manifest.get("checkpoint_manifest_sha256") != checkpoint_sha:
        raise ValueError("A8 current checkpoint changed after training")

    summary_path = _resolve(manifest.get("last_evaluation_summary_path", ""))
    summary_sha = sha256_file(summary_path)
    if summary_sha != manifest.get("last_evaluation_summary_sha256"):
        raise ValueError("A8 current evaluation summary changed after scoring")
    eval_root = summary_path.parent
    expected_eval_root = _resolve("outputs/evaluation/e03_a8") / config["arm"] / run_id / f"step-{from_step:04d}"
    if eval_root != expected_eval_root:
        raise ValueError("A8 current evaluation artifacts are outside the frozen Run namespace")
    stage_manifest = _read_json(eval_root / "stage_manifest.json")
    summary = _read_json(summary_path)
    predictions_path = eval_root / "predictions.jsonl"
    predictions_sha = sha256_file(predictions_path)
    if (
        stage_manifest.get("status") != "complete"
        or stage_manifest.get("run_id") != run_id
        or stage_manifest.get("arm") != config["arm"]
        or stage_manifest.get("step") != from_step
        or stage_manifest.get("checkpoint_manifest_sha256") != checkpoint_sha
        or stage_manifest.get("evaluation_summary_sha256") != summary_sha
        or stage_manifest.get("predictions_sha256") != predictions_sha
        or summary.get("run_id") != run_id
        or summary.get("arm") != config["arm"]
        or summary.get("checkpoint_step") != from_step
        or summary.get("checkpoint_manifest_sha256") != checkpoint_sha
        or summary.get("prediction_sha256") != predictions_sha
        or summary.get("evaluation_code_sha256") != evaluation_code
    ):
        raise ValueError("A8 evaluation artifacts do not form a complete hash-consistent node")
    node = manifest.get("nodes", {}).get(str(from_step), {})
    if (
        node.get("evaluation_status") != "complete"
        or node.get("evaluation_engineering_gate") != "passed"
        or node.get("evaluation_summary_sha256") != summary_sha
        or node.get("evaluation_predictions_sha256") != predictions_sha
    ):
        raise ValueError("A8 Run Manifest does not record a passed evaluation node")

    previous_summary = None
    previous_summary_sha = None
    previous_step = {1024: 512}.get(from_step)
    if previous_step is not None:
        previous_path = expected_eval_root.parent / f"step-{previous_step:04d}" / "evaluation.json"
        previous_summary = _read_json(previous_path)
        previous_summary_sha = sha256_file(previous_path)
        previous_stage = _read_json(previous_path.parent / "stage_manifest.json")
        if (
            previous_summary.get("run_id") != run_id
            or previous_summary.get("arm") != config["arm"]
            or previous_summary.get("checkpoint_step") != previous_step
            or previous_stage.get("evaluation_summary_sha256") != previous_summary_sha
            or manifest.get("nodes", {}).get(str(previous_step), {}).get("evaluation_summary_sha256") != previous_summary_sha
        ):
            raise ValueError("A8 previous stage is missing or does not match this Run")

    quality = conditional_quality_decision(summary, previous_summary)
    if summary.get("quality_decision") != quality or quality.get("decision") != "conditionally_authorized":
        raise ValueError("A8 loose quality gate failed; leave this node for user review")
    if from_step >= AUTHORIZED_MAX_STEP or next_step > AUTHORIZED_MAX_STEP:
        raise ValueError("A8 training above Step 2048 requires new explicit user approval")

    data_manifest_sha = sha256_file(_resolve(config["data"]["dataset_manifest"]))
    code_sha = canonical_sha256({"training": training_code, "evaluation": evaluation_code})
    receipt_path = run_root / "approvals" / f"step-{from_step:04d}_to_{next_step:04d}.json"
    receipt_sha = write_conditional_authorization(
        receipt_path,
        run_id=run_id,
        arm=config["arm"],
        from_step=from_step,
        to_step=next_step,
        checkpoint_manifest_sha256=checkpoint_sha,
        evaluation_summary_sha256=summary_sha,
        previous_evaluation_summary_sha256=previous_summary_sha,
        config_sha256=config_sha256(config),
        data_manifest_sha256=data_manifest_sha,
        code_sha256=code_sha,
        quality_decision=quality,
    )
    manifest.setdefault("authorization_receipts", {})[str(from_step)] = {
        "path": str(receipt_path.relative_to(run_root)),
        "sha256": receipt_sha,
        "to_step": next_step,
        "created_at_utc": _now(),
    }
    node["authorization_status"] = "conditionally_authorized"
    node["authorized_next_step"] = next_step
    manifest.update(status="conditionally_authorized", stage="quality_gate_authorized_for_resume")
    _write_json(manifest_path, manifest)
    return receipt_path
