"""Isolated E01 QLoRA training for the pre-encoded D0.4.3 dataset.

The command deliberately keeps experiment outputs below ``outputs/training``.
It never writes model files into ``participant/model`` and never loads an
Adapter or Trainer checkpoint for the E01-A baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from participant import run_inference as inference
from src.data.sft_collator import SFTDataCollator
from src.shared.prompting import build_messages, encode_inference_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACKAGES = ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate")
TRAIN_ROW_FIELDS = {
    "sample_id", "conversation_id", "source_split", "source", "prompt_length",
    "completion_length", "input_ids", "labels",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(path_value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = _resolve(config_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"训练配置不存在: {resolved}")
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E01-training-v1":
        raise ValueError("训练配置 version 必须为 E01-training-v1")
    required = {"experiment_id", "status", "base_model", "data", "protocol", "quantization", "lora", "optimizer", "training", "evaluation"}
    if not required <= set(config):
        raise ValueError(f"训练配置缺少字段: {sorted(required - set(config))}")
    if config["experiment_id"] not in {"E01-A", "E01-B"}:
        raise ValueError("experiment_id 只允许 E01-A 或 E01-B")
    if config["data"].get("init_adapter") is not None or config["data"].get("resume_from_checkpoint") is not None:
        raise ValueError("E01 baseline 必须从 Base + 新 LoRA 初始化；init_adapter/resume_from_checkpoint 必须为空")
    if config["data"].get("split") != "train":
        raise ValueError("QLoRA 更新数据必须限定为 D0.4.3 Train")
    if config["protocol"].get("packing") is not False or config["protocol"].get("padding_free") is not False:
        raise ValueError("E01 必须沿用 D0.4.3 packing=false、padding_free=false")
    if config["experiment_id"] == "E01-B" and config.get("status") != "draft_manual_gate":
        raise ValueError("E01-B 配置必须保持 draft_manual_gate，不能自动启动")
    return config, resolved


def validate_training_mode(config: dict[str, Any], mode: str) -> None:
    if config["experiment_id"] == "E01-B" and mode != "preflight":
        raise ValueError("E01-B 仍是待审批草案；当前入口只允许 preflight，禁止启动全量训练")
    if config["experiment_id"] == "E01-A" and config.get("status") != "ready_for_e01_a":
        raise ValueError("E01-A 配置 status 必须为 ready_for_e01_a")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: JSONL 不允许空行")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL 行必须是 object")
            yield row


def _count_jsonl(path: Path) -> int:
    count = 0
    for _row in _iter_jsonl(path):
        count += 1
    return count


def validate_train_row(row: dict[str, Any], *, expected_eos_token_id: int | None = None) -> None:
    if set(row) != TRAIN_ROW_FIELDS:
        raise ValueError(f"D0.4.3 row fields differ from locked contract: {sorted(row)}")
    if row.get("source_split") != "train" or not str(row.get("sample_id", "")).startswith("train:"):
        raise ValueError("Training row is not from the D0.4.3 Train split")
    input_ids = row.get("input_ids")
    labels = row.get("labels")
    prompt_length = row.get("prompt_length")
    completion_length = row.get("completion_length")
    if not isinstance(input_ids, list) or not input_ids or not isinstance(labels, list) or len(labels) != len(input_ids):
        raise ValueError(f"{row.get('sample_id')}: input_ids/labels must be non-empty and equal length")
    if not isinstance(prompt_length, int) or isinstance(prompt_length, bool) or prompt_length <= 0:
        raise ValueError(f"{row.get('sample_id')}: prompt_length must be positive")
    if not isinstance(completion_length, int) or isinstance(completion_length, bool) or completion_length <= 0:
        raise ValueError(f"{row.get('sample_id')}: completion_length must be positive")
    if prompt_length + completion_length != len(input_ids):
        raise ValueError(f"{row.get('sample_id')}: prompt/completion lengths do not sum to input length")
    if labels[:prompt_length] != [-100] * prompt_length:
        raise ValueError(f"{row.get('sample_id')}: prompt labels are not all -100")
    if labels[prompt_length:] != input_ids[prompt_length:]:
        raise ValueError(f"{row.get('sample_id')}: completion labels differ from pre-encoded input tokens")
    if expected_eos_token_id is not None and expected_eos_token_id not in input_ids[prompt_length:]:
        raise ValueError(f"{row.get('sample_id')}: completion labels do not contain the locked EOS token")


def deterministic_select_samples(
    sft_metadata: list[dict[str, Any]],
    normalized_metadata: list[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Prefer short, unique conversations while covering emotion/profile classes."""
    if count < 8 or count > 16:
        raise ValueError("E01-A micro-overfit sample count must be between 8 and 16")
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in normalized_metadata:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in source_by_id:
            raise ValueError(f"D0.3 Train sample_id missing or duplicated: {sample_id!r}")
        source_by_id[sample_id] = row

    candidates: list[dict[str, Any]] = []
    seen_sft: set[str] = set()
    for row in sft_metadata:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in seen_sft:
            raise ValueError(f"D0.4.3 Train sample_id missing or duplicated: {sample_id!r}")
        seen_sft.add(sample_id)
        source = source_by_id.get(sample_id)
        if source is None:
            raise ValueError(f"D0.4.3 Train sample not found in D0.3 Train: {sample_id}")
        target = source.get("target", {})
        profile = target.get("user_profile", {})
        if not isinstance(profile, dict):
            raise ValueError(f"D0.3 target profile invalid: {sample_id}")
        is_empty_profile = all(not profile.get(name) for name in ("personality_traits", "interests", "style"))
        conversation_id = row.get("conversation_id")
        if not isinstance(conversation_id, str) or conversation_id != source.get("conversation_id"):
            raise ValueError(f"D0.4.3 and D0.3 conversation identity differs: {sample_id}")
        candidates.append({
            "sample_id": sample_id,
            "conversation_id": conversation_id,
            "emotion_label": target.get("emotion_label"),
            "profile_empty": is_empty_profile,
            "prompt_length": int(row.get("prompt_length", 0)),
            "completion_length": int(row.get("completion_length", 0)),
            "sequence_length": int(row.get("prompt_length", 0)) + int(row.get("completion_length", 0)),
        })
    if len(candidates) < count:
        raise ValueError(f"D0.4.3 Train has only {len(candidates)} matched rows; need {count}")

    selected: list[dict[str, Any]] = []
    used_conversations: set[str] = set()
    emotions: set[str] = set()
    profiles: set[bool] = set()
    remaining = list(candidates)
    while remaining and len(selected) < count:
        eligible = [row for row in remaining if row["conversation_id"] not in used_conversations]
        if not eligible:
            raise ValueError("could not choose unique conversations for E01-A samples")
        eligible.sort(key=lambda row: (
            row["emotion_label"] in emotions,
            row["profile_empty"] in profiles,
            row["sequence_length"],
            row["sample_id"],
        ))
        chosen = eligible[0]
        selected.append(chosen)
        used_conversations.add(chosen["conversation_id"])
        emotions.add(chosen["emotion_label"])
        profiles.add(chosen["profile_empty"])
        remaining = [row for row in remaining if row["sample_id"] != chosen["sample_id"]]
    selected.sort(key=lambda row: row["sample_id"])
    if len(emotions) < 4:
        raise ValueError(f"E01-A selected set covers fewer than four emotion labels: {sorted(emotions)}")
    if profiles != {False, True}:
        raise ValueError("E01-A selected set must include empty and non-empty profile targets")
    return selected


def _training_rows(config: dict[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    data = config["data"]
    train_path = _resolve(data["train_path"])
    dev_path = _resolve(data["dev_path"])
    d043_manifest_path = _resolve(data["sft_manifest"])
    for path in (train_path, dev_path, d043_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required D0.4.3 file missing: {path}")
    manifest = json.loads(d043_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "D0.4.3-v1":
        raise ValueError("D0.4.3 manifest version must remain D0.4.3-v1")
    if manifest.get("outputs", {}).get("dataset_dir") != "data/processed/sft/v1":
        raise ValueError("D0.4.3 data directory differs from the locked v1 dataset")
    if manifest.get("config", {}).get("profile", {}).get("max_train_seq_length") != config["protocol"]["max_train_seq_length"]:
        raise ValueError("D0.4.3 S differs from E01 config")
    if manifest.get("config", {}).get("profile", {}).get("max_prompt_tokens") != config["protocol"]["max_prompt_tokens"]:
        raise ValueError("D0.4.3 P differs from E01 config")
    files = manifest.get("outputs", {}).get("files", {})
    for name, path in (("train.jsonl", train_path), ("dev.jsonl", dev_path)):
        expected = files.get(name, {}).get("sha256")
        if not expected or sha256_file(path) != expected:
            raise ValueError(f"D0.4.3 {name} hash does not match its manifest")
    d03_manifest_path = _resolve(data["d03_manifest"])
    d042_manifest_path = _resolve(data["d042_manifest"])
    d03_sha = sha256_file(d03_manifest_path)
    d042_sha = sha256_file(d042_manifest_path)
    if d03_sha != manifest.get("d03_manifest_sha256") or d042_sha != manifest.get("d042_manifest_sha256"):
        raise ValueError("D0.4.3 source manifests do not match the recorded D0.3/D0.4.2 hashes")
    d03 = json.loads(d03_manifest_path.read_text(encoding="utf-8"))
    if d03.get("pipeline_version") != "D0.3-v1":
        raise ValueError("D0.3 manifest version must remain D0.3-v1")
    normalized_train_path = _resolve(data["normalized_train_path"])
    output = next((item for item in d03.get("outputs", []) if item.get("path") == data["normalized_train_path"]), None)
    if output is None or sha256_file(normalized_train_path) != output.get("sha256"):
        raise ValueError("D0.3 Train data is missing or differs from its manifest")
    if output.get("records") != int(data["expected_source_train_rows"]):
        raise ValueError("D0.3 Train row count differs from E01 config")
    return train_path, dev_path, normalized_train_path, {"sft": manifest, "d03": d03, "d03_manifest_path": d03_manifest_path}


def _select_real_samples(config: dict[str, Any], train_path: Path, normalized_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sft_metadata = []
    for row in _iter_jsonl(train_path):
        sft_metadata.append({key: row.get(key) for key in ("sample_id", "conversation_id", "source_split", "prompt_length", "completion_length")})
    normalized_metadata = []
    for row in _iter_jsonl(normalized_path):
        normalized_metadata.append({key: row.get(key) for key in ("sample_id", "conversation_id", "target")})
    selected = deterministic_select_samples(
        sft_metadata,
        normalized_metadata,
        count=int(config["training"]["sample_selection"]["count"]),
    )
    selected_ids = {row["sample_id"] for row in selected}
    selected_sources = {row["sample_id"]: row for row in _iter_jsonl(normalized_path) if row.get("sample_id") in selected_ids}
    selected_encoded = {row["sample_id"]: row for row in _iter_jsonl(train_path) if row.get("sample_id") in selected_ids}
    if set(selected_sources) != selected_ids or set(selected_encoded) != selected_ids:
        raise ValueError("selected Train IDs were not found in both D0.3 and D0.4.3")
    eos = int(config["protocol"]["eos_token_id"])
    result = []
    for metadata in selected:
        sample_id = metadata["sample_id"]
        encoded = selected_encoded[sample_id]
        validate_train_row(encoded, expected_eos_token_id=eos)
        source = selected_sources[sample_id]
        if not isinstance(source.get("history"), list) or not source["history"]:
            raise ValueError(f"selected D0.3 history is invalid: {sample_id}")
        result.append({"selection": metadata, "source": source, "encoded": encoded})
    selected_manifest = [{key: row[key] for key in (
        "sample_id", "conversation_id", "emotion_label", "profile_empty", "prompt_length",
        "completion_length", "sequence_length",
    )} for row in selected]
    return result, selected_manifest


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in EXPECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _preflight(config: dict[str, Any], *, include_model_hash: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    checks["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    try:
        import torch
        checks["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        checks["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not checks["cuda"]["available"]:
        checks["cuda"]["status"] = "blocked_gpu_unavailable"

    try:
        train_path, dev_path, normalized_path, manifests = _training_rows(config)
        train_count = _count_jsonl(train_path)
        dev_count = _count_jsonl(dev_path)
        expected_train = int(config["data"]["expected_train_rows"])
        expected_dev = int(config["data"]["expected_dev_rows"])
        if train_count != expected_train or dev_count != expected_dev:
            raise ValueError(f"D0.4.3 counts differ: train={train_count}, dev={dev_count}")
        if config["experiment_id"] == "E01-A":
            samples, selected_metadata = _select_real_samples(config, train_path, normalized_path)
        else:
            samples, selected_metadata = [], []
        p0_path = _resolve(config["protocol"]["prompt_path"])
        p0_hash = sha256_file(p0_path)
        if p0_hash != config["protocol"]["p0_sha256"]:
            raise ValueError("P0 fingerprint differs from the locked E01 protocol")
        runner_config = json.loads(_resolve(config["protocol"]["runner_config"]).read_text(encoding="utf-8"))
        if runner_config.get("prompt", {}).get("sha256") != p0_hash:
            raise ValueError("runner P0 fingerprint differs from the locked E01 protocol")
        base_path = _resolve(config["base_model"]["local_path"])
        for required_name in ("config.json", "model.safetensors.index.json"):
            if not (base_path / required_name).is_file():
                raise FileNotFoundError(f"base model missing {required_name}: {base_path}")
        model_identity = inference._weight_identity(base_path, {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"].get("revision"),
        }) if include_model_hash else {"identity_status": "not_computed"}
        observed_hash = model_identity.get("observed_weight_hash")
        if include_model_hash and observed_hash != config["base_model"]["expected_weight_sha256"]:
            raise ValueError(f"Base model SHA-256 differs: expected={config['base_model']['expected_weight_sha256']}, actual={observed_hash}")
        tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runner_config, base_path)
        if tokenizer_facts.get("signature_sha256") != config["protocol"]["tokenizer_signature_sha256"]:
            raise ValueError("runtime tokenizer signature differs from D0.4.3")
        if tokenizer_facts.get("template_sha256") != config["protocol"]["chat_template_sha256"]:
            raise ValueError("runtime chat template differs from D0.4.3")
        if tokenizer.eos_token_id != int(config["protocol"]["eos_token_id"]):
            raise ValueError("runtime tokenizer EOS ID differs from D0.4.3")
        checks["data"] = {
            "status": "passed",
            "train_path": str(train_path.relative_to(PROJECT_ROOT)),
            "train_sha256": sha256_file(train_path),
            "train_rows": train_count,
            "dev_path": str(dev_path.relative_to(PROJECT_ROOT)),
            "dev_sha256": sha256_file(dev_path),
            "dev_rows": dev_count,
            "normalized_train_path": str(normalized_path.relative_to(PROJECT_ROOT)),
            "normalized_train_sha256": sha256_file(normalized_path),
            "d03_manifest_sha256": sha256_file(manifests["d03_manifest_path"]),
            "d043_manifest_sha256": sha256_file(_resolve(config["data"]["sft_manifest"])),
            "selected_samples": selected_metadata,
            "selected_samples_sha256": canonical_sha256(selected_metadata),
            "only_train_used_for_selection_and_updates": True,
            "dev_loaded_for_training": False,
            "official_val_used": False,
        }
        checks["protocol"] = {
            "p0_sha256": p0_hash,
            "tokenizer": tokenizer_facts,
            "sequence_budget_S": int(config["protocol"]["max_train_seq_length"]),
            "prompt_budget_P": int(config["protocol"]["max_prompt_tokens"]),
            "packing": False,
            "padding_free": False,
            "labels_preencoded": True,
        }
        checks["base_model"] = {
            "path": str(base_path),
            "model_id": config["base_model"]["model_id"],
            "revision": config["base_model"].get("revision"),
            "expected_weight_sha256": config["base_model"]["expected_weight_sha256"],
            "identity": model_identity,
            "status": "verified" if observed_hash == config["base_model"]["expected_weight_sha256"] else "not_computed",
        }
        checks["output"] = {
            "root": str(_resolve(config["data"]["output_root"])),
            "base_model_is_read_only_source": True,
            "adapter_output_is_separate": True,
        }
        checks["datasets"] = {"train_rows": train_count, "dev_rows": dev_count}
        # The loaded tokenizer and selected examples are intentionally not serialized.
        del tokenizer, samples
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        checks["data_and_identity_error"] = errors[-1]
    if errors:
        status = "failed"
    elif not checks.get("cuda", {}).get("available"):
        status = "blocked_gpu_unavailable"
    elif checks.get("cuda", {}).get("bf16_supported") is not True:
        status = "blocked_bf16_unavailable"
    else:
        status = "passed"
    return {
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "errors": errors,
        "checks": checks,
    }, []


def _run_root(config: dict[str, Any], run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}", run_id):
        raise ValueError("run-id must use 3-96 ASCII letters, digits, dots, underscores, or hyphens")
    root = _resolve(config["data"]["output_root"]) / run_id
    allowed_root = _resolve(config["data"]["output_root"])
    if not root.is_relative_to(allowed_root):
        raise ValueError("run directory escaped outputs/training")
    if root.exists():
        raise FileExistsError(f"Run ID already exists; refusing to overwrite: {root}")
    root.mkdir(parents=True)
    return root


def _write_run_manifest(run_root: Path, manifest: dict[str, Any]) -> None:
    _write_json(run_root / "manifest.json", manifest)


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


def _load_train_model(config: dict[str, Any], base_path: Path, *, adapter_path: Path | None = None) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("blocked_gpu_unavailable: CUDA is required for real QLoRA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_bf16_unavailable: E01-A requires BF16 CUDA support")
    quant_cfg = config["quantization"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg["quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bool(quant_cfg["use_double_quant"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        device_map=config["quantization"].get("device_map", "auto"),
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
    )
    model.config.use_cache = False
    from peft import PeftModel, prepare_model_for_kbit_training
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False, local_files_only=True)
        model.eval()
        facts = inference.inspect_quantization_runtime(model, _training_quantization_inspection_config(config), torch)
        return model, torch, facts
    prepare_kwargs: dict[str, Any] = {"use_gradient_checkpointing": bool(config["training"]["gradient_checkpointing"])}
    try:
        import inspect
        signature = inspect.signature(prepare_model_for_kbit_training)
        if "gradient_checkpointing_kwargs" in signature.parameters:
            prepare_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    except (TypeError, ValueError):
        pass
    model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    if bool(config["training"]["gradient_checkpointing"]):
        if not getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if getattr(model, "_require_grads_hook", None) is None:
            model.enable_input_require_grads()
    from peft import LoraConfig, get_peft_model
    target_modules = list(config["lora"]["target_modules"])
    observed_targets = sorted({name.rsplit(".", 1)[-1] for name, _module in model.named_modules() if name.rsplit(".", 1)[-1] in target_modules})
    if observed_targets != sorted(target_modules):
        raise RuntimeError(f"LoRA target modules not all present: expected={sorted(target_modules)}, actual={observed_targets}")
    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias=config["lora"].get("bias", "none"),
        base_model_name_or_path=config["base_model"]["model_id"],
    )
    model = get_peft_model(model, lora_config)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name, _parameter in trainable if ".lora_A." not in name and ".lora_B." not in name]
    if not trainable or unexpected:
        raise RuntimeError(f"trainable parameters are not limited to LoRA: {unexpected[:10]}")
    # Parameter names end in lora_A/default/weight; use module paths to check all targets.
    target_names = {
        name.rsplit(".lora_A.", 1)[0].rsplit(".lora_B.", 1)[0].rsplit(".", 1)[-1]
        for name, _parameter in trainable
    }
    if not set(target_modules) <= target_names:
        raise RuntimeError(f"LoRA target modules were not injected: expected={target_modules}, observed={sorted(target_names)}")
    model.train()
    facts = inference.inspect_quantization_runtime(model, _training_quantization_inspection_config(config), torch)
    if not facts.get("real_nf4_verified") or facts.get("offload_devices"):
        raise RuntimeError("NF4/BF16/CUDA placement failed or offload was observed")
    facts["trainable_parameter_count"] = sum(parameter.numel() for _name, parameter in trainable)
    facts["trainable_parameters"] = [name for name, _parameter in trainable]
    facts["frozen_parameter_count"] = sum(parameter.numel() for _name, parameter in model.named_parameters() if not parameter.requires_grad)
    facts["total_parameter_count"] = facts["trainable_parameter_count"] + facts["frozen_parameter_count"]
    facts["all_non_lora_parameters_frozen"] = facts["frozen_parameter_count"] > 0 and facts["total_parameter_count"] >= facts["trainable_parameter_count"]
    facts["observed_lora_targets"] = sorted(target_names)
    facts["all_trainable_parameters_are_lora"] = True
    return model, torch, facts


def _training_quantization_inspection_config(config: dict[str, Any]) -> dict[str, Any]:
    quantization = config["quantization"]
    if quantization.get("method") != "bitsandbytes_4bit" or quantization.get("quant_type") != "nf4":
        raise ValueError("E01 training runtime inspection requires bitsandbytes NF4")
    return {
        "quantization": {
            "enabled": True,
            "use_double_quant": bool(quantization["use_double_quant"]),
        }
    }


def _trainable_parameters(model: Any) -> list[tuple[str, Any]]:
    values = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not values or any(".lora_A." not in name and ".lora_B." not in name for name, _parameter in values):
        raise RuntimeError("all trainable parameters must be LoRA A/B parameters")
    return values


def _model_device(model: Any) -> Any:
    return model.get_input_embeddings().weight.device


def _build_sft_trainer(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    collator: SFTDataCollator,
    config: dict[str, Any],
    *,
    max_steps: int,
    callback: Any,
    run_root: Path,
) -> Any:
    """Build the pinned TRL path without allowing it to retokenize or truncate."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    optimizer = config["optimizer"]
    training = config["training"]
    args = SFTConfig(
        output_dir=str(run_root / "trainer_tmp"),
        do_train=True,
        do_eval=False,
        use_cpu=False,
        per_device_train_batch_size=int(training["per_device_micro_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_steps=max_steps,
        num_train_epochs=1,
        learning_rate=float(optimizer["learning_rate"]),
        lr_scheduler_type=optimizer.get("lr_scheduler", "constant"),
        weight_decay=float(optimizer.get("weight_decay", 0.0)),
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        packing=False,
        padding_free=False,
        completion_only_loss=False,
        assistant_only_loss=False,
        remove_unused_columns=False,
        train_sampling_strategy="sequential",
        dataloader_num_workers=0,
        include_num_input_tokens_seen=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    if not trainer._skip_prepare_dataset or trainer.data_collator is not collator:
        raise RuntimeError("SFTTrainer must preserve the pre-tokenized D0.4.3 dataset and custom collator")
    if trainer.args.max_length is not None or trainer.args.packing or trainer.args.padding_free:
        raise RuntimeError("SFTTrainer must keep max_length=None, packing=False, and padding_free=False")
    if trainer.args.do_eval or trainer.args.resume_from_checkpoint:
        raise RuntimeError("training must not evaluate SFT Dev or resume a checkpoint")
    import torch
    batch_size = int(training["per_device_micro_batch_size"])
    verify_rows = min(len(rows), batch_size * int(training["gradient_accumulation_steps"]))
    iterator = iter(trainer.get_train_dataloader())
    for offset in range(0, verify_rows, batch_size):
        observed_batch = next(iterator)
        expected_batch = collator(rows[offset:offset + batch_size])
        if set(observed_batch) != {"input_ids", "attention_mask", "labels"}:
            raise RuntimeError(f"SFTTrainer passed unexpected fields to model: {sorted(observed_batch)}")
        for name in ("input_ids", "attention_mask", "labels"):
            if not torch.equal(observed_batch[name].cpu(), expected_batch[name].cpu()):
                raise RuntimeError(f"SFTTrainer changed pre-encoded {name} at train row {offset}")
    trainer.e01_data_path_facts = {
        "skip_prepare_dataset": trainer._skip_prepare_dataset,
        "max_length": trainer.args.max_length,
        "packing": trainer.args.packing,
        "padding_free": trainer.args.padding_free,
        "micro_batches_verified_against_collator": verify_rows // batch_size,
        "input_ids_equal": True,
        "attention_mask_equal": True,
        "labels_equal": True,
        "model_batch_fields": ["input_ids", "attention_mask", "labels"],
    }
    return trainer


def _snapshot_trainer_data_path_facts(trainer: Any) -> dict[str, Any]:
    facts = getattr(trainer, "e01_data_path_facts", None)
    if not isinstance(facts, dict):
        raise RuntimeError("SFTTrainer data path evidence is missing")
    return dict(facts)


def _optimizer_evidence_callback(
    model: Any,
    trainable: list[tuple[str, Any]],
    *,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    tokenizer: Any,
    prompt: str,
    schema: dict[str, Any] | None,
    enums: dict[str, list[str]] | None,
    run_root: Path,
    snapshot_adapter: bool,
) -> Any:
    """Collect gradient/update evidence around the actual Trainer optimizer step."""
    from transformers import TrainerCallback

    class EvidenceCallback(TrainerCallback):
        def __init__(self):
            self.model = model
            self.trainable = trainable
            self.config = config
            self.samples = samples
            self.tokenizer = tokenizer
            self.prompt = prompt
            self.schema = schema
            self.enums = enums
            self.run_root = run_root
            self.snapshot_adapter = snapshot_adapter
            self.started = 0.0
            self.before: dict[str, Any] = {}
            self.current: dict[str, Any] = {}
            self.steps: list[dict[str, Any]] = []
            self.logged_steps: set[int] = set()
            self.token_cursor = 0

        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            import torch
            gradients = [parameter.grad for _name, parameter in self.trainable if parameter.grad is not None]
            finite = bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
            nonzero = sum(bool(torch.count_nonzero(gradient).item()) for gradient in gradients)
            if not finite or not nonzero:
                raise RuntimeError("Trainer optimizer step has missing, non-finite, or zero LoRA gradients")
            norm = torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradients))
            if not bool(torch.isfinite(norm).item()) or float(norm.item()) == 0:
                raise RuntimeError("Trainer LoRA gradient norm is invalid")
            self.before = {
                name: parameter.detach().to(device="cpu", copy=True)
                for name, parameter in self.trainable
            }
            self.current = {
                "finite_gradients": finite,
                "nonzero_gradient_parameter_count": nonzero,
                "gradient_norm": float(norm.detach().cpu().item()),
            }
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            import torch
            changed_names = []
            changed_elements = 0
            for name, parameter in self.trainable:
                previous = self.before.get(name)
                current = parameter.detach().to(device="cpu")
                if previous is not None:
                    count = int(previous.ne(current).sum().item())
                    if count:
                        changed_names.append(name)
                        changed_elements += count
            if not changed_names:
                raise RuntimeError("Trainer optimizer.step did not change sampled LoRA parameters")
            next_step = int(state.global_step) + 1
            selected_rows = []
            for _ in range(int(self.config["training"]["gradient_accumulation_steps"])):
                selected_rows.append(self.samples[self.token_cursor % len(self.samples)]["encoded"])
                self.token_cursor += 1
            self.current.update({
                "optimizer_step": next_step,
                "completion_tokens": sum(
                    sum(label != -100 for label in row["labels"]) for row in selected_rows
                ),
                "updated_parameter_count": len(changed_names),
                "updated_parameter_names": changed_names,
                "updated_element_count": changed_elements,
                "optimizer_step_seconds": time.perf_counter() - self.started,
                "cuda_memory": inference.cuda_memory_facts(torch),
            })
            self.steps.append(self.current)
            self.before = {}
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs or not self.steps:
                return control
            evidence = self.steps[-1]
            current_step = int(state.global_step)
            if evidence.get("optimizer_step") != current_step or current_step in self.logged_steps:
                return control
            row = {
                **evidence,
                "loss": float(logs["loss"]),
                "learning_rate": logs.get("learning_rate"),
                "total_input_tokens_seen": logs.get("num_input_tokens_seen"),
            }
            if not math.isfinite(row["loss"]):
                raise FloatingPointError("Trainer reported a non-finite logged loss")
            log_path = self.run_root / "train_log.jsonl"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            evidence["loss"] = row["loss"]
            self.logged_steps.add(current_step)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            interval = int(self.config["training"]["evaluation_interval_steps"])
            if schema is not None and enums is not None and step % interval == 0:
                was_training = self.model.training
                self.model.eval()
                try:
                    _evaluate_train_samples(
                        self.model, self.tokenizer, self.prompt, self.schema, self.enums,
                        self.samples, self.config, step=step, run_root=self.run_root,
                    )
                finally:
                    self.model.train(was_training)
                if self.snapshot_adapter:
                    snapshot = self.run_root / "adapter_snapshots" / f"step_{step:04d}"
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(snapshot, safe_serialization=True)
            return control

    return EvidenceCallback()


def _capture_probe(model: Any, tokenizer: Any, prompt_ids: list[int], *, max_new_tokens: int) -> dict[str, Any]:
    import torch
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=_model_device(model))
    attention_mask = torch.ones_like(input_ids)
    previous_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0, input_ids.shape[1]:].detach().cpu().tolist()
        return {
            "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
            "generated_token_ids": generated_ids,
            "raw_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "generated_tokens": len(generated_ids),
        }
    finally:
        model.train(previous_mode)


def _check_cuda_placement(model: Any) -> dict[str, Any]:
    import torch
    facts = inference.inspect_quantization_runtime(model, {"quantization": {"enabled": True, "use_double_quant": True}}, torch)
    if not facts.get("real_nf4_verified") or facts.get("offload_devices"):
        raise RuntimeError("model placement is not verified NF4/BF16/CUDA without offload")
    return facts


def _step_update(
    model: Any,
    torch: Any,
    optimizer: Any,
    rows: list[dict[str, Any]],
    collator: SFTDataCollator,
    accumulation_steps: int,
) -> dict[str, Any]:
    trainable = _trainable_parameters(model)
    before = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer.zero_grad(set_to_none=True)
    micro_losses: list[float] = []
    micro_tokens: list[int] = []
    start = time.perf_counter()
    for start_index in range(0, len(rows), accumulation_steps):
        window = rows[start_index:start_index + accumulation_steps]
        counts = [sum(label != -100 for label in row["labels"]) for row in window]
        total_tokens = sum(counts)
        for row, token_count in zip(window, counts):
            batch = collator([row])
            device = _model_device(model)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite QLoRA loss")
            micro_losses.append(float(loss.detach().float().item()))
            micro_tokens.append(token_count)
            (loss * (token_count / total_tokens)).backward()
    gradients = [parameter.grad for _name, parameter in trainable if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("no LoRA gradients were produced")
    finite_gradients = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    nonzero_gradient_parameters = sum(bool(torch.count_nonzero(gradient).item()) for gradient in gradients)
    if not finite_gradients or not nonzero_gradient_parameters:
        raise RuntimeError("LoRA gradients are non-finite or all zero")
    gradient_norm = torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradients))
    if not bool(torch.isfinite(gradient_norm).item()) or float(gradient_norm.item()) == 0:
        raise RuntimeError("LoRA gradient norm is non-finite or zero")
    optimizer.step()
    changed_parameters = []
    total_changed_elements = 0
    for name, parameter in trainable:
        changed = before[name].ne(parameter.detach()).sum().item()
        if changed:
            changed_parameters.append(name)
            total_changed_elements += int(changed)
    optimizer.zero_grad(set_to_none=True)
    if not changed_parameters:
        raise RuntimeError("optimizer.step completed but no LoRA parameter changed")
    elapsed = time.perf_counter() - start
    import torch as torch_module
    memory = inference.cuda_memory_facts(torch_module)
    return {
        "loss": sum(loss * count for loss, count in zip(micro_losses, micro_tokens)) / sum(micro_tokens),
        "micro_batch_losses": micro_losses,
        "completion_tokens": sum(micro_tokens),
        "gradient_norm": float(gradient_norm.detach().cpu().item()),
        "finite_gradients": finite_gradients,
        "nonzero_gradient_parameter_count": nonzero_gradient_parameters,
        "updated_parameter_count": len(changed_parameters),
        "updated_parameter_names": changed_parameters,
        "updated_element_count": total_changed_elements,
        "optimizer_step_seconds": elapsed,
        "cuda_memory": memory,
    }


def _serialize_selected_samples(path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [{key: item["selection"][key] for key in (
        "sample_id", "conversation_id", "emotion_label", "profile_empty", "prompt_length",
        "completion_length", "sequence_length",
    )} for item in samples]
    _write_json(path, {"selection_rule": "unique conversation; greedily cover emotion and empty/non-empty profile, then shortest sequence, stable sample_id tie-break", "rows": rows, "sha256": canonical_sha256(rows)})
    return rows


def _load_protocol(config: dict[str, Any], base_path: Path) -> tuple[Any, str, dict[str, Any], dict[str, list[str]]]:
    runner_config = json.loads(_resolve(config["protocol"]["runner_config"]).read_text(encoding="utf-8"))
    tokenizer, _facts = inference.load_runtime_tokenizer(runner_config, base_path)
    prompt_path = _resolve(config["protocol"]["prompt_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_facts = inference.load_runtime_protocol(runner_config)[3]
    schema = inference.load_submission_schema()
    enums = inference.submission_enums(schema)
    return tokenizer, prompt, schema, enums


def _evaluate_train_samples(
    model: Any,
    tokenizer: Any,
    prompt: str,
    schema: dict[str, Any],
    enums: dict[str, list[str]],
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    step: int,
    run_root: Path,
) -> dict[str, Any]:
    from src.evaluation.metrics import emotion_metrics, profile_metrics, raw_diagnostics

    raw_rows = []
    predictions = []
    schema_validity = []
    targets = []
    generation = {"do_sample": False, "max_new_tokens": int(config["evaluation"]["max_new_tokens"])}
    was_training = model.training
    model.eval()
    try:
        for item in samples:
            source = item["source"]
            sample_id = source["sample_id"]
            selection, _messages = inference.prepare_sample_messages(
                {"id": sample_id, "history": source["history"]}, prompt, tokenizer,
                int(config["protocol"]["max_prompt_tokens"]),
            )
            raw_text, details = inference.generate_transformers_sample(
                model, __import__("torch"), tokenizer, selection["prompt_ids"], generation,
            )
            prediction, parse_failed, invalid_label = inference.parse_prediction(sample_id, raw_text, enums)
            try:
                inference.validate_submission_prediction(prediction, schema)
                schema_valid = True
            except Exception:
                schema_valid = False
            raw_rows.append({
                "id": sample_id,
                "raw_text": raw_text,
                "status": "parse_failure" if parse_failed else "invalid_label" if invalid_label else "success",
                "prompt_tokens": selection["prompt_tokens"],
                "generated_tokens": details["generated_tokens"],
                "stop_reason": details["stop_reason"],
                "latency_ms": None,
                "removed_history_messages": selection["selected_start_index"],
                "backend": "transformers",
                "error_type": None,
            })
            predictions.append(prediction)
            schema_validity.append(schema_valid)
            targets.append(source["target"])
    finally:
        model.train(was_training)
    gold_emotions = [target["emotion_label"] for target in targets]
    predicted_emotions = [row["emotion_label"] for row in predictions]
    gold_profiles = [target["user_profile"] for target in targets]
    predicted_profiles = [row["user_profile"] for row in predictions]
    diagnostics = {
        "scope": "train_fitting_only",
        "step": step,
        "sample_ids": [item["source"]["sample_id"] for item in samples],
        "generation_config": generation,
        "emotion": emotion_metrics(gold_emotions, predicted_emotions, enums["emotion_label"]),
        "profile": profile_metrics(gold_profiles, predicted_profiles, enums),
        "raw": raw_diagnostics(raw_rows, enums, predictions, schema_validity, generation_cap=generation["max_new_tokens"]),
        "empty_or_missing_field_count": sum(
            not row["response_text"] or not row["emotion_label"] or any(
                key not in row for key in ("emotion_label", "user_profile", "response_text", "memory_refs")
            ) for row in predictions
        ),
        "generated_raw_sha256": canonical_sha256([row["raw_text"] for row in raw_rows]),
    }
    path = run_root / "evaluation" / f"train_diagnostics_step_{step:04d}.json"
    _write_json(path, diagnostics)
    raw_path = run_root / "evaluation" / f"train_raw_step_{step:04d}.jsonl"
    with raw_path.open("x", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps({key: row[key] for key in ("id", "raw_text", "status")}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return diagnostics


def _make_probe_prompt(samples: list[dict[str, Any]], tokenizer: Any, prompt: str, prompt_cap: int) -> list[int]:
    item = samples[0]
    selection, _messages = inference.prepare_sample_messages(
        {"id": item["source"]["sample_id"], "history": item["source"]["history"]},
        prompt, tokenizer, prompt_cap,
    )
    return list(selection["prompt_ids"])


def _adapter_identity(adapter_dir: Path) -> dict[str, Any]:
    identity, _config = inference.adapter_weight_identity(adapter_dir)
    return identity


def _save_adapter(model: Any, adapter_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    base_path = _resolve(config["base_model"]["local_path"])
    if adapter_dir.resolve() == base_path or base_path.is_relative_to(adapter_dir.resolve()):
        raise ValueError("Adapter output must never contain or replace the read-only Base model")
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = config["base_model"]["model_id"]
    adapter_config_path.write_text(json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    identity = _adapter_identity(adapter_dir)
    identity["config_sha256"] = sha256_file(adapter_config_path)
    return identity


def _verify_step_proof(path: Path, run_root: Path, expected_base_sha: str) -> dict[str, Any]:
    resolved = path.resolve()
    proof = json.loads(resolved.read_text(encoding="utf-8"))
    if proof.get("status") != "passed" or proof.get("optimizer_steps") != 1 or not proof.get("lora_parameter_changed"):
        raise ValueError("E01-A train requires a passing one-step QLoRA proof")
    if proof.get("base_weight_sha256") != expected_base_sha:
        raise ValueError("one-step proof Base weight SHA does not match the configured Base")
    if resolved.is_relative_to(run_root.resolve()):
        raise ValueError("E01-A training run must not reuse the one-step proof Run directory")
    return proof


def execute_one_step(config: dict[str, Any], config_path: Path, run_id: str) -> Path:
    validate_training_mode(config, "one-step")
    run_root = _run_root(config, run_id)
    config_snapshot = run_root / "config_snapshot.yaml"
    config_snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {"run_id": run_id, "experiment_id": "E01-STEP", "status": "running", "created_at_utc": _now(), "config_sha256": sha256_file(config_snapshot), "init_adapter": None, "resume_from_checkpoint": None}
    _write_run_manifest(run_root, manifest)
    preflight, _ = _preflight(config)
    _write_json(run_root / "preflight.json", preflight)
    if preflight["status"] != "passed":
        manifest.update(status=preflight["status"], ended_at_utc=_now())
        _write_run_manifest(run_root, manifest)
        return run_root
    try:
        torch = __import__("torch")
        seed = int(config["training"]["seed"])
        _seed_everything(seed, torch)
        train_path, _dev_path, normalized_path, _ = _training_rows(config)
        samples, selected = _select_real_samples(config, train_path, normalized_path)
        _serialize_selected_samples(run_root / "selected_samples.json", samples)
        base_path = _resolve(config["base_model"]["local_path"])
        tokenizer, prompt, _schema, _enums = _load_protocol(config, base_path)
        model, torch, model_facts = _load_train_model(config, base_path)
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=int(config["protocol"]["max_train_seq_length"]))
        trainable = _trainable_parameters(model)
        sample_rows = [item["encoded"] for item in samples[:int(config["training"]["gradient_accumulation_steps"])]]
        callback = _optimizer_evidence_callback(
            model, trainable, config=config, samples=samples, tokenizer=tokenizer, prompt="",
            schema=None, enums=None, run_root=run_root, snapshot_adapter=False,
        )
        trainer = _build_sft_trainer(
            model, tokenizer, sample_rows, collator, config, max_steps=1,
            callback=callback, run_root=run_root,
        )
        trainer_data_path_facts = _snapshot_trainer_data_path_facts(trainer)
        trainer.train(resume_from_checkpoint=None)
        if len(callback.steps) != 1:
            raise RuntimeError(f"expected one real optimizer update, observed {len(callback.steps)}")
        proof = callback.steps[0]
        if proof.get("loss") is None or not proof.get("finite_gradients") or not proof.get("updated_parameter_count"):
            raise RuntimeError("Trainer one-step did not record finite loss, gradient, and LoRA update evidence")
        base_identity = inference._weight_identity(base_path, {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"].get("revision")})
        if base_identity.get("observed_weight_hash") != config["base_model"]["expected_weight_sha256"]:
            raise RuntimeError("Base model weights changed during the one-step run")
        adapter_facts = _save_adapter(model, run_root / "adapter", config)
        step_proof = {
            "status": "passed",
            "run_id": run_id,
            "optimizer_steps": 1,
            "micro_batches": len(sample_rows),
            "loss": proof["loss"],
            "gradient_norm": proof["gradient_norm"],
            "finite_gradients": proof["finite_gradients"],
            "nonzero_gradient_parameter_count": proof["nonzero_gradient_parameter_count"],
            "lora_parameter_changed": bool(proof["updated_parameter_count"]),
            "updated_parameter_count": proof["updated_parameter_count"],
            "updated_element_count": proof["updated_element_count"],
            "updated_parameter_names": proof["updated_parameter_names"],
            "base_weight_sha256": base_identity["observed_weight_hash"],
            "base_weights_unchanged": True,
            "model_facts": model_facts,
            "trainer_data_path": trainer_data_path_facts,
            "adapter_identity": adapter_facts,
            "optimizer_step_seconds": proof["optimizer_step_seconds"],
            "cuda_memory": proof["cuda_memory"],
            "selected_sample_ids": [row["sample_id"] for row in selected[:len(sample_rows)]],
            "init_adapter": None,
            "resume_from_checkpoint": None,
        }
        _write_json(run_root / "step_proof.json", step_proof)
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        manifest.update(status="passed", ended_at_utc=_now(), base_weight_sha256=base_identity["observed_weight_hash"], step_proof_sha256=sha256_file(run_root / "step_proof.json"))
    except Exception as exc:
        manifest.update(status="failed", ended_at_utc=_now(), error=f"{type(exc).__name__}: {exc}")
        _write_run_manifest(run_root, manifest)
        raise
    _write_run_manifest(run_root, manifest)
    return run_root


def execute_training(config: dict[str, Any], config_path: Path, run_id: str, step_proof_path: Path) -> Path:
    validate_training_mode(config, "train")
    run_root = _run_root(config, run_id)
    proof_record = _verify_step_proof(step_proof_path, run_root, config["base_model"]["expected_weight_sha256"])
    config_snapshot = run_root / "config_snapshot.yaml"
    config_snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "run_id": run_id, "experiment_id": "E01-A", "status": "running", "created_at_utc": _now(),
        "config_sha256": sha256_file(config_snapshot), "one_step_proof_path": str(step_proof_path.resolve()),
        "one_step_proof_sha256": sha256_file(step_proof_path.resolve()), "one_step_proof_status": proof_record["status"],
        "init_adapter": None, "resume_from_checkpoint": None,
    }
    _write_run_manifest(run_root, manifest)
    preflight, _ = _preflight(config)
    _write_json(run_root / "preflight.json", preflight)
    if preflight["status"] != "passed":
        manifest.update(status=preflight["status"], ended_at_utc=_now())
        _write_run_manifest(run_root, manifest)
        return run_root
    try:
        torch = __import__("torch")
        seed = int(config["training"]["seed"])
        _seed_everything(seed, torch)
        train_path, _dev_path, normalized_path, _ = _training_rows(config)
        samples, _selected_metadata = _select_real_samples(config, train_path, normalized_path)
        selected_metadata = _serialize_selected_samples(run_root / "selected_samples.json", samples)
        base_path = _resolve(config["base_model"]["local_path"])
        tokenizer, prompt, schema, enums = _load_protocol(config, base_path)
        pre_metrics = _preflight_model_metrics(config, samples, tokenizer, prompt, schema, enums, run_root, step=0)
        _seed_everything(seed, torch)
        model, torch, model_facts = _load_train_model(config, base_path)
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=int(config["protocol"]["max_train_seq_length"]))
        trainable = _trainable_parameters(model)
        callback = _optimizer_evidence_callback(
            model, trainable, config=config, samples=samples, tokenizer=tokenizer, prompt=prompt,
            schema=schema, enums=enums, run_root=run_root, snapshot_adapter=True,
        )
        rows = [item["encoded"] for item in samples]
        max_steps = int(config["training"]["max_optimizer_steps"])
        trainer = _build_sft_trainer(
            model, tokenizer, rows,
            SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=int(config["protocol"]["max_train_seq_length"])),
            config, max_steps=max_steps, callback=callback, run_root=run_root,
        )
        trainer_data_path_facts = _snapshot_trainer_data_path_facts(trainer)
        probe_prompt = _make_probe_prompt(samples, tokenizer, prompt, int(config["protocol"]["max_prompt_tokens"]))
        probe_token_budget = int(config["evaluation"]["probe_max_new_tokens"])
        trainer.train(resume_from_checkpoint=None)
        if len(callback.steps) != max_steps or len(callback.logged_steps) != max_steps:
            raise RuntimeError(
                f"Trainer optimizer evidence is incomplete: steps={len(callback.steps)}, logs={len(callback.logged_steps)}, expected={max_steps}"
            )
        log_rows = callback.steps
        train_diagnostics = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((run_root / "evaluation").glob("train_diagnostics_step_*.json"))
        ]
        if not any(item.get("step") == max_steps for item in train_diagnostics):
            raise RuntimeError("final optimizer step has no train_fitting_only generation evaluation")
        probe_before = _capture_probe(model, tokenizer, probe_prompt, max_new_tokens=probe_token_budget)
        _write_json(run_root / "generation_probe.json", {"before_release": probe_before, "prompt_token_ids": probe_prompt, "max_new_tokens": probe_token_budget, "do_sample": False})
        adapter_dir = run_root / "adapter"
        adapter_facts = _save_adapter(model, adapter_dir, config)
        base_before = inference._weight_identity(base_path, {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"].get("revision")})
        del trainer, model, callback, trainable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded, torch, reload_model_facts = _load_train_model(config, base_path, adapter_path=adapter_dir)
        probe_after = _capture_probe(reloaded, tokenizer, probe_prompt, max_new_tokens=probe_token_budget)
        base_after = inference._weight_identity(base_path, {"model_id": config["base_model"]["model_id"], "model_revision": config["base_model"].get("revision")})
        exact_match = probe_before["generated_token_ids"] == probe_after["generated_token_ids"]
        base_unchanged = base_before.get("observed_weight_hash") == base_after.get("observed_weight_hash") == config["base_model"]["expected_weight_sha256"]
        reload_check = {
            "status": "passed" if exact_match and base_unchanged else "failed",
            "token_ids_exact_match": exact_match,
            "before_release": probe_before,
            "after_reload": probe_after,
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "base_weights_unchanged": base_unchanged,
            "adapter_identity": adapter_facts,
            "reload_model_facts": reload_model_facts,
        }
        _write_json(run_root / "reload_check.json", reload_check)
        final_train_diagnostics = train_diagnostics[-1]
        window = min(10, len(log_rows))
        training_loss_start = (sum(row["loss"] for row in log_rows[:window]) / window) if log_rows else None
        training_loss_end = (sum(row["loss"] for row in log_rows[-window:]) / window) if log_rows else None
        fit_status = "engineering_passed_learning_inconclusive"
        if final_train_diagnostics.get("emotion", {}).get("accuracy", 0) >= float(config["evaluation"]["train_fitting_thresholds"]["emotion_accuracy"]) and final_train_diagnostics.get("raw", {}).get("raw_schema_valid_rate", 0) >= float(config["evaluation"]["train_fitting_thresholds"]["raw_schema_valid_rate"]) and training_loss_end is not None and training_loss_start is not None and training_loss_end < training_loss_start:
            fit_status = "engineering_passed_fit_evidence"
        manifest.update(
            status="passed" if reload_check["status"] == "passed" else "failed",
            ended_at_utc=_now(),
            base_weight_sha256=base_after.get("observed_weight_hash"),
            adapter_identity=adapter_facts,
            selected_samples_sha256=canonical_sha256(selected_metadata),
            trainer_data_path=trainer_data_path_facts,
            optimizer_steps=len(log_rows),
            fit_status=fit_status,
            initial_generation_metrics=pre_metrics,
            final_generation_metrics=final_train_diagnostics,
            training_loss_start=training_loss_start,
            training_loss_end=training_loss_end,
            reload_check_status=reload_check["status"],
            base_weights_unchanged=base_unchanged,
            dev_smoke_status="not_run",
        )
        _write_run_manifest(run_root, manifest)
        del reloaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        manifest.update(status="failed", ended_at_utc=_now(), error=f"{type(exc).__name__}: {exc}")
        _write_run_manifest(run_root, manifest)
        raise
    return run_root


def _preflight_model_metrics(
    config: dict[str, Any], samples: list[dict[str, Any]], tokenizer: Any, prompt: str,
    schema: dict[str, Any], enums: dict[str, list[str]], run_root: Path, *, step: int,
) -> dict[str, Any]:
    import torch
    # Build the initial model only after the preflight gate has passed.
    model, torch, _facts = _load_train_model(config, _resolve(config["base_model"]["local_path"]))
    result = _evaluate_train_samples(model, tokenizer, prompt, schema, enums, samples, config, step=step, run_root=run_root)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def execute_preflight(config: dict[str, Any], config_path: Path, run_id: str) -> Path:
    validate_training_mode(config, "preflight")
    run_root = _run_root(config, run_id)
    snapshot = run_root / "config_snapshot.yaml"
    snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    result, _ = _preflight(config)
    _write_json(run_root / "preflight.json", result)
    manifest = {
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "status": result["status"],
        "created_at_utc": _now(),
        "config_sha256": sha256_file(snapshot),
        "preflight_sha256": sha256_file(run_root / "preflight.json"),
        "init_adapter": None,
        "resume_from_checkpoint": None,
    }
    _write_run_manifest(run_root, manifest)
    return run_root


def execute_reload_check(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    if not root.is_relative_to(_resolve(config["data"]["output_root"])):
        raise ValueError("run-root must be under outputs/training")
    manifest_path = root / "manifest.json"
    probe_path = root / "generation_probe.json"
    adapter_dir = root / "adapter"
    for path in (manifest_path, probe_path, adapter_dir / "adapter_config.json"):
        if not path.exists():
            raise FileNotFoundError(f"reload-check input missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E01-A" or manifest.get("init_adapter") is not None or manifest.get("resume_from_checkpoint") is not None:
        raise ValueError("reload-check accepts only an independent E01-A run")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    base_path = _resolve(config["base_model"]["local_path"])
    base_before = inference._weight_identity(base_path, {
        "model_id": config["base_model"]["model_id"],
        "model_revision": config["base_model"].get("revision"),
    })
    runner_config = json.loads(_resolve(config["protocol"]["runner_config"]).read_text(encoding="utf-8"))
    tokenizer, _ = inference.load_runtime_tokenizer(runner_config, base_path)
    model, torch, facts = _load_train_model(config, base_path, adapter_path=adapter_dir)
    reloaded = _capture_probe(model, tokenizer, probe["prompt_token_ids"], max_new_tokens=int(probe["max_new_tokens"]))
    before = probe["before_release"]
    base_after = inference._weight_identity(base_path, {
        "model_id": config["base_model"]["model_id"],
        "model_revision": config["base_model"].get("revision"),
    })
    identity, _ = inference.adapter_weight_identity(adapter_dir)
    exact_match = before["generated_token_ids"] == reloaded["generated_token_ids"]
    base_unchanged = (
        base_before.get("observed_weight_hash")
        == base_after.get("observed_weight_hash")
        == config["base_model"]["expected_weight_sha256"]
    )
    result = {
        "status": "passed" if exact_match and base_unchanged else "failed",
        "token_ids_exact_match": exact_match,
        "base_weights_unchanged": base_unchanged,
        "base_weight_sha256_before": base_before.get("observed_weight_hash"),
        "base_weight_sha256_after": base_after.get("observed_weight_hash"),
        "before_release": before,
        "after_reload": reloaded,
        "adapter_identity": identity,
        "reload_model_facts": facts,
    }
    _write_json(root / "reload_check_standalone.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated E01 QLoRA preflight and training")
    parser.add_argument("--config", default="configs/training/e01_a.yaml")
    parser.add_argument("--mode", choices=("preflight", "one-step", "train", "reload-check"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--run-root")
    parser.add_argument("--one-step-proof")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_path = load_config(args.config)
    validate_training_mode(config, args.mode)
    if args.mode == "reload-check":
        if not args.run_root:
            raise ValueError("reload-check requires --run-root")
        result = execute_reload_check(config, _resolve(args.run_root))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "passed" else 2
    if not args.run_id:
        raise ValueError(f"--run-id is required for mode={args.mode}")
    if args.mode == "preflight":
        run_root = execute_preflight(config, config_path, args.run_id)
    elif args.mode == "one-step":
        run_root = execute_one_step(config, config_path, args.run_id)
    else:
        if not args.one_step_proof:
            raise ValueError("train mode requires --one-step-proof from an independent E01-STEP run")
        run_root = execute_training(config, config_path, args.run_id, _resolve(args.one_step_proof))
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"run_root": str(run_root), "status": manifest["status"]}, ensure_ascii=False))
    return 0 if manifest["status"] in {"passed", "blocked_gpu_unavailable", "blocked_bf16_unavailable"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
