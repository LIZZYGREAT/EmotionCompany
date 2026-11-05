#!/usr/bin/env python3
"""Run only bounded E03-A6 F3 training stages; this CLI has no Full mode."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.f3_protocol import parse_f3
from src.training.e03_a6_contract import (
    canonical_sha256,
    f3_prefix_through_r_marker,
    validate_microfit_config,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def _package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("microfit config must be a YAML object")
    validate_microfit_config(config)
    return config


def _project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "base_model": config["base_model"],
        "quantization": config["quantization"],
        "lora": config["lora"],
        "training": config["training"],
    }


def _base_identity(config: dict[str, Any]) -> dict[str, Any]:
    from participant import run_inference as inference

    return inference._weight_identity(
        _project_path(config["base_model"]["local_path"]),
        {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"]["revision"],
        },
    )


def _load_microfit_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_dir = _project_path(config["data"]["dataset_dir"])
    normalized_path = _project_path(config["data"]["normalized_train"])
    manifest_path = _project_path(config["data"]["manifest"])
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("status") != "cpu_preparation_passed":
        raise ValueError("F3 dataset manifest is not a passing CPU preparation artifact")
    if dataset_manifest.get("format_version") != "F3":
        raise ValueError("F3 dataset manifest format identity changed")

    targets: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(normalized_path):
        target = row.get("target", {})
        profile = target.get("user_profile", {}) if isinstance(target, dict) else {}
        labels = {
            name: list(profile.get(name, []))
            for name in ("personality_traits", "interests", "style")
        }
        if any(labels.values()):
            targets[str(row["sample_id"])] = {
                "conversation_id": row["conversation_id"],
                "labels": labels,
                "emotion_label": target.get("emotion_label"),
            }
    if len(targets) < int(config["training"]["sample_count"]):
        raise ValueError("F3 Train has too few non-empty profile rows for microfit")

    encoded_by_id: dict[str, dict[str, Any]] = {}
    target_ids = set(targets)
    for row in _read_jsonl(dataset_dir / "train.jsonl"):
        sample_id = row.get("sample_id")
        if sample_id in target_ids:
            encoded_by_id[sample_id] = row
    candidates_in_f3 = set(targets) & set(encoded_by_id)
    if len(candidates_in_f3) < int(config["training"]["sample_count"]):
        raise ValueError("F3 SFT data has too few non-empty rows for microfit")

    candidates = sorted(
        (
            encoded_by_id[sample_id]
            for sample_id in candidates_in_f3
        ),
        key=lambda row: (len(row["input_ids"]), row["sample_id"]),
    )
    selected: list[dict[str, Any]] = []
    conversations: set[str] = set()
    for row in candidates:
        conversation_id = targets[row["sample_id"]]["conversation_id"]
        if conversation_id in conversations:
            continue
        selected.append(row)
        conversations.add(conversation_id)
        if len(selected) == int(config["training"]["sample_count"]):
            break
    if len(selected) != int(config["training"]["sample_count"]):
        raise ValueError("could not select enough distinct non-empty Train conversations")
    metadata = [{
        "sample_id": row["sample_id"],
        "conversation_id": targets[row["sample_id"]]["conversation_id"],
        "sequence_length": len(row["input_ids"]),
        "emotion_label": targets[row["sample_id"]]["emotion_label"],
        "gold_nonempty_dimensions": [name for name, labels in targets[row["sample_id"]]["labels"].items() if labels],
        "gold_profile_labels": targets[row["sample_id"]]["labels"],
    } for row in selected]
    return selected, metadata


def _load_tokenizer(config: dict[str, Any]):
    from participant import run_inference as inference

    path = _project_path(config["protocol"]["inference_config"])
    runner_config = json.loads(path.read_text(encoding="utf-8"))
    tokenizer, facts = inference.load_runtime_tokenizer(
        runner_config, _project_path(config["base_model"]["local_path"])
    )
    return tokenizer, facts


def _set_cuda_budget(torch: Any, config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("blocked_gpu_unavailable: CUDA is required for bounded F3 microfit")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_bf16_unavailable: F3 microfit requires BF16 CUDA")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    fraction = float(config["quantization"]["max_cuda_memory_fraction"])
    if not 0.5 <= fraction <= 0.95:
        raise ValueError("CUDA memory fraction must stay between 0.5 and 0.95")
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    minimum_free = 5 * 1024**3
    if free_bytes < minimum_free:
        raise RuntimeError(
            f"blocked_gpu_memory: only {free_bytes} free bytes before model load; at least {minimum_free} required"
        )
    return {
        "device_index": device,
        "gpu_name": properties.name,
        "total_memory_bytes": total_bytes,
        "initial_free_memory_bytes": free_bytes,
        "process_memory_fraction": fraction,
        "process_allocator_limit_bytes": int(total_bytes * fraction),
        "bf16_supported": True,
    }


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _generate(model: Any, tokenizer: Any, prompt_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    from src.training.training_sft import _capture_probe

    captured = _capture_probe(model, tokenizer, prompt_ids, max_new_tokens=max_new_tokens)
    generated = captured["generated_token_ids"]
    captured["eos_generated"] = tokenizer.eos_token_id in generated
    try:
        captured["parsed_f3"] = parse_f3(captured["raw_text"], _f3_enums())
        captured["parse_status"] = "success"
    except Exception as exc:
        captured["parsed_f3"] = None
        captured["parse_status"] = getattr(exc, "code", "format_parse_failure")
        captured["parse_error"] = f"{type(exc).__name__}: {exc}"
    return captured


def _protocol_prefix_ids(generation: dict[str, Any], tokenizer: Any) -> tuple[list[int], str]:
    prefix_ids, prefix_text = f3_prefix_through_r_marker(
        generation["generated_token_ids"],
        lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
    )
    parse_f3(prefix_text, _f3_enums())
    return prefix_ids, prefix_text


def _f3_enums() -> dict[str, list[str]]:
    from participant import run_inference as inference

    schema = inference.load_submission_schema()
    from src.shared.f3_protocol import submission_enums
    return submission_enums(schema)


def execute_microfit(config_path: Path, run_id: str) -> Path:
    config = _load_config(config_path)
    run_root = _project_path("outputs/training") / run_id
    if not run_id.startswith("E03-A6-F3-MICROFIT-") or run_root.exists():
        raise ValueError("microfit Run ID must be unique and start with E03-A6-F3-MICROFIT-")
    run_root.mkdir(parents=True)
    config_sha = _sha256(config_path)
    config_snapshot = run_root / "config_snapshot.yaml"
    config_snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    rows, selected_metadata = _load_microfit_rows(config)
    selected_ids = [row["sample_id"] for row in rows]
    dataset_manifest_path = _project_path(config["data"]["manifest"])
    dataset_dir = _project_path(config["data"]["dataset_dir"])
    data_facts = {
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "f3_train_sha256": _sha256(dataset_dir / "train.jsonl"),
        "normalized_train_sha256": _sha256(_project_path(config["data"]["normalized_train"])),
        "selected_sample_ids": selected_ids,
        "selected_sample_ids_sha256": canonical_sha256(selected_ids),
        "selected_samples": selected_metadata,
        "train_sample_count": len(rows),
    }
    code_files = {
        "scripts/e03_a6_train.py": _sha256(Path(__file__).resolve()),
        "src/training/e03_a6_contract.py": _sha256(ROOT / "src/training/e03_a6_contract.py"),
        "src/training/training_sft.py": _sha256(ROOT / "src/training/training_sft.py"),
        "participant/run_inference.py": _sha256(ROOT / "participant/run_inference.py"),
    }
    _write_json(run_root / "selected_samples.json", data_facts)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "training_scope": config["training_scope"],
        "status": "running",
        "stage": "preflight",
        "created_at_utc": _now(),
        "config_sha256": config_sha,
        "training_code_files": code_files,
        "training_code_sha256": canonical_sha256(code_files),
        "data": data_facts,
        "training_budget": {
            "max_optimizer_steps": int(config["training"]["max_optimizer_steps"]),
            "max_training_seconds": int(config["training"]["max_training_seconds"]),
            "cuda_memory_fraction": float(config["quantization"]["max_cuda_memory_fraction"]),
        },
        "approval_boundary": "bounded_microfit_only; does not authorize Pilot or Full training",
        "full_training_status": "not_authorized_not_run",
    }
    _write_json(run_root / "manifest.json", manifest)
    model = tokenizer = None
    try:
        import torch
        from participant import run_inference as inference
        from src.data.sft_collator import SFTDataCollator
        from src.training.training_sft import _load_train_model, _save_adapter, _step_update

        cuda_facts = _set_cuda_budget(torch, config)
        base_path = _project_path(config["base_model"]["local_path"])
        base_before = _base_identity(config)
        expected_base_sha = config["base_model"]["expected_weight_sha256"]
        if base_before.get("observed_weight_hash") != expected_base_sha:
            raise ValueError("local Qwen3-1.7B Base weight SHA differs from the frozen config")
        tokenizer, tokenizer_facts = _load_tokenizer(config)
        training_config = _training_config(config)
        model, torch, model_facts = _load_train_model(training_config, base_path)
        trained_lora_dtypes = sorted({
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.requires_grad
        })
        trainable_before = {
            name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        collator = SFTDataCollator(
            tokenizer.pad_token_id,
            max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
        )
        _seed_everything(int(config["training"]["seed"]), torch)
        prefit = [
            _generate(
                model, tokenizer, row["input_ids"][:row["prompt_length"]],
                int(config["evaluation"]["max_new_tokens"]),
            )
            for row in rows
        ]
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        step_log: list[dict[str, Any]] = []
        training_started = time.perf_counter()
        for step in range(1, int(config["training"]["max_optimizer_steps"]) + 1):
            if time.perf_counter() - training_started >= int(config["training"]["max_training_seconds"]):
                raise TimeoutError("bounded microfit reached its frozen wall-clock budget")
            row = rows[(step - 1) % len(rows)]
            evidence = _step_update(model, torch, optimizer, [row], collator, accumulation_steps=1)
            if not math.isfinite(float(evidence["loss"])) or not evidence["updated_parameter_count"]:
                raise RuntimeError("microfit optimizer step did not produce finite loss and LoRA updates")
            step_log.append({"optimizer_step": step, **evidence})
            _write_json(run_root / "training_steps.json", step_log)
        training_seconds = time.perf_counter() - training_started
        adapter_facts = _save_adapter(model, run_root / "adapter", training_config)
        trained = [
            _generate(
                model, tokenizer, row["input_ids"][:row["prompt_length"]],
                int(config["evaluation"]["max_new_tokens"]),
            )
            for row in rows
        ]
        _write_json(run_root / "generation_probe.json", {
            "sample_ids": selected_ids,
            "before_training": prefit,
            "after_training_before_reload": trained,
            "do_sample": False,
            "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
        })
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, torch, reload_facts = _load_train_model(
            training_config,
            base_path,
            adapter_path=run_root / "adapter",
            autocast_adapter_dtype=False,
        )
        after_reload = [
            _generate(
                reloaded, tokenizer, row["input_ids"][:row["prompt_length"]],
                int(config["evaluation"]["max_new_tokens"]),
            )
            for row in rows
        ]
        reload_match = [
            before["generated_token_ids"] == after["generated_token_ids"]
            for before, after in zip(trained, after_reload)
        ]
        prefix_pairs = [
            (_protocol_prefix_ids(before, tokenizer), _protocol_prefix_ids(after, tokenizer))
            for before, after in zip(trained, after_reload)
        ]
        prefix_match = [before[0] == after[0] for before, after in prefix_pairs]
        prefix_text_match = [before[1] == after[1] for before, after in prefix_pairs]
        reloaded_lora_dtypes = sorted({
            str(parameter.dtype)
            for name, parameter in reloaded.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        })
        dtype_match = trained_lora_dtypes == reloaded_lora_dtypes
        base_after = _base_identity(config)
        base_unchanged = base_after.get("observed_weight_hash") == expected_base_sha == base_before.get("observed_weight_hash")
        nonempty_predictions = sum(
            bool(prediction.get("parsed_f3") and any(prediction["parsed_f3"]["user_profile"].values()))
            for prediction in after_reload
        )
        _write_json(run_root / "reload_check.json", {
            "status": "passed" if all(prefix_match) and all(prefix_text_match) and dtype_match and base_unchanged else "failed",
            "full_generation_token_ids_exact_match_per_sample": reload_match,
            "full_generation_token_ids_exact_match": all(reload_match),
            "f3_protocol_prefix_token_ids_exact_match_per_sample": prefix_match,
            "f3_protocol_prefix_text_exact_match_per_sample": prefix_text_match,
            "f3_protocol_prefix_token_ids_exact_match": all(prefix_match),
            "f3_protocol_prefix_text_exact_match": all(prefix_text_match),
            "trainable_lora_dtypes_before_save": trained_lora_dtypes,
            "lora_dtypes_after_reload": reloaded_lora_dtypes,
            "lora_dtype_exact_match": dtype_match,
            "base_weights_unchanged": base_unchanged,
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "adapter_identity": adapter_facts,
            "reload_runtime": reload_facts,
            "adapter_dtype_autocast": False,
        })
        _write_json(run_root / "generation_reload.json", {
            "sample_ids": selected_ids,
            "after_reload": after_reload,
        })
        losses = [float(row["loss"]) for row in step_log]
        engineering_passed = (
            all(prefix_match) and all(prefix_text_match) and dtype_match and base_unchanged
        )
        microfit_passed = (
            len(step_log) == int(config["training"]["max_optimizer_steps"])
            and all(math.isfinite(loss) for loss in losses)
            and engineering_passed
            and nonempty_predictions >= int(config["evaluation"]["expected_nonempty_profile_predictions"])
        )
        result = {
            "status": "passed" if microfit_passed else "failed",
            "optimizer_steps": len(step_log),
            "training_seconds": training_seconds,
            "loss_first_4_mean": sum(losses[:4]) / min(4, len(losses)),
            "loss_last_4_mean": sum(losses[-4:]) / min(4, len(losses)),
            "finite_loss_steps": sum(math.isfinite(loss) for loss in losses),
            "all_steps_updated_lora": all(bool(row["updated_parameter_count"]) for row in step_log),
            "f3_strict_parse_successes_after_reload": sum(row["parse_status"] == "success" for row in after_reload),
            "f3_nonempty_profile_predictions_after_reload": nonempty_predictions,
            "selected_nonempty_train_samples": len(rows),
            "reload_token_ids_exact_match": all(reload_match),
            "reload_f3_protocol_prefix_token_ids_exact_match": all(prefix_match),
            "reload_lora_dtype_exact_match": dtype_match,
            "full_generation_response_token_ids_mismatches": sum(not value for value in reload_match),
            "training_code_sha256": manifest["training_code_sha256"],
            "base_weights_unchanged": base_unchanged,
            "adapter_identity": adapter_facts,
            "cuda_preflight": cuda_facts,
            "cuda_peak_memory": inference.cuda_memory_facts(torch),
            "model_facts": model_facts,
            "tokenizer": tokenizer_facts,
            "package_versions": _package_versions(),
            "full_training_status": "not_authorized_not_run",
        }
        _write_json(run_root / "microfit_result.json", result)
        manifest.update(
            status=result["status"],
            stage="microfit_complete" if microfit_passed else "microfit_failed_gate",
            ended_at_utc=_now(),
            training_seconds=training_seconds,
            base_weight_sha256=base_after.get("observed_weight_hash"),
            adapter_weight_sha256=adapter_facts["weight_sha256"],
            reload_check_sha256=_sha256(run_root / "reload_check.json"),
            result_sha256=_sha256(run_root / "microfit_result.json"),
        )
        _write_json(run_root / "manifest.json", manifest)
        if not microfit_passed:
            raise RuntimeError(
                "microfit did not meet the learning/reload gate; Pilot training remains blocked"
            )
        del reloaded, optimizer
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        manifest.update(
            status="failed",
            stage="microfit_failed",
            ended_at_utc=_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(run_root / "manifest.json", manifest)
        raise
    return run_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("microfit",), required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/training/e03_a6_f3_microfit.yaml"
    )
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_root = execute_microfit(args.config.resolve(), args.run_id)
    print(json.dumps({"status": "passed", "run_root": str(run_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
