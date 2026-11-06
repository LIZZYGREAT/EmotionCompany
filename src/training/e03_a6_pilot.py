"""Isolated, fixed-budget E03-A6 F3 Pilot training and checkpoint validation."""

from __future__ import annotations

import gc
import copy
import hashlib
import importlib.metadata
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from participant import run_inference as inference
from src.shared.f3_protocol import parse_f3
from src.training.e03_a6_contract import (
    canonical_sha256,
    validate_historical_pilot_selection,
    validate_expanded_pilot_config,
    validate_expanded_pilot_resume_manifest,
    validate_pilot_config,
    validate_pilot_resume_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_SELECTION_SHA256 = "b53336224071914500aa91a4ad82c6f9e6334b11eb5a69231a6b4e20ba09314e"
MAX_F3_TRAIN_ROWS = 16192
MAX_F3_DEV_ROWS = 1821
EXPECTED_PILOT_PAUSE_STEP = 32
EXPECTED_PILOT_STEPS = 128
PRECHECKPOINT_TRAINING_CODE_SHA256 = "8999cd1465cd447fc483731e8902cd3b231a99ce21661368ba8e02398e6a0e2e"
PRECHECKPOINT_PILOT_MODULE_SHA256 = "7cb5e9468b061faae5c10422d5acd28ad2bd0f47ae6434fd16c88c25ede052e0"
RECOVERABLE_RESUME_FAILURE = "NameError: name '_read_training_log' is not defined"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
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


def _project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Pilot config must be a YAML object")
    if config.get("training_scope") == "supplemental_pilot":
        validate_expanded_pilot_config(config)
    else:
        validate_pilot_config(config)
    return config


def _training_code_facts() -> dict[str, Any]:
    paths = (
        "scripts/e03_a6_train.py",
        "src/training/e03_a6_pilot.py",
        "src/training/e03_a6_contract.py",
        "src/training/training_sft.py",
        "src/data/e03_a6_sft.py",
        "src/shared/f3_protocol.py",
        "src/shared/prompting.py",
        "participant/run_inference.py",
    )
    files = {name: _sha256(ROOT / name) for name in paths}
    return {"files": files, "sha256": canonical_sha256(files)}


def _package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _expanded_selection(config: dict[str, Any], historical_selection: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    from src.training.training_sft import deterministic_select_pilot_samples

    data = config["data"]
    f2_train_path = _project_path(data["historical_f2_train"])
    normalized_path = _project_path(data["normalized_train"])
    f2_rows = [
        {key: row.get(key) for key in (
            "sample_id", "conversation_id", "source_split", "prompt_length", "completion_length",
        )}
        for row in _read_jsonl(f2_train_path)
    ]
    normalized_rows = [
        {key: row.get(key) for key in ("sample_id", "conversation_id", "target")}
        for row in _read_jsonl(normalized_path)
    ]
    selection_config = config["training"]["sample_selection"]
    rows = deterministic_select_pilot_samples(
        f2_rows,
        normalized_rows,
        count=int(selection_config["count"]),
        seed=int(selection_config["seed"]),
        length_bins=int(selection_config["length_bins"]),
    )
    rows_sha256 = canonical_sha256(rows)
    if rows_sha256 != selection_config["expected_rows_sha256"]:
        raise ValueError("expanded Pilot ordered 1,024-row selection hash changed")
    base_ids = {row["sample_id"] for row in historical_selection["rows"]}
    selected_ids = {row["sample_id"] for row in rows}
    if len(rows) != 1024 or len(selected_ids) != 1024 or len(base_ids) != 512:
        raise ValueError("expanded Pilot selection must contain 1,024 unique IDs and the 512-row base")
    if not base_ids <= selected_ids:
        raise ValueError("expanded Pilot selection does not preserve every historical 512-row ID")
    return rows, rows_sha256


def _load_pilot_assets(config: dict[str, Any]) -> dict[str, Any]:
    from src.training.training_sft import _select_sft_dev_rows, validate_train_row

    data_config = config["data"]
    dataset_dir = _project_path(data_config["dataset_dir"])
    manifest_path = _project_path(data_config["manifest"])
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("status") != "cpu_preparation_passed" or data_manifest.get("format_version") != "F3":
        raise ValueError("A6 F3 data manifest is not a passing F3 preparation artifact")
    if data_manifest.get("outputs", {}).get("train_count") != MAX_F3_TRAIN_ROWS:
        raise ValueError("F3 Train row count differs from the frozen 16,192-row cohort")
    if data_manifest.get("outputs", {}).get("dev_count") != MAX_F3_DEV_ROWS:
        raise ValueError("F3 SFT Dev row count differs from the frozen 1,821-row cohort")

    selection_path = _project_path(data_config["historical_pilot_selection"])
    historical_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pilot_ids = validate_historical_pilot_selection(historical_selection)
    selection_sha = canonical_sha256(historical_selection["rows"])
    if selection_sha != data_config["expected_historical_selection_sha256"]:
        raise ValueError("historical E01-B Pilot ordered sample identity changed")
    if historical_selection.get("sha256") != HISTORICAL_SELECTION_SHA256:
        raise ValueError("historical E01-B selected_samples.json identity changed")
    expanded = config.get("training_scope") == "supplemental_pilot"
    if expanded:
        selection_rows, selection_rows_sha256 = _expanded_selection(config, historical_selection)
        selected_ids = [row["sample_id"] for row in selection_rows]
    else:
        selection_rows = historical_selection["rows"]
        selection_rows_sha256 = selection_sha
        selected_ids = pilot_ids
    wanted = set(selected_ids)

    normalized_by_id: dict[str, dict[str, Any]] = {}
    normalized_path = _project_path(data_config["normalized_train"])
    for row in _read_jsonl(normalized_path):
        if row.get("sample_id") in wanted:
            target = row.get("target", {})
            profile = target.get("user_profile", {})
            normalized_by_id[row["sample_id"]] = {
                "conversation_id": row.get("conversation_id"),
                "emotion_label": target.get("emotion_label"),
                "profile_empty": not any(profile.get(dim, []) for dim in (
                    "personality_traits", "interests", "style",
                )),
            }
    if set(normalized_by_id) != wanted:
        raise ValueError("selected Pilot IDs contain rows absent from normalized F3 Train")

    encoded_by_id: dict[str, dict[str, Any]] = {}
    actual_train_ids: list[str] = []
    for row in _read_jsonl(dataset_dir / "train.jsonl"):
        sample_id = row.get("sample_id")
        actual_train_ids.append(sample_id)
        if sample_id in wanted:
            encoded_by_id[sample_id] = row
    if len(actual_train_ids) != MAX_F3_TRAIN_ROWS or len(set(actual_train_ids)) != MAX_F3_TRAIN_ROWS:
        raise ValueError("F3 Train dataset contains unexpected row count or duplicate IDs")
    if canonical_sha256(actual_train_ids) != data_manifest["split_summary"]["train"]["ids_sha256"]:
        raise ValueError("F3 Train ordered ID hash differs from its passing manifest")
    if set(encoded_by_id) != wanted:
        raise ValueError("selected Pilot IDs contain rows absent from the frozen F3 Train set")

    audit_by_id: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(dataset_dir / "audit_train.jsonl"):
        if row.get("sample_id") in wanted:
            audit_by_id[row["sample_id"]] = row
    if set(audit_by_id) != wanted:
        raise ValueError("F3 audit rows do not cover every selected Pilot Train ID")

    ordered_rows = []
    selected_metadata = []
    for old_row in selection_rows:
        sample_id = old_row["sample_id"]
        encoded = encoded_by_id[sample_id]
        normalized = normalized_by_id[sample_id]
        audit = audit_by_id[sample_id]
        if normalized["conversation_id"] != old_row["conversation_id"]:
            raise ValueError(f"{sample_id}: selected conversation identity changed")
        if normalized["emotion_label"] != old_row["emotion_label"]:
            raise ValueError(f"{sample_id}: selected emotion identity changed")
        if normalized["profile_empty"] != old_row["profile_empty"]:
            raise ValueError(f"{sample_id}: selected empty/non-empty profile identity changed")
        validate_train_row(
            encoded,
            expected_eos_token_id=int(config["protocol"]["eos_token_id"]),
            expected_split="train",
        )
        if not all(audit.get(key) is True for key in (
            "p0_selected_history_matches_f2", "completion_round_trip_lossless",
            "prompt_labels_masked", "terminal_eos_supervised",
        )):
            raise ValueError(f"{sample_id}: F3 audit invariants do not pass")
        if int(encoded["prompt_length"]) > int(config["protocol"]["max_prompt_tokens"]):
            raise ValueError(f"{sample_id}: F3 Pilot prompt exceeds P=1280")
        if len(encoded["input_ids"]) > int(config["protocol"]["max_train_seq_length"]):
            raise ValueError(f"{sample_id}: F3 Pilot training sequence exceeds S=2048")
        ordered_rows.append(encoded)
        selected_metadata.append({
            **old_row,
            "f3_prompt_length": int(encoded["prompt_length"]),
            "f3_completion_length": int(encoded["completion_length"]),
            "f3_sequence_length": len(encoded["input_ids"]),
            "p0_selected_history_matches_f2": True,
            "terminal_eos_supervised": True,
        })

    if len(ordered_rows) != len(selected_ids):
        raise ValueError("Pilot rows do not cover the exact frozen selected Train IDs")

    dev_path = dataset_dir / "dev.jsonl"
    dev_rows, dev_facts = _select_sft_dev_rows(config, dev_path)
    dev_id_order = [row["sample_id"] for row in dev_rows]
    dev_selection = {
        **dev_facts,
        "sample_ids_sha256": canonical_sha256(dev_id_order),
        "sample_ids": dev_id_order,
    }
    source_paths = {
        "f3_manifest": manifest_path,
        "f3_train": dataset_dir / "train.jsonl",
        "f3_dev": dev_path,
        "f3_audit_train": dataset_dir / "audit_train.jsonl",
        "historical_selection": selection_path,
        "normalized_train": normalized_path,
        "normalized_dev": _project_path(data_config["normalized_dev"]),
    }
    if expanded:
        source_paths["historical_f2_train"] = _project_path(data_config["historical_f2_train"])
    facts = {f"{name}_sha256": _sha256(path) for name, path in source_paths.items()}
    facts.update({
        "historical_selection_rows_sha256": selection_sha,
        "selected_selection_rows_sha256": selection_rows_sha256,
        "selected_sample_ids_sha256": canonical_sha256(selected_ids),
        "selected_sample_count": len(ordered_rows),
        "selected_sample_ids": selected_ids,
        "selected_unique_conversation_count": len({row["conversation_id"] for row in selected_metadata}),
        "f3_train_count": len(actual_train_ids),
        "f3_train_ids_sha256": canonical_sha256(actual_train_ids),
        "sft_dev_count": len(dev_rows),
        "sft_dev_selection": {
            key: value for key, value in dev_selection.items() if key != "sample_ids"
        },
        "sft_dev_ids_sha256": dev_selection["sample_ids_sha256"],
        "sft_dev_sample_ids": dev_id_order,
        "only_train_rows_used_for_optimizer_updates": True,
        "normalized_dev_used_only_for_identity": True,
        "official_val_used": False,
    })
    if expanded:
        base_ids = set(pilot_ids)
        added_ids = [sample_id for sample_id in selected_ids if sample_id not in base_ids]
        facts.update({
            "base_sample_count": len(base_ids),
            "base_sample_ids_sha256": canonical_sha256(pilot_ids),
            "added_sample_count": len(added_ids),
            "added_sample_ids_sha256": canonical_sha256(added_ids),
        })
    return {
        "train_rows": ordered_rows,
        "selected_metadata": selected_metadata,
        "historical_selection": historical_selection,
        "selected_selection_rows": selection_rows,
        "selected_selection_rows_sha256": selection_rows_sha256,
        "supplemental_selection": expanded,
        "dev_rows": dev_rows,
        "dev_selection": dev_selection,
        "data_facts": facts,
    }


def _cuda_preflight(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("blocked_gpu_unavailable: A6 Pilot requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_bf16_unavailable: A6 Pilot requires BF16 CUDA")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    memory_fraction = float(config["quantization"]["max_cuda_memory_fraction"])
    if not 0.5 <= memory_fraction <= 0.95:
        raise ValueError("Pilot CUDA allocator fraction is outside the reviewed 0.5-0.95 range")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    minimum_free = 5 * 1024**3
    if free_bytes < minimum_free:
        raise RuntimeError(
            f"blocked_gpu_memory: {free_bytes} free bytes before model load; {minimum_free} required"
        )
    output_path = _project_path(config["data"]["output_root"])
    output_path.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(output_path).free
    required_disk = float(config["training"]["min_free_disk_gb"]) * 1024**3
    if disk_free < required_disk:
        raise RuntimeError(
            f"blocked_disk_space: {disk_free} free bytes; {required_disk:.0f} required"
        )
    return torch, {
        "status": "passed",
        "gpu_name": properties.name,
        "device_index": device,
        "bf16_supported": True,
        "free_bytes_before_load": free_bytes,
        "total_memory_bytes": total_bytes,
        "allocator_fraction": memory_fraction,
        "allocator_limit_bytes": int(total_bytes * memory_fraction),
        "disk_free_bytes": disk_free,
        "disk_required_bytes": int(required_disk),
    }


def _base_identity(config: dict[str, Any]) -> dict[str, Any]:
    return inference._weight_identity(
        _project_path(config["base_model"]["local_path"]),
        {
            "model_id": config["base_model"]["model_id"],
            "model_revision": config["base_model"]["revision"],
        },
    )


def _tokenizer(config: dict[str, Any]):
    inference_config_path = _project_path(config["protocol"]["inference_config"])
    runner_config = json.loads(inference_config_path.read_text(encoding="utf-8"))
    tokenizer, facts = inference.load_runtime_tokenizer(
        runner_config,
        _project_path(config["base_model"]["local_path"]),
    )
    return tokenizer, facts


def _source_identity(
    config_path: Path,
    config: dict[str, Any],
    assets: dict[str, Any],
    tokenizer_facts: dict[str, Any],
) -> dict[str, Any]:
    code_facts = _training_code_facts()
    f3_prompt_path = _project_path("configs/prompts/p_a6_f3.txt")
    p0_prompt_path = _project_path("configs/prompts/p0.txt")
    runner_config_path = _project_path(config["protocol"]["inference_config"])
    runner_config = json.loads(runner_config_path.read_text(encoding="utf-8"))
    if runner_config.get("format_version") != "F3":
        raise ValueError("Pilot inference configuration must explicitly select F3")
    if runner_config.get("prompt", {}).get("sha256") != _sha256(f3_prompt_path):
        raise ValueError("Pilot Runner F3 prompt identity differs from the prepared data")
    if runner_config.get("selection_prompt", {}).get("version") != "P0" or runner_config.get("selection_prompt", {}).get("sha256") != _sha256(p0_prompt_path):
        raise ValueError("Pilot Runner history selector differs from the frozen P0/T1 identity")
    identity = {
        "run_config_sha256": _sha256(config_path),
        "training_code_files": code_facts["files"],
        "training_code_sha256": code_facts["sha256"],
        "data": assets["data_facts"],
        "prompt_f3_sha256": _sha256(f3_prompt_path),
        "prompt_p0_sha256": _sha256(p0_prompt_path),
        "runner_config_sha256": _sha256(runner_config_path),
        "tokenizer_signature_sha256": tokenizer_facts["signature_sha256"],
        "tokenizer_template_sha256": tokenizer_facts["template_sha256"],
        "base_weight_sha256": _base_identity(config)["observed_weight_hash"],
        "training_scope": config["training_scope"],
        "format_version": "F3",
        "init_adapter": None,
        "resume_from_checkpoint": None,
    }
    if identity["base_weight_sha256"] != config["base_model"]["expected_weight_sha256"]:
        raise ValueError("Pilot Base model weight SHA differs from the frozen Qwen3-1.7B identity")
    if tokenizer_facts["signature_sha256"] != "1662c9e99c6185441026a168b2dd6b09af103dc6dc3563b72f598411f14d1913":
        raise ValueError("Pilot Tokenizer signature differs from the frozen D0.4.2 identity")
    if tokenizer_facts["template_sha256"] != "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8":
        raise ValueError("Pilot Chat Template differs from the frozen D0.4.2 identity")
    if identity["prompt_f3_sha256"] != "5779cc2ede7d1d14991b6760b7975680381293df2947b3eb6ca3a0f33d14e8a4":
        raise ValueError("Pilot F3 prompt differs from the prepared dataset identity")
    if identity["prompt_p0_sha256"] != "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2":
        raise ValueError("Pilot P0 history-selection prompt differs from the frozen baseline")
    return identity


def _run_root(run_id: str) -> Path:
    if not run_id.startswith("E03-A6-F3-PILOT-") or len(run_id) > 96:
        raise ValueError("Pilot Run ID must start with E03-A6-F3-PILOT- and remain under 96 characters")
    root = _project_path("outputs/training") / run_id
    if not root.is_relative_to(_project_path("outputs/training")):
        raise ValueError("Pilot Run path escaped outputs/training")
    return root


def _load_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot Run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Pilot Run manifest must be a JSON object")
    return manifest


def _same_source_identity(expected: dict[str, Any], manifest: dict[str, Any]) -> None:
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Pilot source identity changed since Run creation: {key}")


def _adopt_checkpoint_compatible_resume_code(
    source_identity: dict[str, Any], manifest: dict[str, Any],
) -> str:
    """Allow one frozen resume-orchestration repair without changing checkpoint identity."""
    old_code_sha = manifest.get("training_code_sha256")
    current_code_sha = source_identity["training_code_sha256"]
    old_files = manifest.get("training_code_files")
    current_files = source_identity["training_code_files"]
    compatibility = manifest.get("resume_code_compatibility")
    if compatibility:
        if (
            old_code_sha != current_code_sha
            or old_files != current_files
            or compatibility.get("continuation_training_code_sha256") != current_code_sha
            or compatibility.get("continuation_training_code_files") != current_files
        ):
            raise ValueError("Pilot resume code changed again after its one-time compatibility repair")
        return str(compatibility["checkpoint_training_code_sha256"])
    if old_code_sha == current_code_sha and old_files == current_files:
        return current_code_sha

    changed_files = sorted(
        name for name in set(old_files or {}) | set(current_files or {})
        if (old_files or {}).get(name) != (current_files or {}).get(name)
    )
    expected_files = set(old_files or {})
    if (
        manifest.get("run_id") != "E03-A6-F3-PILOT-20261106-01"
        or old_code_sha != PRECHECKPOINT_TRAINING_CODE_SHA256
        or manifest.get("checkpoint_identity", {}).get("training_code_sha256") != PRECHECKPOINT_TRAINING_CODE_SHA256
        or (old_files or {}).get("src/training/e03_a6_pilot.py") != PRECHECKPOINT_PILOT_MODULE_SHA256
        or set(current_files or {}) != expected_files
        or changed_files != ["src/training/e03_a6_pilot.py"]
    ):
        raise ValueError("Pilot resume source differs from the one reviewed pre-checkpoint compatibility repair")

    manifest["training_code_at_checkpoint"] = {
        "sha256": old_code_sha,
        "files": dict(old_files),
    }
    manifest["resume_code_compatibility"] = {
        "reason": "restore the shared JSONL log reader in resume orchestration; no optimizer step, model update, data, config, or checkpoint algorithm changed",
        "changed_files": changed_files,
        "original_training_code_sha256": old_code_sha,
        "original_training_code_files": dict(old_files),
        "continuation_training_code_sha256": current_code_sha,
        "continuation_training_code_files": dict(current_files),
        "checkpoint_training_code_sha256": manifest["checkpoint_identity"]["training_code_sha256"],
        "checkpoint_identity_preserved": True,
        "optimizer_steps_executed_after_checkpoint_before_repair": 0,
    }
    manifest["training_code_sha256"] = current_code_sha
    manifest["training_code_files"] = dict(current_files)
    return str(manifest["checkpoint_identity"]["training_code_sha256"])


def _recover_pretrain_resume_failure(run_root: Path, manifest: dict[str, Any]) -> bool:
    """Restore the exact checkpoint-32 state after a failure before resumed Trainer.train()."""
    if manifest.get("status") != "failed":
        return False
    if (
        manifest.get("run_id") != "E03-A6-F3-PILOT-20261106-01"
        or manifest.get("stage") != "training_failed"
        or manifest.get("optimizer_steps") != 0
        or manifest.get("error") != RECOVERABLE_RESUME_FAILURE
        or manifest.get("last_checkpoint") != str((run_root / "trainer" / "checkpoint-32").resolve())
        or manifest.get("checkpoint_manifest_sha256") != _sha256(run_root / "trainer" / "checkpoint-32" / "checkpoint_manifest.json")
    ):
        raise ValueError("Pilot Run failed outside the one recoverable pre-train checkpoint-resume path")

    from src.training.training_sft import _verify_checkpoint

    checkpoint = run_root / "trainer" / "checkpoint-32"
    _verify_checkpoint(
        checkpoint,
        expected=manifest["checkpoint_identity"],
        run_id=manifest["run_id"],
    )
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    log_rows = _verify_training_log(run_root, EXPECTED_PILOT_PAUSE_STEP)
    if state.get("global_step") != EXPECTED_PILOT_PAUSE_STEP:
        raise ValueError("recoverable Pilot Checkpoint does not point to optimizer step 32")
    if not manifest.get("training_attempts") or manifest["training_attempts"][-1].get("ending_optimizer_step") != EXPECTED_PILOT_PAUSE_STEP:
        raise ValueError("Pilot Run has no successful step-32 attempt before the pre-train failure")

    failure = {
        "status": manifest["status"],
        "stage": manifest["stage"],
        "error": manifest["error"],
        "failed_at_utc": manifest.get("failed_at_utc"),
        "optimizer_steps_after_checkpoint": 0,
        "checkpoint_global_step_verified": EXPECTED_PILOT_PAUSE_STEP,
        "verified_log_rows": len(log_rows),
        "checkpoint_manifest_sha256": manifest["checkpoint_manifest_sha256"],
    }
    manifest["recovered_pretrain_resume_failure"] = failure
    manifest.update(
        status="interrupted",
        stage="checkpoint_saved",
        optimizer_steps=EXPECTED_PILOT_PAUSE_STEP,
        max_optimizer_steps=EXPECTED_PILOT_STEPS,
        stop_reason="requested_checkpoint_pause",
        recovered_at_utc=_now(),
    )
    manifest.pop("error", None)
    manifest.pop("failed_at_utc", None)
    _write_json(run_root / "manifest.json", manifest)
    return True


def _write_selected_files(run_root: Path, assets: dict[str, Any]) -> None:
    rows = assets["selected_metadata"]
    selection_file = {
        "selection_rule": assets["historical_selection"]["selection_rule"],
        "rows": rows,
        "historical_f2_rows_sha256": assets["historical_selection"]["sha256"],
        "f3_enriched_rows_sha256": canonical_sha256(rows),
        "sample_ids_sha256": assets["data_facts"]["selected_sample_ids_sha256"],
    }
    if assets.get("supplemental_selection"):
        selection_file.update(
            selection_method="historical_512_plus_seeded_length_deciles_1024_v1",
            source_f2_selection_rows_sha256=assets["selected_selection_rows_sha256"],
            base_sample_count=512,
            added_sample_count=512,
            added_sample_ids_sha256=assets["data_facts"]["added_sample_ids_sha256"],
        )
    _write_json(run_root / "selected_samples.json", selection_file)
    _write_json(run_root / "sft_dev_loss_selection.json", {
        **assets["dev_selection"],
    })


def _checkpoint_identity(source_identity: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "training_code_sha256": source_identity["training_code_sha256"],
        "config_sha256": source_identity["run_config_sha256"],
        "dataset_manifest_sha256": source_identity["data"]["f3_manifest_sha256"],
        "train_data_sha256": source_identity["data"]["f3_train_sha256"],
        "dev_data_sha256": source_identity["data"]["f3_dev_sha256"],
        "selected_samples_sha256": source_identity["data"].get(
            "selected_selection_rows_sha256",
            source_identity["data"]["historical_selection_rows_sha256"],
        ),
        "selected_ids_sha256": source_identity["data"]["selected_sample_ids_sha256"],
        "sft_dev_ids_sha256": source_identity["data"]["sft_dev_ids_sha256"],
        "base_weight_sha256": source_identity["base_weight_sha256"],
        "tokenizer_signature_sha256": source_identity["tokenizer_signature_sha256"],
        "tokenizer_template_sha256": source_identity["tokenizer_template_sha256"],
        "prompt_f3_sha256": source_identity["prompt_f3_sha256"],
        "format_version": "F3",
        "init_adapter": None,
    }


def _new_run_manifest(
    run_id: str,
    config: dict[str, Any],
    source_identity: dict[str, Any],
    gpu_facts: dict[str, Any],
    microfit_facts: dict[str, Any],
    dry_step_facts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "training_scope": config["training_scope"],
        "format_version": "F3",
        "status": "running",
        "stage": "preflight_passed",
        "created_at_utc": _now(),
        **source_identity,
        "training_budget": {
            "max_optimizer_steps": int(config["training"]["max_optimizer_steps"]),
            "pause_after_step": int(config["training"]["pause_after_step"]),
            "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
            "per_device_micro_batch_size": int(config["training"]["per_device_micro_batch_size"]),
            "max_training_seconds": int(config["training"]["max_training_seconds"]),
            "max_cuda_memory_fraction": float(config["quantization"]["max_cuda_memory_fraction"]),
        },
        "gpu_preflight": gpu_facts,
        "checkpoint_identity": _checkpoint_identity(source_identity, run_id),
        "init_adapter": None,
        "resume_from_checkpoint": None,
        "microfit_prerequisite": microfit_facts,
        "microfit_gate_status": "passed",
        "dry_step_prerequisite": dry_step_facts,
        "full_training_status": "not_authorized_not_run",
        "approval_boundary": "Bounded A6 Pilot only; does not authorize A6 Full training",
        "training_attempts": [],
        "training_seconds": 0.0,
    }


def _probe_samples(model: Any, tokenizer: Any, samples: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    from src.training.training_sft import _capture_probe

    rows = []
    for sample in samples:
        prompt_ids = list(sample["input_ids"][:sample["prompt_length"]])
        result = _capture_probe(model, tokenizer, prompt_ids, max_new_tokens=max_new_tokens)
        parsed = None
        status = "success"
        try:
            from src.shared.f3_protocol import submission_enums

            parsed = parse_f3(result["raw_text"], submission_enums(inference.load_submission_schema()))
        except Exception as exc:
            status = getattr(exc, "code", "format_parse_failure")
        rows.append({
            "sample_id": sample["sample_id"],
            "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": result["generated_token_ids"],
            "generated_tokens": result["generated_tokens"],
            "raw_text": result["raw_text"],
            "eos_generated": int(tokenizer.eos_token_id) in result["generated_token_ids"],
            "parse_status": status,
            "parsed_f3": parsed,
        })
    return rows


def _verify_microfit_gate(config: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    run_id = config["data"]["microfit_run_id"]
    if not run_id.startswith("E03-A6-F3-MICROFIT-"):
        raise ValueError("Pilot prerequisite must reference the isolated A6 Microfit scope")
    run_root = _project_path("outputs/training") / run_id
    manifest_path = run_root / "manifest.json"
    result_path = run_root / "microfit_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("the configured passing Microfit Run evidence is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or manifest.get("status") != "passed" or result.get("status") != "passed":
        raise ValueError("Pilot prerequisite Microfit Run did not pass")
    required_result_checks = (
        "all_steps_updated_lora", "reload_token_ids_exact_match",
        "reload_lora_dtype_exact_match", "base_weights_unchanged",
    )
    if any(result.get(key) is not True for key in required_result_checks):
        raise ValueError("Pilot prerequisite Microfit evidence has a failed learning/reload invariant")
    if result.get("f3_strict_parse_successes_after_reload") != 4 or result.get("f3_nonempty_profile_predictions_after_reload") < 1:
        raise ValueError("Pilot prerequisite Microfit did not show parsable, non-empty F3 learning signal")
    if manifest.get("base_weight_sha256") != config["base_model"]["expected_weight_sha256"]:
        raise ValueError("Microfit Base identity differs from the Pilot Base")
    expected_data_manifest_sha = assets["data_facts"]["f3_manifest_sha256"]
    if manifest.get("data", {}).get("dataset_manifest_sha256") != expected_data_manifest_sha:
        raise ValueError("Microfit and Pilot use different F3 dataset manifests")
    if result.get("full_training_status") != "not_authorized_not_run":
        raise ValueError("Microfit evidence changed the Full training authorization boundary")
    return {
        "run_id": run_id,
        "manifest_sha256": _sha256(manifest_path),
        "result_sha256": _sha256(result_path),
        "status": "passed",
        "adapter_weight_sha256": manifest.get("adapter_weight_sha256"),
        "base_weight_sha256": manifest.get("base_weight_sha256"),
        "dataset_manifest_sha256": expected_data_manifest_sha,
    }


def execute_pilot_dry_step(config_path: Path, dry_step_run_id: str) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    if not dry_step_run_id.startswith("E03-A6-F3-PILOT-DRYSTEP-") or len(dry_step_run_id) > 96:
        raise ValueError("dry-step Run ID must start with E03-A6-F3-PILOT-DRYSTEP-")
    root = _project_path("outputs/e03_a6/pilot_preflight") / dry_step_run_id
    if root.exists():
        raise FileExistsError(f"dry-step Run ID already exists; refusing overwrite: {root}")
    assets = _load_pilot_assets(config)
    microfit = _verify_microfit_gate(config, assets)
    torch, gpu_facts = _cuda_preflight(config)
    tokenizer, tokenizer_facts = _tokenizer(config)
    source_identity = _source_identity(config_path, config, assets, tokenizer_facts)
    base_before = _base_identity(config)
    root.mkdir(parents=True)
    _write_json(root / "dry_step_manifest.json", {
        "run_id": dry_step_run_id,
        "experiment_id": config["experiment_id"],
        "training_scope": f"{config['training_scope']}_dry_step_only",
        "status": "running",
        "created_at_utc": _now(),
        "source_identity": source_identity,
        "microfit_prerequisite": microfit,
        "gpu_preflight": gpu_facts,
        "optimizer_steps_budget": 1,
        "train_sample_ids": assets["data_facts"]["selected_sample_ids"][:4],
        "adapter_output": "none; dry-step weights are discarded",
        "full_training_status": "not_authorized_not_run",
    })

    model = trainer = callback = None
    try:
        from participant import run_inference as local_inference
        from src.data.sft_collator import SFTDataCollator
        from src.training.training_sft import (
            _TrainingCounters,
            _build_sft_trainer,
            _load_train_model,
            _optimizer_evidence_callback,
            _seed_everything,
            _snapshot_trainer_data_path_facts,
            _trainable_parameters,
        )

        _seed_everything(int(config["training"]["seed"]), torch)
        model, torch, model_facts = _load_train_model(config, _project_path(config["base_model"]["local_path"]))
        torch.cuda.reset_peak_memory_stats()
        dry_config = copy.deepcopy(config)
        dry_config["training"]["max_optimizer_steps"] = 1
        dry_config["training"]["checkpointing"] = False
        dry_config["training"]["evaluation_interval_steps"] = 0
        dry_config["training"]["save_interval_steps"] = 0
        dry_config["evaluation"]["sft_dev_loss_rows"] = 0
        counters = _TrainingCounters()
        collator = SFTDataCollator(
            tokenizer.pad_token_id,
            max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
        )
        callback = _optimizer_evidence_callback(
            model,
            _trainable_parameters(model),
            config=dry_config,
            samples=[],
            tokenizer=tokenizer,
            prompt="",
            schema=None,
            enums=None,
            run_root=root,
            snapshot_adapter=False,
            max_optimizer_steps=1,
            collator=counters,
        )
        trainer = _build_sft_trainer(
            model,
            tokenizer,
            assets["train_rows"][:4],
            collator,
            dry_config,
            max_steps=1,
            callback=callback,
            run_root=root,
            eval_rows=None,
            training_counts=counters,
        )
        trainer_data_path = _snapshot_trainer_data_path_facts(trainer)
        if trainer.args.max_steps != 1 or trainer.args.gradient_accumulation_steps != 4:
            raise RuntimeError("GPU dry-step Trainer setup differs from the Pilot's microbatch schedule")
        trainer.train()
        if int(trainer.state.global_step) != 1 or len(callback.steps) != 1:
            raise RuntimeError("GPU dry-step did not perform exactly one Trainer optimizer update")
        step = callback.steps[0]
        if not step.get("finite_gradients") or not step.get("updated_parameter_count") or not math.isfinite(float(step.get("loss", float("nan")))):
            raise RuntimeError("GPU dry-step did not verify finite gradients, loss, and LoRA parameter changes")
        base_after = _base_identity(config)
        base_unchanged = (
            base_before.get("observed_weight_hash")
            == base_after.get("observed_weight_hash")
            == config["base_model"]["expected_weight_sha256"]
        )
        if not base_unchanged:
            raise RuntimeError("GPU dry-step changed the frozen Base model weights")
        evidence = {
            "status": "passed",
            "run_id": dry_step_run_id,
            "optimizer_steps": 1,
            "loss": float(step["loss"]),
            "finite_gradients": bool(step["finite_gradients"]),
            "gradient_norm": float(step["gradient_norm"]),
            "nonzero_gradient_parameter_count": int(step["nonzero_gradient_parameter_count"]),
            "updated_parameter_count": int(step["updated_parameter_count"]),
            "updated_element_count": int(step["updated_element_count"]),
            "completion_tokens": int(step["completion_tokens"]),
            "training_sample_count": 4,
            "train_sample_ids": assets["data_facts"]["selected_sample_ids"][:4],
            "trainer_data_path": trainer_data_path,
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "base_weights_unchanged": base_unchanged,
            "runtime_model_facts": model_facts,
            "gpu_preflight": gpu_facts,
            "cuda_peak_memory": local_inference.cuda_memory_facts(torch),
            "package_versions": _package_versions(),
            "training_code_sha256": source_identity["training_code_sha256"],
            "config_sha256": source_identity["run_config_sha256"],
            "dataset_manifest_sha256": assets["data_facts"]["f3_manifest_sha256"],
            "f3_train_sha256": assets["data_facts"]["f3_train_sha256"],
            "tokenizer_signature_sha256": tokenizer_facts["signature_sha256"],
            "microfit_run_id": microfit["run_id"],
            "microfit_result_sha256": microfit["result_sha256"],
            "adapter_saved": False,
            "full_training_status": "not_authorized_not_run",
        }
        _write_json(root / "dry_step_result.json", evidence)
        manifest = json.loads((root / "dry_step_manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            status="passed",
            ended_at_utc=_now(),
            result_sha256=_sha256(root / "dry_step_result.json"),
        )
        _write_json(root / "dry_step_manifest.json", manifest)
    except Exception as exc:
        manifest_path = root / "dry_step_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(status="failed", failed_at_utc=_now(), error=f"{type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        raise
    finally:
        if trainer is not None:
            del trainer
        if callback is not None:
            del callback
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return root


def _verify_dry_step_gate(
    config: dict[str, Any],
    assets: dict[str, Any],
    source_identity: dict[str, Any],
    dry_step_run_id: str,
    *,
    expected_training_code_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(dry_step_run_id, str) or not dry_step_run_id.startswith("E03-A6-F3-PILOT-DRYSTEP-"):
        raise ValueError("Pilot Train requires a passing A6 Trainer dry-step Run")
    result_path = _project_path("outputs/e03_a6/pilot_preflight") / dry_step_run_id / "dry_step_result.json"
    if not result_path.is_file():
        raise FileNotFoundError("Pilot Trainer dry-step result is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required = (
        "finite_gradients", "base_weights_unchanged", "adapter_saved",
    )
    if result.get("status") != "passed" or result.get("optimizer_steps") != 1:
        raise ValueError("Pilot Trainer dry-step did not pass one optimizer update")
    if any(result.get(key) is not expected for key, expected in (
        ("finite_gradients", True), ("base_weights_unchanged", True), ("adapter_saved", False),
    )) or not result.get("updated_parameter_count"):
        raise ValueError("Pilot Trainer dry-step evidence is missing required LoRA/Base invariants")
    expected = {
        "training_code_sha256": expected_training_code_sha256 or source_identity["training_code_sha256"],
        "config_sha256": source_identity["run_config_sha256"],
        "dataset_manifest_sha256": assets["data_facts"]["f3_manifest_sha256"],
        "f3_train_sha256": assets["data_facts"]["f3_train_sha256"],
        "tokenizer_signature_sha256": source_identity["tokenizer_signature_sha256"],
        "base_weight_sha256_after": source_identity["base_weight_sha256"],
        "microfit_run_id": config["data"]["microfit_run_id"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Pilot Trainer dry-step identity mismatch: {key}")
    return {
        "run_id": dry_step_run_id,
        "result_sha256": _sha256(result_path),
        "status": "passed",
        "optimizer_steps": 1,
        "updated_parameter_count": result["updated_parameter_count"],
    }


def _verify_training_log(run_root: Path, expected_step: int) -> list[dict[str, Any]]:
    from src.training.training_sft import _read_training_log

    rows = _read_training_log(run_root)
    steps = [int(row.get("optimizer_step", -1)) for row in rows]
    if steps != list(range(1, expected_step + 1)):
        raise RuntimeError("Pilot optimizer-step log is incomplete, duplicated, or out of order")
    for row in rows:
        loss = float(row.get("loss", float("nan")))
        if not math.isfinite(loss):
            raise RuntimeError("Pilot optimizer-step log contains a non-finite loss")
        if not row.get("finite_gradients") or not row.get("nonzero_gradient_parameter_count"):
            raise RuntimeError("Pilot optimizer-step log is missing finite nonzero LoRA gradient evidence")
        if row.get("parameter_update_checked") and not row.get("updated_parameter_count"):
            raise RuntimeError("Pilot optimizer-step log contains a checked step with no LoRA update")
    return rows


def _execute_pilot_training(
    config_path: Path,
    run_id: str,
    *,
    resume: bool,
    dry_step_run_id: str | None = None,
) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    max_steps = int(config["training"]["max_optimizer_steps"])
    pause_step = int(config["training"]["pause_after_step"])
    expanded = config.get("training_scope") == "supplemental_pilot"
    run_root = _run_root(run_id)
    assets = _load_pilot_assets(config)
    torch, gpu_facts = _cuda_preflight(config)
    tokenizer, tokenizer_facts = _tokenizer(config)
    source_identity = _source_identity(config_path, config, assets, tokenizer_facts)

    if resume:
        manifest = _load_manifest(run_root)
        dry_step_run_id = manifest.get("dry_step_prerequisite", {}).get("run_id")
        if expanded:
            validate_expanded_pilot_resume_manifest(
                manifest,
                run_id=run_id,
                checkpoint_name=f"checkpoint-{pause_step}",
                pause_step=pause_step,
                max_steps=max_steps,
            )
        else:
            _recover_pretrain_resume_failure(run_root, manifest)
            checkpoint_name = "checkpoint-32"
            validate_pilot_resume_manifest(manifest, run_id=run_id, checkpoint_name=checkpoint_name)
        from src.training.training_sft import _verify_checkpoint

        checkpoint = run_root / "trainer" / (f"checkpoint-{pause_step}" if expanded else checkpoint_name)
        _verify_checkpoint(
            checkpoint,
            expected=manifest["checkpoint_identity"],
            run_id=run_id,
        )
        checkpoint_training_code_sha256 = _adopt_checkpoint_compatible_resume_code(
            source_identity, manifest,
        )
        _same_source_identity(source_identity, manifest)
        microfit_facts = _verify_microfit_gate(config, assets)
        dry_step_facts = _verify_dry_step_gate(
            config,
            assets,
            source_identity,
            dry_step_run_id,
            expected_training_code_sha256=checkpoint_training_code_sha256,
        )
        if manifest.get("dry_step_prerequisite", {}).get("result_sha256") != dry_step_facts["result_sha256"]:
            raise ValueError("Pilot dry-step result changed after checkpoint-32 was created")
        if manifest.get("gpu_preflight", {}).get("gpu_name") != gpu_facts["gpu_name"]:
            raise ValueError("Pilot GPU device changed between checkpoint pause and resume")
        if manifest.get("status") != "interrupted":
            raise ValueError("only an interrupted Pilot Run may be resumed")
    else:
        if not dry_step_run_id:
            raise ValueError("Pilot Train requires --dry-step-run-id referencing a passing GPU Trainer dry-step")
        microfit_facts = _verify_microfit_gate(config, assets)
        dry_step_facts = _verify_dry_step_gate(config, assets, source_identity, dry_step_run_id)
        if run_root.exists():
            raise FileExistsError(f"Pilot Run ID already exists; refusing overwrite: {run_root}")
        run_root.mkdir(parents=True)
        config_snapshot = run_root / "config_snapshot.yaml"
        config_snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = _new_run_manifest(
            run_id, config, source_identity, gpu_facts, microfit_facts, dry_step_facts,
        )
        _write_selected_files(run_root, assets)
        _write_json(run_root / "preflight.json", {
            "status": "passed",
            "created_at_utc": _now(),
            "experiment_id": config["experiment_id"],
            "run_id": run_id,
            "gpu": gpu_facts,
            "source_identity": source_identity,
            "package_versions": _package_versions(),
            "selected_train_rows": len(assets["train_rows"]),
            "selected_dev_loss_rows": len(assets["dev_rows"]),
            "only_new_base_lora": True,
            "no_official_val": True,
        })
        _write_json(run_root / "manifest.json", manifest)

    from participant import run_inference as local_inference
    from src.data.sft_collator import SFTDataCollator
    from src.training.training_sft import (
        _TrainingCounters,
        _build_sft_trainer,
        _load_train_model,
        _optimizer_evidence_callback,
        _prepare_resume_logs,
        _read_training_log,
        _seed_everything,
        _snapshot_trainer_data_path_facts,
        _trainable_parameters,
        _verify_checkpoint,
        _write_checkpoint_manifest,
        _write_training_progress,
    )

    previous_training_seconds = float(manifest.get("training_seconds", 0.0)) if resume else 0.0
    checkpoint_to_resume = run_root / "trainer" / f"checkpoint-{pause_step}" if resume else None
    if resume:
        _prepare_resume_logs(run_root, pause_step)

    training_started = None
    model = trainer = callback = None
    try:
        _seed_everything(int(config["training"]["seed"]), torch)
        base_path = _project_path(config["base_model"]["local_path"])
        model, torch, model_facts = _load_train_model(config, base_path)
        torch.cuda.reset_peak_memory_stats()
        model_facts["trainable_lora_dtypes"] = sorted({
            str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad
        })
        model_facts["cuda_peak_memory_scope"] = "Pilot Trainer setup, training, teacher-forced SFT Dev Loss, and checkpoints"
        collator = SFTDataCollator(
            tokenizer.pad_token_id,
            max_sequence_length=int(config["protocol"]["max_train_seq_length"]),
        )
        counters = _TrainingCounters(_read_training_log(run_root) if resume else None)
        checkpoint_facts = manifest["checkpoint_identity"]
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
            max_optimizer_steps=max_steps,
            collator=counters,
            checkpoint_facts=checkpoint_facts,
            stop_after_step=pause_step if not resume else None,
            previous_training_seconds=previous_training_seconds,
        )
        trainer = _build_sft_trainer(
            model,
            tokenizer,
            assets["train_rows"],
            collator,
            config,
            max_steps=max_steps,
            callback=callback,
            run_root=run_root,
            eval_rows=assets["dev_rows"],
            training_counts=counters,
        )
        trainer_data_path = _snapshot_trainer_data_path_facts(trainer)
        if trainer_data_path.get("train_sampling_strategy") != "random":
            raise RuntimeError("Pilot Trainer did not construct the frozen random sampler")
        if (
            trainer.args.gradient_accumulation_steps != int(config["training"]["gradient_accumulation_steps"])
            or trainer.args.max_steps != max_steps
        ):
            raise RuntimeError("Pilot Trainer budget differs from its frozen configuration")
        manifest.update(
            stage="training",
            trainer_data_path=trainer_data_path,
            model_facts=model_facts,
            current_gpu_preflight=gpu_facts,
            current_package_versions=_package_versions(),
            resume_checkpoint=str(checkpoint_to_resume) if checkpoint_to_resume else None,
            resumed_from_optimizer_step=pause_step if resume else 0,
        )
        _write_json(run_root / "manifest.json", manifest)
        start_step = pause_step if resume else 0
        training_started = time.perf_counter()
        trainer.train(resume_from_checkpoint=str(checkpoint_to_resume) if checkpoint_to_resume else None)
        attempt_seconds = time.perf_counter() - training_started
        end_step = int(trainer.state.global_step)
        stop_reason = callback.stop_reason
        manifest["training_seconds"] = previous_training_seconds + attempt_seconds
        manifest.setdefault("training_attempts", []).append({
            "starting_optimizer_step": start_step,
            "ending_optimizer_step": end_step,
            "training_seconds": attempt_seconds,
            "stop_reason": stop_reason,
            "cuda_memory": local_inference.cuda_memory_facts(torch),
            "dry_step_update_verified": bool(callback.steps and callback.steps[0].get("updated_parameter_count")),
        })
        expected_end = pause_step if not resume else max_steps
        if end_step != expected_end:
            raise RuntimeError(f"Pilot reached optimizer step {end_step}; expected {expected_end}")
        if not resume:
            if stop_reason != "requested_checkpoint_pause":
                raise RuntimeError(f"Pilot step-{pause_step} stop reason is unexpected: {stop_reason}")
            checkpoint = run_root / "trainer" / f"checkpoint-{pause_step}"
            _verify_checkpoint(checkpoint, expected=checkpoint_facts, run_id=run_id)
            _verify_training_log(run_root, pause_step)
            manifest.update(
                status="interrupted",
                stage="checkpoint_saved",
                optimizer_steps=end_step,
                max_optimizer_steps=max_steps,
                stop_reason=stop_reason,
                last_checkpoint=str(checkpoint),
                checkpoint_manifest_sha256=_sha256(checkpoint / "checkpoint_manifest.json"),
                paused_at_utc=_now(),
            )
            _write_json(run_root / "manifest.json", manifest)
            return run_root

        if stop_reason is not None:
            raise RuntimeError(f"Pilot resume stopped before step {max_steps}: {stop_reason}")
        log_rows = _verify_training_log(run_root, max_steps)
        if not all(row.get("updated_parameter_count") for row in log_rows if row.get("parameter_update_checked")):
            raise RuntimeError("one or more Pilot parameter verification steps did not update LoRA")
        adapter_dir = run_root / "adapter"
        adapter_facts = __import__("src.training.training_sft", fromlist=["_save_adapter"])._save_adapter(
            model, adapter_dir, config,
        )
        base_before_reload = _base_identity(config)
        if base_before_reload.get("observed_weight_hash") != config["base_model"]["expected_weight_sha256"]:
            raise RuntimeError("Pilot training changed or replaced the read-only Base model")
        probes = _probe_samples(
            model,
            tokenizer,
            assets["train_rows"][:4],
            int(config["evaluation"]["probe_max_new_tokens"]),
        )
        _write_json(run_root / "generation_probe.json", {
            "run_id": run_id,
            "sample_ids": [row["sample_id"] for row in assets["train_rows"][:4]],
            "max_new_tokens": int(config["evaluation"]["probe_max_new_tokens"]),
            "do_sample": False,
            "model_training_mode": "final_trained_adapter_before_release",
            "probe_rows": probes,
        })
        prior_training_seconds = previous_training_seconds + attempt_seconds
        memory = local_inference.cuda_memory_facts(torch)
        training_log_summary = {
            "optimizer_steps": len(log_rows),
            "first_10_loss_mean": sum(float(row["loss"]) for row in log_rows[:10]) / 10,
            "last_10_loss_mean": sum(float(row["loss"]) for row in log_rows[-10:]) / 10,
            "finite_loss_steps": len(log_rows),
            "updated_parameter_checked_steps": sum(bool(row.get("parameter_update_checked")) for row in log_rows),
            "checked_steps_with_update": sum(bool(row.get("updated_parameter_count")) for row in log_rows if row.get("parameter_update_checked")),
            "samples_seen": sum(int(row.get("samples_seen", 0)) for row in log_rows),
            "input_tokens_seen": sum(int(row.get("input_tokens_seen", 0)) for row in log_rows),
            "completion_tokens_seen": sum(int(row.get("completion_tokens", 0)) for row in log_rows),
            "micro_batches_seen": sum(int(row.get("micro_batches_seen", 0)) for row in log_rows),
            "sft_dev_loss_rows": len(assets["dev_rows"]),
            "sft_dev_loss_observations": 0,
            "training_seconds": prior_training_seconds,
            "cuda_peak_memory": memory,
            "model_facts": model_facts,
            "adapter_identity": adapter_facts,
            "trainable_lora_dtypes": model_facts["trainable_lora_dtypes"],
            "trainer_data_path": trainer_data_path,
            "package_versions": _package_versions(),
        }
        eval_log_path = run_root / "eval_log.jsonl"
        if eval_log_path.is_file():
            eval_rows = list(_read_jsonl(eval_log_path))
            training_log_summary["sft_dev_loss_observations"] = len(eval_rows)
            training_log_summary["sft_dev_loss_first"] = eval_rows[0].get("sft_dev_loss") if eval_rows else None
            training_log_summary["sft_dev_loss_last"] = eval_rows[-1].get("sft_dev_loss") if eval_rows else None
        _write_json(run_root / "training_evidence.json", training_log_summary)
        checkpoint_root = run_root / "trainer"
        retained_checkpoints = sorted(
            (path for path in checkpoint_root.glob("checkpoint-[0-9]*") if path.is_dir()),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        if not retained_checkpoints or retained_checkpoints[-1].name != f"checkpoint-{max_steps}":
            raise RuntimeError("final optimizer-step checkpoint is missing after checkpoint rotation")
        latest_checkpoint = retained_checkpoints[-1]
        _verify_checkpoint(latest_checkpoint, expected=checkpoint_facts, run_id=run_id)
        manifest.update(
            status="trained_waiting_reload_check",
            stage="training_complete_adapter_saved",
            optimizer_steps=max_steps,
            max_optimizer_steps=max_steps,
            stop_reason=None,
            last_checkpoint=str(latest_checkpoint.resolve()),
            checkpoint_manifest_sha256=_sha256(latest_checkpoint / "checkpoint_manifest.json"),
            retained_checkpoints=[path.name for path in retained_checkpoints],
            adapter_identity=adapter_facts,
            adapter_weight_sha256=adapter_facts["weight_sha256"],
            adapter_config_sha256=adapter_facts["config_sha256"],
            base_weight_sha256_after_training=base_before_reload["observed_weight_hash"],
            training_evidence_sha256=_sha256(run_root / "training_evidence.json"),
            generation_probe_sha256=_sha256(run_root / "generation_probe.json"),
            training_seconds=prior_training_seconds,
            ended_training_at_utc=_now(),
            full_training_status="not_authorized_not_run",
        )
        _write_json(run_root / "manifest.json", manifest)
        return run_root
    except Exception as exc:
        manifest.update(
            status="failed",
            stage="training_failed",
            optimizer_steps=(int(trainer.state.global_step) if trainer is not None else 0),
            error=f"{type(exc).__name__}: {exc}",
            failed_at_utc=_now(),
        )
        _write_json(run_root / "manifest.json", manifest)
        raise
    finally:
        if training_started is not None:
            training_started = None
        if trainer is not None:
            del trainer
        if callback is not None:
            del callback
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute_pilot_train(config_path: Path, run_id: str, dry_step_run_id: str) -> Path:
    return _execute_pilot_training(
        config_path, run_id, resume=False, dry_step_run_id=dry_step_run_id,
    )


def execute_pilot_resume(config_path: Path, run_id: str) -> Path:
    return _execute_pilot_training(config_path, run_id, resume=True)


def execute_pilot_reload_check(config_path: Path, run_id: str) -> Path:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    max_steps = int(config["training"]["max_optimizer_steps"])
    run_root = _run_root(run_id)
    manifest = _load_manifest(run_root)
    if (
        manifest.get("experiment_id") != config["experiment_id"]
        or manifest.get("training_scope") != config["training_scope"]
    ):
        raise ValueError("reload-check Run scope differs from the frozen Pilot config")
    if manifest.get("status") != "trained_waiting_reload_check" or manifest.get("optimizer_steps") != max_steps:
        raise ValueError(f"reload-check requires the completed {max_steps}-step Pilot adapter")
    assets = _load_pilot_assets(config)
    torch, gpu_facts = _cuda_preflight(config)
    tokenizer, tokenizer_facts = _tokenizer(config)
    source_identity = _source_identity(config_path, config, assets, tokenizer_facts)
    _same_source_identity(source_identity, manifest)
    if source_identity["run_config_sha256"] != _sha256(run_root / "config_snapshot.yaml"):
        raise ValueError("Pilot config snapshot changed before independent reload-check")
    base_before = _base_identity(config)
    if base_before.get("observed_weight_hash") != manifest.get("base_weight_sha256_after_training"):
        raise ValueError("Base model identity changed before independent Pilot reload-check")
    adapter_path = run_root / "adapter"
    adapter_identity, adapter_config = inference.adapter_weight_identity(adapter_path)
    if adapter_identity["weight_sha256"] != manifest.get("adapter_weight_sha256"):
        raise ValueError("saved Pilot Adapter weights changed after training")
    if adapter_identity["config_sha256"] != manifest.get("adapter_config_sha256"):
        raise ValueError("saved Pilot Adapter config changed after training")
    inference.validate_adapter_base_reference(
        adapter_config,
        config["base_model"]["model_id"],
        _project_path(config["base_model"]["local_path"]),
    )
    previous_probe = json.loads((run_root / "generation_probe.json").read_text(encoding="utf-8"))
    expected_ids = previous_probe.get("sample_ids", [])
    if expected_ids != assets["data_facts"]["selected_sample_ids"][:4]:
        raise ValueError("Pilot reload probe sample IDs differ from the frozen Train selection")
    by_id = {row["sample_id"]: row for row in assets["train_rows"]}
    loaded, torch, runtime_facts = __import__(
        "src.training.training_sft", fromlist=["_load_train_model"],
    )._load_train_model(
        config,
        _project_path(config["base_model"]["local_path"]),
        adapter_path=adapter_path,
        autocast_adapter_dtype=bool(config["evaluation"]["autocast_adapter_dtype"]),
    )
    actual_probes = []
    prompt_ids_match = []
    for old_probe in previous_probe.get("probe_rows", []):
        sample_id = old_probe["sample_id"]
        row = by_id[sample_id]
        prompt_ids = list(row["input_ids"][:row["prompt_length"]])
        prompt_ids_match.append(
            canonical_sha256(prompt_ids) == old_probe["prompt_token_ids_sha256"]
            and prompt_ids == old_probe["prompt_token_ids"]
        )
        actual = _probe_samples(loaded, tokenizer, [row], int(previous_probe["max_new_tokens"]))[0]
        actual_probes.append(actual)
    token_match = [
        old["generated_token_ids"] == actual["generated_token_ids"]
        for old, actual in zip(previous_probe.get("probe_rows", []), actual_probes)
    ]
    trained_dtypes = sorted(manifest.get("model_facts", {}).get("trainable_lora_dtypes", []))
    reloaded_dtypes = sorted({
        str(parameter.dtype)
        for name, parameter in loaded.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    })
    base_after = _base_identity(config)
    base_unchanged = (
        base_before.get("observed_weight_hash")
        == base_after.get("observed_weight_hash")
        == config["base_model"]["expected_weight_sha256"]
    )
    status = "passed" if all(prompt_ids_match) and all(token_match) and trained_dtypes == reloaded_dtypes and base_unchanged else "failed"
    result = {
        "status": status,
        "run_id": run_id,
        "optimizer_steps": max_steps,
        "probe_sample_count": len(actual_probes),
        "prompt_token_ids_exact_match_per_sample": prompt_ids_match,
        "greedy_token_ids_exact_match_per_sample": token_match,
        "greedy_token_ids_exact_match": all(token_match),
        "trained_lora_dtypes": trained_dtypes,
        "reloaded_lora_dtypes": reloaded_dtypes,
        "lora_dtype_exact_match": trained_dtypes == reloaded_dtypes,
        "adapter_identity": adapter_identity,
        "adapter_autocast_dtype": bool(config["evaluation"]["autocast_adapter_dtype"]),
        "base_weight_sha256_before": base_before.get("observed_weight_hash"),
        "base_weight_sha256_after": base_after.get("observed_weight_hash"),
        "base_weights_unchanged": base_unchanged,
        "cuda_preflight": gpu_facts,
        "reload_runtime": runtime_facts,
        "cuda_peak_memory": inference.cuda_memory_facts(torch),
        "source_identity_sha256": canonical_sha256(source_identity),
    }
    _write_json(run_root / "reload_check_standalone.json", result)
    manifest.update(
        status=status,
        stage="pilot_training_reloaded" if status == "passed" else "pilot_reload_failed",
        reload_check_status=status,
        reload_check_sha256=_sha256(run_root / "reload_check_standalone.json"),
        reload_greedy_token_ids_exact_match=all(token_match),
        reload_lora_dtype_exact_match=trained_dtypes == reloaded_dtypes,
        base_weights_unchanged=base_unchanged,
        ended_at_utc=_now(),
    )
    _write_json(run_root / "manifest.json", manifest)
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if status != "passed":
        raise RuntimeError("independent Pilot Adapter reload-check did not pass")
    return run_root
