#!/usr/bin/env python3
"""Isolated A8 P fine-tuning on D0.3 Dev followed by target-free Val scoring."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from scripts.e03_a8_staged_eval import _append_jsonl_fsynced, _generate_rows
from src.data.e03_a8_devft import build_devft_training_item
from src.data.e03_a8_sft import canonical_sha256
from src.data.sft_collator import SFTDataCollator
from src.evaluation.e03_a6_f4_staged_metrics import _conversation_id, selected_ids_sha256
from src.evaluation.e03_a8_staged_metrics import bootstrap_conversation_micro_f1, score_a8_cohort
from src.shared.e03_a8_protocol import PROFILE_DIMS, validate_a8_prompt_enum_order
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
    _row_contract,
    _runtime_tokenizer,
    _set_seed,
    _write_json,
    sha256_file,
)
from src.training.e03_a8_completion_loss import completion_only_forward_loss
from src.training.e03_a8_v2_staged import _gradient_summary, _verify_v2_checkpoint
from src.training.training_sft import _load_train_model


DATA_CONFIG_PATH = ROOT / "configs/data/e03_a8_devft_v1.yaml"
TRAIN_CONFIG_PATH = ROOT / "configs/training/e03_a8_devft_p.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/training/e03_a8_v2_p_staged.yaml"
RUN_ID = "E03-A8-DEVFT-20261108-01-PDEV-1EPOCH"
PARENT_RUN_ID = "E03-A8-V2-20261107-04-P4048-R1"
VAL_EXPOSURE_PATH = ROOT / "outputs/diagnostics/e03_a2/imports/e01-b-209"
VAL_INPUT_VERSION = "E03-A8-DEVFT-target-free-val-v1"
VAL_LABEL_EXCLUSION_VERSION = "E03-A2-exact-point-exclusion-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"DevFT path escapes the project root: {value}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _read_jsonl_path(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _unique_by(rows: list[dict[str, Any]], key: str, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{context} has a missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def prepare_devft_data() -> Path:
    data_cfg = _read_yaml(DATA_CONFIG_PATH)
    p_model_config = _read_yaml(MODEL_CONFIG_PATH)
    output_root = _resolve(data_cfg["outputs"]["dataset_dir"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite DevFT dataset: {output_root}")

    paths = {key: _resolve(value) for key, value in data_cfg["inputs"].items()}
    sft_manifest = _read_json(paths["sft_manifest"])
    d03_manifest = _read_json(paths["d03_manifest"])
    d042_manifest = _read_json(paths["d042_manifest"])
    if d03_manifest.get("pipeline_version") != "D0.3-v1" or d042_manifest.get("audit_version") != "D0.4.2-v1":
        raise ValueError("DevFT source data protocol identity differs")
    if sft_manifest.get("config", {}).get("profile", {}).get("max_prompt_tokens") != 1280 or sft_manifest.get("config", {}).get("profile", {}).get("max_train_seq_length") != 2048:
        raise ValueError("DevFT requires the existing D0.4.3 P=1280/S=2048 profile")

    d03_dev_record = next((row for row in d03_manifest.get("outputs", []) if row.get("path") == "data/processed/v1/dev.jsonl"), None)
    expected_dev_sha = sft_manifest.get("inputs", {}).get("normalized_dev", {}).get("sha256")
    expected_budget_sha = sft_manifest.get("inputs", {}).get("d042_budget_rows", {}).get("sha256")
    expected_d03_sha = sft_manifest.get("d03_manifest_sha256")
    expected_d042_sha = sft_manifest.get("d042_manifest_sha256")
    if not d03_dev_record or d03_dev_record.get("sha256") != expected_dev_sha:
        raise ValueError("D0.3 Dev does not match the D0.4.3 normalized Dev source")
    if sha256_file(paths["d03_manifest"]) != expected_d03_sha or sha256_file(paths["d042_manifest"]) != expected_d042_sha:
        raise ValueError("D0.4.3 source manifest identities differ")
    if sha256_file(paths["normalized_dev"]) != expected_dev_sha or sha256_file(paths["sft_dev"]) != sft_manifest.get("outputs", {}).get("files", {}).get("dev.jsonl", {}).get("sha256"):
        raise ValueError("D0.3 or D0.4.3 Dev data hash differs from its Manifest")
    if sha256_file(paths["d042_budget_rows"]) != expected_budget_sha:
        raise ValueError("D0.4.2 budget audit hash differs from the D0.4.3 Manifest")
    if sft_manifest.get("inputs", {}).get("train_schema", {}).get("sha256") != sha256_file(paths["train_schema"]):
        raise ValueError("official Train Schema differs from the D0.4.3 frozen Schema")
    if sft_manifest.get("inputs", {}).get("prompt_path", {}).get("sha256") != sha256_file(paths["p0_prompt"]):
        raise ValueError("protected P0 Prompt differs from the D0.4.3 source identity")

    normalized_rows = _unique_by(_read_jsonl_path(paths["normalized_dev"]), "sample_id", "D0.3 Dev")
    sft_rows = _unique_by(_read_jsonl_path(paths["sft_dev"]), "sample_id", "D0.4.3 Dev")
    budget_rows = {}
    for row in _read_jsonl_path(paths["d042_budget_rows"]):
        if row.get("split") != "dev" or row.get("max_prompt_tokens") != 1280 or row.get("max_train_seq_length") != 2048:
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in budget_rows:
            raise ValueError(f"D0.4.2 Dev audit has a missing/duplicate ID: {sample_id!r}")
        budget_rows[sample_id] = row

    if len(normalized_rows) != 1833 or len(sft_rows) != 1821 or len(budget_rows) != 1833:
        raise ValueError(f"D0.3/D0.4.2/D0.4.3 counts differ: normalized={len(normalized_rows)}, audit={len(budget_rows)}, sft={len(sft_rows)}")
    accepted_ids = {sample_id for sample_id, row in budget_rows.items() if row.get("outcome") in {"full_fit", "trimmed_fit"}}
    if accepted_ids != set(sft_rows):
        raise ValueError("D0.4.3 Dev IDs do not equal all D0.4.2 1280/2048 accepted Dev IDs")
    if set(sft_rows) - set(normalized_rows) or set(sft_rows) - set(budget_rows):
        raise ValueError("D0.4.3 Dev includes an ID outside normalized Dev or D0.4.2")

    tokenizer, tokenizer_facts = _runtime_tokenizer(p_model_config)
    schema = json.loads(paths["train_schema"].read_text(encoding="utf-8"))
    p_prompt = _resolve(p_model_config["protocol"]["prompt_path"]).read_text(encoding="utf-8")
    ep_prompt_path = ROOT / "configs/prompts/p_a8_v2_emotion_profile.txt"
    ep_prompt = ep_prompt_path.read_text(encoding="utf-8")
    validate_a8_prompt_enum_order(p_prompt, schema, "P")
    validate_a8_prompt_enum_order(ep_prompt, schema, "EP")
    if sha256_file(_resolve(p_model_config["protocol"]["prompt_path"])) != p_model_config["protocol"]["prompt_sha256"]:
        raise ValueError("A8-V2 P Prompt hash differs from the frozen P configuration")

    p_rows: list[dict[str, Any]] = []
    ep_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for sample_id, sft_item in sft_rows.items():
        sample = normalized_rows[sample_id]
        paired, audit = build_devft_training_item(
            sample,
            sft_item,
            budget_rows[sample_id],
            p0_prompt=paths["p0_prompt"].read_text(encoding="utf-8"),
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            prompt_cap=int(data_cfg["profile"]["max_prompt_tokens"]),
            sequence_cap=int(data_cfg["profile"]["max_train_seq_length"]),
            tokenizer=tokenizer,
            official_schema=schema,
        )
        p_rows.append(paired["P"])
        ep_rows.append(paired["EP"])
        audit_rows.append(audit)

    ids = [row["sample_id"] for row in p_rows]
    if len(ids) != 1821 or len(set(ids)) != len(ids):
        raise ValueError("DevFT preparation did not produce one unique P row for every accepted D0.4.3 Dev ID")
    seed = int(data_cfg["schedule"]["seed"])
    accumulation = int(data_cfg["schedule"]["gradient_accumulation_steps"])
    shuffled_ids = list(ids)
    random.Random(seed).shuffle(shuffled_ids)
    schedule = [
        {"optimizer_step": index + 1, "sample_ids": shuffled_ids[index:index + accumulation]}
        for index in range(0, len(shuffled_ids), accumulation)
    ]
    if len(schedule) != 456 or sum(len(group["sample_ids"]) for group in schedule) != len(ids) or any(not 1 <= len(group["sample_ids"]) <= accumulation for group in schedule):
        raise ValueError("DevFT one-epoch schedule count or accumulation contract differs")
    if len({sample_id for group in schedule for sample_id in group["sample_ids"]}) != len(ids):
        raise ValueError("DevFT schedule repeats or omits a Dev sample")

    row_by_id = {row["sample_id"]: row for row in p_rows}
    dimension_positive = Counter()
    label_frequency = {dimension: Counter() for dimension in PROFILE_DIMS}
    nonempty = 0
    sequence_lengths = []
    prompt_lengths = []
    completion_lengths = []
    conversation_ids = set()
    for sample_id in ids:
        sample = normalized_rows[sample_id]
        profile = sample["target"]["user_profile"]
        nonempty += any(profile[dimension] for dimension in PROFILE_DIMS)
        for dimension in PROFILE_DIMS:
            values = profile[dimension]
            if values:
                dimension_positive[dimension] += 1
            label_frequency[dimension].update(values)
        row = row_by_id[sample_id]
        conversation_ids.add(row["conversation_id"])
        prompt_lengths.append(row["prompt_length"])
        completion_lengths.append(row["completion_length"])
        sequence_lengths.append(len(row["input_ids"]))

    def percentiles(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "p50": ordered[(len(ordered) - 1) // 2],
            "p90": ordered[math.ceil(0.90 * len(ordered)) - 1],
            "max": ordered[-1],
        }

    output_root.mkdir(parents=True, exist_ok=False)
    _write_jsonl_exclusive(output_root / "p_devft.jsonl", p_rows)
    _write_jsonl_exclusive(output_root / "ep_common_history_devft.jsonl", ep_rows)
    _write_jsonl_exclusive(output_root / "common_history_audit.jsonl", audit_rows)
    _write_jsonl_exclusive(output_root / "sample_order.jsonl", schedule)
    code_paths = [
        "configs/data/e03_a8_devft_v1.yaml",
        "configs/training/e03_a8_v2_p_staged.yaml",
        "configs/prompts/p_a8_v2_profile.txt",
        "configs/prompts/p_a8_v2_emotion_profile.txt",
        "src/data/e03_a8_devft.py",
        "src/data/e03_a8_sft.py",
        "src/shared/e03_a8_protocol.py",
        "src/shared/prompting.py",
        "scripts/e03_a8_devft.py",
    ]
    manifest = {
        "version": data_cfg["version"],
        "experiment_id": data_cfg["experiment_id"],
        "status": "cpu_preparation_passed",
        "created_at_utc": _now(),
        "source_role_change": {
            "source_split": "D0.3 Dev",
            "new_role": "single-epoch P fine-tuning data",
            "protocol_version": "E03-A8-DEVFT-v1",
            "changes_D0_3_or_D0_4_semantics": False,
            "independent_dev_evaluation_after_training": False,
        },
        "inputs": {key: _input_record(path) for key, path in paths.items()},
        "source_manifest_sha256": {
            "d03": sha256_file(paths["d03_manifest"]),
            "d042": sha256_file(paths["d042_manifest"]),
            "d043": sha256_file(paths["sft_manifest"]),
        },
        "official_val_read": False,
        "prompts": {
            "P": {"path": str(_resolve(p_model_config["protocol"]["prompt_path"]).relative_to(ROOT)), "sha256": sha256_file(_resolve(p_model_config["protocol"]["prompt_path"]))},
            "EP_common_history_only": {"path": str(ep_prompt_path.relative_to(ROOT)), "sha256": sha256_file(ep_prompt_path)},
        },
        "p0_sha256": sha256_file(paths["p0_prompt"]),
        "schema_sha256": sha256_file(paths["train_schema"]),
        "tokenizer": tokenizer_facts,
        "profile": data_cfg["profile"],
        "schedule": {
            **data_cfg["schedule"],
            "rows": len(ids),
            "steps": len(schedule),
            "final_group_size": len(schedule[-1]["sample_ids"]),
            "ordered_ids_sha256": canonical_sha256(ids),
            "scheduled_ids_sha256": canonical_sha256([sample_id for group in schedule for sample_id in group["sample_ids"]]),
            "schedule_sha256": canonical_sha256(schedule),
        },
        "counts": {
            "normalized_dev": len(normalized_rows),
            "d042_accepted_dev": len(accepted_ids),
            "d042_excluded_dev": len(normalized_rows) - len(accepted_ids),
            "effective_devft_rows": len(p_rows),
            "nonempty_profile_rows": nonempty,
            "all_empty_profile_rows": len(p_rows) - nonempty,
            "nonempty_by_dimension": dict(dimension_positive),
            "unique_conversations": len(conversation_ids),
            "repeated_conversation_rows": len(ids) - len(conversation_ids),
        },
        "label_frequency": {dimension: dict(sorted(counter.items())) for dimension, counter in label_frequency.items()},
        "lengths": {
            "prompt_tokens": percentiles(prompt_lengths),
            "completion_tokens": percentiles(completion_lengths),
            "sequence_tokens": percentiles(sequence_lengths),
        },
        "code_sha256": {name: sha256_file(_resolve(name)) for name in code_paths},
        "files": {
            name: sha256_file(output_root / name)
            for name in ("p_devft.jsonl", "ep_common_history_devft.jsonl", "common_history_audit.jsonl", "sample_order.jsonl")
        },
        "runtime_packages": _package_versions(),
    }
    manifest["dataset_manifest_sha256"] = canonical_sha256(manifest)
    _write_json_exclusive(output_root / "manifest.json", manifest)
    del tokenizer
    return output_root


def _load_devft_artifacts() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path]:
    data_cfg = _read_yaml(DATA_CONFIG_PATH)
    root = _resolve(data_cfg["outputs"]["dataset_dir"])
    manifest = _read_json(root / "manifest.json")
    if manifest.get("version") != "E03-A8-DEVFT-data-v1" or manifest.get("status") != "cpu_preparation_passed":
        raise ValueError("DevFT data Manifest is missing or invalid")
    files = {"p_devft.jsonl": _read_jsonl_path(root / "p_devft.jsonl"), "sample_order.jsonl": _read_jsonl_path(root / "sample_order.jsonl")}
    for name in ("p_devft.jsonl", "ep_common_history_devft.jsonl", "common_history_audit.jsonl", "sample_order.jsonl"):
        if sha256_file(root / name) != manifest.get("files", {}).get(name):
            raise ValueError(f"DevFT prepared file changed after freeze: {name}")
    ids = [row.get("sample_id") for row in files["p_devft.jsonl"]]
    schedule_ids = [sample_id for group in files["sample_order.jsonl"] for sample_id in group.get("sample_ids", [])]
    if len(ids) != 1821 or len(set(ids)) != len(ids) or len(files["sample_order.jsonl"]) != 456 or schedule_ids != ids and set(schedule_ids) != set(ids):
        raise ValueError("DevFT data count or sample-order coverage differs")
    return manifest, files["p_devft.jsonl"], files["sample_order.jsonl"], root


def _validate_devft_row(row: dict[str, Any], model_config: dict[str, Any]) -> None:
    legacy = dict(row)
    if legacy.get("source_split") != "dev_ft":
        raise ValueError(f"{row.get('sample_id')}: training row is not stamped dev_ft")
    legacy["source_split"] = "train"
    _row_contract(
        legacy,
        eos_token_id=int(model_config["protocol"]["eos_token_id"]),
        prompt_cap=int(model_config["protocol"]["max_prompt_tokens"]),
        sequence_cap=int(model_config["protocol"]["max_train_seq_length"]),
    )


def _validate_parent(model_config: dict[str, Any], train_cfg: dict[str, Any]) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    from src.training.e03_a8_v2_staged import _config_sha256, _resolve

    parent_root = _resolve(model_config["data"]["output_root"]) / train_cfg["parent_run_id"]
    parent_manifest = _read_json(parent_root / "manifest.json")
    checkpoint = _resolve(train_cfg["parent_checkpoint"])
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    node = parent_manifest.get("nodes", {}).get("4048", {})
    if (
        parent_manifest.get("run_id") != train_cfg["parent_run_id"]
        or parent_manifest.get("arm") != "P"
        or parent_manifest.get("optimizer_steps") != 4048
        or parent_manifest.get("authorized_max_step") != 4048
        or node.get("training_status") != "passed"
        or node.get("evaluation_status") != "complete"
        or parent_manifest.get("config_sha256") != _config_sha256(model_config)
        or parent_manifest.get("checkpoint_manifest_sha256") != sha256_file(checkpoint_manifest_path)
        or node.get("checkpoint_manifest_sha256") != sha256_file(checkpoint_manifest_path)
    ):
        raise ValueError("P4048 parent Run identity, completion, or evaluation state differs")
    record = _verify_v2_checkpoint(checkpoint, {
        "run_id": train_cfg["parent_run_id"],
        "arm": "P",
        "optimizer_step": 4048,
        "config_sha256": parent_manifest["config_sha256"],
        "dataset_manifest_sha256": parent_manifest["dataset_manifest_sha256"],
        "sample_order_sha256": parent_manifest["sample_order_sha256"],
        "training_code_sha256": parent_manifest["training_code_sha256"],
    })
    trainer_state = _read_json(checkpoint / "trainer_state.json")
    if (
        trainer_state.get("global_step") != 4048
        or trainer_state.get("sample_cursor") != 16192
        or trainer_state.get("run_id") != train_cfg["parent_run_id"]
        or trainer_state.get("arm") != "P"
        or trainer_state.get("scheduler_last_epoch") != 4048
    ):
        raise ValueError("P4048 optimizer recovery state is not at the verified boundary")
    return parent_manifest, checkpoint, record


def _save_devft_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    run_root: Path,
    run_id: str,
    step: int,
    data_manifest_sha256: str,
    train_file_sha256: str,
    schedule_sha256: str,
    consumed_ids: list[str],
    parent_checkpoint_sha256: str,
    base_sha256: str,
    loss_mean: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    checkpoint = run_root / "trainer" / f"checkpoint-{step:04d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    adapter_config_path = checkpoint / "adapter_config.json"
    adapter_config = _read_json(adapter_config_path)
    adapter_config["base_model_name_or_path"] = "Qwen/Qwen3-1.7B"
    _write_json_atomic(adapter_config_path, adapter_config)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(_rng_state(torch), checkpoint / "rng_state.pth")
    cursor = min(step * 4, 1821)
    state = {
        "run_id": run_id,
        "arm": "P",
        "devft_step": step,
        "global_optimizer_step": 4048 + step,
        "sample_cursor": cursor,
        "last_loss_mean": float(loss_mean),
        "elapsed_training_seconds": float(elapsed_seconds),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "rng_state_saved": True,
    }
    _write_json(checkpoint / "trainer_state.json", state, exclusive=True)
    adapter_identity, restored_config = inference.adapter_weight_identity(checkpoint)
    if restored_config.get("r") != 8 or restored_config.get("lora_alpha") != 16:
        raise RuntimeError("DevFT checkpoint adapter config differs from P QLoRA r=8/alpha=16")
    files = _checkpoint_files(checkpoint)
    record = {
        "status": "complete",
        "version": "E03-A8-DEVFT-checkpoint-v1",
        "experiment_id": "E03-A8-DEVFT",
        "arm": "P",
        "run_id": run_id,
        "devft_step": step,
        "global_optimizer_step": 4048 + step,
        "sample_cursor": cursor,
        "dataset_manifest_sha256": data_manifest_sha256,
        "train_file_sha256": train_file_sha256,
        "sample_order_sha256": schedule_sha256,
        "consumed_sample_ids_sha256": canonical_sha256(consumed_ids),
        "parent_checkpoint_manifest_sha256": parent_checkpoint_sha256,
        "base_model_weight_sha256": base_sha256,
        "adapter_identity": adapter_identity,
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "loss_mean_through_step": float(loss_mean),
        "elapsed_training_seconds": float(elapsed_seconds),
        "files": files,
        "files_sha256": canonical_sha256(files),
        "created_at_utc": _now(),
    }
    _write_json(checkpoint / "checkpoint_manifest.json", record, exclusive=True)
    return record


def train_devft(run_id: str = RUN_ID) -> Path:
    import torch

    train_cfg = _read_yaml(TRAIN_CONFIG_PATH)
    data_manifest, train_rows, schedule, dataset_root = _load_devft_artifacts()
    model_config, _ = __import__("src.training.e03_a8_v2_staged", fromlist=["_load_config"])._load_config(MODEL_CONFIG_PATH, "P")
    parent_manifest, parent_checkpoint, parent_checkpoint_record = _validate_parent(model_config, train_cfg)
    parent_root = _resolve(model_config["data"]["output_root"]) / train_cfg["parent_run_id"]
    if train_cfg.get("version") != "E03-A8-DEVFT-staged-training-v1" or train_cfg.get("arm") != "P":
        raise ValueError("unsupported DevFT training config")
    if train_cfg.get("parent_run_id") != PARENT_RUN_ID:
        raise ValueError("DevFT parent Run differs from the frozen P4048 source")
    if len(schedule) != int(train_cfg["training"]["max_optimizer_steps"]):
        raise ValueError("DevFT training steps differ from its frozen one-epoch schedule")

    for row in train_rows:
        _validate_devft_row(row, model_config)
    schedule_ids = [sample_id for item in schedule for sample_id in item["sample_ids"]]
    if len(schedule_ids) != 1821 or len(set(schedule_ids)) != len(schedule_ids) or set(schedule_ids) != {row["sample_id"] for row in train_rows}:
        raise ValueError("DevFT schedule is not an exact one-epoch permutation of Dev data")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_gpu_unavailable_or_bf16_unsupported: DevFT requires CUDA BF16")

    run_root = _resolve(train_cfg["data"]["output_root"]) / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite DevFT Run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "trainer").mkdir()
    model_config_sha = canonical_sha256(model_config)
    train_path = _resolve(train_cfg["data"]["train_path"])
    schedule_path = _resolve(train_cfg["data"]["sample_order_path"])
    data_manifest_path = _resolve(train_cfg["data"]["dataset_manifest"])
    train_code_paths = [
        "configs/data/e03_a8_devft_v1.yaml",
        "configs/training/e03_a8_devft_p.yaml",
        "configs/training/e03_a8_v2_p_staged.yaml",
        "src/data/e03_a8_devft.py",
        "src/data/e03_a8_sft.py",
        "src/shared/e03_a8_protocol.py",
        "src/shared/prompting.py",
        "src/training/e03_a8_completion_loss.py",
        "src/training/e03_a8_v2_staged.py",
        "src/training/e03_a8_staged.py",
        "src/training/training_sft.py",
        "src/data/sft_collator.py",
        "scripts/e03_a8_devft.py",
        "participant/run_inference.py",
    ]
    training_code = {name: sha256_file(_resolve(name)) for name in train_code_paths}
    parent_checkpoint_sha = sha256_file(parent_checkpoint / "checkpoint_manifest.json")
    _write_json_exclusive(run_root / "config_snapshot.json", train_cfg)
    run_manifest = {
        "version": "E03-A8-DEVFT-run-v1",
        "experiment_id": "E03-A8-DEVFT",
        "status": "running",
        "stage": "initializing_from_P4048",
        "arm": "P",
        "run_id": run_id,
        "config_sha256": canonical_sha256(train_cfg),
        "model_config_sha256": model_config_sha,
        "dataset_manifest_sha256": sha256_file(data_manifest_path),
        "train_file_sha256": sha256_file(train_path),
        "sample_order_sha256": canonical_sha256(schedule),
        "training_code_sha256": training_code,
        "parent": {
            "run_id": train_cfg["parent_run_id"],
            "optimizer_step": 4048,
            "run_manifest_sha256": sha256_file(_resolve(model_config["data"]["output_root"]) / train_cfg["parent_run_id"] / "manifest.json"),
            "checkpoint_path": str(parent_checkpoint.relative_to(ROOT)),
            "checkpoint_manifest_sha256": parent_checkpoint_sha,
            "adapter_weight_sha256": parent_checkpoint_record["adapter_identity"]["weight_sha256"],
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_policy": "reset_to_seed_42_for_independent_dev_epoch",
        },
        "effective_rows": len(train_rows),
        "planned_steps": len(schedule),
        "checkpoint_interval_steps": int(train_cfg["training"]["checkpoint_interval_steps"]),
        "optimizer_state_restored": True,
        "official_val_read": False,
        "authorization_source": train_cfg["authorization"]["source"],
        "optimizer_steps": 0,
        "nodes": {},
        "created_at_utc": _now(),
    }
    _write_json_exclusive(run_root / "manifest.json", run_manifest)
    (run_root / "sample_order.jsonl").write_bytes(schedule_path.read_bytes())

    tokenizer = model = ledger = None
    base_before = None
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(model_config)
        _set_seed(int(train_cfg["training"]["seed"]), torch)
        model, torch, model_facts = _load_train_model(
            model_config,
            _base_path(model_config),
            adapter_path=parent_checkpoint,
            is_trainable_adapter=True,
            autocast_adapter_dtype=True,
        )
        trainable = _lora_parameters(model)
        lora_sha_start = _lora_tensor_sha256(model, torch)
        parent_training_summary = _read_json(parent_root / parent_manifest["nodes"]["4048"]["training_summary_path"])
        expected_parent_tensor_sha = parent_training_summary.get("independent_reload_check", {}).get("adapter_tensor_sha256_after_reload")
        if not isinstance(expected_parent_tensor_sha, str) or lora_sha_start != expected_parent_tensor_sha:
            raise ValueError("loaded DevFT LoRA tensors differ from the verified P4048 parent")

        optimizer = torch.optim.AdamW(
            [parameter for _name, parameter in trainable],
            lr=float(model_config["optimizer"]["learning_rate"]),
            weight_decay=float(model_config["optimizer"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        optimizer.load_state_dict(torch.load(parent_checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
        scheduler.load_state_dict(torch.load(parent_checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
        if len(optimizer.state) != len(trainable) or scheduler.last_epoch != 4048:
            raise ValueError("P4048 AdamW or constant scheduler state did not restore at Step 4048")
        observed_optimizer_steps = []
        for state in optimizer.state.values():
            state_step = state.get("step")
            if state_step is not None:
                observed_optimizer_steps.append(int(state_step.item()) if hasattr(state_step, "item") else int(state_step))
        if observed_optimizer_steps and set(observed_optimizer_steps) != {4048}:
            raise ValueError("P4048 AdamW moments do not report exactly 4,048 optimizer updates")
        _set_seed(int(train_cfg["training"]["seed"]), torch)
        base_before = _base_identity(model_config)
        torch.cuda.reset_peak_memory_stats()
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        ledger_path = run_root / "step_ledger.jsonl"
        ledger = ledger_path.open("xb")
        pending: list[dict[str, Any]] = []
        losses_all: list[float] = []
        step_gpu_seconds: list[float] = []
        consumed_ids: list[str] = []
        step_started = time.monotonic()
        last_checkpoint_time = step_started
        last_checkpoint_step = 0
        with JsonlById(train_path) as train_index:
            for step, group in enumerate(schedule, 1):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                completion_tokens = 0
                sequence_tokens = 0
                prompt_tokens = 0
                started = time.monotonic()
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
                for sample_id in group["sample_ids"]:
                    row = train_index.get(sample_id)
                    _validate_devft_row(row, model_config)
                    batch = collator([row])
                    device = model.get_input_embeddings().weight.device
                    batch = {key: value.to(device) for key, value in batch.items()}
                    loss = completion_only_forward_loss(
                        model,
                        batch,
                        prompt_length=int(row["prompt_length"]),
                        sequence_length=len(row["input_ids"]),
                    )
                    if not bool(torch.isfinite(loss.detach()).all().item()):
                        raise FloatingPointError(f"non-finite DevFT completion loss at step {step}")
                    losses.append(loss.detach().float())
                    completion_tokens += int(row["completion_length"])
                    sequence_tokens += len(row["input_ids"])
                    prompt_tokens += int(row["prompt_length"])
                    (loss / len(group["sample_ids"])).backward()
                    consumed_ids.append(sample_id)
                gradient = _gradient_summary(
                    trainable,
                    torch,
                    full=(step % int(train_cfg["training"]["diagnostic_interval_steps"]) == 0 or step == len(schedule)),
                    loss_tensors=losses,
                )
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                event_end.record()
                torch.cuda.synchronize()
                gpu_seconds = event_start.elapsed_time(event_end) / 1000.0
                step_gpu_seconds.append(gpu_seconds)
                loss_values = [float(value.detach().cpu()) for value in losses]
                step_loss = sum(loss_values) / len(loss_values)
                losses_all.append(step_loss)
                pending.append({
                    "run_id": run_id,
                    "arm": "P",
                    "devft_step": step,
                    "global_optimizer_step": 4048 + step,
                    "sample_ids": list(group["sample_ids"]),
                    "sample_ids_sha256": canonical_sha256(group["sample_ids"]),
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "sequence_tokens": sequence_tokens,
                    "mean_completion_loss": step_loss,
                    "micro_sample_losses": loss_values,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient": gradient,
                    "gpu_step_seconds": gpu_seconds,
                    "host_step_seconds": time.monotonic() - started,
                    "created_at_utc": _now(),
                })

                interval = int(train_cfg["training"]["checkpoint_interval_steps"])
                if step % interval == 0 or step == len(schedule):
                    payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for row in pending)
                    ledger.write(payload)
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    pending.clear()
                    elapsed = time.monotonic() - step_started
                    segment_elapsed = time.monotonic() - last_checkpoint_time
                    checkpoint_record = _save_devft_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        torch=torch,
                        run_root=run_root,
                        run_id=run_id,
                        step=step,
                        data_manifest_sha256=sha256_file(data_manifest_path),
                        train_file_sha256=sha256_file(train_path),
                        schedule_sha256=canonical_sha256(schedule),
                        consumed_ids=consumed_ids,
                        parent_checkpoint_sha256=parent_checkpoint_sha,
                        base_sha256=base_before["observed_weight_hash"],
                        loss_mean=sum(losses_all) / len(losses_all),
                        elapsed_seconds=elapsed,
                    )
                    node_root = run_root / f"step-{step:04d}"
                    node_root.mkdir(parents=False, exist_ok=False)
                    segment_steps = step - last_checkpoint_step
                    recent = _read_jsonl(run_root / "step_ledger.jsonl")[-segment_steps:]
                    segment_rows = sum(len(group_row["sample_ids"]) for group_row in schedule[last_checkpoint_step:step])
                    _write_json_exclusive(node_root / "training.json", {
                        "run_id": run_id,
                        "arm": "P",
                        "devft_step": step,
                        "global_optimizer_step": 4048 + step,
                        "rows_consumed": len(consumed_ids),
                        "segment_rows": segment_rows,
                        "cumulative_loss_mean": sum(losses_all) / len(losses_all),
                        "segment_loss_mean": sum(float(row["mean_completion_loss"]) for row in recent) / len(recent),
                        "last_loss": losses_all[-1],
                        "segment_training_seconds": segment_elapsed,
                        "cumulative_training_seconds": elapsed,
                        "segment_gpu_step_seconds": {
                            "mean": sum(float(row["gpu_step_seconds"]) for row in recent) / len(recent),
                            "p50": sorted(float(row["gpu_step_seconds"]) for row in recent)[(len(recent) - 1) // 2],
                            "p90": sorted(float(row["gpu_step_seconds"]) for row in recent)[math.ceil(0.90 * len(recent)) - 1],
                        },
                        "cumulative_gpu_step_seconds": {
                            "mean": sum(step_gpu_seconds) / len(step_gpu_seconds),
                            "p50": sorted(step_gpu_seconds)[(len(step_gpu_seconds) - 1) // 2],
                            "p90": sorted(step_gpu_seconds)[math.ceil(0.90 * len(step_gpu_seconds)) - 1],
                        },
                        "cuda_memory": inference.cuda_memory_facts(torch),
                        "checkpoint_manifest_sha256": canonical_sha256(checkpoint_record),
                        "checkpoint_path": str((run_root / "trainer" / f"checkpoint-{step:04d}").relative_to(run_root)),
                        "created_at_utc": _now(),
                    })
                    run_manifest.update({
                        "status": "running" if step < len(schedule) else "checkpoint_saved",
                        "stage": f"devft_step_{step}",
                        "optimizer_steps": 4048 + step,
                        "devft_steps": step,
                        "last_checkpoint": f"trainer/checkpoint-{step:04d}",
                        "last_checkpoint_manifest_sha256": sha256_file(run_root / "trainer" / f"checkpoint-{step:04d}" / "checkpoint_manifest.json"),
                        "nodes": {**run_manifest.get("nodes", {}), str(step): {"training_status": "passed", "checkpoint_manifest_sha256": sha256_file(run_root / "trainer" / f"checkpoint-{step:04d}" / "checkpoint_manifest.json")}},
                        "updated_at_utc": _now(),
                    })
                    _write_json_atomic(run_root / "manifest.json", run_manifest)
                    segment_gpu = sum(row["gpu_step_seconds"] for row in recent)
                    print(
                        f"[E03-A8-DEVFT P] step={step}/{len(schedule)} rows={len(consumed_ids)}/1821 "
                        f"segment_loss={sum(float(row['mean_completion_loss']) for row in recent) / len(recent):.5f} "
                        f"avg_gpu_step={segment_gpu / len(recent):.2f}s elapsed={elapsed:.0f}s checkpoint=complete",
                        flush=True,
                    )
                    last_checkpoint_time = time.monotonic()
                    last_checkpoint_step = step
                    if elapsed > float(train_cfg["training"]["max_training_seconds"]):
                        raise TimeoutError("DevFT exceeded its configured training-time limit")

        if pending:
            raise RuntimeError("DevFT ledger has unflushed steps at the final checkpoint")
        ledger.close()
        ledger = None
        final_checkpoint = run_root / "trainer" / f"checkpoint-{len(schedule):04d}"
        model.eval()
        probe_row = train_rows[0]
        probe_prompt = probe_row["input_ids"][:probe_row["prompt_length"]]
        lora_sha_before_reload = _lora_tensor_sha256(model, torch)
        next_token_before = _next_token_id(model, torch, probe_prompt)
        final_adapter_identity, _ = inference.adapter_weight_identity(final_checkpoint)
        model_facts_reload = None
        reloaded, torch, model_facts_reload = _load_train_model(
            model_config,
            _base_path(model_config),
            adapter_path=final_checkpoint,
            autocast_adapter_dtype=True,
        )
        reloaded.eval()
        next_token_after = _next_token_id(reloaded, torch, probe_prompt)
        lora_sha_after_reload = _lora_tensor_sha256(reloaded, torch)
        if next_token_before != next_token_after or lora_sha_before_reload != lora_sha_after_reload:
            raise RuntimeError("DevFT independent Adapter reload does not reproduce its exact tensors and next token")
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        base_after = _base_identity(model_config)
        if base_before["observed_weight_hash"] != base_after["observed_weight_hash"]:
            raise RuntimeError("DevFT changed the frozen Base model weights")
        total_elapsed = time.monotonic() - step_started
        ledger_rows = _read_jsonl(run_root / "step_ledger.jsonl")
        total_loss = sum(float(row["mean_completion_loss"]) for row in ledger_rows) / len(ledger_rows)
        final_training = {
            "status": "passed",
            "run_id": run_id,
            "arm": "P",
            "parent_run_id": train_cfg["parent_run_id"],
            "parent_optimizer_step": 4048,
            "devft_steps": len(schedule),
            "global_optimizer_step": 4048 + len(schedule),
            "rows_consumed": len(consumed_ids),
            "rows_unique": len(set(consumed_ids)),
            "consumed_ids_sha256": canonical_sha256(consumed_ids),
            "loss_mean": total_loss,
            "last_loss": float(ledger_rows[-1]["mean_completion_loss"]),
            "training_seconds": total_elapsed,
            "gpu_step_seconds": {
                "mean": sum(float(row["gpu_step_seconds"]) for row in ledger_rows) / len(ledger_rows),
                "p50": sorted(float(row["gpu_step_seconds"]) for row in ledger_rows)[(len(ledger_rows) - 1) // 2],
                "p90": sorted(float(row["gpu_step_seconds"]) for row in ledger_rows)[math.ceil(0.90 * len(ledger_rows)) - 1],
            },
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "scheduler_last_epoch": int(scheduler.last_epoch),
            "initial_adapter_tensor_sha256": lora_sha_start,
            "final_adapter_tensor_sha256": lora_sha_before_reload,
            "final_adapter_weight_sha256": final_adapter_identity.get("weight_sha256"),
            "independent_reload": {
                "status": "passed",
                "next_token_exact_match": next_token_before == next_token_after,
                "adapter_tensor_sha256_before_reload": lora_sha_before_reload,
                "adapter_tensor_sha256_after_reload": lora_sha_after_reload,
                "model_facts": model_facts_reload,
            },
            "base_weights_unchanged": True,
            "base_weight_sha256": base_before["observed_weight_hash"],
            "model_facts": model_facts,
            "tokenizer_facts": tokenizer_facts,
            "created_at_utc": _now(),
        }
        final_node = run_root / f"step-{len(schedule):04d}"
        _write_json_atomic(final_node / "training.json", final_training)
        final_manifest = _read_json(run_root / "manifest.json")
        final_manifest.update({
            "status": "awaiting_val_prediction",
            "stage": "devft_complete_awaiting_target_free_val_generation",
            "optimizer_steps": 4048 + len(schedule),
            "devft_steps": len(schedule),
            "training_summary_path": str((final_node / "training.json").relative_to(run_root)),
            "training_summary_sha256": sha256_file(final_node / "training.json"),
            "final_adapter_tensor_sha256": lora_sha_before_reload,
            "final_adapter_weight_sha256": final_adapter_identity.get("weight_sha256"),
            "base_weights_unchanged": True,
            "official_val_read": False,
            "ended_at_utc": _now(),
        })
        _write_json_atomic(run_root / "manifest.json", final_manifest)
        return run_root
    except BaseException as exc:
        if ledger is not None:
            ledger.close()
        manifest_path = run_root / "manifest.json"
        if manifest_path.is_file():
            broken = _read_json(manifest_path)
            broken.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", stage="devft_interrupted" if isinstance(exc, KeyboardInterrupt) else "devft_failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
            _write_json_atomic(manifest_path, broken)
        raise
    finally:
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _eval_root(run_id: str) -> Path:
    return _resolve("outputs/evaluation/e03_a8_devft_v1") / run_id


def prepare_val_inputs(run_id: str = RUN_ID) -> Path:
    """Freeze Val histories and exact prior A2 point IDs without opening Gold fields."""
    output_root = _eval_root(run_id)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite DevFT Val evaluation directory: {output_root}")
    val_path = _resolve("data/processed/v1/val.jsonl")
    d03_path = _resolve("outputs/data_pipeline/v1/manifest.json")
    d03 = _read_json(d03_path)
    val_record = next((row for row in d03.get("outputs", []) if row.get("path") == "data/processed/v1/val.jsonl"), None)
    if d03.get("pipeline_version") != "D0.3-v1" or not val_record or sha256_file(val_path) != val_record.get("sha256"):
        raise ValueError("normalized Val does not match the frozen D0.3 Manifest")
    raw_rows = _read_jsonl_path(val_path)
    ids = []
    target_free = []
    mapping_rows = []
    seen = set()
    for row in raw_rows:
        sample_id = row.get("sample_id")
        conversation_id = row.get("conversation_id")
        history = row.get("history")
        source = row.get("source")
        if (
            not isinstance(sample_id, str) or not sample_id.startswith("val:")
            or not isinstance(conversation_id, str) or not isinstance(history, list)
            or not isinstance(source, dict) or source.get("split") != "val"
            or sample_id in seen
        ):
            raise ValueError("normalized Val has an invalid ID/history/source contract")
        seen.add(sample_id)
        ids.append(sample_id)
        target_free.append({"id": sample_id, "history": history})
        mapping_rows.append({
            "eval_id": sample_id,
            "sample_id": sample_id,
            "conversation_id": conversation_id,
            "target_user_turn_id": source.get("target_user_turn_id"),
        })
    if len(ids) != 2354 or len(set(ids)) != len(ids):
        raise ValueError(f"D0.3 Val identity count differs: {len(ids)}")

    exposure_dir = VAL_EXPOSURE_PATH
    selection_path = exposure_dir / "sample_ids.json"
    selection = _read_json(selection_path)
    prior_point_ids = selection.get("point_ids")
    if selection.get("points") != 209 or not isinstance(prior_point_ids, list) or len(prior_point_ids) != 209 or len(set(prior_point_ids)) != 209:
        raise ValueError("A2 prior exposure identity does not contain exactly 209 unique point IDs")
    by_source_point = {}
    for row in mapping_rows:
        raw_point_id = f"{row['conversation_id']}_{row['target_user_turn_id']}"
        if raw_point_id in by_source_point:
            raise ValueError(f"D0.3 Val maps multiple samples to one A2 target turn: {raw_point_id}")
        by_source_point[raw_point_id] = row["eval_id"]
    missing_exposures = set(prior_point_ids) - set(by_source_point)
    if missing_exposures:
        raise ValueError(f"A2 prior exploration IDs do not map exactly into D0.3 Val: {len(missing_exposures)} missing")
    exposed_ids = [by_source_point[value] for value in prior_point_ids]
    exposed_id_set = set(exposed_ids)
    if len(exposed_ids) != 209 or len(exposed_id_set) != 209:
        raise ValueError("A2 prior exploration did not map one-to-one to exactly 209 Val points")

    data_manifest, train_rows, _schedule, _dataset_root = _load_devft_artifacts()
    devft_ids = {row["sample_id"] for row in train_rows}
    if devft_ids & set(ids):
        raise ValueError("DevFT training IDs overlap normalized Val IDs")
    prior_conversations = {
        row["conversation_id"] for row in mapping_rows if row["eval_id"] in exposed_id_set
    }
    output_root.mkdir(parents=True, exist_ok=False)
    _write_jsonl_exclusive(output_root / "val_target_free.jsonl", target_free)
    _write_jsonl_exclusive(output_root / "val_id_map.jsonl", mapping_rows)
    _write_json_exclusive(output_root / "a2_exposed_ids.json", {
        "version": VAL_LABEL_EXCLUSION_VERSION,
        "source_path": str(selection_path.relative_to(ROOT)),
        "source_sha256": sha256_file(selection_path),
        "ids": exposed_ids,
        "ids_sha256": canonical_sha256(exposed_ids),
    })
    manifest = {
        "version": VAL_INPUT_VERSION,
        "run_id": run_id,
        "status": "target_free_val_frozen",
        "created_at_utc": _now(),
        "source": {
            "d03_manifest_path": str(d03_path.relative_to(ROOT)),
            "d03_manifest_sha256": sha256_file(d03_path),
            "normalized_val_path": str(val_path.relative_to(ROOT)),
            "normalized_val_sha256": sha256_file(val_path),
            "normalized_val_row_count": len(raw_rows),
        },
        "devft_data_manifest_sha256": data_manifest["dataset_manifest_sha256"],
        "training_rows": len(train_rows),
        "target_free_input_count": len(target_free),
        "target_free_input_ids_sha256": selected_ids_sha256(ids),
        "devft_val_id_overlap_count": 0,
        "prior_a2_exact_point_exposure": {
            "version": VAL_LABEL_EXCLUSION_VERSION,
            "point_count": len(exposed_ids),
            "unique_conversation_count": len(prior_conversations),
            "point_ids_sha256": canonical_sha256(exposed_ids),
            "point_ids_path": "a2_exposed_ids.json",
            "same_conversation_context_may_overlap": True,
        },
        "files": {
            name: sha256_file(output_root / name)
            for name in ("val_target_free.jsonl", "val_id_map.jsonl", "a2_exposed_ids.json")
        },
        "references_read": False,
        "target_fields_serialized": False,
        "code_sha256": {
            "scripts/e03_a8_devft.py": sha256_file(Path(__file__).resolve()),
            "src/data/e03_a8_sft.py": sha256_file(_resolve("src/data/e03_a8_sft.py")),
            "src/shared/prompting.py": sha256_file(_resolve("src/shared/prompting.py")),
        },
    }
    _write_json_exclusive(output_root / "val_input_manifest.json", manifest)
    return output_root


def _load_target_free_val(eval_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = _read_json(eval_root / "val_input_manifest.json")
    if manifest.get("version") != VAL_INPUT_VERSION or manifest.get("status") != "target_free_val_frozen" or manifest.get("references_read") is not False:
        raise ValueError("DevFT Val input Manifest is not frozen target-free")
    for name, expected in manifest.get("files", {}).items():
        if sha256_file(eval_root / name) != expected:
            raise ValueError(f"DevFT Val target-free input changed after freeze: {name}")
    inputs = _read_jsonl_path(eval_root / "val_target_free.jsonl")
    if any(set(row) != {"id", "history"} for row in inputs):
        raise ValueError("DevFT Val generation inputs contain fields outside id/history")
    mapping_rows = _read_jsonl_path(eval_root / "val_id_map.jsonl")
    mapping = _unique_by(mapping_rows, "eval_id", "DevFT Val ID map")
    ids = [row["id"] for row in inputs]
    if len(ids) != 2354 or len(set(ids)) != len(ids) or set(ids) != set(mapping):
        raise ValueError("DevFT Val target-free inputs and ID map do not have exact 2,354 ID coverage")
    if selected_ids_sha256(ids) != manifest.get("target_free_input_ids_sha256"):
        raise ValueError("DevFT Val target-free ordered IDs differ from their Manifest")
    return manifest, inputs, mapping


def _validate_devft_checkpoint(checkpoint: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    record = _read_json(checkpoint / "checkpoint_manifest.json")
    if record.get("status") != "complete" or record.get("run_id") != run_manifest.get("run_id") or record.get("arm") != "P":
        raise ValueError("DevFT checkpoint status or Run identity differs")
    files = _checkpoint_files(checkpoint)
    if files != record.get("files") or canonical_sha256(files) != record.get("files_sha256"):
        raise ValueError("DevFT checkpoint file identity differs from its Manifest")
    state = _read_json(checkpoint / "trainer_state.json")
    step = int(checkpoint.name.removeprefix("checkpoint-"))
    if state.get("devft_step") != step or state.get("sample_cursor") != min(step * 4, 1821) or record.get("devft_step") != step:
        raise ValueError("DevFT checkpoint Step and consumed sample cursor differ")
    return record


def predict_val(run_id: str = RUN_ID) -> Path:
    import torch

    eval_root = _eval_root(run_id)
    if not eval_root.is_dir():
        raise FileNotFoundError("target-free Val must be frozen before prediction")
    val_manifest, val_inputs, mapping = _load_target_free_val(eval_root)
    generation_manifest_path = eval_root / "generation_manifest.json"
    if generation_manifest_path.exists() or (eval_root / "predictions.jsonl").exists():
        raise FileExistsError("refusing to overwrite frozen DevFT Val predictions")
    train_cfg = _read_yaml(TRAIN_CONFIG_PATH)
    model_config, _ = __import__("src.training.e03_a8_v2_staged", fromlist=["_load_config"])._load_config(MODEL_CONFIG_PATH, "P")
    run_root = _resolve(train_cfg["data"]["output_root"]) / run_id
    training_manifest = _read_json(run_root / "manifest.json")
    if (
        training_manifest.get("status") != "awaiting_val_prediction"
        or training_manifest.get("devft_steps") != 456
        or training_manifest.get("official_val_read") is not False
    ):
        raise ValueError("DevFT Run is not complete or has already read Val Gold")
    checkpoint = run_root / "trainer" / "checkpoint-0456"
    checkpoint_record = _validate_devft_checkpoint(checkpoint, training_manifest)
    tokenizer, tokenizer_facts = _runtime_tokenizer(model_config)
    prompt_path = _resolve(model_config["protocol"]["prompt_path"])
    p_prompt = prompt_path.read_text(encoding="utf-8")
    ep_prompt_path = ROOT / "configs/prompts/p_a8_v2_emotion_profile.txt"
    ep_prompt = ep_prompt_path.read_text(encoding="utf-8")
    p0_path = _resolve(model_config["protocol"]["selection_prompt_path"])
    schema_path = _resolve("data/train/train_public.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enums = validate_a8_prompt_enum_order(p_prompt, schema, "P")
    validate_a8_prompt_enum_order(ep_prompt, schema, "EP")
    if sha256_file(prompt_path) != model_config["protocol"]["prompt_sha256"] or sha256_file(p0_path) != model_config["protocol"]["p0_sha256"]:
        raise ValueError("A8 P Prompt or protected P0 identity changed before Val prediction")
    base_before = _base_identity(model_config)
    started = time.monotonic()
    model = None
    try:
        model, torch, model_facts = _load_train_model(
            model_config,
            _base_path(model_config),
            adapter_path=checkpoint,
            autocast_adapter_dtype=True,
        )
        model.eval()
        adapter_identity, _ = inference.adapter_weight_identity(checkpoint)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        cohorts = {
            "cohorts": {"full_val": {"name": "full_val", "mapping": mapping}},
            "union_ids": [row["id"] for row in val_inputs],
            "union_inputs": val_inputs,
            "union_count": len(val_inputs),
        }
        eval_config = {
            "generation": {"seed": 42, "max_new_tokens": 256, "max_run_seconds": 21600},
            "protocol": {"max_prompt_tokens": 1280},
        }
        generated, generation_facts = _generate_rows(
            config=model_config,
            eval_config=eval_config,
            run_id=run_id,
            step=456,
            cohorts=cohorts,
            tokenizer=tokenizer,
            model=model,
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            p0_prompt=p0_path.read_text(encoding="utf-8"),
            schema=schema,
        )
        if len(generated) != 2354 or {row["id"] for row in generated} != set(cohorts["union_ids"]):
            raise RuntimeError("DevFT Val prediction coverage is not exactly 2,354 unique IDs")
        _append_jsonl_fsynced(eval_root / "predictions.jsonl", generated)
        prediction_sha = sha256_file(eval_root / "predictions.jsonl")
        base_after = _base_identity(model_config)
        if base_before["observed_weight_hash"] != base_after["observed_weight_hash"]:
            raise RuntimeError("Val prediction changed the frozen Base model weights")
        generation_manifest = {
            "version": "E03-A8-DEVFT-val-generation-v1",
            "experiment_id": "E03-A8-DEVFT",
            "run_id": run_id,
            "arm": "P",
            "checkpoint_step": 456,
            "checkpoint_global_optimizer_step": 4504,
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
            "checkpoint_adapter_weight_sha256": adapter_identity.get("weight_sha256"),
            "val_input_manifest_sha256": sha256_file(eval_root / "val_input_manifest.json"),
            "val_input_sha256": val_manifest["files"]["val_target_free.jsonl"],
            "val_id_map_sha256": val_manifest["files"]["val_id_map.jsonl"],
            "prediction_count": len(generated),
            "prediction_ids_sha256": selected_ids_sha256([row["id"] for row in generated]),
            "prediction_sha256": prediction_sha,
            "generation_facts": generation_facts,
            "tokenizer_facts": tokenizer_facts,
            "model_facts": model_facts,
            "base_weight_sha256_before": base_before["observed_weight_hash"],
            "base_weight_sha256_after": base_after["observed_weight_hash"],
            "base_weights_unchanged": True,
            "references_read": False,
            "generation_seconds_including_model_load": time.monotonic() - started,
            "created_at_utc": _now(),
            "status": "predictions_frozen",
        }
        _write_json_exclusive(generation_manifest_path, generation_manifest)
        final_run_manifest = _read_json(run_root / "manifest.json")
        final_run_manifest.update({
            "status": "predictions_frozen_awaiting_local_score",
            "stage": "val_predictions_frozen_references_not_read",
            "val_prediction_count": len(generated),
            "val_prediction_sha256": prediction_sha,
            "val_generation_manifest_sha256": sha256_file(generation_manifest_path),
            "official_val_read": False,
            "updated_at_utc": _now(),
        })
        _write_json_atomic(run_root / "manifest.json", final_run_manifest)
        return eval_root
    finally:
        if model is not None:
            del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_val(run_id: str = RUN_ID) -> Path:
    eval_root = _eval_root(run_id)
    generation_manifest_path = eval_root / "generation_manifest.json"
    generation_manifest = _read_json(generation_manifest_path)
    predictions_path = eval_root / "predictions.jsonl"
    if (
        generation_manifest.get("status") != "predictions_frozen"
        or generation_manifest.get("references_read") is not False
        or sha256_file(predictions_path) != generation_manifest.get("prediction_sha256")
        or generation_manifest.get("prediction_count") != 2354
    ):
        raise ValueError("Val predictions are not frozen or their identity changed")
    val_manifest, val_inputs, mapping = _load_target_free_val(eval_root)
    val_path = _resolve("data/processed/v1/val.jsonl")
    d03 = _read_json("outputs/data_pipeline/v1/manifest.json")
    val_record = next((row for row in d03.get("outputs", []) if row.get("path") == "data/processed/v1/val.jsonl"), None)
    if not val_record or sha256_file(val_path) != val_record.get("sha256") or sha256_file(val_path) != val_manifest["source"]["normalized_val_sha256"]:
        raise ValueError("Val Gold source changed after target-free prediction freeze")

    # This is the first point at which the scoring process reads target fields.
    gold_rows = _read_jsonl_path(val_path)
    gold_by_id = _unique_by(gold_rows, "sample_id", "D0.3 Val Gold")
    prediction_rows = _read_jsonl_path(predictions_path)
    predictions = _unique_by(prediction_rows, "id", "frozen DevFT Val predictions")
    ids = [row["id"] for row in val_inputs]
    if set(gold_by_id) != set(ids) or set(predictions) != set(ids):
        raise ValueError("D0.3 Val Gold, target-free input, and predictions do not share exact IDs")
    references = {
        sample_id: {
            "eval_id": sample_id,
            "sample_id": sample_id,
            "target": gold_by_id[sample_id]["target"],
        }
        for sample_id in ids
    }
    for sample_id in ids:
        if mapping[sample_id].get("sample_id") != sample_id or gold_by_id[sample_id].get("conversation_id") != mapping[sample_id].get("conversation_id"):
            raise ValueError(f"Val Gold and frozen ID map differ for {sample_id}")

    exposure = _read_json(eval_root / "a2_exposed_ids.json")
    exposed_ids = exposure.get("ids")
    if exposure.get("version") != VAL_LABEL_EXCLUSION_VERSION or not isinstance(exposed_ids, list) or len(exposed_ids) != 209 or not set(exposed_ids) <= set(ids):
        raise ValueError("A2 exact-point exposure set differs after prediction freeze")
    exposed_set = set(exposed_ids)
    cohort_specs = {
        "full_val_2354": ids,
        "a2_previously_explored_209_exact_points": [sample_id for sample_id in ids if sample_id in exposed_set],
        "outside_a2_exact_points_2145": [sample_id for sample_id in ids if sample_id not in exposed_set],
    }
    schema = json.loads(_resolve("data/train/train_public.schema.json").read_text(encoding="utf-8"))
    p_prompt = _resolve("configs/prompts/p_a8_v2_profile.txt").read_text(encoding="utf-8")
    enums = validate_a8_prompt_enum_order(p_prompt, schema, "P")
    metrics = {}
    bootstraps = {}
    for name, subset_ids in cohort_specs.items():
        cohort = {
            "name": name,
            "ids": subset_ids,
            "ids_sha256": selected_ids_sha256(subset_ids),
            "conversation_count": len({_conversation_id(mapping[sample_id]) for sample_id in subset_ids}),
            "mapping": mapping,
        }
        metrics[name] = score_a8_cohort(cohort, references, predictions, enums, arm="P")
        bootstraps[name] = bootstrap_conversation_micro_f1(
            cohort,
            references,
            predictions,
            enums,
            replicates=1000,
            seed=42,
        )

    scoring = {
        "version": "E03-A8-DEVFT-val-local-proxy-v1",
        "experiment_id": "E03-A8-DEVFT",
        "run_id": run_id,
        "status": "completed_local_proxy",
        "metric_scope": "local_proxy_not_official_score",
        "dev_was_used_for_training": True,
        "val_prediction_freeze_sha256": generation_manifest["prediction_sha256"],
        "references_read_after_prediction_freeze": True,
        "prediction_count": len(prediction_rows),
        "prior_a2_exact_exposure_count": len(exposed_set),
        "prior_a2_conversation_count": len({mapping[sample_id]["conversation_id"] for sample_id in exposed_set}),
        "outside_a2_exact_points_may_share_conversations_with_exposed_points": True,
        "metrics": metrics,
        "conversation_cluster_bootstrap": bootstraps,
        "created_at_utc": _now(),
    }
    score_path = eval_root / "scoring.json"
    _write_json_exclusive(score_path, scoring)
    score_manifest = {
        "version": "E03-A8-DEVFT-score-manifest-v1",
        "run_id": run_id,
        "status": "scored_after_prediction_freeze",
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "prediction_sha256": generation_manifest["prediction_sha256"],
        "val_gold_source": str(val_path.relative_to(ROOT)),
        "val_gold_source_sha256": sha256_file(val_path),
        "d03_manifest_sha256": sha256_file(_resolve("outputs/data_pipeline/v1/manifest.json")),
        "scoring_sha256": sha256_file(score_path),
        "a2_exposed_ids_sha256": sha256_file(eval_root / "a2_exposed_ids.json"),
        "references_read_after_prediction_freeze": True,
        "created_at_utc": _now(),
    }
    _write_json_exclusive(eval_root / "scoring_manifest.json", score_manifest)
    run_root = _resolve(_read_yaml(TRAIN_CONFIG_PATH)["data"]["output_root"]) / run_id
    run_manifest = _read_json(run_root / "manifest.json")
    run_manifest.update({
        "status": "scored_local_proxy",
        "stage": "val_prediction_frozen_and_scored",
        "official_val_read": True,
        "val_scoring_manifest_sha256": sha256_file(eval_root / "scoring_manifest.json"),
        "updated_at_utc": _now(),
    })
    _write_json_atomic(run_root / "manifest.json", run_manifest)
    return score_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continue A8 P on D0.3 Dev and predict D0.3 Val in an isolated protocol version.")
    parser.add_argument("mode", choices=("prepare-devft", "train-devft", "prepare-val", "predict-val", "score-val"))
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args(argv)
    if args.mode == "prepare-devft":
        result = prepare_devft_data()
    elif args.mode == "train-devft":
        result = train_devft(args.run_id)
    elif args.mode == "prepare-val":
        result = prepare_val_inputs(args.run_id)
    elif args.mode == "predict-val":
        result = predict_val(args.run_id)
    else:
        result = score_val(args.run_id)
    print(json.dumps({"status": "completed", "path": str(result)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
