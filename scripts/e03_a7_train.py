#!/usr/bin/env python3
"""Prepare and run E03-A7's isolated Prompt-only P/EP classifier experiments."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e03_a5_evaluate, e03_a5_probe
from src.data.e03_a5_labels import canonical_sha256 as a5_canonical_sha256
from src.data.e03_a5_labels import iter_jsonl, schema_label_order, sha256_file
from src.data.e03_a7_classification import (
    SFTPromptIndex,
    build_schedule,
    build_supervision,
    emotion_label_order,
    read_splits,
    sft_prompt_train_ids,
    validate_conversation_splits,
    verify_prompt_equivalence,
    verify_schedule,
)
from src.evaluation.e03_a7_classification_metrics import DIMENSIONS
from src.training.e03_a7_classifier import (
    arm_loss,
    build_heads,
    count_parameters,
    forward_prompt_end,
    tensor_state_sha256,
    trainable_parameter_groups,
    validate_arm,
)
from src.training.e03_a7_staged import (
    STAGE_STEPS,
    StageAuthorizationError,
    canonical_sha256,
    capture_rng_state,
    checkpoint_manifest,
    create_start_ticket,
    load_training_state,
    mark_receipt_consumed,
    restore_rng_state,
    save_training_state,
    sha256_file as staged_sha256_file,
    validate_stage_transition,
    verify_checkpoint_manifest,
    verify_continuation_receipt,
    verify_start_ticket,
    write_manifest,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/e03_a7_task_split.yaml"
CODE_PATHS = (
    "scripts/e03_a7_train.py",
    "scripts/e03_a7_evaluate.py",
    "scripts/e03_a7_authorize.py",
    "src/data/e03_a7_classification.py",
    "src/training/e03_a7_classifier.py",
    "src/training/e03_a7_staged.py",
    "src/evaluation/e03_a7_classification_metrics.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        payload = encoded.encode("utf-8")
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "E03-A7-classifier-split-v1" or config.get("experiment_id") != "E03-A7-CLASSIFIER-SPLIT":
        raise ValueError("unsupported E03-A7 experiment config")
    if config.get("authorization", {}).get("execution_enabled") is not False:
        raise ValueError("A7 config cannot directly authorize GPU training")
    if config.get("authorization", {}).get("official_val") != "prohibited":
        raise ValueError("A7 must prohibit Official Val access")
    identity = config.get("identity", {})
    for key in (
        "base_weight_sha256", "tokenizer_signature_sha256", "chat_template_sha256", "p0_sha256",
        "train_schema_sha256", "output_schema_sha256", "d03_manifest_sha256", "normalized_train_sha256",
        "d042_manifest_sha256", "d043_manifest_sha256", "sft_train_sha256", "a5_config_sha256",
        "a5_audit_manifest_sha256", "a5_supervision_sha256", "a5_split_sha256", "frozen_dev_ids_sha256",
        "frozen_dev_input_sha256", "frozen_dev_references_sha256", "frozen_dev_id_map_sha256",
        "excluded_ledger_sha256", "a4_config_sha256", "baseline_config_sha256",
    ):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"A7 identity.{key} must be a pinned lowercase SHA-256")
    training = config.get("training", {})
    if (
        training.get("stage_steps") != [512, 1024, 2048, 4048]
        or training.get("fit_epoch_boundary_step") != 2826
        or training.get("fit_sample_count") != 11304
        or training.get("max_optimizer_steps") != 4048
        or training.get("micro_batch_size") != 1
        or training.get("gradient_accumulation_steps") != 4
        or training.get("lambda_emotion") != 0.08
        or training.get("lambda_status") != "calibrated_before_training_by_long_prompt_gradient_norm"
        or training.get("lambda_target_lora_gradient_norm_ratio") != 1.0
        or training.get("lambda_gradient_norm_ratio_bounds") != [0.9, 1.1]
        or training.get("source_adapter") is not None
        or training.get("response_loss") is not False
        or training.get("emotion_only_arm") is not False
    ):
        raise ValueError("A7 schedule, task separation, or source Adapter policy changed")
    model = config.get("model", {})
    lora = model.get("lora", {})
    if (
        model.get("quantization") != "NF4"
        or model.get("compute_dtype") != "bfloat16"
        or lora.get("r") != 8
        or lora.get("alpha") != 16
        or lora.get("dropout") != 0.0
        or set(lora.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ):
        raise ValueError("A7 frozen model or LoRA settings changed")
    evaluation = config.get("evaluation", {})
    if (
        evaluation.get("calibration_split") != "train_calibration"
        or evaluation.get("calibration_min_micro_precision") != 0.45
        or evaluation.get("frozen_dev_points") != 224
        or evaluation.get("official_val_use") != "prohibited"
    ):
        raise ValueError("A7 calibration or fixed Dev evaluation contract changed")


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("A7 config must be a YAML object")
    _validate_config(config)
    return config, sha256_file(path)


def _validate_pair_id(config: Mapping[str, Any], requested_pair_id: str | None) -> str:
    expected = str(config["training"]["pair_id"])
    if requested_pair_id is not None and requested_pair_id != expected:
        raise ValueError("A7 pair ID must match the immutable config identity")
    return expected


def _git_facts(*, require_clean: bool = True) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    status = run("status", "--porcelain")
    if require_clean and status:
        raise ValueError("A7 preflight and GPU runs require a clean committed Git worktree")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _code_facts() -> dict[str, str]:
    missing = [name for name in CODE_PATHS if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"A7 implementation files are incomplete: {missing}")
    return {name: sha256_file(ROOT / name) for name in CODE_PATHS}


def _code_sha256() -> str:
    return canonical_sha256(_code_facts())


def _source_path_facts(config: Mapping[str, Any]) -> dict[str, str]:
    identity = config["identity"]
    data = config["data"]
    expected = {
        "normalized_train": (data["normalized_train"], identity["normalized_train_sha256"]),
        "sft_train": (data["sft_train"], identity["sft_train_sha256"]),
        "train_schema": (data["train_schema"], identity["train_schema_sha256"]),
        "output_schema": (data["output_schema"], identity["output_schema_sha256"]),
        "prompt": (data["prompt"], identity["p0_sha256"]),
        "a5_config": (data["a5_config"], identity["a5_config_sha256"]),
        "a5_audit_manifest": (f"{data['a5_audit_root']}/manifest.json", identity["a5_audit_manifest_sha256"]),
        "a5_supervision": (data["a5_supervision"], identity["a5_supervision_sha256"]),
        "a5_splits": (data["a5_splits"], identity["a5_split_sha256"]),
        "excluded_ledger": (data["excluded_ledger"], identity["excluded_ledger_sha256"]),
        "a4_config": (data["a4_config"], identity["a4_config_sha256"]),
        "baseline_config": (data["baseline_config"], identity["baseline_config_sha256"]),
        "frozen_dev_input": (data["frozen_dev_input"], identity["frozen_dev_input_sha256"]),
        "frozen_dev_references": (data["frozen_dev_references"], identity["frozen_dev_references_sha256"]),
        "frozen_dev_id_map": (data["frozen_dev_id_map"], identity["frozen_dev_id_map_sha256"]),
    }
    for manifest_key in ("d03_manifest", "d042_manifest", "d043_manifest"):
        expected[manifest_key] = (
            f"outputs/{ {'d03_manifest':'data_pipeline/v1','d042_manifest':'token_audit/v1','d043_manifest':'sft_data/v1'}[manifest_key] }/manifest.json",
            identity[manifest_key.replace("_manifest", "_manifest") + "_sha256"],
        )
    result: dict[str, str] = {}
    for name, (relative, expected_sha) in expected.items():
        path = _resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"A7 frozen source is missing: {name} ({path})")
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ValueError(f"A7 frozen source hash changed: {name}")
        result[name] = actual
    return result


def _a5_assets(config: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    a5_path = _resolve(config["data"]["a5_config"])
    a5_config, a5_config_sha = e03_a5_probe._load_config(a5_path)
    if a5_config_sha != config["identity"]["a5_config_sha256"]:
        raise ValueError("A7 does not bind the reviewed A5 config")
    audit_id = config["identity"]["a5_audit_run_id"]
    audit = e03_a5_probe._verify_audit_run(a5_config, a5_config_sha, audit_id)
    if audit["manifest_sha256"] != config["identity"]["a5_audit_manifest_sha256"]:
        raise ValueError("A5 audit Manifest differs from the A7 frozen source identity")
    for artifact, identity_key in (
        ("supervision.jsonl", "a5_supervision_sha256"),
        ("conversation_splits.json", "a5_split_sha256"),
    ):
        if audit["manifest_value"]["outputs"][artifact]["sha256"] != config["identity"][identity_key]:
            raise ValueError(f"A5 audit does not bind the expected {artifact}")
    return a5_config, a5_config_sha, audit


def _load_source_data(config: Mapping[str, Any], a5_config: Mapping[str, Any]) -> dict[str, Any]:
    source_facts = _source_path_facts(config)
    train_schema = _json(_resolve(config["data"]["train_schema"]))
    output_schema = _json(_resolve(config["data"]["output_schema"]))
    supervision, supervision_summary = build_supervision(
        _resolve(config["data"]["normalized_train"]),
        _resolve(config["data"]["a5_supervision"]),
        train_schema,
        output_schema,
        expected_rows=int(config["data"]["expected_d03_train_rows"]),
    )
    splits = read_splits(_resolve(config["data"]["a5_splits"]))
    split_summary = validate_conversation_splits(splits, supervision)
    expected_counts = {
        "fit": config["data"]["expected_fit_rows"],
        "calibration": config["data"]["expected_calibration_rows"],
        "internal_holdout": config["data"]["expected_internal_holdout_rows"],
    }
    for name, expected_count in expected_counts.items():
        if len(splits[name]) != int(expected_count):
            raise ValueError(f"A5 {name} partition has {len(splits[name])} rows, expected {expected_count}")
    prompt_index = SFTPromptIndex(
        _resolve(config["data"]["sft_train"]),
        list(supervision),
        max_prompt_tokens=int(config["protocol"]["max_prompt_tokens"]),
    )
    if prompt_index.row_count != int(config["data"]["expected_sft_train_rows"]):
        raise ValueError("A7 SFT row count differs from the D0.4.3 contract")
    if supervision_summary["sft_excluded_train_rows"] != 120:
        raise ValueError("A7 D0.4.3 exclusions differ from the reviewed 120-row ledger")
    schedule = build_schedule(
        splits["fit"],
        optimizer_steps=int(config["training"]["max_optimizer_steps"]),
        micro_batch_size=int(config["training"]["micro_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        seed=int(config["training"]["schedule_seed"]),
    )
    schedule_facts = verify_schedule(schedule, splits["fit"])
    if set(schedule["steps"][0]["sample_ids"]) - set(splits["fit"]):
        raise ValueError("A7 schedule contains non-Fit IDs")

    runtime_config = _json(_resolve(a5_config["data"]["runtime_inference_config"]))
    from participant import run_inference as inference

    prompt_text, _output_schema, _output_enums, prompt_facts = inference.load_runtime_protocol(runtime_config)
    if prompt_facts.get("sha256") != config["identity"]["p0_sha256"]:
        raise ValueError("A7 P0 prompt differs from the protected protocol hash")
    tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(
        runtime_config, local_source=_resolve(config["identity"]["base_model_path"])
    )
    if (
        tokenizer_facts.get("signature_sha256") != config["identity"]["tokenizer_signature_sha256"]
        or tokenizer_facts.get("template_sha256") != config["identity"]["chat_template_sha256"]
    ):
        raise ValueError("A7 runtime Tokenizer/Chat Template differs from D0.4.2")
    prompt_equivalence = verify_prompt_equivalence(
        _resolve(config["data"]["normalized_train"]), prompt_index, tokenizer, prompt_text,
        prompt_cap=int(config["protocol"]["max_prompt_tokens"]),
    )

    eval_assets = e03_a5_evaluate._load_eval_assets(a5_config)
    frozen_ids = list(eval_assets["frozen_ids"])
    if len(frozen_ids) != int(config["data"]["expected_dev_rows"]):
        raise ValueError("A7 frozen Dev cohort does not contain exactly 224 IDs")
    for asset_key, identity_key in (
        ("frozen_input", "frozen_dev_input_sha256"),
        ("references", "frozen_dev_references_sha256"),
        ("id_map", "frozen_dev_id_map_sha256"),
    ):
        if eval_assets["source_hashes"][asset_key] != config["identity"][identity_key]:
            raise ValueError(f"A7 frozen Dev {asset_key} hash changed")

    base_identity = inference._weight_identity(
        _resolve(config["identity"]["base_model_path"]),
        {"model_id": config["identity"]["base_model_id"], "model_revision": config["identity"]["base_model_revision"]},
    )
    if base_identity.get("observed_weight_hash") != config["identity"]["base_weight_sha256"]:
        raise ValueError("A7 frozen Base model weights differ from E01-B")

    eval_hashes = {
        "frozen_evaluation_sources": eval_assets["source_hashes"],
        "baseline_outputs": eval_assets["baseline_hashes"],
        "a4_config": eval_assets["a4_config_sha256"],
        "fast_source_config": eval_assets["fast_source_config_sha256"],
        "e01b_full_eval_manifest": eval_assets["e01b_full_eval_manifest_sha256"],
        "e01b_inference_config": eval_assets["e01b_inference_sha256"],
    }
    source_facts["evaluation_assets_sha256"] = canonical_sha256(eval_hashes)
    data_sha256 = canonical_sha256(source_facts)
    return {
        "source_facts": source_facts,
        "data_sha256": data_sha256,
        "supervision": supervision,
        "supervision_summary": supervision_summary,
        "splits": splits,
        "split_summary": split_summary,
        "prompt_index": prompt_index,
        "schedule": schedule,
        "schedule_facts": schedule_facts,
        "prompt_equivalence": prompt_equivalence,
        "prompt_text": prompt_text,
        "tokenizer": tokenizer,
        "tokenizer_facts": tokenizer_facts,
        "eval_assets": eval_assets,
        "base_identity": base_identity,
        "emotion_labels": emotion_label_order(train_schema, output_schema),
        "profile_labels": schema_label_order(train_schema, output_schema)[0],
    }


def _frozen_dev_conversation_count(eval_assets: Mapping[str, Any]) -> int:
    """Validate the A5 eval-ID map and count its frozen conversation clusters."""
    expected_ids = list(eval_assets["frozen_ids"])
    expected = set(expected_ids)
    indexed: dict[str, str] = {}
    for line_number, row, _sha in iter_jsonl(eval_assets["source_paths"]["id_map"]):
        eval_id = row.get("eval_id")
        conversation_id = row.get("conversation_id")
        if not isinstance(eval_id, str) or not eval_id or eval_id in indexed:
            raise ValueError(f"A7 frozen Dev ID map row {line_number} has a missing or duplicate eval_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"A7 frozen Dev ID map row {line_number} has no conversation_id")
        indexed[eval_id] = conversation_id
    if set(indexed) != expected or len(indexed) != len(expected_ids):
        raise ValueError("A7 frozen Dev ID map IDs differ from the fixed 224-point cohort")
    return len(set(indexed.values()))


def _pair_root(config: Mapping[str, Any]) -> Path:
    root = _resolve(config["outputs"]["training_root"]) / "pairs" / str(config["training"]["pair_id"])
    if not root.resolve().is_relative_to(_resolve("outputs")):
        raise ValueError("A7 pair artifacts must remain under ignored outputs/")
    return root


def _write_schedule(path: Path, schedule: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = sha256_file(path)
        expected_rows = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            for row in schedule["steps"]
        ).encode("utf-8")
        if existing != hashlib.sha256(expected_rows).hexdigest():
            raise FileExistsError("A7 immutable Schedule already exists with a different SHA")
        return existing
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in schedule["steps"]:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _preflight(args: argparse.Namespace) -> int:
    config_path = _resolve(args.config)
    config, config_sha = _load_config(config_path)
    _validate_pair_id(config, args.pair_id)
    git = _git_facts()
    code = _code_facts()
    a5_config, a5_config_sha, audit = _a5_assets(config)
    data = _load_source_data(config, a5_config)
    root = _pair_root(config)
    root.mkdir(parents=True, exist_ok=True)
    schedule_path = root / "schedule.jsonl"
    schedule_file_sha = _write_schedule(schedule_path, data["schedule"])
    schedule_manifest = {
        "version": "E03-A7-schedule-manifest-v1",
        "pair_id": config["training"]["pair_id"],
        "config_sha256": config_sha,
        "code_sha256": canonical_sha256(code),
        "data_sha256": data["data_sha256"],
        "a5_audit_manifest_sha256": audit["manifest_sha256"],
        "source_split": "train_fit",
        "schedule_sha256": data["schedule"]["schedule_sha256"],
        "schedule_file_sha256": schedule_file_sha,
        "sample_ids_sha256": data["schedule"]["ordered_draw_ids_sha256"],
        "unique_fit_ids": len(data["splits"]["fit"]),
        "draw_count": data["schedule"]["draw_count"],
        "fit_epoch_boundary_step": data["schedule"]["fit_epoch_boundary_step"],
        "dev_and_val_in_schedule": False,
    }
    schedule_manifest_path = root / "schedule_manifest.json"
    if schedule_manifest_path.exists():
        existing = _json(schedule_manifest_path)
        if existing != schedule_manifest:
            raise FileExistsError("A7 Schedule Manifest differs from the frozen preflight identity")
    else:
        _write_once(schedule_manifest_path, schedule_manifest)
    preflight = {
        "version": "E03-A7-cpu-preflight-v1",
        "status": "G0_passed",
        "experiment_id": config["experiment_id"],
        "pair_id": config["training"]["pair_id"],
        "created_at_utc": _now(),
        "git": git,
        "config_sha256": config_sha,
        "code_sha256": canonical_sha256(code),
        "code_files": code,
        "a5_config_sha256": a5_config_sha,
        "a5_audit_run_id": config["identity"]["a5_audit_run_id"],
        "a5_audit_manifest_sha256": audit["manifest_sha256"],
        "source_hashes": data["source_facts"],
        "data_sha256": data["data_sha256"],
        "supervision": data["supervision_summary"],
        "partitions": data["split_summary"],
        "prompt_contract": data["prompt_equivalence"],
        "tokenizer": data["tokenizer_facts"],
        "model_base_identity": data["base_identity"],
        "schedule": schedule_manifest,
        "frozen_dev": {
            "count": len(data["eval_assets"]["frozen_ids"]),
            "conversation_count": _frozen_dev_conversation_count(data["eval_assets"]),
            "ids_sha256": config["identity"]["frozen_dev_ids_sha256"],
            "gold_read": False,
            "official_val_read": False,
        },
        "max_fit_prompt_tokens": data["prompt_index"].max_prompt_length,
        "max_fit_prompt_sample_id": max(
            data["splits"]["fit"], key=lambda sample_id: (data["prompt_index"].prompt_lengths[sample_id], sample_id)
        ),
        "profile_label_order_sha256": canonical_sha256(data["profile_labels"]),
        "emotion_label_order_sha256": canonical_sha256(data["emotion_labels"]),
        "gpu_training_started": False,
    }
    preflight_path = root / "preflight.json"
    if preflight_path.exists():
        existing = _json(preflight_path)
        for key in ("config_sha256", "code_sha256", "data_sha256", "a5_audit_manifest_sha256"):
            if existing.get(key) != preflight.get(key):
                raise FileExistsError("A7 pair already has a different immutable CPU preflight")
        preflight = existing
    else:
        _write_once(preflight_path, preflight)
    print(json.dumps({"status": preflight["status"], "pair_id": preflight["pair_id"], "schedule_sha256": data["schedule"]["schedule_sha256"], "prompt_equivalence": data["prompt_equivalence"], "output_dir": str(root)}, ensure_ascii=False))
    return 0


def _load_preflight(config: Mapping[str, Any], config_sha: str, *, require_code_match: bool = True) -> tuple[dict[str, Any], Path]:
    root = _pair_root(config)
    path = root / "preflight.json"
    value = _json(path)
    if value.get("status") != "G0_passed" or value.get("config_sha256") != config_sha:
        raise ValueError("A7 CPU preflight is missing or bound to a different config")
    if require_code_match and value.get("code_sha256") != _code_sha256():
        raise ValueError("A7 code changed after CPU preflight; rerun preflight with a new immutable pair ID")
    return value, root


def _load_preflight_for_training(config: Mapping[str, Any], config_sha: str) -> tuple[dict[str, Any], Path]:
    """Accept only the original code or the narrowly recorded reload-only repair."""
    preflight, root = _load_preflight(config, config_sha, require_code_match=False)
    current_code_sha = _code_sha256()
    if preflight.get("code_sha256") == current_code_sha:
        return preflight, root
    receipt_path = root / "reload_compatibility.json"
    if not receipt_path.is_file():
        raise ValueError("A7 code changed after preflight without a verified reload-compatibility receipt")
    receipt = _json(receipt_path)
    _validate_reload_compatibility_receipt(
        preflight, receipt, current_code_sha=current_code_sha, config_sha=config_sha,
    )
    repaired_run = _json(_resolve(config["outputs"]["training_root"]) / receipt["run_id"] / "run_manifest.json")
    repaired_node = repaired_run.get("nodes", {}).get("512", {})
    if (
        receipt.get("pair_id") != config["training"]["pair_id"]
        or receipt.get("arm") not in {"P", "EP"}
        or repaired_run.get("run_id") != receipt.get("run_id")
        or repaired_run.get("arm") != receipt.get("arm")
        or repaired_run.get("status") not in {"awaiting_evaluation", "awaiting_manual_review"}
        or repaired_run.get("optimizer_step") != 512
        or repaired_node.get("training_status") != "passed"
        or repaired_node.get("reload_check_sha256") != receipt.get("reload_check_sha256")
    ):
        raise ValueError("A7 reload-compatibility receipt is not backed by a finalized Step 512 Run")
    return preflight, root


def _validate_reload_compatibility_receipt(
    preflight: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    current_code_sha: str,
    config_sha: str,
) -> None:
    if (
        receipt.get("status") != "reload_compatibility_repaired"
        or receipt.get("scope") != "inference_base_dtype_reload_only"
        or receipt.get("training_code_sha256") != preflight.get("code_sha256")
        or receipt.get("compatible_code_sha256") != current_code_sha
        or receipt.get("config_sha256") != config_sha
        or receipt.get("optimizer_steps_added") != 0
    ):
        raise ValueError("A7 reload-compatibility receipt does not authorize the current code identity")


def _load_schedule(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    manifest = _json(root / "schedule_manifest.json")
    schedule_path = root / "schedule.jsonl"
    if sha256_file(schedule_path) != manifest.get("schedule_file_sha256"):
        raise ValueError("A7 frozen Schedule file SHA differs from its Manifest")
    steps = [row for _line, row, _sha in iter_jsonl(schedule_path)]
    schedule = {"steps": steps, "schedule_sha256": manifest.get("schedule_sha256"), "fit_epoch_boundary_step": manifest.get("fit_epoch_boundary_step")}
    if canonical_sha256(steps) != schedule["schedule_sha256"]:
        raise ValueError("A7 frozen Schedule content SHA differs from its Manifest")
    return schedule


def _gpu_facts(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("blocked_gpu_unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_bf16_unavailable")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if props.total_memory < 8 * 1024**3 - 16 * 1024**2:
        raise RuntimeError(f"blocked_total_vram_below_8_gib:{props.total_memory}")
    if free_bytes < 6 * 1024**3:
        raise RuntimeError(f"blocked_free_vram_below_6_gib:{free_bytes}")
    return {
        "device_index": int(device),
        "device_name": props.name,
        "total_vram_bytes": int(total_bytes),
        "free_vram_bytes_before_load": int(free_bytes),
        "bf16_supported": True,
        "cuda_version": torch.version.cuda,
    }


def _seed_all(torch: Any, seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_base(config: Mapping[str, Any], torch: Any, *, trainable: bool) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from participant import run_inference as inference

    base_path = _resolve(config["identity"]["base_model_path"])
    identity = inference._weight_identity(base_path, {
        "model_id": config["identity"]["base_model_id"],
        "model_revision": config["identity"]["base_model_revision"],
    })
    if identity.get("observed_weight_hash") != config["identity"]["base_weight_sha256"]:
        raise ValueError("A7 Base weight SHA differs from the frozen E01-B Base")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        device_map={"": int(torch.cuda.current_device())},
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    from peft import prepare_model_for_kbit_training

    if trainable:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        # Keep frozen LayerNorms in the same FP32 compute dtype used during
        # training so Adapter+Head reload logits are directly comparable.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return model


def _new_model_and_heads(config: Mapping[str, Any], arm: str, torch: Any, *, trainable: bool) -> tuple[Any, Any, Any, Any]:
    from peft import LoraConfig, get_peft_model

    arm = validate_arm(arm)
    device = torch.device("cuda", torch.cuda.current_device())
    base = _load_base(config, torch, trainable=trainable)
    lora = config["model"]["lora"]
    torch.manual_seed(int(lora["init_seed"]))
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, peft_config)
    model.config.use_cache = False
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
    hidden_size = int(model.config.hidden_size)
    heads = build_heads(
        hidden_size,
        arm,
        profile_seed=int(config["model"]["heads"]["profile_seed"]),
        emotion_seed=int(config["model"]["heads"]["emotion_seed"]),
    ).to(device)
    return model, heads, device, hidden_size


def _peft_state(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    return dict(get_peft_model_state_dict(model, adapter_name="default"))


def _initial_state_identity(model: Any, heads: Any, arm: str) -> dict[str, Any]:
    lora_state = _peft_state(model)
    profile_state = heads.profile_head.state_dict()
    emotion_state = heads.emotion_head.state_dict() if arm == "EP" else None
    return {
        "lora_sha256": tensor_state_sha256(lora_state),
        "profile_head_sha256": tensor_state_sha256(profile_state),
        "emotion_head_sha256": tensor_state_sha256(emotion_state) if emotion_state is not None else None,
        "initial_sha256": canonical_sha256({
            "lora": tensor_state_sha256(lora_state),
            "profile_head": tensor_state_sha256(profile_state),
            "emotion_head": tensor_state_sha256(emotion_state) if emotion_state is not None else None,
        }),
        "lora_dtypes": sorted({str(value.dtype).removeprefix("torch.") for value in lora_state.values()}),
        "profile_head_dtype": str(next(iter(profile_state.values())).dtype).removeprefix("torch."),
        "emotion_head_dtype": str(next(iter(emotion_state.values())).dtype).removeprefix("torch.") if emotion_state else None,
    }


def _gradient_summary(parameters: Sequence[tuple[str, Any]], description: str) -> dict[str, Any]:
    import torch

    observed = nonzero = missing = 0
    squared_norm = 0.0
    for _name, parameter in parameters:
        observed += int(parameter.requires_grad)
        gradient = parameter.grad
        if gradient is None:
            missing += 1
            continue
        if not torch.isfinite(gradient).all():
            raise ValueError(f"{description} gradient is non-finite")
        norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        squared_norm += norm * norm
        nonzero += norm > 0
    norm = squared_norm**0.5
    if observed == 0 or nonzero == 0 or not norm > 0:
        raise ValueError(f"{description} has no finite nonzero gradient")
    return {"parameter_count": observed, "nonzero_gradient_parameter_count": int(nonzero), "missing_gradient_parameter_count": missing, "gradient_norm": norm}


def _zero_grad(model: Any, heads: Any) -> None:
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)


def _forward_loss(
    model: Any,
    heads: Any,
    arm: str,
    torch: Any,
    input_ids: Any,
    attention_mask: Any,
    prompt_length: int,
    target: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    hidden = forward_prompt_end(model, input_ids, attention_mask, torch.tensor([prompt_length], device=input_ids.device))
    output = heads(hidden)
    return arm_loss(
        output,
        torch.tensor([target["profile_target"]], dtype=torch.float32, device=input_ids.device),
        torch.tensor([target["profile_mask"]], dtype=torch.float32, device=input_ids.device),
        torch.tensor([target["emotion_target"]], dtype=torch.long, device=input_ids.device) if arm == "EP" else None,
        arm,
        lambda_emotion=float(config["training"]["lambda_emotion"]),
    )


def _gpu_preflight(args: argparse.Namespace) -> int:
    import torch

    config, config_sha = _load_config(_resolve(args.config))
    _validate_pair_id(config, args.pair_id)
    arm = validate_arm(args.arm)
    preflight, root = _load_preflight(config, config_sha)
    git = _git_facts()
    gpu = _gpu_facts(torch)
    _a5, _a5_sha, _audit = _a5_assets(config)
    splits = read_splits(_resolve(config["data"]["a5_splits"]))
    prompt_train_ids = sft_prompt_train_ids(splits)
    index = SFTPromptIndex(_resolve(config["data"]["sft_train"]), prompt_train_ids)
    sample_id = max(splits["fit"], key=lambda value: (index.prompt_lengths[value], value))
    prompt = index.prompt_ids(sample_id)
    prompt_length = len(prompt)
    if prompt_length > int(config["protocol"]["max_prompt_tokens"]):
        raise ValueError("A7 worst real Prompt exceeds the frozen P=1280 budget")
    supervision, _summary = build_supervision(
        _resolve(config["data"]["normalized_train"]),
        _resolve(config["data"]["a5_supervision"]),
        _json(_resolve(config["data"]["train_schema"])),
        _json(_resolve(config["data"]["output_schema"])),
        expected_rows=int(config["data"]["expected_d03_train_rows"]),
    )
    validate_conversation_splits(splits, supervision)
    if set(prompt_train_ids) != set(supervision):
        raise ValueError("A7 GPU preflight Prompt IDs differ from the complete supervised Train set")
    torch.cuda.reset_peak_memory_stats()
    seed = int(config["model"]["lora"]["init_seed"])
    _seed_all(torch, seed)
    model, heads, device, hidden_size = _new_model_and_heads(config, arm, torch, trainable=True)
    groups = trainable_parameter_groups(model, heads, arm)
    params = [parameter for values in groups.values() for _name, parameter in values]
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name):
        raise ValueError("A7 Base/LM Head must remain frozen")
    init_before = _initial_state_identity(model, heads, arm)
    padded_length = int(config["protocol"]["max_prompt_tokens"])
    input_ids = torch.full((1, padded_length), int(config["identity"]["pad_token_id"]), dtype=torch.long, device=device)
    input_ids[0, :prompt_length] = torch.tensor(prompt, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[0, :prompt_length] = 1
    target = supervision[sample_id]

    _zero_grad(model, heads)
    losses = _forward_loss(model, heads, arm, torch, input_ids, attention_mask, prompt_length, target, config)
    losses["total"].backward()
    combined_grad = _gradient_summary(groups["lora"], "A7 LoRA")
    profile_grad = _gradient_summary(groups["profile_head"], "A7 ProfileHead")
    emotion_grad = _gradient_summary(groups["emotion_head"], "A7 EmotionHead") if arm == "EP" else None
    profile_only_lora_norm = combined_grad["gradient_norm"]
    emotion_only_lora_norm = None
    lambda_effective_ratio = None
    if arm == "EP":
        _zero_grad(model, heads)
        profile_loss = _forward_loss(model, heads, arm, torch, input_ids, attention_mask, prompt_length, target, config)["profile_bce"]
        profile_loss.backward()
        profile_only_lora_norm = _gradient_summary(groups["lora"], "A7 Profile-only LoRA")["gradient_norm"]
        _zero_grad(model, heads)
        emotion_loss = _forward_loss(model, heads, arm, torch, input_ids, attention_mask, prompt_length, target, config)["emotion_ce"]
        emotion_loss.backward()
        emotion_only_lora_norm = _gradient_summary(groups["lora"], "A7 Emotion-only LoRA")["gradient_norm"]
        lambda_effective_ratio = float(config["training"]["lambda_emotion"]) * emotion_only_lora_norm / max(profile_only_lora_norm, 1e-30)
    _zero_grad(model, heads)
    init_after = _initial_state_identity(model, heads, arm)
    if init_before != init_after:
        raise ValueError("A7 GPU dry-step changed initial LoRA or head weights without an optimizer step")
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise ValueError("A7 frozen Base/LM parameters received gradients")
    facts = {
        "version": "E03-A7-gpu-preflight-v1",
        "status": "G1_passed",
        "pair_id": config["training"]["pair_id"],
        "arm": arm,
        "run_id": f"{config['training']['pair_id']}-{arm}",
        "created_at_utc": _now(),
        "git": git,
        "config_sha256": config_sha,
        "code_sha256": preflight["code_sha256"],
        "data_sha256": preflight["data_sha256"],
        "schedule_sha256": preflight["schedule"]["schedule_sha256"],
        "base_weight_sha256": config["identity"]["base_weight_sha256"],
        "dry_prompt": {"sample_id": sample_id, "real_prompt_tokens": prompt_length, "padded_sequence_length": padded_length, "model_input_contains_completion": False},
        "hidden_size": hidden_size,
        "heads": {"profile_logits_shape": [1, 33], "emotion_logits_shape": [1, 16] if arm == "EP" else None},
        "initial_identity": init_before,
        "parameter_groups": count_parameters(groups),
        "losses": {key: float(value.detach().float().item()) if value is not None else None for key, value in losses.items()},
        "gradients": {"lora_combined": combined_grad, "profile_head": profile_grad, "emotion_head": emotion_grad},
        "lambda_dry_run": {
            "fixed_value": float(config["training"]["lambda_emotion"]),
            "profile_only_lora_gradient_norm": profile_only_lora_norm,
            "emotion_only_lora_gradient_norm": emotion_only_lora_norm,
            "weighted_emotion_to_profile_lora_gradient_norm_ratio": lambda_effective_ratio,
        },
        "cuda": {
            **gpu,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "bf16_lora_or_adapter_dtypes": init_before["lora_dtypes"],
            "head_dtypes": {"profile": init_before["profile_head_dtype"], "emotion": init_before["emotion_head_dtype"]},
        },
        "optimizer_step_called": False,
        "base_and_initial_weights_unchanged": True,
    }
    output = root / "gpu_preflight" / f"{arm}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"A7 {arm} GPU preflight is immutable and already exists: {output}")
    _write_once(output, facts)
    del model, heads, groups, params
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": facts["status"], "arm": arm, "init_sha256": init_before["initial_sha256"], "peak_allocated_bytes": facts["cuda"]["peak_allocated_bytes"], "manifest": str(output)}, ensure_ascii=False))
    return 0


def _freeze_preflights(args: argparse.Namespace) -> int:
    config, config_sha = _load_config(_resolve(args.config))
    _validate_pair_id(config, args.pair_id)
    preflight, root = _load_preflight(config, config_sha)
    reports = {arm: _json(root / "gpu_preflight" / f"{arm}.json") for arm in ("P", "EP")}
    for arm, report in reports.items():
        if (
            report.get("status") != "G1_passed"
            or report.get("arm") != arm
            or report.get("config_sha256") != config_sha
            or report.get("code_sha256") != preflight["code_sha256"]
            or report.get("data_sha256") != preflight["data_sha256"]
            or report.get("schedule_sha256") != preflight["schedule"]["schedule_sha256"]
            or report.get("base_weight_sha256") != config["identity"]["base_weight_sha256"]
        ):
            raise ValueError(f"A7 {arm} GPU preflight is incomplete or bound to different frozen sources")
    if reports["P"]["initial_identity"]["lora_sha256"] != reports["EP"]["initial_identity"]["lora_sha256"]:
        raise ValueError("A7 P and EP initial LoRA tensors differ")
    if reports["P"]["initial_identity"]["profile_head_sha256"] != reports["EP"]["initial_identity"]["profile_head_sha256"]:
        raise ValueError("A7 P and EP initial ProfileHead tensors differ")
    lambda_value = float(config["training"]["lambda_emotion"])
    effective_ratio = reports["EP"]["lambda_dry_run"]["weighted_emotion_to_profile_lora_gradient_norm_ratio"]
    if effective_ratio is None or not math.isfinite(effective_ratio) or not 0 < effective_ratio:
        raise ValueError(f"A7 fixed lambda has no finite positive measured gradient ratio: {effective_ratio}")
    ratio_bounds = [float(value) for value in config["training"]["lambda_gradient_norm_ratio_bounds"]]
    if not ratio_bounds[0] <= effective_ratio <= ratio_bounds[1]:
        raise ValueError(
            f"A7 lambda gradient ratio {effective_ratio:.6f} is outside frozen bounds {ratio_bounds}; "
            "do not start training until the loss weight is recalibrated"
        )
    freeze = {
        "version": "E03-A7-gpu-preflight-freeze-v1",
        "status": "G1_passed_lambda_frozen",
        "pair_id": config["training"]["pair_id"],
        "config_sha256": config_sha,
        "code_sha256": preflight["code_sha256"],
        "data_sha256": preflight["data_sha256"],
        "schedule_sha256": preflight["schedule"]["schedule_sha256"],
        "p_gpu_preflight_manifest_sha256": sha256_file(root / "gpu_preflight/P.json"),
        "ep_gpu_preflight_manifest_sha256": sha256_file(root / "gpu_preflight/EP.json"),
        "shared_initial_lora_sha256": reports["P"]["initial_identity"]["lora_sha256"],
        "shared_initial_profile_head_sha256": reports["P"]["initial_identity"]["profile_head_sha256"],
        "p_initial_identity_sha256": reports["P"]["initial_identity"]["initial_sha256"],
        "ep_initial_identity_sha256": reports["EP"]["initial_identity"]["initial_sha256"],
        "lambda_emotion": lambda_value,
        "lambda_status": str(config["training"]["lambda_status"]),
        "effective_emotion_to_profile_lora_gradient_norm_ratio": effective_ratio,
        "target_emotion_to_profile_lora_gradient_norm_ratio": float(config["training"]["lambda_target_lora_gradient_norm_ratio"]),
        "accepted_emotion_to_profile_lora_gradient_norm_ratio": ratio_bounds,
        "lambda_review_facts": {
            "profile_bce": reports["EP"]["losses"]["profile_bce"],
            "emotion_cross_entropy": reports["EP"]["losses"]["emotion_ce"],
            "weighted_emotion_to_profile_loss_ratio": (
                lambda_value * reports["EP"]["losses"]["emotion_ce"]
                / max(reports["EP"]["losses"]["profile_bce"], 1e-30)
            ),
            "profile_only_lora_gradient_norm": reports["EP"]["lambda_dry_run"]["profile_only_lora_gradient_norm"],
            "emotion_only_lora_gradient_norm": reports["EP"]["lambda_dry_run"]["emotion_only_lora_gradient_norm"],
            "weighted_emotion_to_profile_lora_gradient_norm_ratio": effective_ratio,
            "review_basis": "record component loss scale and separately measured LoRA gradient norms before first optimizer update",
        },
        "both_arms_preflight_passed_before_start_authorization": True,
        "created_at_utc": _now(),
    }
    freeze_path = root / "lambda_freeze.json"
    if freeze_path.exists():
        raise FileExistsError("A7 lambda and initial-stage tickets have already been frozen")
    write_manifest(freeze_path, freeze)
    for arm in ("P", "EP"):
        report = reports[arm]
        run_id = f"{config['training']['pair_id']}-{arm}"
        create_start_ticket(
            root / "start_tickets" / f"{arm}.json",
            run_id=run_id,
            arm=arm,
            config_sha256=config_sha,
            code_sha256=preflight["code_sha256"],
            data_sha256=preflight["data_sha256"],
            init_sha256=report["initial_identity"]["initial_sha256"],
            dry_run_manifest_sha256=sha256_file(root / "gpu_preflight" / f"{arm}.json"),
        )
    print(json.dumps({"status": freeze["status"], "lambda_emotion": lambda_value, "weighted_gradient_ratio": effective_ratio, "start_tickets": [str(root / "start_tickets/P.json"), str(root / "start_tickets/EP.json")]}, ensure_ascii=False))
    return 0


def _read_schedule_for_step(schedule: Mapping[str, Any], step: int) -> dict[str, Any]:
    row = schedule["steps"][step - 1]
    if row["optimizer_step"] != step:
        raise ValueError("A7 Schedule row index does not equal the optimizer step")
    return row


def _export_inference_artifacts(model: Any, heads: Any, arm: str, checkpoint_dir: Path) -> dict[str, str]:
    from safetensors.torch import save_file

    adapter_dir = checkpoint_dir / "inference_export" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    head_dir = checkpoint_dir / "inference_export"
    profile_path = head_dir / "profile_head.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in heads.profile_head.state_dict().items()}, str(profile_path))
    result = {
        "adapter_model.safetensors": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "adapter_config.json": sha256_file(adapter_dir / "adapter_config.json"),
        "profile_head.safetensors": sha256_file(profile_path),
    }
    if arm == "EP":
        emotion_path = head_dir / "emotion_head.safetensors"
        save_file({name: value.detach().cpu().contiguous() for name, value in heads.emotion_head.state_dict().items()}, str(emotion_path))
        result["emotion_head.safetensors"] = sha256_file(emotion_path)
    return result


def _expected_reload_logits(model: Any, heads: Any, arm: str, torch: Any, sample_rows: Sequence[tuple[str, list[int]]], pad_token_id: int) -> list[dict[str, Any]]:
    from src.training.e03_a7_classifier import forward_prompt_end

    values = []
    for sample_id, prompt in sample_rows:
        input_ids = torch.tensor([prompt], dtype=torch.long, device=torch.device("cuda", torch.cuda.current_device()))
        mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            hidden = forward_prompt_end(model, input_ids, mask, torch.tensor([len(prompt)], device=input_ids.device))
            output = heads(hidden)
        values.append({
            "sample_id": sample_id,
            "profile_logits": output["profile_logits"].detach().float().cpu(),
            "emotion_logits": output["emotion_logits"].detach().float().cpu() if arm == "EP" else None,
        })
    return values


def _verify_export_reload(config: Mapping[str, Any], arm: str, torch: Any, checkpoint_dir: Path, sample_rows: Sequence[tuple[str, list[int]]], expected_logits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from peft import PeftModel
    from safetensors.torch import load_file

    base = _load_base(config, torch, trainable=False)
    adapter_dir = checkpoint_dir / "inference_export/adapter"
    model = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        is_trainable=False,
        local_files_only=True,
        autocast_adapter_dtype=False,
    )
    model.eval()
    head_module = build_heads(
        int(model.config.hidden_size), arm,
        profile_seed=int(config["model"]["heads"]["profile_seed"]),
        emotion_seed=int(config["model"]["heads"]["emotion_seed"]),
    ).to(torch.device("cuda", torch.cuda.current_device()))
    head_root = checkpoint_dir / "inference_export"
    profile_state = load_file(str(head_root / "profile_head.safetensors"), device="cuda:0")
    head_module.profile_head.load_state_dict(profile_state, strict=True)
    if arm == "EP":
        emotion_state = load_file(str(head_root / "emotion_head.safetensors"), device="cuda:0")
        head_module.emotion_head.load_state_dict(emotion_state, strict=True)
    reloaded = _expected_reload_logits(model, head_module, arm, torch, sample_rows, int(config["identity"]["pad_token_id"]))
    comparisons = []
    for expected, actual in zip(expected_logits, reloaded):
        if expected["sample_id"] != actual["sample_id"]:
            raise ValueError("A7 independent Adapter reload changed its fixed sample order")
        profile_match = torch.allclose(expected["profile_logits"], actual["profile_logits"], atol=1e-4, rtol=1e-4)
        emotion_match = arm == "P" or torch.allclose(expected["emotion_logits"], actual["emotion_logits"], atol=1e-4, rtol=1e-4)
        if not profile_match or not emotion_match:
            raise ValueError(f"A7 independent Adapter+Head reload logits differ for {expected['sample_id']}")
        comparisons.append({"sample_id": expected["sample_id"], "profile_logits_match": True, "emotion_logits_match": True if arm == "EP" else "not_applicable"})
    return {
        "status": "independent_adapter_head_reload_passed",
        "sample_count": len(comparisons),
        "all_profile_logits_match": True,
        "emotion_logits_match": True if arm == "EP" else "not_applicable",
        "comparisons": comparisons,
    }


def _write_checkpoint(
    *,
    config: Mapping[str, Any],
    run_root: Path,
    arm: str,
    run_id: str,
    step: int,
    schedule: Mapping[str, Any],
    consumed_ids: Sequence[str],
    model: Any,
    heads: Any,
    optimizer: Any,
    scheduler: Any,
    identity: Mapping[str, Any],
    torch: Any,
    prompt_index: SFTPromptIndex,
) -> dict[str, Any]:
    checkpoint_dir = run_root / "trainer" / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model_state = _peft_state(model)
    head_state = {key: value.detach().cpu() for key, value in heads.state_dict().items()}
    state_path = checkpoint_dir / "training_state.pt"
    saved = save_training_state(
        state_path,
        identity=identity,
        arm=arm,
        run_id=run_id,
        optimizer_step=step,
        schedule_sha256=str(schedule["schedule_sha256"]),
        consumed_ids=consumed_ids,
        model_state=model_state,
        head_state=head_state,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        rng_state=capture_rng_state(),
    )
    export_hashes = _export_inference_artifacts(model, heads, arm, checkpoint_dir)
    identities = _initial_state_identity(model, heads, arm)
    checkpoint_value = checkpoint_manifest(
        state_path,
        arm=arm,
        run_id=run_id,
        optimizer_step=step,
        identity=identity,
        schedule_sha256=str(schedule["schedule_sha256"]),
        consumed_ids_sha256=str(saved["consumed_ids_sha256"]),
        sample_count=len(consumed_ids),
        adapter_sha256=export_hashes["adapter_model.safetensors"],
        profile_head_sha256=export_hashes["profile_head.safetensors"],
        emotion_head_sha256=export_hashes.get("emotion_head.safetensors"),
    )
    checkpoint_value["files"] = export_hashes
    checkpoint_value["initial_identity_after_training"] = identities
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    write_manifest(manifest_path, checkpoint_value)
    return {"checkpoint_dir": checkpoint_dir, "checkpoint_manifest": checkpoint_value, "checkpoint_manifest_path": manifest_path}


def _optimizer_parameters(model: Any, heads: Any, arm: str) -> tuple[dict[str, list[tuple[str, Any]]], list[Any]]:
    groups = trainable_parameter_groups(model, heads, arm)
    params = [parameter for group in ("lora", "profile_head", "emotion_head") for _name, parameter in groups[group]]
    return groups, params


def _train_segment(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    arm: str,
    run_id: str,
    from_step: int,
    stop_at_step: int,
    run_root: Path,
    schedule: Mapping[str, Any],
    supervision: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, Sequence[str]],
    prompt_index: SFTPromptIndex,
    initial_identity: Mapping[str, Any],
    data_sha256: str,
    code_sha256: str,
    start_ticket_path: Path | None = None,
    approval_file: Path | None = None,
) -> dict[str, Any]:
    import torch
    from peft import set_peft_model_state_dict

    validate_arm(arm)
    validate_stage_transition(from_step if from_step else None, stop_at_step) if from_step == 0 else validate_stage_transition(from_step, stop_at_step)
    run_manifest_path = run_root / "run_manifest.json"
    if from_step == 0:
        if run_root.exists():
            raise FileExistsError(f"A7 Run already exists and cannot be restarted: {run_root}")
        run_root.mkdir(parents=True, exist_ok=False)
        run_manifest = {
            "version": "E03-A7-run-manifest-v1",
            "status": "training",
            "run_id": run_id,
            "arm": arm,
            "pair_id": config["training"]["pair_id"],
            "config_sha256": config_sha,
            "code_sha256": code_sha256,
            "data_sha256": data_sha256,
            "schedule_sha256": schedule["schedule_sha256"],
            "initial_identity": dict(initial_identity),
            "stages": [],
            "nodes": {},
            "created_at_utc": _now(),
            "authorization": {"first_stage_ticket": str(start_ticket_path), "ticket_sha256": sha256_file(start_ticket_path)},
            "training_contract": {
                "source_split": "train_fit",
                "profile_loss": "unweighted_closed_world_bce",
                "emotion_loss": "cross_entropy" if arm == "EP" else "not_applicable",
                "lambda_emotion": float(config["training"]["lambda_emotion"]) if arm == "EP" else 0.0,
                "response_loss": False,
                "gold_completion_model_input": False,
                "optimizer": "AdamW",
                "learning_rate": float(config["training"]["learning_rate"]),
                "scheduler": "constant",
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
            },
        }
        ticket_value = _json(start_ticket_path)
        ticket_value.update(status="consumed", consumed_at_utc=_now())
        write_manifest(
            _resolve(config["outputs"]["training_root"]) / "pairs" / config["training"]["pair_id"] / "start_tickets" / f"{arm}.json",
            ticket_value,
            overwrite=True,
        )
        _write_text_once(run_root / "config_snapshot.yaml", yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False))
        _write_once(run_root / "initial_identity.json", dict(initial_identity))
    else:
        run_manifest = _json(run_manifest_path)
        if (
            run_manifest.get("run_id") != run_id
            or run_manifest.get("arm") != arm
            or run_manifest.get("status") != "awaiting_manual_review"
            or run_manifest.get("optimizer_step") != from_step
            or run_manifest.get("config_sha256") != config_sha
            or run_manifest.get("code_sha256") != code_sha256
            or run_manifest.get("data_sha256") != data_sha256
            or run_manifest.get("schedule_sha256") != schedule["schedule_sha256"]
        ):
            raise StageAuthorizationError("A7 Run is not awaiting the requested same-Run continuation")

    _seed_all(torch, int(config["training"]["schedule_seed"]))
    model, heads, device, hidden_size = _new_model_and_heads(config, arm, torch, trainable=True)
    groups, params = _optimizer_parameters(model, heads, arm)
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    current_identity = _initial_state_identity(model, heads, arm) if from_step == 0 else dict(initial_identity)
    if from_step == 0 and current_identity != dict(initial_identity):
        raise ValueError("A7 initialized model differs from the validated GPU dry-run identity")
    consumed_ids = [sample_id for row in schedule["steps"][:from_step] for sample_id in row["sample_ids"]]
    if from_step:
        checkpoint_dir = run_root / "trainer" / f"checkpoint-{from_step}"
        checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
        checkpoint = verify_checkpoint_manifest(
            checkpoint_manifest_path,
            expected_run_id=run_id,
            expected_arm=arm,
            expected_step=from_step,
            expected_identity=run_manifest["checkpoint_identity"],
            expected_schedule_sha256=str(schedule["schedule_sha256"]),
        )
        training_state = load_training_state(
            checkpoint_dir / "training_state.pt",
            expected_identity=run_manifest["checkpoint_identity"],
            expected_arm=arm,
            expected_run_id=run_id,
            expected_step=from_step,
            expected_schedule_sha256=str(schedule["schedule_sha256"]),
            expected_consumed_ids=consumed_ids,
        )
        set_peft_model_state_dict(model, training_state["model_state"], adapter_name="default")
        heads.load_state_dict(training_state["head_state"], strict=True)
        optimizer.load_state_dict(training_state["optimizer_state"])
        scheduler.load_state_dict(training_state["scheduler_state"])
        restore_rng_state(training_state["rng_state"])
        if approval_file is None:
            raise StageAuthorizationError("A7 resume requires the user continuation receipt")
        mark_receipt_consumed(approval_file, run_id=run_id, arm=arm, from_step=from_step)
        run_manifest.setdefault("resume_history", []).append({
            "from_step": from_step,
            "to_step": stop_at_step,
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "approval_file": str(approval_file),
            "approval_sha256": sha256_file(approval_file),
            "started_at_utc": _now(),
        })
    else:
        run_manifest["checkpoint_identity"] = {
            "config_sha256": config_sha,
            "code_sha256": code_sha256,
            "data_sha256": data_sha256,
            "base_weight_sha256": config["identity"]["base_weight_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "initial_identity": dict(current_identity),
            "lambda_emotion": float(config["training"]["lambda_emotion"]),
        }
    write_manifest(run_manifest_path, run_manifest, overwrite=True)

    train_steps_path = run_root / "train_steps.jsonl"
    existing_steps = [row for _line, row, _sha in iter_jsonl(train_steps_path)] if train_steps_path.exists() else []
    if [row.get("optimizer_step") for row in existing_steps] != list(range(1, from_step + 1)):
        raise ValueError("A7 train step ledger does not match the resume checkpoint prefix")
    previous_consumed = [sample_id for row in existing_steps for sample_id in row.get("sample_ids", [])]
    if from_step and previous_consumed != consumed_ids:
        raise ValueError("A7 train step ledger consumed-ID sequence differs from the Schedule")
    sample_order_path = run_root / "sample_order.jsonl"
    existing_sample_order = [row for _line, row, _sha in iter_jsonl(sample_order_path)] if sample_order_path.exists() else []
    if [row.get("optimizer_step") for row in existing_sample_order] != list(range(1, from_step + 1)):
        raise ValueError("A7 sample-order ledger does not match the resume checkpoint prefix")
    if from_step and [row.get("sample_ids") for row in existing_sample_order] != [row["sample_ids"] for row in schedule["steps"][:from_step]]:
        raise ValueError("A7 sample-order ledger differs from the frozen Schedule prefix")

    _zero_grad(model, heads)
    start_time = time.monotonic()
    checkpoint_records: list[dict[str, Any]] = []
    for optimizer_step in range(from_step + 1, stop_at_step + 1):
        schedule_row = _read_schedule_for_step(schedule, optimizer_step)
        step_ids = schedule_row["sample_ids"]
        step_losses: dict[str, list[float]] = {"profile_bce": [], "emotion_ce": []}
        for sample_id in step_ids:
            prompt = prompt_index.prompt_ids(sample_id)
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            losses = _forward_loss(model, heads, arm, torch, input_ids, attention_mask, len(prompt), supervision[sample_id], config)
            (losses["total"] / 4.0).backward()
            step_losses["profile_bce"].append(float(losses["profile_bce"].detach().float().item()))
            if losses["emotion_ce"] is not None:
                step_losses["emotion_ce"].append(float(losses["emotion_ce"].detach().float().item()))
        grad_summary = _gradient_summary([(name, p) for values in groups.values() for name, p in values], "A7 trainable parameters")
        torch.nn.utils.clip_grad_norm_(params, float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        _zero_grad(model, heads)
        consumed_ids.extend(step_ids)
        ledger_row = {
            "optimizer_step": optimizer_step,
            "epoch": schedule_row["epoch"],
            "sample_ids": list(step_ids),
            "profile_bce_mean": sum(step_losses["profile_bce"]) / 4,
            "emotion_ce_mean": sum(step_losses["emotion_ce"]) / 4 if step_losses["emotion_ce"] else None,
            "total_loss_mean": sum(step_losses["profile_bce"]) / 4 + float(config["training"]["lambda_emotion"]) * (sum(step_losses["emotion_ce"]) / 4 if step_losses["emotion_ce"] else 0.0),
            "pre_clip_gradient_norm": grad_summary["gradient_norm"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        _append_jsonl(train_steps_path, ledger_row)
        _append_jsonl(run_root / "sample_order.jsonl", {"optimizer_step": optimizer_step, "sample_ids": list(step_ids), "epoch": schedule_row["epoch"]})

        if optimizer_step in {*STAGE_STEPS, 2826} and (optimizer_step == stop_at_step or optimizer_step == 2826):
            record = _write_checkpoint(
                config=config,
                run_root=run_root,
                arm=arm,
                run_id=run_id,
                step=optimizer_step,
                schedule=schedule,
                consumed_ids=consumed_ids,
                model=model,
                heads=heads,
                optimizer=optimizer,
                scheduler=scheduler,
                identity=run_manifest["checkpoint_identity"],
                torch=torch,
                prompt_index=prompt_index,
            )
            checkpoint_records.append({
                "step": optimizer_step,
                "checkpoint_manifest_path": str(record["checkpoint_manifest_path"]),
                "checkpoint_manifest_sha256": sha256_file(record["checkpoint_manifest_path"]),
                "training_state_sha256": record["checkpoint_manifest"]["training_state_sha256"],
                "sample_count": optimizer_step * 4,
            })
            if optimizer_step == 2826 and optimizer_step != stop_at_step:
                continue

    if len(consumed_ids) != stop_at_step * 4:
        raise ValueError("A7 optimizer-step sample ledger count differs from 4x accumulation")
    stage_checkpoint = run_root / "trainer" / f"checkpoint-{stop_at_step}"
    stage_manifest_path = stage_checkpoint / "checkpoint_manifest.json"
    stage_manifest = verify_checkpoint_manifest(
        stage_manifest_path,
        expected_run_id=run_id,
        expected_arm=arm,
        expected_step=stop_at_step,
        expected_identity=run_manifest["checkpoint_identity"],
        expected_schedule_sha256=str(schedule["schedule_sha256"]),
    )
    # Evaluate a fixed pair of training prompts before and after an independent Adapter+Head reload.
    reload_ids = list(dict.fromkeys(sample_id for row in schedule["steps"][:1] for sample_id in row["sample_ids"]))[:2]
    reload_rows = [(sample_id, prompt_index.prompt_ids(sample_id)) for sample_id in reload_ids]
    model.eval()
    heads.eval()
    expected_logits = _expected_reload_logits(model, heads, arm, torch, reload_rows, int(config["identity"]["pad_token_id"]))
    del optimizer, scheduler, model, heads, groups, params
    gc.collect()
    torch.cuda.empty_cache()
    reload_check = _verify_export_reload(config, arm, torch, stage_checkpoint, reload_rows, expected_logits)
    write_manifest(stage_checkpoint / "reload_check.json", reload_check)
    del expected_logits
    gc.collect()
    torch.cuda.empty_cache()

    training_node = {
        "training_status": "passed",
        "optimizer_step": stop_at_step,
        "consumed_sample_exposures": stop_at_step * 4,
        "consumed_ids_sha256": canonical_sha256(consumed_ids),
        "consumed_ids_set_sha256": canonical_sha256(sorted(set(consumed_ids))),
        "schedule_sha256": schedule["schedule_sha256"],
        "checkpoint_manifest_sha256": sha256_file(stage_manifest_path),
        "checkpoint_manifest_path": str(stage_manifest_path),
        "training_state_sha256": stage_manifest["training_state_sha256"],
        "reload_check_sha256": sha256_file(stage_checkpoint / "reload_check.json"),
        "independent_reload_status": reload_check["status"],
        "elapsed_seconds": time.monotonic() - start_time,
        "evaluation_status": "not_run",
    }
    run_manifest.setdefault("nodes", {})[str(stop_at_step)] = training_node
    run_manifest.update({
        "status": "awaiting_evaluation",
        "optimizer_step": stop_at_step,
        "sample_count": stop_at_step * 4,
        "last_checkpoint_manifest_sha256": training_node["checkpoint_manifest_sha256"],
        "last_updated_at_utc": _now(),
        "checkpoint_records": (run_manifest.get("checkpoint_records") or []) + checkpoint_records,
        "training_elapsed_seconds": run_manifest.get("training_elapsed_seconds", 0.0) + (time.monotonic() - start_time),
    })
    write_manifest(run_manifest_path, run_manifest, overwrite=True)
    print(json.dumps({"status": run_manifest["status"], "arm": arm, "run_id": run_id, "optimizer_steps": stop_at_step, "sample_exposures": stop_at_step * 4, "checkpoint_manifest_sha256": training_node["checkpoint_manifest_sha256"], "reload_check": reload_check["status"]}, ensure_ascii=False))
    return run_manifest


def _prepare_training_inputs(config: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]], SFTPromptIndex]:
    supervision, _summary = build_supervision(
        _resolve(config["data"]["normalized_train"]),
        _resolve(config["data"]["a5_supervision"]),
        _json(_resolve(config["data"]["train_schema"])),
        _json(_resolve(config["data"]["output_schema"])),
        expected_rows=int(config["data"]["expected_d03_train_rows"]),
    )
    splits = read_splits(_resolve(config["data"]["a5_splits"]))
    if len(supervision) != int(config["data"]["expected_sft_train_rows"]):
        raise ValueError("A7 supervised Train row count changed")
    prompt_index = SFTPromptIndex(_resolve(config["data"]["sft_train"]), list(supervision))
    return supervision, {"schedule": _load_schedule(config, root)}, splits, prompt_index


def _verify_training_ledger_prefix(
    schedule: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    order_rows: Sequence[Mapping[str, Any]],
    *,
    expected_step: int,
) -> list[str]:
    """Require both append-only ledgers to equal the immutable schedule prefix."""
    if (
        len(train_rows) != expected_step
        or len(order_rows) != expected_step
        or expected_step > len(schedule["steps"])
    ):
        raise ValueError("A7 training ledgers do not cover the requested optimizer-step prefix")
    consumed: list[str] = []
    for step, (train_row, order_row, schedule_row) in enumerate(
        zip(train_rows, order_rows, schedule["steps"][:expected_step]), 1
    ):
        expected_ids = list(schedule_row["sample_ids"])
        if (
            train_row.get("optimizer_step") != step
            or order_row.get("optimizer_step") != step
            or train_row.get("sample_ids") != expected_ids
            or order_row.get("sample_ids") != expected_ids
            or train_row.get("epoch") != schedule_row["epoch"]
            or order_row.get("epoch") != schedule_row["epoch"]
        ):
            raise ValueError(f"A7 training ledger row {step} differs from the frozen schedule")
        consumed.extend(expected_ids)
    return consumed


def _repair_reload(args: argparse.Namespace) -> int:
    """Revalidate a saved node's Adapter reload after the frozen-base dtype fix."""
    import torch
    from peft import set_peft_model_state_dict

    config, config_sha = _load_config(_resolve(args.config))
    pair_id = _validate_pair_id(config, args.pair_id)
    arm = validate_arm(args.arm)
    step = int(args.from_step or 512)
    if step != 512:
        raise ValueError("A7 reload recovery is limited to the incomplete Step 512 node")
    preflight, pair_root = _load_preflight(config, config_sha, require_code_match=False)
    run_id = args.run_id or f"{pair_id}-{arm}"
    if run_id != f"{pair_id}-{arm}":
        raise ValueError("A7 reload recovery Run ID must bind the frozen pair and Arm")
    run_root = _resolve(config["outputs"]["training_root"]) / run_id
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = _json(run_manifest_path)
    if (
        run_manifest.get("status") != "training"
        or run_manifest.get("run_id") != run_id
        or run_manifest.get("arm") != arm
        or run_manifest.get("config_sha256") != config_sha
        or run_manifest.get("code_sha256") != preflight.get("code_sha256")
        or run_manifest.get("data_sha256") != preflight.get("data_sha256")
        or run_manifest.get("schedule_sha256") != preflight.get("schedule", {}).get("schedule_sha256")
        or run_manifest.get("optimizer_step") is not None
    ):
        raise StageAuthorizationError("A7 reload recovery requires the failed, unfinalized same-Run Step 512 state")
    reload_check_path = run_root / "trainer" / f"checkpoint-{step}" / "reload_check.json"
    if reload_check_path.exists():
        raise FileExistsError("A7 Step 512 reload recovery has already been recorded")

    schedule = _load_schedule(config, pair_root)
    train_rows = [row for _line, row, _sha in iter_jsonl(run_root / "train_steps.jsonl")]
    order_rows = [row for _line, row, _sha in iter_jsonl(run_root / "sample_order.jsonl")]
    consumed_ids = _verify_training_ledger_prefix(
        schedule, train_rows, order_rows, expected_step=step,
    )
    checkpoint_dir = run_root / "trainer" / f"checkpoint-{step}"
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest_value = verify_checkpoint_manifest(
        checkpoint_manifest_path,
        expected_run_id=run_id,
        expected_arm=arm,
        expected_step=step,
        expected_identity=run_manifest["checkpoint_identity"],
        expected_schedule_sha256=schedule["schedule_sha256"],
    )
    training_state = load_training_state(
        checkpoint_dir / "training_state.pt",
        expected_identity=run_manifest["checkpoint_identity"],
        expected_arm=arm,
        expected_run_id=run_id,
        expected_step=step,
        expected_schedule_sha256=schedule["schedule_sha256"],
        expected_consumed_ids=consumed_ids,
    )
    supervision, _wrapper, _splits, prompt_index = _prepare_training_inputs(config, pair_root)
    model, heads, _device, _hidden = _new_model_and_heads(config, arm, torch, trainable=True)
    set_peft_model_state_dict(model, training_state["model_state"], adapter_name="default")
    heads.load_state_dict(training_state["head_state"], strict=True)
    model.eval()
    heads.eval()
    reload_ids = list(dict.fromkeys(schedule["steps"][0]["sample_ids"]))[:2]
    reload_rows = [(sample_id, prompt_index.prompt_ids(sample_id)) for sample_id in reload_ids]
    expected_logits = _expected_reload_logits(
        model, heads, arm, torch, reload_rows, int(config["identity"]["pad_token_id"]),
    )
    del model, heads, training_state
    gc.collect()
    torch.cuda.empty_cache()
    reload_check = _verify_export_reload(
        config, arm, torch, checkpoint_dir, reload_rows, expected_logits,
    )
    reload_check["recovery"] = {
        "status": "revalidated_after_frozen_base_dtype_fix",
        "scope": "inference_base_dtype_reload_only",
        "training_code_sha256": preflight["code_sha256"],
        "compatible_code_sha256": _code_sha256(),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "optimizer_steps_added": 0,
        "saved_optimizer_and_rng_state_preserved": True,
    }
    write_manifest(reload_check_path, reload_check)
    compatibility = {
        "version": "E03-A7-reload-compatibility-v1",
        "status": "reload_compatibility_repaired",
        "scope": "inference_base_dtype_reload_only",
        "pair_id": pair_id,
        "run_id": run_id,
        "arm": arm,
        "checkpoint_step": step,
        "config_sha256": config_sha,
        "training_code_sha256": preflight["code_sha256"],
        "compatible_code_sha256": _code_sha256(),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "reload_check_sha256": sha256_file(reload_check_path),
        "optimizer_steps_added": 0,
        "training_semantics_unchanged": True,
        "created_at_utc": _now(),
    }
    compatibility_path = pair_root / "reload_compatibility.json"
    if compatibility_path.exists():
        raise FileExistsError("A7 reload compatibility has already been recorded")
    write_manifest(compatibility_path, compatibility)

    training_node = {
        "training_status": "passed",
        "optimizer_step": step,
        "consumed_sample_exposures": step * 4,
        "consumed_ids_sha256": canonical_sha256(consumed_ids),
        "consumed_ids_set_sha256": canonical_sha256(sorted(set(consumed_ids))),
        "schedule_sha256": schedule["schedule_sha256"],
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "training_state_sha256": checkpoint_manifest_value["training_state_sha256"],
        "reload_check_sha256": sha256_file(reload_check_path),
        "independent_reload_status": reload_check["status"],
        "reload_recovery": str(reload_check_path),
        "evaluation_status": "not_run",
    }
    run_manifest.setdefault("nodes", {})[str(step)] = training_node
    checkpoint_record = {
        "step": step,
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": training_node["checkpoint_manifest_sha256"],
        "training_state_sha256": training_node["training_state_sha256"],
        "sample_count": step * 4,
        "reload_recovery": True,
    }
    run_manifest.update({
        "status": "awaiting_evaluation",
        "optimizer_step": step,
        "sample_count": step * 4,
        "last_checkpoint_manifest_sha256": training_node["checkpoint_manifest_sha256"],
        "last_updated_at_utc": _now(),
        "checkpoint_records": (run_manifest.get("checkpoint_records") or []) + [checkpoint_record],
        "reload_compatibility_path": str(compatibility_path),
        "reload_compatibility_sha256": sha256_file(compatibility_path),
        "optimizer_steps_added_by_repair": 0,
    })
    write_manifest(run_manifest_path, run_manifest, overwrite=True)
    print(json.dumps({
        "status": run_manifest["status"], "arm": arm, "run_id": run_id,
        "optimizer_steps": step, "sample_exposures": step * 4,
        "reload_check": reload_check["status"],
        "training_code_sha256": preflight["code_sha256"],
        "compatible_code_sha256": _code_sha256(),
    }, ensure_ascii=False))
    return 0


def _start(args: argparse.Namespace) -> int:
    import torch

    config, config_sha = _load_config(_resolve(args.config))
    _validate_pair_id(config, args.pair_id)
    arm = validate_arm(args.arm)
    if args.stop_at_step != 512 or not args.allow_staged_training:
        raise ValueError("Initial A7 start only allows the explicitly authorized 0→512 stage and requires --allow-staged-training")
    preflight, root = _load_preflight_for_training(config, config_sha)
    freeze = _json(root / "lambda_freeze.json")
    if freeze.get("status") != "G1_passed_lambda_frozen" or freeze.get("lambda_emotion") != config["training"]["lambda_emotion"]:
        raise StageAuthorizationError("A7 cannot start before both Arm GPU preflights and lambda freeze pass")
    run_id = args.run_id or f"{config['training']['pair_id']}-{arm}"
    if run_id != f"{config['training']['pair_id']}-{arm}":
        raise ValueError("A7 Run ID must bind the frozen pair ID and exact Arm")
    ticket_path = _resolve(args.ticket) if args.ticket else root / "start_tickets" / f"{arm}.json"
    ticket = verify_start_ticket(ticket_path, expected={
        "run_id": run_id,
        "arm": arm,
        "from_step": 0,
        "to_step": 512,
        "config_sha256": config_sha,
        "code_sha256": preflight["code_sha256"],
        "data_sha256": preflight["data_sha256"],
        "init_sha256": freeze[f"{arm.lower()}_initial_identity_sha256"] if f"{arm.lower()}_initial_identity_sha256" in freeze else _json(root / "gpu_preflight" / f"{arm}.json")["initial_identity"]["initial_sha256"],
        "dry_run_manifest_sha256": sha256_file(root / "gpu_preflight" / f"{arm}.json"),
    })
    if sha256_file(ticket_path) != sha256_file(root / "start_tickets" / f"{arm}.json"):
        raise StageAuthorizationError("A7 start ticket path differs from the frozen single-use ticket")
    schedule = _load_schedule(config, root)
    _supervision, _schedule_wrapper, splits, prompt_index = _prepare_training_inputs(config, root)
    _a5, _a5_sha, _audit = _a5_assets(config)
    initial_identity = _json(root / "gpu_preflight" / f"{arm}.json")["initial_identity"]
    result = _train_segment(
        config=config,
        config_sha=config_sha,
        arm=arm,
        run_id=run_id,
        from_step=0,
        stop_at_step=args.stop_at_step,
        run_root=_resolve(config["outputs"]["training_root"]) / run_id,
        schedule=schedule,
        supervision=_supervision,
        splits=splits,
        prompt_index=prompt_index,
        initial_identity=initial_identity,
        data_sha256=preflight["data_sha256"],
        code_sha256=preflight["code_sha256"],
        start_ticket_path=ticket_path,
    )
    return 0 if result.get("status") == "awaiting_evaluation" else 1


def _resume(args: argparse.Namespace) -> int:
    config, config_sha = _load_config(_resolve(args.config))
    _validate_pair_id(config, args.pair_id)
    arm = validate_arm(args.arm)
    validate_stage_transition(args.from_step, args.stop_at_step)
    if not args.allow_staged_training:
        raise ValueError("A7 resume requires --allow-staged-training after user node approval")
    preflight, pair_root = _load_preflight_for_training(config, config_sha)
    run_id = args.run_id or f"{config['training']['pair_id']}-{arm}"
    run_root = _resolve(config["outputs"]["training_root"]) / run_id
    run_manifest = _json(run_root / "run_manifest.json")
    if run_manifest.get("status") != "awaiting_manual_review" or run_manifest.get("optimizer_step") != args.from_step:
        raise StageAuthorizationError("A7 resume requires a reviewed checkpoint and awaiting_manual_review state")
    checkpoint_path = run_root / "trainer" / f"checkpoint-{args.from_step}" / "checkpoint_manifest.json"
    evaluation_manifest_path = _resolve(config["outputs"]["evaluation_root"]) / run_id / f"step-{args.from_step:04d}" / "evaluation_manifest.json"
    if not evaluation_manifest_path.is_file():
        raise FileNotFoundError("A7 resume requires the completed node evaluation Manifest")
    receipt_path = _resolve(args.approval_file) if args.approval_file else run_root / "reviews" / f"step-{args.from_step:04d}.json"
    receipt = verify_continuation_receipt(receipt_path, expected={
        "run_id": run_id,
        "arm": arm,
        "from_step": args.from_step,
        "to_step": args.stop_at_step,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "config_sha256": config_sha,
        "code_sha256": preflight["code_sha256"],
    })
    schedule = _load_schedule(config, pair_root)
    supervision, _schedule_wrapper, splits, prompt_index = _prepare_training_inputs(config, pair_root)
    initial_identity = run_manifest["initial_identity"]
    _train_segment(
        config=config,
        config_sha=config_sha,
        arm=arm,
        run_id=run_id,
        from_step=args.from_step,
        stop_at_step=args.stop_at_step,
        run_root=run_root,
        schedule=schedule,
        supervision=supervision,
        splits=splits,
        prompt_index=prompt_index,
        initial_identity=initial_identity,
        data_sha256=preflight["data_sha256"],
        code_sha256=preflight["code_sha256"],
        approval_file=receipt_path,
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    config, _config_sha = _load_config(_resolve(args.config))
    pair_id = _validate_pair_id(config, args.pair_id)
    result = {"pair_id": pair_id, "preflight": None, "arms": {}}
    root = _resolve(config["outputs"]["training_root"]) / "pairs" / pair_id
    if (root / "preflight.json").is_file():
        result["preflight"] = _json(root / "preflight.json").get("status")
    for arm in ("P", "EP"):
        run_id = f"{pair_id}-{arm}"
        path = _resolve(config["outputs"]["training_root"]) / run_id / "run_manifest.json"
        if path.is_file():
            manifest = _json(path)
            result["arms"][arm] = {
                "run_id": run_id,
                "status": manifest.get("status"),
                "optimizer_step": manifest.get("optimizer_step", 0),
                "sample_count": manifest.get("sample_count", 0),
                "node_steps": sorted(int(step) for step in manifest.get("nodes", {})),
            }
        else:
            result["arms"][arm] = {"run_id": run_id, "status": "not_started"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", required=True, choices=("preflight", "gpu-preflight", "freeze-preflights", "start", "repair-reload", "resume", "status"))
    parser.add_argument("--arm", choices=("P", "EP"))
    parser.add_argument("--pair-id")
    parser.add_argument("--run-id")
    parser.add_argument("--ticket", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--from-step", type=int)
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--allow-staged-training", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode in {"gpu-preflight", "start", "repair-reload", "resume"} and args.arm is None:
        raise SystemExit(f"--mode {args.mode} requires --arm P|EP")
    if args.mode == "resume" and (args.from_step is None or args.stop_at_step is None):
        raise SystemExit("--mode resume requires --from-step and --stop-at-step")
    if args.mode == "start" and args.stop_at_step is None:
        args.stop_at_step = 512
    if args.mode == "preflight":
        return _preflight(args)
    if args.mode == "gpu-preflight":
        return _gpu_preflight(args)
    if args.mode == "freeze-preflights":
        return _freeze_preflights(args)
    if args.mode == "start":
        return _start(args)
    if args.mode == "repair-reload":
        return _repair_reload(args)
    if args.mode == "resume":
        return _resume(args)
    return _status(args)


if __name__ == "__main__":
    raise SystemExit(main())
