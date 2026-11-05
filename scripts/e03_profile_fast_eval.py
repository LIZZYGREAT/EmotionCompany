#!/usr/bin/env python3
"""CPU audit and isolated, budgeted Greedy comparison for profile-only output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
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

from src.evaluation import read_jsonl
from src.evaluation.metrics import profile_metrics
from src.evaluation.profile_fast_eval import (
    compare_generated_prefix,
    profile_nonempty_recall,
    run_profile_fast_only_attempts,
    score_profile_attempts,
    selected_id_sha256,
    select_frozen_id_samples,
    select_unique_conversation_samples,
)
from src.evaluation.profile_prefix import ProfilePrefixStopper, parse_profile_prefix
from src.evaluation.profile_reconcile import index_profile_prediction_rows
from src.evaluation.source_integrity import (
    audit_prediction_point_sources,
    build_source_manifest,
    unique_index,
    validate_target_free_inputs,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _read_selected_jsonl_lines(path: Path, selected_lines: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number not in selected_lines:
                continue
            if not line.strip():
                raise ValueError(f"selected source line {line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"selected source line {line_number} is not an object")
            rows[line_number] = value
    missing = selected_lines - set(rows)
    if missing:
        raise ValueError("one or more selected source lines are missing")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl_fsynced(path: Path, row: dict[str, Any]) -> None:
    """Persist one complete profile-only record before the next sample starts."""
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _minimal_profile_prediction(attempt: dict[str, Any]) -> dict[str, Any]:
    sample_id = attempt.get("id")
    status = attempt.get("status")
    reason = attempt.get("reason")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("profile-only prediction is missing a sample ID")
    if not isinstance(status, str) or not isinstance(reason, str) or not reason:
        raise ValueError(f"profile-only prediction {sample_id} has no status/reason")
    if status == "valid":
        profile = attempt.get("user_profile")
        emotion = attempt.get("emotion_label")
        if not isinstance(profile, dict) or not isinstance(emotion, str):
            raise ValueError(f"valid profile-only prediction {sample_id} has no complete parsed result")
        return {
            "id": sample_id,
            "status": status,
            "emotion_label": emotion,
            "user_profile": profile,
            "reason": reason,
        }
    return {"id": sample_id, "status": status, "reason": reason}


def _profile_prediction_artifact(
    path: Path,
    expected_ids: list[str],
    enums: dict[str, list[str]],
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "sha256": None,
            "rows": 0,
            "expected_rows": len(expected_ids),
            "ids_sha256": None,
            "status_counts": {},
            "complete_id_coverage": False,
            "all_selected_rows_valid": False,
            "strict_quality_comparison_ready": False,
            "validation_error": "prediction_file_missing",
            "write_mode": "append_fsync_per_record",
            "response_text_saved": False,
        }
    try:
        rows = _read_jsonl(path)
    except (OSError, ValueError) as exc:
        payload = path.read_bytes()
        row_count = payload.count(b"\n") + int(bool(payload) and not payload.endswith(b"\n"))
        return {
            "path": str(path),
            "sha256": _sha256_file(path),
            "rows": row_count,
            "expected_rows": len(expected_ids),
            "ids_sha256": None,
            "status_counts": {},
            "complete_id_coverage": False,
            "all_selected_rows_valid": False,
            "strict_quality_comparison_ready": False,
            "validation_error": f"{type(exc).__name__}: {exc}",
            "write_mode": "append_fsync_per_record",
            "response_text_saved": False,
        }
    indexed: dict[str, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    validation_error: str | None = None
    try:
        indexed, status_counts = index_profile_prediction_rows(rows, expected_ids, enums)
    except (KeyError, TypeError, ValueError) as exc:
        validation_error = f"{type(exc).__name__}: {exc}"
    all_valid = (
        validation_error is None
        and status_counts.get("valid") == len(expected_ids)
    )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "rows": len(rows),
        "expected_rows": len(expected_ids),
        "ids_sha256": selected_id_sha256(list(indexed)) if validation_error is None else None,
        "status_counts": status_counts,
        "complete_id_coverage": validation_error is None,
        "all_selected_rows_valid": all_valid,
        "strict_quality_comparison_ready": all_valid,
        "validation_error": validation_error,
        "write_mode": "append_fsync_per_record",
        "response_text_saved": False,
    }


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "E03-profile-fast-v1":
        raise ValueError("unsupported E03 profile fast config")
    if value.get("protocol", {}).get("format_version") != "F2" or value.get("protocol", {}).get("truncation_version") != "T1":
        raise ValueError("profile fast evaluation must use the locked F2/T1 protocol")
    if value.get("generation", {}).get("do_sample") is not False:
        raise ValueError("profile fast evaluation requires Greedy generation")
    budget = int(value.get("gpu_budget", {}).get("max_seconds", 0))
    if not 1 <= budget <= 900:
        raise ValueError("GPU budget must be between 1 and 900 seconds")
    experiment_version = value.get("experiment_version")
    if not isinstance(experiment_version, str) or not experiment_version.strip():
        raise ValueError("profile fast config requires an explicit experiment_version")
    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("profile fast config requires an explicit selection mode")
    selection_mode = selection.get("mode", "mechanism_smoke")
    if selection_mode == "mechanism_smoke":
        count = selection.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 16 <= count <= 32:
            raise ValueError("mechanism_smoke selection count must be between 16 and 32")
    elif selection_mode == "frozen_ids":
        expected_count = selection.get("expected_count")
        expected_sha = selection.get("expected_ids_sha256")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError("frozen_ids selection requires a positive expected_count")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha.lower()
        ):
            raise ValueError("frozen_ids selection requires a pinned expected_ids_sha256")
        if not isinstance(selection.get("frozen_ids_path"), str) or not selection["frozen_ids_path"].strip():
            raise ValueError("frozen_ids selection requires a frozen_ids_path")
        expected_conversations = selection.get("expected_conversation_count")
        if (
            isinstance(expected_conversations, bool)
            or not isinstance(expected_conversations, int)
            or expected_conversations < 1
        ):
            raise ValueError("frozen_ids selection requires a positive expected_conversation_count")
    else:
        raise ValueError("selection.mode must be mechanism_smoke or frozen_ids")
    adoption_gates = value.get("adoption_gates")
    if not isinstance(adoption_gates, dict):
        raise ValueError("profile fast config requires explicit adoption_gates")
    minimum_coverage = adoption_gates.get("minimum_valid_coverage")
    if (
        isinstance(minimum_coverage, bool)
        or not isinstance(minimum_coverage, (int, float))
        or not 0 < minimum_coverage <= 1
    ):
        raise ValueError("adoption_gates.minimum_valid_coverage must be in (0, 1]")
    if adoption_gates.get("require_fixed_denominator_metrics") is not True:
        raise ValueError("profile fast config must require fixed-denominator scoring")
    return value


def _load_enums() -> dict[str, list[str]]:
    from src.evaluation import load_submission_schema, schema_enums

    return schema_enums(load_submission_schema(ROOT))


def _output_path(config: dict[str, Any], run_id: str, override: str | None) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
        raise ValueError("run-id must contain only letters, digits, dot, underscore, and dash")
    output_root = _resolve(override or config["output_root"]) / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root / "profile_fast_metrics.json"


def _select_profile_samples(
    config: dict[str, Any],
    input_rows: list[dict[str, Any]],
    id_map_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = config["selection"]
    mode = selection.get("mode", "mechanism_smoke")
    if mode == "mechanism_smoke":
        selected = select_unique_conversation_samples(
            input_rows,
            id_map_rows,
            count=int(selection["count"]),
            seed=int(selection["seed"]),
        )
        id_sha256 = selected_id_sha256([row["id"] for row in selected])
        method = selection.get("method", "seeded_hash_order_one_point_per_conversation_v1")
        frozen_ids_path = None
        frozen_ids_file_sha256 = None
    else:
        frozen_ids_path = _resolve(selection["frozen_ids_path"])
        frozen_rows = _read_jsonl(frozen_ids_path)
        selected = select_frozen_id_samples(
            input_rows,
            id_map_rows,
            frozen_rows,
            expected_ids_sha256=str(selection["expected_ids_sha256"]).lower(),
            expected_count=int(selection["expected_count"]),
            expected_conversation_count=int(selection["expected_conversation_count"]),
            require_complete_conversations=True,
        )
        id_sha256 = selected_id_sha256([row["id"] for row in selected])
        method = "pre_frozen_target_free_id_list_v1"
        frozen_ids_file_sha256 = _sha256_file(frozen_ids_path)
    map_index = unique_index(id_map_rows, "eval_id", "ID map")
    conversation_ids = {
        str(map_index[row["id"]]["conversation_id"])
        for row in selected
    }
    return selected, {
        "mode": mode,
        "method": method,
        "seed": selection.get("seed") if mode == "mechanism_smoke" else None,
        "selected_count": len(selected),
        "selected_conversation_count": len(conversation_ids),
        "selected_ids_sha256": id_sha256,
        "expected_ids_sha256": selection.get("expected_ids_sha256") if mode == "frozen_ids" else None,
        "frozen_ids_path": str(frozen_ids_path) if frozen_ids_path is not None else None,
        "frozen_ids_file_sha256": frozen_ids_file_sha256,
    }


def parse_existing_raw(
    config: dict[str, Any],
    raw_path: Path,
    output_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    enums = _load_enums()
    rows = _read_jsonl(raw_path)
    raw_index = unique_index(rows, "id", "Raw predictions")
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in raw_index.values():
        raw_text = row.get("raw_text")
        if not isinstance(raw_text, str):
            counts["invalid"] += 1
            reasons["missing_raw_text"] += 1
            continue
        result = parse_profile_prefix(raw_text, enums)
        counts[result.status] += 1
        reasons[result.reason] += 1
    manifest_path = _resolve(config["sources"]["dataset_manifest_path"])
    source = {
        "raw_path": str(raw_path),
        "raw_sha256": _sha256_file(raw_path),
        "raw_rows": len(rows),
        "unique_raw_ids": len(raw_index),
        "source_identity_path": str(manifest_path),
        "source_identity_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else None,
        "parser_source_sha256": _sha256_file(Path(__file__).resolve()),
        "fast_eval_config_sha256": _sha256_file(config_path),
    }
    report = {
        "status": "cpu_raw_parse_completed",
        "created_at_utc": _now(),
        "evaluation_scope": "prefix_parse_only_no_early_stop_equivalence_claim",
        "source_manifest": source,
        "sample_count": len(rows),
        "parse_status_counts": {key: counts.get(key, 0) for key in ("valid", "invalid", "incomplete", "unsupported_order")},
        "parse_reason_counts": dict(sorted(reasons.items())),
        "valid_parse_rate": counts.get("valid", 0) / len(rows) if rows else 0.0,
        "gpu_used": False,
        "references_read": False,
    }
    _write_json(output_path, report)
    return report


def audit_selected_source_history(
    config: dict[str, Any],
    selected_inputs: list[dict[str, Any]],
    selected_id_maps: list[dict[str, Any]],
) -> dict[str, Any]:
    source_cfg = config["sources"]
    normalized_path = _resolve(source_cfg["normalized_dev_path"])
    raw_source_path = _resolve(source_cfg["raw_source_path"])
    normalized_rows = _read_jsonl(normalized_path)
    normalized_index = unique_index(normalized_rows, "sample_id", "normalized Dev")
    id_map = unique_index(selected_id_maps, "eval_id", "selected ID map")
    selected_sample_ids: set[str] = set()
    for row in selected_inputs:
        mapping = id_map.get(row["id"])
        sample_id = mapping.get("sample_id") if mapping else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("selected ID map has a missing sample ID")
        selected_sample_ids.add(sample_id)
    if len(selected_sample_ids) != len(selected_inputs):
        raise ValueError("selected ID map has duplicate sample mappings")
    try:
        selected_normalized = [normalized_index[sample_id] for sample_id in sorted(selected_sample_ids)]
    except KeyError as exc:
        raise ValueError("a selected sample ID is absent from normalized Dev") from exc
    normalized_by_id = {str(row["sample_id"]): row for row in selected_normalized}
    for eval_id, mapping in id_map.items():
        source_row = normalized_by_id.get(str(mapping.get("sample_id")))
        if source_row is None or mapping.get("conversation_id") != source_row.get("conversation_id"):
            raise ValueError("selected sample conversation ID differs from normalized Dev")
    selected_source_lines = {
        row.get("source", {}).get("source_line")
        for row in selected_normalized
        if isinstance(row.get("source"), dict)
    }
    if any(isinstance(line, bool) or not isinstance(line, int) or line < 1 for line in selected_source_lines):
        raise ValueError("selected normalized Dev row has an invalid source line")
    raw_source_rows = _read_selected_jsonl_lines(raw_source_path, selected_source_lines)
    history_audit = audit_prediction_point_sources(selected_normalized, raw_source_rows)
    dataset_manifest = _resolve(source_cfg["dataset_manifest_path"])
    if not dataset_manifest.is_file():
        raise FileNotFoundError(f"D0.3 dataset manifest is missing: {dataset_manifest}")
    dataset_manifest_value = _read_json(dataset_manifest)
    if dataset_manifest_value.get("pipeline_version") != "D0.3-v1":
        raise ValueError("unsupported D0.3 data pipeline version")
    train_fact = dataset_manifest_value.get("inputs", {}).get("train", {})
    dev_outputs = [
        row for row in dataset_manifest_value.get("outputs", [])
        if row.get("path") == "data/processed/v1/dev.jsonl"
    ]
    raw_sha = _sha256_file(raw_source_path)
    normalized_sha = _sha256_file(normalized_path)
    if train_fact.get("sha256_after") != raw_sha or train_fact.get("sha256_before") != raw_sha:
        raise ValueError("raw Train SHA differs from the D0.3 source Manifest")
    if (
        len(dev_outputs) != 1
        or dev_outputs[0].get("sha256") != normalized_sha
        or dev_outputs[0].get("records") != len(normalized_rows)
    ):
        raise ValueError("normalized Dev SHA/count differs from the D0.3 output Manifest")
    return {
        **history_audit,
        "normalized_dev_path": str(normalized_path),
        "normalized_dev_sha256": normalized_sha,
        "raw_source_path": str(raw_source_path),
        "raw_source_sha256": raw_sha,
        "dataset_manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest),
    }


def _enrich_id_map_conversations(
    config: dict[str, Any],
    id_map_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows = _read_jsonl(_resolve(config["sources"]["normalized_dev_path"]))
    normalized_index = unique_index(normalized_rows, "sample_id", "normalized Dev")
    enriched: list[dict[str, Any]] = []
    for row in id_map_rows:
        sample_id = row.get("sample_id")
        normalized = normalized_index.get(sample_id) if isinstance(sample_id, str) else None
        if normalized is None:
            raise ValueError("ID map sample ID is absent from normalized Dev")
        conversation_id = normalized.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("normalized Dev row has no conversation_id")
        enriched.append({**row, "conversation_id": conversation_id})
    return enriched


def run_source_audit(
    config: dict[str, Any],
    output_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    input_rows = _read_jsonl(_resolve(config["sources"]["input_path"]))
    id_map_rows = _enrich_id_map_conversations(
        config, _read_jsonl(_resolve(config["sources"]["id_map_path"]))
    )
    try:
        selected, selection_facts = _select_profile_samples(config, input_rows, id_map_rows)
        input_index = validate_target_free_inputs(input_rows)
        map_index = unique_index(id_map_rows, "eval_id", "ID map")
        selected_maps = [dict(map_index[row["id"]]) for row in selected]
        history_audit = audit_selected_source_history(config, selected, selected_maps)
    except (KeyError, OSError, ValueError) as exc:
        report = {
            "status": "preflight_failed",
            "created_at_utc": _now(),
            "reason": str(exc),
            "gpu_used": False,
            "generation_started": False,
        }
        _write_json(output_path, report)
        return report
    selected_rows = [input_index[row["id"]] for row in selected]
    source_manifest = build_source_manifest(
        source_path=history_audit["raw_source_path"],
        source_sha256=history_audit["raw_source_sha256"],
        input_rows=selected_rows,
        id_map_rows=selected_maps,
        prompt_version=config["protocol"]["prompt_version"],
        prompt_sha256=config["protocol"]["prompt_sha256"],
        format_version=config["protocol"]["format_version"],
        truncation_version=config["protocol"]["truncation_version"],
        exporter_source_sha256=_sha256_file(Path(__file__).resolve()),
        selection_method=selection_facts["method"],
        selection_seed=selection_facts["seed"],
    )
    source_manifest["selection"] = selection_facts
    source_manifest["history_audit"] = history_audit
    source_manifest["normalized_dev_path"] = history_audit["normalized_dev_path"]
    source_manifest["normalized_dev_sha256"] = history_audit["normalized_dev_sha256"]
    source_manifest["generation_input_path"] = str(_resolve(config["sources"]["input_path"]))
    source_manifest["generation_input_sha256"] = _sha256_file(_resolve(config["sources"]["input_path"]))
    source_manifest["id_map_path"] = str(_resolve(config["sources"]["id_map_path"]))
    source_manifest["id_map_sha256"] = _sha256_file(_resolve(config["sources"]["id_map_path"]))
    report = {
        "status": "source_preflight_passed",
        "created_at_utc": _now(),
        "fast_eval_config_sha256": _sha256_file(config_path),
        "source_manifest": source_manifest,
        "selected_points": len(selected),
        "selected_conversations": selection_facts["selected_conversation_count"],
        "gpu_used": False,
        "generation_started": False,
    }
    _write_json(output_path, report)
    return report


class _BudgetCriterion:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.exhausted = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        if time.monotonic() >= self.deadline:
            self.exhausted = True
            return True
        return False


def _load_model(config: dict[str, Any], runner: Any, torch: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    local_model = runner.model_path(config)
    runner.require_cuda_for_quantization(config, torch)
    trust_remote_code = bool(config.get("trust_remote_code", False))
    model_config = AutoConfig.from_pretrained(
        local_model, trust_remote_code=trust_remote_code, local_files_only=True
    )
    model_loader = (
        AutoModelForImageTextToText
        if model_config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        else AutoModelForCausalLM
    )
    tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(config, local_model)
    weight_identity = runner._weight_identity(local_model, config)
    expected_base_sha = config.get("expected_weight_sha256")
    if expected_base_sha and weight_identity.get("observed_weight_hash") != expected_base_sha:
        raise ValueError("Base weight identity differs from the frozen inference config")
    adapter_config = runner.validate_adapter_config(config.get("adapter"), "transformers")
    adapter_identity: dict[str, Any] = {"enabled": False}
    adapter_path: Path | None = None
    if adapter_config["enabled"]:
        if weight_identity.get("identity_status") != "observed":
            raise ValueError("Base weight identity is not observed")
        if weight_identity.get("observed_weight_hash") != adapter_config["expected_base_weight_sha256"]:
            raise ValueError("Adapter is bound to another Base weight identity")
        adapter_path = Path(adapter_config["path"]).expanduser()
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
        adapter_path = adapter_path.resolve()
        adapter_identity, adapter_metadata = runner.adapter_weight_identity(adapter_path)
        if adapter_identity["weight_sha256"] != adapter_config["expected_adapter_weight_sha256"]:
            raise ValueError("Adapter SHA differs from the frozen inference config")
        if (
            adapter_config.get("expected_adapter_config_sha256")
            and adapter_identity["config_sha256"] != adapter_config["expected_adapter_config_sha256"]
        ):
            raise ValueError("Adapter config SHA differs from the frozen inference config")
        runner.validate_adapter_base_reference(adapter_metadata, config.get("model_id", ""), local_model)

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": True,
        "device_map": config.get("device_map", "auto"),
        "torch_dtype": config.get("torch_dtype", "auto"),
    }
    quantization_config = runner.build_quantization_config(config, torch)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = model_loader.from_pretrained(local_model, **load_kwargs)
    if adapter_path is not None:
        model = runner.load_peft_adapter(
            model, adapter_path, prepare_quantized_base=quantization_config is not None,
            is_trainable=False, local_files_only=True,
        )
    model.eval()
    quantization_facts = runner.inspect_quantization_runtime(model, config, torch)
    if quantization_config is not None and not quantization_facts["real_nf4_verified"]:
        raise ValueError("runtime did not pass NF4/BF16/CUDA/no-offload verification")
    facts = {
        "model_path_name": local_model.name,
        "model_type": model_config.model_type,
        "model_weight_identity_sha256": weight_identity.get("observed_weight_hash"),
        "adapter_weight_sha256": adapter_identity.get("weight_sha256"),
        "adapter_config_sha256": adapter_identity.get("config_sha256"),
        "tokenizer_facts": tokenizer_facts,
        "quantization": quantization_facts,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    return model, tokenizer, model_config, facts


def run_gpu_comparison(
    config: dict[str, Any],
    output_path: Path,
    run_config_path: Path | None = None,
    fast_config_path: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    from participant import run_inference as runner

    input_path = _resolve(config["sources"]["input_path"])
    id_map_path = _resolve(config["sources"]["id_map_path"])
    references_path = _resolve(config["sources"]["references_path"])
    inputs = _read_jsonl(input_path)
    id_maps = _enrich_id_map_conversations(config, _read_jsonl(id_map_path))
    input_index = validate_target_free_inputs(inputs)
    map_index = unique_index(id_maps, "eval_id", "ID map")
    if set(input_index) != set(map_index):
        raise ValueError("input and ID map IDs differ")
    count = int(config["selection"]["count"])
    selected = select_unique_conversation_samples(
        inputs, id_maps, count=count, seed=int(config["selection"]["seed"]),
    )
    selected_ids = {row["id"] for row in selected}
    selected_maps = [dict(map_index[eval_id]) for eval_id in sorted(selected_ids)]
    history_audit = audit_selected_source_history(config, selected, selected_maps)
    selected_input_rows = [input_index[eval_id] for eval_id in sorted(selected_ids)]
    source_manifest = build_source_manifest(
        source_path=history_audit["raw_source_path"],
        source_sha256=history_audit["raw_source_sha256"],
        input_rows=selected_input_rows,
        id_map_rows=selected_maps,
        prompt_version=config["protocol"]["prompt_version"],
        prompt_sha256=config["protocol"]["prompt_sha256"],
        format_version=config["protocol"]["format_version"],
        truncation_version=config["protocol"]["truncation_version"],
        exporter_source_sha256=_sha256_file(Path(__file__).resolve()),
        selection_method=config["selection"]["method"],
        selection_seed=int(config["selection"]["seed"]),
    )
    source_manifest["history_audit"] = history_audit
    source_manifest["normalized_dev_path"] = history_audit["normalized_dev_path"]
    source_manifest["normalized_dev_sha256"] = history_audit["normalized_dev_sha256"]
    source_manifest["generation_input_path"] = str(input_path)
    source_manifest["generation_input_sha256"] = _sha256_file(input_path)
    source_manifest["id_map_path"] = str(id_map_path)
    source_manifest["id_map_sha256"] = _sha256_file(id_map_path)
    inference_path = _resolve(run_config_path or config["sources"]["inference_config_path"])
    inference_config = _read_json(inference_path)
    inference_config["generation"] = {
        **inference_config["generation"],
        **config["generation"],
    }
    if inference_config.get("backend") != "transformers":
        raise ValueError("fast profile comparison requires the frozen Transformers runner config")
    prompt_text, _submission_schema, enums, prompt_facts = runner.load_runtime_protocol(inference_config)
    if prompt_facts.get("sha256") != config["protocol"]["prompt_sha256"]:
        raise ValueError("P0 prompt SHA differs from the fast-eval config")
    if inference_config["generation"].get("do_sample") is not False:
        raise ValueError("the paired run must use Greedy decoding")

    budget_seconds = int(config["gpu_budget"]["max_seconds"])
    started = time.monotonic()
    deadline = started + budget_seconds
    if not torch.cuda.is_available():
        report = {
            "status": "blocked_environment",
            "created_at_utc": _now(),
            "reason": "cuda_unavailable",
            "gpu_used": False,
            "sample_count": count,
            "budget_seconds": budget_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "source_manifest": source_manifest,
        }
        _write_json(output_path, report)
        return report

    model, tokenizer, _model_config, model_facts = _load_model(inference_config, runner, torch)
    load_seconds = time.monotonic() - started
    if time.monotonic() >= deadline:
        report = {
            "status": "budget_exhausted",
            "created_at_utc": _now(),
            "reason": "model_load_exhausted_budget",
            "gpu_used": True,
            "model_load_seconds": load_seconds,
            "budget_seconds": budget_seconds,
            "source_manifest": source_manifest,
        }
        _write_json(output_path, report)
        return report

    class BudgetCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.inner = _BudgetCriterion(deadline)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return self.inner(input_ids, scores, **kwargs)

    full_seconds: list[float] = []
    fast_seconds: list[float] = []
    full_peak_bytes = 0
    fast_peak_bytes = 0
    case_status: Counter[str] = Counter()
    comparisons: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    full_prefix_predictions: dict[str, dict[str, Any]] = {}
    budget_exhausted = False
    generation = dict(inference_config["generation"])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def invoke(prompt_ids: list[int], criteria: Any) -> tuple[list[int], str, bool]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.get_input_embeddings().weight.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stopping_criteria=criteria,
                **generation,
            )
        generated = [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        exhausted = any(getattr(item, "inner", None) and item.inner.exhausted for item in criteria)
        return generated, text, exhausted

    for sample in selected:
        if time.monotonic() >= deadline:
            budget_exhausted = True
            break
        selection, _messages = runner.prepare_sample_messages(
            sample, prompt_text, tokenizer, inference_config["context"]["max_prompt_tokens"]
        )
        if selection.get("floor_exceeds_cap"):
            case_status["invalid_prompt_floor_exceeds_cap"] += 1
            continue
        prompt_ids = list(selection["prompt_ids"])
        runner.synchronize(torch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        full_budget = BudgetCriteria()
        full_ids, full_text, full_exhausted = invoke(prompt_ids, StoppingCriteriaList([full_budget]))
        runner.synchronize(torch)
        full_seconds.append(time.monotonic() - before)
        full_peak_bytes = max(full_peak_bytes, int(torch.cuda.max_memory_allocated()))
        if full_exhausted:
            case_status["budget_exhausted_during_full"] += 1
            budget_exhausted = True
            break
        full_prefix = parse_profile_prefix(full_text, enums)
        if not full_prefix.valid:
            case_status[f"full_{full_prefix.status}"] += 1
        if time.monotonic() >= deadline:
            budget_exhausted = True
            break

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        prefix_stopper = ProfilePrefixStopper(
            lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True),
            enums,
            deadline=deadline,
        )

        class PrefixCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                generated = input_ids[0, len(prompt_ids):].detach().cpu().tolist()
                return prefix_stopper.inspect(generated)

        prefix_criteria = PrefixCriteria()
        fast_ids, _fast_text, fast_exhausted = invoke(prompt_ids, StoppingCriteriaList([prefix_criteria]))
        runner.synchronize(torch)
        fast_seconds.append(time.monotonic() - before)
        fast_peak_bytes = max(fast_peak_bytes, int(torch.cuda.max_memory_allocated()))
        if fast_exhausted or prefix_stopper.stop_reason == "budget_exhausted":
            case_status["budget_exhausted_during_fast"] += 1
            budget_exhausted = True
            break
        stop_reason = (
            "profile_prefix_closed"
            if prefix_stopper.result is not None and prefix_stopper.result.valid
            else "eos"
            if eos_token_id is not None and eos_token_id in fast_ids
            else "max_new_tokens"
            if len(fast_ids) >= int(generation["max_new_tokens"])
            else "unknown"
        )
        parsed_fast = prefix_stopper.result or prefix_stopper.finish(stop_reason, fast_ids)
        if not parsed_fast.valid:
            case_status[f"fast_{parsed_fast.status}:{parsed_fast.reason}"] += 1
            continue
        token_ids_for_prefix = prefix_stopper.generated_token_ids or tuple(fast_ids)
        comparison = compare_generated_prefix(
            full_ids,
            token_ids_for_prefix,
            full_prefix.emotion_label,
            parsed_fast.emotion_label,
            full_prefix.user_profile,
            parsed_fast.user_profile,
        )
        if not full_prefix.valid:
            comparison["emotion_and_profile_match"] = False
        comparison.update({
            "fast_status": parsed_fast.status,
            "full_status": full_prefix.status,
            "stop_reason": prefix_stopper.stop_reason or stop_reason,
        })
        comparisons.append(comparison)
        case_status["valid"] += 1
        if not comparison["token_prefix_match"] or not comparison["emotion_and_profile_match"]:
            case_status["mismatch"] += 1
            continue
        predictions[sample["id"]] = {
            "emotion_label": parsed_fast.emotion_label,
            "user_profile": dict(parsed_fast.user_profile or {}),
        }
        full_prefix_predictions[sample["id"]] = {
            "emotion_label": full_prefix.emotion_label,
            "user_profile": dict(full_prefix.user_profile or {}),
        }

    # References are opened only after all generation has finished.
    references = _read_jsonl(references_path)
    reference_index = unique_index(references, "eval_id", "References")
    profile_gold: list[dict[str, list[str]]] = []
    profile_predicted: list[dict[str, list[str]]] = []
    emotion_equal_count = 0
    for sample_id in sorted(predictions):
        if sample_id not in reference_index:
            raise ValueError(f"missing Reference for selected eval ID {sample_id}")
        target = reference_index[sample_id].get("target")
        if not isinstance(target, dict):
            raise ValueError(f"invalid Reference target for {sample_id}")
        gold_profile = target.get("user_profile")
        prediction = predictions[sample_id]
        if not isinstance(gold_profile, dict):
            raise ValueError(f"invalid Reference profile for {sample_id}")
        profile_gold.append(gold_profile)
        profile_predicted.append(prediction["user_profile"])
        emotion_equal_count += target.get("emotion_label") == prediction["emotion_label"]
    metrics = profile_metrics(profile_gold, profile_predicted, enums) if profile_gold else None
    recalls = profile_nonempty_recall(profile_gold, profile_predicted) if profile_gold else None
    source_manifest["references_sha256"] = _sha256_file(references_path)
    valid_count = case_status.get("valid", 0)
    total_selected = count
    valid_coverage = valid_count / total_selected if total_selected else 0.0
    full_total = sum(full_seconds)
    fast_total = sum(fast_seconds)
    saving = 1.0 - fast_total / full_total if full_total else None
    exact_prefix_count = sum(bool(row["token_prefix_match"]) for row in comparisons)
    exact_profile_count = sum(bool(row["emotion_and_profile_match"]) for row in comparisons)
    gates = config["adoption_gates"]
    if budget_exhausted:
        status = "budget_exhausted"
    elif case_status.get("mismatch", 0):
        status = "not_adopted"
    elif valid_coverage < float(gates["minimum_valid_coverage"]):
        status = "not_adopted"
    elif saving is None or saving < float(gates["minimum_generation_saving"]):
        status = "not_adopted"
    elif gates.get("require_fast_peak_memory_not_higher") and fast_peak_bytes > full_peak_bytes:
        status = "not_adopted"
    elif len(comparisons) != total_selected:
        status = "not_adopted"
    else:
        status = "passed"

    report = {
        "status": status,
        "created_at_utc": _now(),
        "evaluation_scope": "fixed_dev_unique_conversation_full_vs_profile_prefix_greedy",
        "run_identity": {
            "config_sha256": _sha256_file(fast_config_path or _resolve("configs/evaluation/e03_profile_fast.yaml")),
            "inference_config_path": str(inference_path),
            "inference_config_sha256": _sha256_file(inference_path),
            "source_code_sha256": _sha256_file(Path(__file__).resolve()),
            "prompt_facts": prompt_facts,
            "model_facts": model_facts,
        },
        "source_manifest": source_manifest,
        "sample_count": total_selected,
        "completed_full_count": len(full_seconds),
        "completed_fast_count": len(fast_seconds),
        "valid_fast_count": valid_count,
        "valid_fast_coverage": valid_coverage,
        "exactly_matched_scoring_count": len(predictions),
        "invalid_or_incomplete_count": total_selected - valid_count,
        "case_status_counts": dict(sorted(case_status.items())),
        "consistency": {
            "compared_valid_count": len(comparisons),
            "token_prefix_match_count": exact_prefix_count,
            "emotion_and_profile_match_count": exact_profile_count,
            "all_valid_token_prefixes_match": exact_prefix_count == len(comparisons) and len(comparisons) == valid_count,
            "all_valid_profiles_match": exact_profile_count == len(comparisons) and len(comparisons) == valid_count,
        },
        "profile_fast_metrics": metrics,
        "emotion_accuracy_on_valid_fast_rows": emotion_equal_count / len(predictions) if predictions else None,
        "three_dimension_nonempty_recall": recalls,
        "timing": {
            "budget_seconds_including_model_load": budget_seconds,
            "elapsed_seconds_including_model_load": round(time.monotonic() - started, 6),
            "model_load_seconds": round(load_seconds, 6),
            "full_generation_seconds": round(full_total, 6),
            "fast_generation_seconds": round(fast_total, 6),
            "generation_time_saving_rate": saving,
            "full_generation_sample_seconds": full_seconds,
            "fast_generation_sample_seconds": fast_seconds,
            "full_peak_memory_bytes": full_peak_bytes,
            "fast_peak_memory_bytes": fast_peak_bytes,
        },
        "fallback_used": False,
        "gpu_used": True,
        "references_read_by_generator": False,
        "adoption_gates": gates,
    }
    _write_json(output_path, report)
    return report


def run_gpu_fast_only(
    config: dict[str, Any],
    output_path: Path,
    run_config_path: Path | None = None,
    fast_config_path: Path | None = None,
) -> dict[str, Any]:
    """Generate one profile prefix per selected sample without full-output runs."""
    started = time.monotonic()
    budget_seconds = int(config["gpu_budget"]["max_seconds"])
    deadline = started + budget_seconds
    input_path = _resolve(config["sources"]["input_path"])
    id_map_path = _resolve(config["sources"]["id_map_path"])
    inference_path = _resolve(run_config_path or config["sources"]["inference_config_path"])
    report_identity = {
        "experiment_version": config["experiment_version"],
        "config_path": str(fast_config_path or _resolve("configs/evaluation/e03_profile_fast.yaml")),
        "config_sha256": _sha256_file(fast_config_path or _resolve("configs/evaluation/e03_profile_fast.yaml")),
        "source_code_hashes": {
            "fast_eval_script": _sha256_file(Path(__file__).resolve()),
            "fast_eval_helpers": _sha256_file(ROOT / "src/evaluation/profile_fast_eval.py"),
            "profile_prefix_parser": _sha256_file(ROOT / "src/evaluation/profile_prefix.py"),
            "source_integrity": _sha256_file(ROOT / "src/evaluation/source_integrity.py"),
            "inference_runner": _sha256_file(ROOT / "participant/run_inference.py"),
        },
        "inference_config_path": str(inference_path),
        "inference_config_sha256": _sha256_file(inference_path) if inference_path.is_file() else None,
    }
    progress: dict[str, Any] = {
        "status": "incomplete",
        "created_at_utc": _now(),
        "evaluation_scope": "profile_prefix_candidate_screen_one_generation_per_sample",
        "run_identity": report_identity,
        "selection": None,
        "sample_count": None,
        "attempted_count": 0,
        "completed_attempt_count": 0,
        "accounted_count": 0,
        "gpu_used": False,
        "full_generation_performed": False,
        "response_text_generated_or_saved": False,
        "generation_started": False,
    }
    _write_json(output_path, progress)

    try:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        from participant import run_inference as runner
    except ImportError as exc:
        report = {
            **progress,
            "status": "blocked_environment",
            "reason": f"required_runtime_dependency_missing:{type(exc).__name__}:{exc}",
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        _write_json(output_path, report)
        return report

    try:
        input_rows = _read_jsonl(input_path)
        input_index = validate_target_free_inputs(input_rows)
        id_map_rows = _enrich_id_map_conversations(config, _read_jsonl(id_map_path))
        id_map_index = unique_index(id_map_rows, "eval_id", "ID map")
        selected, selection_facts = _select_profile_samples(config, input_rows, id_map_rows)
        selected_ids = [row["id"] for row in selected]
        selected_maps = [dict(id_map_index[eval_id]) for eval_id in selected_ids]
        history_audit = audit_selected_source_history(config, selected, selected_maps)
        source_manifest = build_source_manifest(
            source_path=history_audit["raw_source_path"],
            source_sha256=history_audit["raw_source_sha256"],
            input_rows=[dict(input_index[eval_id]) for eval_id in selected_ids],
            id_map_rows=selected_maps,
            prompt_version=config["protocol"]["prompt_version"],
            prompt_sha256=config["protocol"]["prompt_sha256"],
            format_version=config["protocol"]["format_version"],
            truncation_version=config["protocol"]["truncation_version"],
            exporter_source_sha256=report_identity["source_code_hashes"]["fast_eval_script"],
            selection_method=selection_facts["method"],
            selection_seed=selection_facts["seed"],
        )
        source_manifest.update({
            "selection": selection_facts,
            "history_audit": history_audit,
            "normalized_dev_path": history_audit["normalized_dev_path"],
            "normalized_dev_sha256": history_audit["normalized_dev_sha256"],
            "generation_input_path": str(input_path),
            "generation_input_sha256": _sha256_file(input_path),
            "id_map_path": str(id_map_path),
            "id_map_sha256": _sha256_file(id_map_path),
        })
        inference_config = _read_json(inference_path)
        inference_config["generation"] = {
            **inference_config["generation"],
            **config["generation"],
        }
        if inference_config.get("backend") != "transformers":
            raise ValueError("gpu-fast-only requires the frozen Transformers runner config")
        if (
            config["protocol"].get("prompt_version") != "P0"
            or inference_config.get("prompt", {}).get("version") != "P0"
            or inference_config.get("prompt", {}).get("sha256") != config["protocol"]["prompt_sha256"]
        ):
            raise ValueError("run config does not match the frozen P0 version and SHA")
        if (
            config["protocol"].get("format_version") != "F2"
            or config["protocol"].get("truncation_version") != "T1"
        ):
            raise ValueError("gpu-fast-only requires the frozen F2/T1 protocol")
        if (
            inference_config["generation"].get("do_sample") is not False
            or inference_config["generation"].get("num_beams", 1) != 1
            or inference_config["generation"].get("num_beam_groups", 1) != 1
            or inference_config["generation"].get("penalty_alpha") not in (None, 0, 0.0)
            or inference_config["generation"].get("temperature", 1.0) != 1.0
            or inference_config["generation"].get("top_p", 1.0) != 1.0
            or inference_config["generation"].get("top_k") not in (None, 0)
        ):
            raise ValueError("gpu-fast-only requires single-beam Greedy decoding")
        if int(inference_config["generation"].get("max_new_tokens", 0)) != int(config["generation"]["max_new_tokens"]):
            raise ValueError("run config max_new_tokens differs from the frozen fast-eval budget")
        prompt_text, _submission_schema, enums, prompt_facts = runner.load_runtime_protocol(inference_config)
        if prompt_facts.get("sha256") != config["protocol"]["prompt_sha256"]:
            raise ValueError("P0 prompt SHA differs from the fast-eval config")
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        report = {
            **progress,
            "status": "preflight_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "gpu_used": False,
        }
        _write_json(output_path, report)
        return report

    progress.update({
        "selection": selection_facts,
        "source_manifest": source_manifest,
        "sample_count": len(selected),
        "run_identity": {
            **report_identity,
            "model_experiment_id": inference_config.get("experiment_id"),
            "prompt_facts": prompt_facts,
            "generation": dict(inference_config["generation"]),
        },
        "gpu_budget_seconds_including_model_load": budget_seconds,
    })
    selected_ids = [row["id"] for row in selected]
    prediction_path = output_path.with_name("profile_predictions.jsonl")
    with prediction_path.open("x", encoding="utf-8", newline="\n"):
        pass
    generation_call_ids: list[str] = []
    prediction_rows_written = 0
    prediction_status_counts: Counter[str] = Counter()
    progress["profile_predictions"] = {
        "path": str(prediction_path),
        "expected_rows": len(selected_ids),
        "rows_written": 0,
        "write_mode": "append_fsync_per_record",
        "contains_only_profile_predictions": True,
        "response_text_saved": False,
    }

    def persist_prediction(attempt: dict[str, Any], index: int) -> None:
        nonlocal prediction_rows_written
        row = _minimal_profile_prediction(attempt)
        if index != prediction_rows_written or row["id"] != selected_ids[index]:
            raise ValueError("profile prediction writes must preserve frozen cohort order")
        _append_jsonl_fsynced(prediction_path, row)
        prediction_rows_written += 1
        prediction_status_counts[row["status"]] += 1
        progress["accounted_count"] = prediction_rows_written
        progress["attempted_count"] = sum(sample_id in generation_call_ids for sample_id in selected_ids[:prediction_rows_written])
        progress["completed_attempt_count"] = sum(
            any(
                previous.get("id") == sample_id and "generation_seconds" in previous
                for previous in attempt_history
            )
            for sample_id in selected_ids[:prediction_rows_written]
        )
        progress["valid_count"] = prediction_status_counts["valid"]
        progress["case_status_counts"] = dict(sorted(prediction_status_counts.items()))
        progress["profile_predictions"]["rows_written"] = prediction_rows_written
        _write_json(output_path, progress)

    attempt_history: list[dict[str, Any]] = []

    def persist_attempt(attempt: dict[str, Any], index: int) -> None:
        attempt_history.append(dict(attempt))
        persist_prediction(attempt, index)

    def persist_unstarted(reason: str) -> None:
        for index, sample in enumerate(selected[prediction_rows_written:], prediction_rows_written):
            persist_attempt(
                {"id": sample["id"], "status": "incomplete", "reason": reason},
                index,
            )

    _write_json(output_path, progress)
    if not torch.cuda.is_available():
        persist_unstarted("cuda_unavailable")
        report = {
            **progress,
            "status": "blocked_environment",
            "reason": "cuda_unavailable",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "profile_predictions": _profile_prediction_artifact(prediction_path, selected_ids, enums),
        }
        _write_json(output_path, report)
        return report

    try:
        model, tokenizer, _model_config, model_facts = _load_model(inference_config, runner, torch)
    except (OSError, RuntimeError, ValueError) as exc:
        persist_unstarted(f"model_load_failed:{type(exc).__name__}")
        report = {
            **progress,
            "status": "blocked_environment",
            "reason": f"model_load_failed:{type(exc).__name__}:{exc}",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "gpu_used": True,
            "profile_predictions": _profile_prediction_artifact(prediction_path, selected_ids, enums),
        }
        _write_json(output_path, report)
        return report
    model_load_seconds = time.monotonic() - started
    if time.monotonic() >= deadline:
        attempts = [
            {"id": sample["id"], "status": "budget_exhausted", "reason": "model_load_exhausted_budget"}
            for sample in selected
        ]
        for index, attempt in enumerate(attempts):
            persist_attempt(attempt, index)
    else:
        generation = dict(inference_config["generation"])
        progress["gpu_used"] = True
        progress["generation_started"] = True
        _write_json(output_path, progress)

        def generate_candidate_once(sample: dict[str, Any]) -> dict[str, Any]:
            if time.monotonic() >= deadline:
                return {"status": "budget_exhausted", "reason": "run_deadline_before_generation"}
            prompt_selection, _messages = runner.prepare_sample_messages(
                sample,
                prompt_text,
                tokenizer,
                inference_config["context"]["max_prompt_tokens"],
            )
            if prompt_selection.get("floor_exceeds_cap"):
                return {
                    "status": "invalid", "reason": "prompt_floor_exceeds_cap",
                    "generated_token_count": 0, "generation_call_count": 0,
                }
            prompt_ids = list(prompt_selection["prompt_ids"])
            input_ids = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=model.get_input_embeddings().weight.device,
            )
            attention_mask = torch.ones_like(input_ids)
            prefix_stopper = ProfilePrefixStopper(
                lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True),
                enums,
                deadline=deadline,
                stop_on_rejection=True,
            )

            class PrefixCriteria(StoppingCriteria):
                def __call__(self, sequences: Any, scores: Any, **kwargs: Any) -> bool:
                    generated = sequences[0, len(prompt_ids):].detach().cpu().tolist()
                    return prefix_stopper.inspect(generated)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            before = time.monotonic()
            generation_call_ids.append(str(sample["id"]))
            try:
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        stopping_criteria=StoppingCriteriaList([PrefixCriteria()]),
                        **generation,
                    )
            except RuntimeError as exc:
                return {
                    "status": "incomplete",
                    "reason": f"generation_error:{type(exc).__name__}:{exc}",
                    "generation_call_count": 1,
                }
            runner.synchronize(torch)
            elapsed = time.monotonic() - before
            generated_ids = [
                int(value)
                for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()
            ]
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            stop_reason = (
                prefix_stopper.stop_reason
                if prefix_stopper.stop_reason is not None
                else "eos"
                if eos_token_id is not None and eos_token_id in generated_ids
                else "token_limit"
                if len(generated_ids) >= int(generation["max_new_tokens"])
                else "generation_ended"
            )
            parsed = prefix_stopper.result or prefix_stopper.finish(stop_reason, generated_ids)
            status = parsed.status
            if parsed.reason == "budget_exhausted":
                status = "budget_exhausted"
            result: dict[str, Any] = {
                "status": status,
                "reason": parsed.reason,
                "stop_reason": stop_reason,
                "generated_token_count": len(generated_ids),
                "generation_call_count": 1,
                "generation_seconds": round(elapsed, 6),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            }
            if parsed.valid:
                result["emotion_label"] = parsed.emotion_label
                result["user_profile"] = dict(parsed.user_profile or {})
                result["prefix_token_ids_sha256"] = hashlib.sha256(
                    json.dumps(generated_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            return result

        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGTERM, _handle_sigterm)
            attempts = run_profile_fast_only_attempts(
                selected,
                generate_candidate_once,
                on_attempt=persist_attempt,
            )
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    if "generation_call_ids" not in locals():
        generation_call_ids = []

    for completed in range(1, len(attempts) + 1):
        current_attempts = attempts[:completed]
        attempted_ids = {
            str(item["id"])
            for item in current_attempts
            if item.get("generation_call_count") == 1
            or str(item.get("id")) in generation_call_ids
        }
        progress["attempted_count"] = len(attempted_ids)
        progress["completed_attempt_count"] = sum("generation_seconds" in item for item in current_attempts)
        progress["accounted_count"] = completed
        progress["case_status_counts"] = dict(sorted(Counter(
            str(item["status"]) for item in current_attempts
        ).items()))
        _write_json(output_path, progress)

    references_path = _resolve(config["sources"]["references_path"])
    references = _read_jsonl(references_path)
    reference_index = unique_index(references, "eval_id", "References")
    if not set(selected_ids) <= set(reference_index):
        report = {
            **progress,
            "status": "preflight_failed",
            "reason": "References do not cover the complete frozen selection",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "gpu_used": True,
        }
        _write_json(output_path, report)
        return report
    ordered_attempts = {attempt["id"]: attempt for attempt in attempts}
    ordered_attempts_list = [ordered_attempts[eval_id] for eval_id in selected_ids]
    gold_profiles: list[dict[str, list[str]]] = []
    emotion_correct = 0
    valid_emotion_correct = 0
    valid_emotion_count = 0
    for eval_id, attempt in zip(selected_ids, ordered_attempts_list):
        target = reference_index[eval_id].get("target")
        if not isinstance(target, dict) or not isinstance(target.get("user_profile"), dict):
            raise ValueError(f"invalid Reference target for {eval_id}")
        gold_profiles.append(target["user_profile"])
        if attempt["status"] == "valid":
            is_correct = target.get("emotion_label") == attempt.get("emotion_label")
            emotion_correct += int(is_correct)
            valid_emotion_correct += int(is_correct)
            valid_emotion_count += 1
    score = score_profile_attempts(gold_profiles, ordered_attempts_list, enums)
    prediction_artifact = _profile_prediction_artifact(prediction_path, selected_ids, enums)
    prediction_validation_error = prediction_artifact["validation_error"]
    source_manifest["references_path"] = str(references_path)
    source_manifest["references_sha256"] = _sha256_file(references_path)
    source_manifest["selected_reference_rows"] = len(selected_ids)
    valid_count = score["status_counts"]["valid"]
    valid_coverage = valid_count / len(selected_ids) if selected_ids else 0.0
    gates = config["adoption_gates"]
    status_counts = score["status_counts"]
    if prediction_validation_error is not None:
        status = "incomplete"
    elif status_counts["budget_exhausted"]:
        status = "budget_exhausted"
    elif status_counts["incomplete"]:
        status = "incomplete"
    elif valid_coverage < float(gates["minimum_valid_coverage"]):
        status = "not_adopted"
    else:
        status = "passed"
    timed_samples = [attempt for attempt in ordered_attempts_list if "generation_seconds" in attempt]
    sample_seconds = [float(attempt["generation_seconds"]) for attempt in timed_samples]
    generation_called_ids = set(generation_call_ids)
    report = {
        **progress,
        "status": status,
        "ended_at_utc": _now(),
        "source_manifest": source_manifest,
        "run_identity": {
            **report_identity,
            "model_experiment_id": inference_config.get("experiment_id"),
            "prompt_facts": prompt_facts,
            "generation": dict(inference_config["generation"]),
            "model_facts": model_facts,
            "base_weight_sha256": model_facts.get("model_weight_identity_sha256"),
            "adapter_weight_sha256": model_facts.get("adapter_weight_sha256"),
            "adapter_config_sha256": model_facts.get("adapter_config_sha256"),
            "tokenizer_facts": model_facts.get("tokenizer_facts"),
        },
        "attempted_count": len(generation_called_ids),
        "completed_attempt_count": len(timed_samples),
        "accounted_count": len(ordered_attempts_list),
        "generation_attempted_sample_count": sum(
            int(attempt["id"] in generation_called_ids) for attempt in ordered_attempts_list
        ),
        "valid_count": valid_count,
        "valid_coverage": valid_coverage,
        "profile_predictions": prediction_artifact,
        "fixed_denominator_profile_metrics": score["profile_metrics"],
        "valid_only_profile_diagnostic": score["valid_only_diagnostic"],
        "case_status_counts": score["status_counts"],
        "failure_reason_counts": score["failure_reason_counts"],
        "emotion_accuracy_fixed_denominator": emotion_correct / len(selected_ids) if selected_ids else 0.0,
        "emotion_accuracy_valid_only": valid_emotion_correct / valid_emotion_count if valid_emotion_count else None,
        "three_dimension_nonempty_recall_fixed_denominator": profile_nonempty_recall(
            gold_profiles,
            [
                attempt.get("user_profile", {"personality_traits": [], "interests": [], "style": []})
                if attempt["status"] == "valid"
                else {"personality_traits": [], "interests": [], "style": []}
                for attempt in ordered_attempts_list
            ],
        ),
        "generation": {
            "one_model_generate_call_per_attempted_sample": True,
            "model_generate_call_count": len(generation_call_ids),
            "full_generation_performed": False,
            "response_text_generated_or_saved": False,
            "max_new_tokens": int(config["generation"]["max_new_tokens"]),
            "do_sample": False,
            "num_beams": 1,
            "generation_seconds": round(sum(sample_seconds), 6),
            "generation_sample_seconds": sample_seconds,
            "peak_memory_bytes": max((int(attempt.get("peak_memory_bytes", 0)) for attempt in timed_samples), default=0),
        },
        "timing": {
            "budget_seconds_including_model_load": budget_seconds,
            "elapsed_seconds_including_model_load": round(time.monotonic() - started, 6),
            "model_load_seconds": round(model_load_seconds, 6),
        },
        "adoption_gates": {
            "minimum_valid_coverage": float(gates["minimum_valid_coverage"]),
            "valid_coverage_gate_passed": valid_coverage >= float(gates["minimum_valid_coverage"]),
            "status_is_execution_coverage_only": True,
            "full_dev_required_for_final_model_choice": True,
        },
        "gpu_used": True,
        "references_read_by_generator": False,
        "submission_written": False,
        "per_sample_attempts": [
            {
                "id": attempt["id"],
                "status": attempt["status"],
                "reason": attempt.get("reason"),
                "stop_reason": attempt.get("stop_reason"),
                "generated_token_count": attempt.get("generated_token_count"),
                "prefix_token_ids_sha256": attempt.get("prefix_token_ids_sha256"),
                "generation_call_count": int(attempt["id"] in generation_called_ids),
            }
            for attempt in ordered_attempts_list
        ],
    }
    _write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("parse-raw", "audit-source", "gpu-compare", "gpu-fast-only"),
        required=True,
    )
    parser.add_argument("--config", default="configs/evaluation/e03_profile_fast.yaml")
    parser.add_argument("--raw", help="completed Raw predictions JSONL for CPU-only prefix audit")
    parser.add_argument("--run-config", help="frozen Transformers inference_config.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root")
    parser.add_argument(
        "--require-passed",
        action="store_true",
        help="return nonzero unless the report status is exactly passed",
    )
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_path = _output_path(config, args.run_id, args.output_root)
    if args.mode == "parse-raw":
        if not args.raw:
            parser.error("--raw is required for parse-raw")
        report = parse_existing_raw(config, _resolve(args.raw), output_path, config_path)
    elif args.mode == "audit-source":
        report = run_source_audit(config, output_path, config_path)
    elif args.mode == "gpu-compare":
        report = run_gpu_comparison(
            config,
            output_path,
            _resolve(args.run_config) if args.run_config else None,
            config_path,
        )
    else:
        report = run_gpu_fast_only(
            config,
            output_path,
            _resolve(args.run_config) if args.run_config else None,
            config_path,
        )
    print(json.dumps({
        "status": report.get("status"),
        "output": str(output_path),
        "sample_count": report.get("sample_count"),
        "valid_fast_count": report.get("valid_fast_count", report.get("valid_count")),
        "valid_parse_rate": report.get("valid_parse_rate", report.get("valid_coverage")),
    }, ensure_ascii=False, sort_keys=True))
    if args.require_passed:
        return 0 if report.get("status") == "passed" else 1
    return 0 if report.get("status") in {
        "cpu_raw_parse_completed", "source_preflight_passed", "passed",
        "not_adopted", "budget_exhausted", "blocked_environment", "incomplete",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
