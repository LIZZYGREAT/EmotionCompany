"""Validate aligned evaluation artifacts and write local proxy metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable

import yaml
from jsonschema import Draft202012Validator

from src.evaluation import (
    jsonl_bytes,
    load_d03_target_validator,
    load_submission_schema,
    read_jsonl,
    schema_enums,
    sha256_file,
    strict_json_loads,
)
from src.evaluation.metrics import emotion_metrics, profile_metrics, raw_diagnostics, response_metrics
from src.evaluation.prepare_dev import adapter_config_identity, source_identity_sha256


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FIELDS = {"eval_id", "sample_id", "target"}
RAW_FIELDS = {
    "id", "raw_text", "status", "prompt_tokens", "generated_tokens", "stop_reason",
    "latency_ms", "removed_history_messages", "backend", "error_type",
}


def _resolve(path: Path | str, base: Path = ROOT) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def _config_sha256(config_path: Path) -> str:
    return sha256_file(config_path)


def _output_fact(manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict) and filename in outputs:
        return outputs[filename]
    if isinstance(outputs, list):
        matches = [item for item in outputs if item.get("path") == filename]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"评测 Manifest 缺少输出指纹: {filename}")


def _assert_file_matches_manifest(path: Path, manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    fact = _output_fact(manifest, filename)
    actual_hash = sha256_file(path)
    if fact.get("sha256") != actual_hash:
        raise ValueError(f"{filename} SHA-256 与评测 Manifest 不符")
    records = fact.get("rows", fact.get("records"))
    if records is not None:
        actual_rows = len(read_jsonl(path))
        if actual_rows != records:
            raise ValueError(f"{filename} 行数与评测 Manifest 不符: expected={records}, actual={actual_rows}")
    return fact


def _validate_target(target: Any, enums: dict[str, list[str]], sample_id: str) -> None:
    if not isinstance(target, dict) or set(target) != {
        "emotion_label", "user_profile", "response_text", "memory_refs"
    }:
        raise ValueError(f"Reference target 字段不符合 D0.3 契约: {sample_id}")
    if target["emotion_label"] not in enums["emotion_label"]:
        raise ValueError(f"Reference target 情绪非法: {sample_id}")
    if not isinstance(target["response_text"], str):
        raise ValueError(f"Reference target response_text 类型无效: {sample_id}")
    profile = target["user_profile"]
    if not isinstance(profile, dict) or set(profile) != {"personality_traits", "interests", "style"}:
        raise ValueError(f"Reference target profile 字段无效: {sample_id}")
    for dimension in ("personality_traits", "interests", "style"):
        values = profile[dimension]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or not set(values) <= set(enums[dimension])
        ):
            raise ValueError(f"Reference target profile 标签无效: {sample_id}: {dimension}")
    memory_refs = target["memory_refs"]
    if (
        not isinstance(memory_refs, list)
        or any(not isinstance(value, str) for value in memory_refs)
        or len(memory_refs) != len(set(memory_refs))
    ):
        raise ValueError(f"Reference target memory_refs 无效: {sample_id}")


def _unique_by_id(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} 第 {index} 行 {key} 缺失或无效")
        if value in mapping:
            raise ValueError(f"{label} 有重复 {key}: {value}")
        mapping[value] = row
    return mapping


def _validate_d03_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    if manifest.get("dataset_scope") != "D0.3_dev":
        return {"status": "not_applicable", "reason": "synthetic_or_non_d03_dataset"}
    source = manifest.get("source", {})
    d03_path = _resolve(source.get("d03_manifest_path", ""))
    if not d03_path.is_file():
        raise FileNotFoundError(f"D0.3 来源 Manifest 缺失: {d03_path}")
    d03_sha = sha256_file(d03_path)
    if d03_sha != source.get("d03_manifest_sha256"):
        raise ValueError("D0.3 来源 Manifest SHA-256 已变化")
    d03 = json.loads(d03_path.read_text(encoding="utf-8"))
    dev_fact = _output_fact(d03, "data/processed/v1/dev.jsonl")
    if dev_fact.get("sha256") != source.get("dev_sha256"):
        raise ValueError("E00 Dev Manifest 与 D0.3 Manifest 来源 hash 不一致")
    return {
        "status": "verified",
        "d03_manifest_path": str(d03_path),
        "d03_manifest_sha256": d03_sha,
        "dev_sha256": dev_fact["sha256"],
        "dev_rows": dev_fact.get("records"),
    }


def _observed_weight_hash_valid(identity: Any) -> bool:
    if not isinstance(identity, dict) or identity.get("identity_status") != "observed":
        return False
    observed_hash = identity.get("observed_weight_hash")
    files = identity.get("files")
    if (
        not isinstance(observed_hash, str)
        or len(observed_hash) != 64
        or any(character not in "0123456789abcdef" for character in observed_hash.lower())
        or not isinstance(files, list)
        or not files
    ):
        return False
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            return False
        path = item.get("path")
        size = item.get("size_bytes")
        file_hash = item.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path in seen_paths
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(file_hash, str)
            or len(file_hash) != 64
            or any(character not in "0123456789abcdef" for character in file_hash.lower())
        ):
            return False
        seen_paths.add(path)
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return aggregate == observed_hash


def _validate_runtime_compatibility(
    run_metadata: dict[str, Any], config: dict[str, Any], dataset_scope: str, *, allow_mock_fixture: bool = False
) -> dict[str, Any]:
    if dataset_scope != "D0.3_dev":
        return {
            "status": "not_applicable_synthetic_fixture",
            "checks": {"backend_is_mock": run_metadata.get("backend") == "mock"},
        }
    if allow_mock_fixture and run_metadata.get("backend") == "mock":
        return {
            "status": "not_applicable_mock_fixture",
            "checks": {"explicit_execution_mode": run_metadata.get("execution_mode") == "mock_fixture"},
        }
    runtime_protocol = run_metadata.get("protocol", {})
    runtime_prompt = runtime_protocol.get("prompt", {})
    runtime_tokenizer = runtime_protocol.get("tokenizer", {})
    expected_protocol = config["protocol"]
    expected_completion_order = ["emotion_label", "user_profile", "response_text", "memory_refs"]
    configured_generation = config["generation"]
    observed_generation = run_metadata.get("generation_config", {})
    checks = {
        "model_id": run_metadata.get("model_id") == config["model"]["model_id"],
        "prompt_sha256": runtime_prompt.get("sha256") == expected_protocol["prompt_sha256"],
        "tokenizer_signature_sha256": runtime_tokenizer.get("signature_sha256") == expected_protocol["tokenizer_signature_sha256"],
        "chat_template_sha256": runtime_tokenizer.get("template_sha256") == expected_protocol["chat_template_sha256"],
        "max_prompt_tokens": run_metadata.get("budget", {}).get("P_max_prompt_tokens") == expected_protocol["max_prompt_tokens"],
        "generation_settings": all(observed_generation.get(key) == value for key, value in configured_generation.items()),
        "history_strategy": runtime_protocol.get("history_strategy") == "T1_recent_contiguous_complete_message_suffix",
        "completion_field_order": runtime_protocol.get("completion_field_order") == expected_completion_order,
        "id_not_in_prompt": runtime_protocol.get("id_in_prompt") is False,
        "backend_is_not_mock": run_metadata.get("backend") != "mock",
    }
    expected_weight_revision = config.get("model", {}).get("weight_revision")
    if expected_weight_revision is not None:
        checks["model_revision"] = run_metadata.get("model_revision") == expected_weight_revision
    if run_metadata.get("backend") == "transformers" and config.get("quantization", {}).get("planned", {}).get("load_in_4bit"):
        observed_quantization = run_metadata.get("quantization", {})
        checks["transformers_nf4_applied"] = (
            observed_quantization.get("applied") is True
            and observed_quantization.get("quant_type") == "nf4"
            and observed_quantization.get("compute_dtype") == "bfloat16"
            and observed_quantization.get("use_double_quant") == config["quantization"]["planned"].get("use_double_quant")
            and observed_quantization.get("real_nf4_verified") is True
            and type(observed_quantization.get("linear4bit_module_count")) is int
            and observed_quantization.get("linear4bit_module_count", 0) > 0
            and observed_quantization.get("device_map_observed") is True
            and not observed_quantization.get("offload_devices")
        )
    if run_metadata.get("backend") == "transformers":
        weight_identity = run_metadata.get("model_weight_identity", {})
        checks["observed_model_weight_hash"] = _observed_weight_hash_valid(weight_identity)
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(f"模型运行协议与当前 E00 配置不一致: {failed_checks}")
    return {"status": "verified", "checks": checks}


def _model_identity_verification(run_metadata: dict[str, Any], dataset_scope: str) -> dict[str, Any]:
    backend = run_metadata.get("backend")
    if dataset_scope != "D0.3_dev":
        return {"status": "not_applicable_synthetic_fixture", "reason": "synthetic_dataset_scope"}
    if backend == "mock":
        return {"status": "not_applicable_mock", "reason": "mock_outputs_do_not_identify_model_weights"}
    identity = run_metadata.get("model_weight_identity")
    if not isinstance(identity, dict):
        return {"status": "unverified", "reason": "model_weight_identity_missing"}
    observed_hash = identity.get("observed_weight_hash")
    if not _observed_weight_hash_valid(identity):
        return {
            "status": "unverified",
            "reason": identity.get("fingerprint_status", "weight_hash_not_observed"),
            "configured_revision": identity.get("configured_revision"),
        }
    return {
        "status": "verified",
        "basis": "SHA-256 of observed local model weight files",
        "observed_weight_hash": observed_hash,
        "configured_revision": identity.get("configured_revision"),
        "revision_status": identity.get("revision_status", "not_configured"),
    }


def _inference_filename(references_path: Path) -> str:
    return "smoke_input.jsonl" if references_path.name == "smoke_references.jsonl" else "inference_input.jsonl"


def _assert_target_free_inference_input(path: Path, expected_ids: set[str]) -> None:
    inputs = read_jsonl(path)
    mapping = _unique_by_id(inputs, "id", "推理输入")
    if set(mapping) != expected_ids:
        raise ValueError("推理输入 ID 集合与 References 不一致")
    for sample_id, row in mapping.items():
        if set(row) != {"id", "history"}:
            raise ValueError(f"推理输入包含非许可字段（可能泄漏真值）: {sample_id}")
        history = row["history"]
        if not isinstance(history, list) or not history:
            raise ValueError(f"推理输入 history 无效: {sample_id}")
        for message in history:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(f"推理输入 history 消息字段无效: {sample_id}")
            if message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str):
                raise ValueError(f"推理输入 history 消息类型无效: {sample_id}")
        if history[-1].get("role") != "user":
            raise ValueError(f"推理输入最后一条消息必须为 user: {sample_id}")


def _case_errors(
    raw_rows: list[dict[str, Any]] | None,
    submission_rows: list[dict[str, Any]],
    schema_validity: list[bool],
    enums: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if raw_rows is None:
        return []
    cases: list[dict[str, Any]] = []
    for raw, submission, is_schema_valid in zip(raw_rows, submission_rows, schema_validity):
        reasons: list[str] = []
        raw_text = raw.get("raw_text")
        raw_object = None
        if isinstance(raw_text, str):
            try:
                value = strict_json_loads(raw_text)
                raw_object = value if isinstance(value, dict) else None
            except (json.JSONDecodeError, ValueError):
                raw_object = None
        if raw_object is None:
            reasons.append("strict_json_parse_failure")
        else:
            if raw_object.get("emotion_label") not in enums["emotion_label"]:
                reasons.append("invalid_emotion")
            profile = raw_object.get("user_profile")
            if not isinstance(profile, dict) or set(profile) != {"personality_traits", "interests", "style"}:
                reasons.append("invalid_profile")
            else:
                for dimension in ("personality_traits", "interests", "style"):
                    labels = profile[dimension]
                    if (
                        not isinstance(labels, list)
                        or any(not isinstance(label, str) for label in labels)
                        or len(labels) != len(set(labels))
                        or not set(labels) <= set(enums[dimension])
                    ):
                        reasons.append("invalid_profile")
                        break
            if not isinstance(raw_object.get("response_text"), str) or not raw_object["response_text"].strip():
                reasons.append("empty_or_invalid_response")
            if raw_object.get("memory_refs") != []:
                reasons.append("memory_refs_not_empty_array")
            if set(raw_object) != {"emotion_label", "user_profile", "response_text", "memory_refs"}:
                reasons.append("raw_fields_missing_or_extra")
        status = raw.get("status")
        if status and status not in {"success"}:
            reasons.append(f"runner_status:{status}")
        if not is_schema_valid:
            reasons.append("submission_schema_invalid")
        if raw.get("stop_reason") in {"max_new_tokens", "server_finish_reason:length", "server_finish_reason:max_tokens"}:
            reasons.append("generation_stopped_at_limit")
        if reasons:
            cases.append({"id": submission["id"], "error_types": sorted(set(reasons))})
    return cases


def _load_train_majority(
    config: dict[str, Any], d03_manifest_path: Path, labels: list[str]
) -> dict[str, Any]:
    train_path = _resolve(config["data"]["train_path"])
    if not train_path.is_file():
        return {"status": "not_run", "reason": "D0.3 Train file missing", "emotion_label": None, "accuracy": None}
    d03 = json.loads(d03_manifest_path.read_text(encoding="utf-8"))
    train_fact = _output_fact(d03, "data/processed/v1/train.jsonl")
    train_hash = sha256_file(train_path)
    if train_hash != train_fact.get("sha256"):
        raise ValueError("D0.3 Train SHA-256 与 Manifest 不符，拒绝计算多数类基线")
    train_rows = read_jsonl(train_path)
    counts = Counter(row["target"]["emotion_label"] for row in train_rows)
    if any(label not in labels for label in counts):
        raise ValueError("D0.3 Train 出现未知情绪标签")
    # Stable schema order resolves ties, and the Dev labels never influence the choice.
    majority = max(labels, key=lambda label: counts.get(label, 0))
    return {
        "status": "computed",
        "training_source": "D0.3 Train only",
        "train_sha256": train_hash,
        "train_sample_count": len(train_rows),
        "emotion_label": majority,
        "train_label_counts": {label: counts.get(label, 0) for label in labels},
        "accuracy": None,
        "label_order_tie_break": "official schema order",
    }


def _current_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_score_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    report_text: str,
    case_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        same_run = (
            previous.get("run_id") == metrics.get("run_id")
            and previous.get("hashes") == metrics.get("hashes")
            and previous.get("experiment_id") == metrics.get("experiment_id")
            and previous.get("scorer_code_commit") == metrics.get("scorer_code_commit")
        )
        if not same_run:
            raise FileExistsError(f"评分输出目录已经属于其他 run，拒绝覆盖: {output_dir}")
    _write_atomic(metrics_path, (json.dumps(metrics, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    _write_atomic(output_dir / "report.md", report_text.encode("utf-8"))
    _write_atomic(output_dir / "case_errors.jsonl", jsonl_bytes(case_rows))


def score_files(
    *,
    config: dict[str, Any],
    config_sha256: str,
    references_path: Path,
    submission_path: Path,
    raw_path: Path | None,
    output_dir: Path,
    id_map_path: Path | None = None,
    manifest_path: Path | None = None,
    run_metadata_path: Path | None = None,
    bert_scorer: Callable[[list[str], list[str]], tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    references_path = references_path.resolve()
    submission_path = submission_path.resolve()
    raw_path = raw_path.resolve() if raw_path else None
    output_dir = output_dir.resolve()
    manifest_path = (manifest_path or references_path.parent / "manifest.json").resolve()
    inference_filename = _inference_filename(references_path)
    references_filename = "references.jsonl" if inference_filename == "inference_input.jsonl" else "smoke_references.jsonl"
    id_map_name = "smoke_id_map.jsonl" if inference_filename == "smoke_input.jsonl" else "id_map.jsonl"
    id_map_path = (id_map_path or references_path.parent / id_map_name).resolve()
    run_metadata_path = (run_metadata_path or submission_path.parent / "performance_report.json").resolve()
    for path, label in (
        (references_path, "References"), (submission_path, "Submission"),
        (manifest_path, "evaluation Manifest"), (id_map_path, "ID map"),
        (run_metadata_path, "performance report"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 文件缺失: {path}")
    if raw_path is not None and not raw_path.is_file():
        raise FileNotFoundError(f"Raw predictions 文件缺失: {raw_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    d03_facts = _validate_d03_manifest(manifest, manifest_path)
    official_target_validator = None
    official_target_schema_facts = None
    if d03_facts.get("status") == "verified":
        official_target_validator, official_target_schema_facts = load_d03_target_validator(
            Path(d03_facts["d03_manifest_path"]), ROOT
        )
        recorded_target_schema = manifest.get("source", {}).get("target_schema")
        if recorded_target_schema and recorded_target_schema.get("sha256") != official_target_schema_facts["sha256"]:
            raise ValueError("评测 Manifest target Schema 指纹与 D0.3 Schema 不一致")
    schema = load_submission_schema(ROOT)
    enums = schema_enums(schema)
    validator = Draft202012Validator(schema)
    references = read_jsonl(references_path)
    submissions = read_jsonl(submission_path)
    id_maps = read_jsonl(id_map_path)
    reference_by_id = _unique_by_id(references, "eval_id", "References")
    submission_by_id = _unique_by_id(submissions, "id", "Submission")
    id_map_by_id = _unique_by_id(id_maps, "eval_id", "ID map")
    expected_ids = set(reference_by_id)
    if not expected_ids:
        raise ValueError("References 必须至少有一条样本")
    if set(submission_by_id) != expected_ids:
        missing = sorted(expected_ids - set(submission_by_id))[:5]
        extra = sorted(set(submission_by_id) - expected_ids)[:5]
        raise ValueError(f"Submission/Reference ID 集合不一致: missing={missing}, extra={extra}")
    if set(id_map_by_id) != expected_ids:
        raise ValueError("ID map/Reference ID 集合不一致")

    for eval_id, reference in reference_by_id.items():
        if set(reference) != REFERENCE_FIELDS:
            raise ValueError(f"Reference 字段不符合隔离契约: {eval_id}")
        id_map = id_map_by_id[eval_id]
        if id_map.get("sample_id") != reference.get("sample_id"):
            raise ValueError(f"Reference sample_id 与 ID map 不一致: {eval_id}")
        if not isinstance(reference.get("sample_id"), str) or not reference["sample_id"]:
            raise ValueError(f"Reference sample_id 缺失: {eval_id}")
        _validate_target(reference.get("target"), enums, eval_id)
        if official_target_validator is not None:
            schema_errors = list(official_target_validator.iter_errors(reference["target"]))
            if schema_errors:
                raise ValueError(
                    f"Reference target 不符合 D0.3 Train Schema: {eval_id}: "
                    f"{schema_errors[0].message}"
                )
        prediction = submission_by_id[eval_id]
        if prediction.get("id") != eval_id:
            raise ValueError(f"Submission id 与对齐 ID 不一致: {eval_id}")

    references_fact = _output_fact(manifest, references_filename)
    if sha256_file(references_path) != references_fact.get("sha256"):
        raise ValueError("References 文件 SHA-256 与 Manifest 不符")
    reference_rows_count = references_fact.get("rows", references_fact.get("records"))
    if reference_rows_count is not None and reference_rows_count != len(references):
        raise ValueError("References 行数与 Manifest 不符")
    id_map_fact = _assert_file_matches_manifest(id_map_path, manifest, id_map_name)
    if sha256_file(id_map_path) != id_map_fact.get("sha256"):
        raise ValueError("ID map SHA-256 与 Manifest 不符")
    input_fact = _output_fact(manifest, inference_filename)
    inference_path = manifest_path.parent / input_fact.get("path", inference_filename)
    if not inference_path.is_file():
        raise FileNotFoundError(f"Manifest 指定的推理输入缺失: {inference_path}")
    _assert_file_matches_manifest(inference_path, manifest, inference_filename)
    _assert_target_free_inference_input(inference_path, expected_ids)
    if manifest.get("dataset_scope") == "D0.3_dev":
        if manifest.get("source_identity_sha256") != source_identity_sha256(manifest.get("source", {})):
            raise ValueError("评测 Manifest 的 D0.3 数据来源指纹不一致")
        expected_dev_count = int(config["data"]["expected_dev_count"])
        if manifest.get("source", {}).get("dev_rows") != d03_facts.get("dev_rows"):
            raise ValueError("E00 Manifest source.dev_rows 与 D0.3 Manifest 行数不一致")
        if manifest.get("source", {}).get("dev_rows") != expected_dev_count:
            raise ValueError(
                f"D0.3 来源 Dev 数量必须为 {expected_dev_count}: "
                f"source={manifest.get('source', {}).get('dev_rows')}"
            )
        expected_adapter, expected_adapter_sha = adapter_config_identity(config)
        if (
            manifest.get("adapter_config_sha256") != expected_adapter_sha
            or manifest.get("adapter_config") != expected_adapter
        ):
            raise ValueError("评测数据适配条件与当前配置不一致（Dev 数量、ID 规则或 Smoke seed/count）")
        if inference_filename == "smoke_input.jsonl":
            selection = manifest.get("smoke_selection", {})
            smoke_ids = selection.get("eval_ids")
            expected_smoke_count = min(int(config["smoke"]["count"]), expected_dev_count)
            if (
                not isinstance(smoke_ids, list)
                or any(not isinstance(sample_id, str) or not sample_id for sample_id in smoke_ids)
                or len(smoke_ids) != len(set(smoke_ids))
                or selection.get("actual_count") != expected_smoke_count
                or selection.get("requested_count") != int(config["smoke"]["count"])
                or set(smoke_ids) != expected_ids
            ):
                raise ValueError("Smoke References ID 集合与 Manifest smoke_selection 不一致")
        elif len(expected_ids) != expected_dev_count:
            raise ValueError(
                f"完整 D0.3 Dev 数量必须为 {expected_dev_count}: references={len(expected_ids)}"
            )
    current_config_hash = config_sha256

    with submission_path.open("rb") as handle:
        submission_hash = hashlib.sha256(handle.read()).hexdigest()
    with references_path.open("rb") as handle:
        references_hash = hashlib.sha256(handle.read()).hexdigest()
    raw_rows_by_id: dict[str, dict[str, Any]] | None = None
    raw_hash = None
    if raw_path is not None:
        raw_rows = read_jsonl(raw_path)
        raw_rows_by_id = _unique_by_id(raw_rows, "id", "Raw predictions")
        if set(raw_rows_by_id) != expected_ids:
            raise ValueError("Raw predictions/Reference ID 集合不一致")
        for eval_id, row in raw_rows_by_id.items():
            if set(row) != RAW_FIELDS:
                raise ValueError(f"Raw prediction 字段不符合 allowlist: {eval_id}")
            if "target" in row or "sample_id" in row:
                raise ValueError(f"Raw prediction 不得包含 target/sample_id: {eval_id}")
            if row["raw_text"] is not None and not isinstance(row["raw_text"], str):
                raise ValueError(f"Raw prediction raw_text 类型无效: {eval_id}")
            if row["backend"] not in {"transformers", "openai_compatible", "mock"}:
                raise ValueError(f"Raw prediction backend 无效: {eval_id}")
        raw_hash = sha256_file(raw_path)

    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if run_metadata.get("test_file_sha256") != input_fact.get("sha256"):
        raise ValueError("模型运行报告的输入 hash 与评测 Manifest 不一致")
    if run_metadata.get("submission_sha256") != submission_hash:
        raise ValueError("模型运行报告的 submission hash 不匹配")
    if run_metadata.get("samples", run_metadata.get("submission_rows")) != len(expected_ids):
        raise ValueError("模型运行报告样本数与所选评测集合不一致")
    if raw_path is not None and run_metadata.get("raw_predictions_sha256") != raw_hash:
        raise ValueError("模型运行报告的 raw hash 不匹配")
    if run_metadata.get("run_id") is None:
        raise ValueError("模型运行报告缺少 run_id，拒绝混合不明来源预测")
    run_experiment_id = run_metadata.get("experiment_id")
    if not isinstance(run_experiment_id, str) or not run_experiment_id.strip() or run_experiment_id == "unspecified":
        raise ValueError("模型运行报告缺少具名 experiment_id")
    execution_mode = run_metadata.get(
        "execution_mode", "mock_fixture" if run_metadata.get("backend") == "mock" else "model_inference"
    )
    if execution_mode == "mock_fixture" and run_metadata.get("backend") != "mock":
        raise ValueError("execution_mode=mock_fixture 必须使用 backend=mock")
    is_mock_fixture = execution_mode == "mock_fixture" and run_metadata.get("backend") == "mock"
    runtime_compatibility = _validate_runtime_compatibility(
        run_metadata,
        config,
        manifest.get("dataset_scope", "unknown"),
        allow_mock_fixture=is_mock_fixture,
    )

    ordered_ids = list(reference_by_id)
    gold_targets = [reference_by_id[eval_id]["target"] for eval_id in ordered_ids]
    predicted_rows = [submission_by_id[eval_id] for eval_id in ordered_ids]
    schema_validity = []
    schema_errors: list[str] = []
    for eval_id, prediction in zip(ordered_ids, predicted_rows):
        errors = sorted(validator.iter_errors(prediction), key=lambda item: list(item.path))
        schema_validity.append(not errors)
        if errors:
            schema_errors.extend(f"{eval_id}: {error.message}" for error in errors)
    if schema_errors:
        raise ValueError("Submission 含不符合官方 Schema 的输出: " + "; ".join(schema_errors[:10]))

    gold_emotions = [target["emotion_label"] for target in gold_targets]
    predicted_emotions = [row["emotion_label"] for row in predicted_rows]
    gold_profiles = [target["user_profile"] for target in gold_targets]
    predicted_profiles = [row["user_profile"] for row in predicted_rows]
    gold_responses = [target["response_text"] for target in gold_targets]
    predicted_responses = [row["response_text"] for row in predicted_rows]
    emotion_result = emotion_metrics(gold_emotions, predicted_emotions, enums["emotion_label"])
    profile_result = profile_metrics(gold_profiles, predicted_profiles, enums)
    response_result = response_metrics(
        gold_responses,
        predicted_responses,
        config["scoring"],
        bert_scorer=bert_scorer,
        workspace_root=ROOT,
    )

    raw_rows_ordered = [raw_rows_by_id[eval_id] for eval_id in ordered_ids] if raw_rows_by_id is not None else None
    raw_result = (
        raw_diagnostics(
            raw_rows_ordered or [], enums, predicted_rows, schema_validity,
            generation_cap=config.get("generation", {}).get("max_new_tokens"),
        )
        if raw_rows_ordered is not None else None
    )
    case_rows = _case_errors(raw_rows_ordered, predicted_rows, schema_validity, enums)
    majority_baseline = None
    if manifest.get("dataset_scope") == "D0.3_dev":
        d03_manifest_path = _resolve(manifest["source"]["d03_manifest_path"])
        majority_baseline = _load_train_majority(config, d03_manifest_path, enums["emotion_label"])
        if majority_baseline["status"] == "computed":
            majority_baseline["accuracy"] = sum(
                majority_baseline["emotion_label"] == target["emotion_label"] for target in gold_targets
            ) / len(gold_targets)

    raw_latencies = [row.get("latency_ms") for row in raw_rows_ordered or []]
    raw_latencies = [float(value) for value in raw_latencies if isinstance(value, (int, float)) and not isinstance(value, bool)]
    performance = {
        "backend": run_metadata.get("backend"),
        "run_id": run_metadata["run_id"],
        "execution_mode": execution_mode,
        "hardware_label": run_metadata.get("hardware_label"),
        "timing_scope": run_metadata.get("timing_scope"),
        "official_performance_rounds": run_metadata.get("rounds"),
        "official_performance_complete": run_metadata.get("complete"),
        "official_average_latency_ms": run_metadata.get("average_latency_ms"),
        "official_median_latency_ms": run_metadata.get("median_latency_ms"),
        "official_p95_latency_ms": run_metadata.get("p95_latency_ms"),
        "all_scored_samples_latency_count": len(raw_latencies),
        "all_scored_samples_average_latency_ms": fmean(raw_latencies) if raw_latencies else None,
        "all_scored_samples_median_latency_ms": median(raw_latencies) if raw_latencies else None,
    }
    metrics = {
        "metrics_scope": "local_proxy",
        "dataset_scope": manifest.get("dataset_scope", "unknown"),
        "experiment_id": run_experiment_id,
        "evaluation_manifest_experiment_id": manifest.get("experiment_id"),
        "run_id": run_metadata["run_id"],
        "execution_mode": execution_mode,
        "sample_count": len(expected_ids),
        "denominators": {
            "emotion": len(expected_ids),
            "profile": len(expected_ids),
            "response": len(expected_ids),
            "raw_diagnostics": len(expected_ids) if raw_rows_ordered is not None else None,
        },
        "hashes": {
            "evaluation_manifest_sha256": sha256_file(manifest_path),
            "d03_manifest_sha256": manifest.get("source", {}).get("d03_manifest_sha256"),
            "d03_dev_sha256": manifest.get("source", {}).get("dev_sha256"),
            "inference_input_sha256": input_fact.get("sha256"),
            "references_sha256": references_hash,
            "id_map_sha256": sha256_file(id_map_path),
            "submission_sha256": submission_hash,
            "raw_predictions_sha256": raw_hash,
            "evaluation_config_sha256": current_config_hash,
        },
        "source_verification": d03_facts,
        "runtime_compatibility": runtime_compatibility,
        "model_identity_verification": _model_identity_verification(
            run_metadata, manifest.get("dataset_scope", "unknown")
        ),
        "model": {
            "model_id": run_metadata.get("model_id"),
            "model_revision": run_metadata.get("model_revision"),
            "model_weight_identity": run_metadata.get("model_weight_identity"),
            "quantization": run_metadata.get("quantization"),
            "generation_config": run_metadata.get("generation_config"),
            "protocol": run_metadata.get("protocol"),
        },
        "emotion": emotion_result,
        "profile": profile_result,
        "response": response_result,
        "raw_diagnostics": raw_result,
        "simple_majority_emotion_baseline": majority_baseline,
        "performance": performance,
        "limitations": [
            "All task metrics are local proxies until the official scoring implementations are verified.",
            "No local submetric is converted into official competition points.",
            *(["Mock fixture predictions are pipeline checks, not model quality evidence."] if is_mock_fixture else []),
        ],
        "scorer_code_commit": _current_commit(),
    }
    identity = metrics["model_identity_verification"]
    identity_summary = identity.get("observed_weight_hash") or identity.get("reason") or identity.get("basis", "not available")
    majority_summary = (
        f"{majority_baseline['status']}, label={majority_baseline['emotion_label']}, accuracy={majority_baseline['accuracy']}"
        if majority_baseline is not None else "not applicable"
    )
    report_lines = [
        "# E00 本地评测报告",
        "",
        f"- 范围：`{metrics['dataset_scope']}`；样本数：{metrics['sample_count']}。",
        f"- 实验 / 运行：`{metrics['experiment_id']}` / `{metrics['run_id']}`；后端：`{performance['backend']}`。",
        f"- 执行模式：`{execution_mode}`。",
        f"- 模型权重身份：`{identity['status']}`（{identity_summary}）；配置 revision 不代表已核验 revision。",
        f"- 范围标记：`{metrics['metrics_scope']}`；不映射为官方总分。",
        f"- 情绪 Accuracy：{emotion_result['accuracy']:.6f}；固定 16 类 Macro-F1：{emotion_result['macro_f1']:.6f}。",
        f"- 画像三维同时 Exact Match：{profile_result['all_three_dimensions_exact_match']:.6f}；33 标签 Micro-F1：{profile_result['overall_33_label_micro_f1']:.6f}。",
        f"- 回复 Corpus BLEU-4：{response_result['local_corpus_bleu4']['score_percent']:.6f}；字符级 ROUGE-L：{response_result['local_rouge_l_macro_f1']['value']:.6f}。",
        f"- BERTScore：`{response_result['bert_score']['status']}`（{response_result['bert_score']['reason'] or 'computed'}）。",
        f"- BERTScore F1：{response_result['bert_score']['f1'] if response_result['bert_score']['f1'] is not None else 'null'}。",
        f"- 原始 JSON 合法率：{raw_result['raw_json_parse_rate']:.6f}；原始 Completion Schema 合法率：{raw_result['raw_schema_valid_rate']:.6f}." if raw_result else "- 未提供 Raw predictions；不计算原始生成合规率。",
        f"- Submission Schema 合法率：{(raw_result['submission_schema_valid_rate'] if raw_result else 1.0):.6f}。",
        f"- 计时范围：{performance['timing_scope'] or '未提供'}。",
        f"- D0.3 来源校验：`{d03_facts['status']}`。",
        f"- Train 多数类基线：`{majority_summary}`。",
        "",
        "情绪及画像完整逐类数据、运行指纹和错误案例见 `metrics.json` 与 `case_errors.jsonl`。",
    ]
    report_text = "\n".join(report_lines) + "\n"
    _write_score_outputs(output_dir, metrics, report_text, case_rows)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对齐 E00 预测并计算可追溯的本地指标")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--id-map", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    metrics = score_files(
        config=config,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        references_path=args.references,
        submission_path=args.submission,
        raw_path=args.raw,
        output_dir=args.output_dir,
        id_map_path=args.id_map,
        manifest_path=args.manifest,
        run_metadata_path=args.run_metadata,
    )
    print(
        f"PASS: scored {metrics['sample_count']} rows ({metrics['dataset_scope']}); "
        f"local accuracy={metrics['emotion']['accuracy']:.6f}; output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
