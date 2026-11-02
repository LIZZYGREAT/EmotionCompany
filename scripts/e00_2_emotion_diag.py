#!/usr/bin/env python3
"""Single-field E00.2-D emotion diagnosis; run never reads references or gold labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference
from src.evaluation import load_submission_schema, read_jsonl, schema_enums, sha256_file
from src.evaluation.metrics import emotion_metrics
from src.shared.prompting import build_messages, encode_inference_prompt, select_recent_complete_suffix

RUNS_ROOT = ROOT / "outputs/evaluation/e00_2/emotion_diag"
BASELINE_WEIGHT_SHA256 = "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
BASELINE_INPUT_SHA256 = "093bc078ddd7e3cc0a874e0fac5b8c987ef80e88852098fdcd67a98b8353a84c"
BASELINE_REFERENCE_SHA256 = "cc65f1faf7d74b0710af7217da3bcbb7fa13fe47816ea93209b12d7a5dd05f8f"
BASELINE_RAW_SHA256 = "07e8c30105afd2267f52d56bd902f8970c562fbeb5832cf119ca135ee5388ba6"
BASELINE_PERFORMANCE_SHA256 = "94923c30b0d05b3cd1c3f829ba1e315a4f9dd63f77388d901789c6ab4d2698e2"
BASELINE_MANIFEST_SHA256 = "3aaf08f4f0a5cac14bbec5c79c4b8f3f76adf18f2ed989572198a51b5197c379"
BASELINE_SMOKE_ID_MAP_SHA256 = "2ce02478cc161e160dc27a305f34ad44354f18fe658cd307c216529ecb5a2090"
BASELINE_D042_BUDGET_SHA256 = "6b0811244bf19d5099f857eda379b61719c530015e45aceafbf3cc93e88d083e"
BASELINE_PROMPT_SHA256 = "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2"
BASELINE_TOKENIZER_SHA256 = "1662c9e99c6185441026a168b2dd6b09af103dc6dc3563b72f598411f14d1913"
BASELINE_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} JSON 根节点必须为 object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} YAML 根节点必须为 object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def _index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        item_key = row.get(key)
        if not isinstance(item_key, str) or not item_key:
            raise ValueError(f"{label} 第 {row_number} 行缺少非空字符串 {key}")
        if item_key in result:
            raise ValueError(f"{label} 存在重复 {key}: {item_key}")
        result[item_key] = row
    return result


def _require_ids(expected: set[str], observed: dict[str, Any], label: str) -> None:
    actual = set(observed)
    if expected != actual:
        raise ValueError(
            f"{label} ID 集合不一致: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 缺失: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 与归档不符: expected={expected}, actual={actual}")
    return actual


def _official_emotions() -> list[str]:
    return schema_enums(load_submission_schema(ROOT))["emotion_label"]


def _validate_diag_prompt(prompt: str, labels: list[str]) -> None:
    enum_lines = re.findall(r"(?m)^\s*([a-z_]+(?:,\s*[a-z_]+)+)\s*$", prompt)
    if not enum_lines or [item.strip() for item in enum_lines[0].split(",")] != labels:
        raise ValueError("Pdiag-E1 枚举及顺序必须与官方提交 Schema 的 16 类情绪一致")
    if "只输出一个英文标签本身" not in prompt:
        raise ValueError("Pdiag-E1 必须要求只输出一个英文标签")


def validate_single_label(raw_text: Any, labels: list[str], stop_reason: str | None = None) -> tuple[str | None, str | None]:
    """Accept only a label after outer whitespace trimming; never extract or map labels."""
    if not isinstance(raw_text, str):
        return None, "raw_text_missing_or_non_string"
    stripped = raw_text.strip()
    if stripped in labels:
        return stripped, None
    if not stripped:
        return None, "empty_output"
    if stop_reason in {"max_new_tokens", "server_finish_reason:length", "server_finish_reason:max_tokens"}:
        return None, "truncated_at_generation_cap"
    if stripped.startswith("```"):
        return None, "code_fence"
    if stripped.startswith(("{", "[")):
        return None, "json_or_container_output"
    if (stripped.startswith("\"") and stripped.endswith("\"")) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return None, "quoted_label"
    matches = [label for label in labels if re.search(rf"(?<![a-z_]){re.escape(label)}(?![a-z_])", stripped)]
    if len(matches) > 1:
        return None, "multiple_labels_or_explanation"
    if matches:
        return None, "label_with_extra_content"
    return None, "non_enumerated_output"


def validate_unlabeled_inputs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = _index(rows, "id", "smoke input")
    for sample_id, row in indexed.items():
        if set(row) != {"id", "history"}:
            raise ValueError(f"推理输入只能包含 id/history，拒绝可能包含标签的字段: {sample_id}")
        history = row["history"]
        if not isinstance(history, list) or not history:
            raise ValueError(f"{sample_id} history 必须为非空数组")
    return indexed


def validate_model_identity(identity: dict[str, Any], expected_sha256: str) -> None:
    observed = identity.get("aggregate_sha256") or identity.get("observed_weight_hash")
    if identity.get("identity_status") != "observed" or observed != expected_sha256:
        raise ValueError(f"实际模型权重身份不符: expected={expected_sha256}, observed={observed}")


def validate_quantization_runtime(facts: dict[str, Any], cuda_available: bool) -> None:
    if not cuda_available or not facts.get("cuda_available"):
        raise RuntimeError("CUDA 不可用；拒绝执行 NF4 诊断")
    if not facts.get("real_nf4_verified") or not facts.get("placement_verified"):
        raise RuntimeError("真实 NF4/BF16/CUDA 设备放置未通过验收")
    if facts.get("offload_devices") or facts.get("accelerate_offload_hook_modules"):
        raise RuntimeError("模型存在 CPU/disk offload，拒绝开始生成")


def confirm_fixed_histories(
    input_rows: list[dict[str, Any]],
    perf_by_id: dict[str, dict[str, Any]],
    budget_by_sample_id: dict[str, dict[str, Any]],
    sample_by_id: dict[str, str],
    tokenizer: Any,
    p0_text: str,
    diag_prompt: str,
    prompt_cap: int,
) -> dict[str, dict[str, Any]]:
    inputs = validate_unlabeled_inputs(input_rows)
    expected_ids = set(inputs)
    _require_ids(expected_ids, perf_by_id, "E00.2 A0 Greedy performance")
    _require_ids(expected_ids, sample_by_id, "E00.1 Smoke ID map")
    confirmed: dict[str, dict[str, Any]] = {}
    for sample_id, row in inputs.items():
        history = row["history"]
        old = perf_by_id[sample_id]
        sample_ref = sample_by_id[sample_id]
        budget = budget_by_sample_id.get(sample_ref)
        if budget is None:
            raise ValueError(f"D0.4.2 预算账本缺少 Smoke 样本: {sample_id}")
        selection = select_recent_complete_suffix(history, p0_text, prompt_cap, tokenizer)
        if selection["status"] != "ok":
            raise ValueError(f"P0/T1 选择失败: {sample_id} ({selection['status']})")
        original = len(history)
        retained = len(selection["selected_history"])
        removed = original - retained
        comparisons = {
            "original_history_messages": original,
            "retained_history_messages": retained,
            "removed_history_messages": removed,
            "client_prompt_tokens": selection["prompt_tokens"],
        }
        for key, value in comparisons.items():
            if old.get(key) != value:
                raise ValueError(f"E00.2 A0 Greedy {key} 不一致: {sample_id} expected={old.get(key)} actual={value}")
        budget_comparisons = {
            "original_history_message_count": original,
            "retained_history_message_count": retained,
            "removed_history_message_count": removed,
            "selected_prompt_tokens": selection["prompt_tokens"],
        }
        for key, value in budget_comparisons.items():
            if budget.get(key) != value:
                raise ValueError(f"D0.4.2 {key} 不一致: {sample_id} expected={budget.get(key)} actual={value}")
        # Re-encode the confirmed A0 history with Pdiag; do not call T1 with the shorter prompt.
        diag_ids = encode_inference_prompt(tokenizer, build_messages(selection["selected_history"], diag_prompt))
        if len(diag_ids) > prompt_cap:
            raise ValueError(f"Pdiag-E1 超出 {prompt_cap} Token 预算: {sample_id} ({len(diag_ids)})")
        confirmed[sample_id] = {
            "selected_history": selection["selected_history"],
            "a0_prompt_tokens": selection["prompt_tokens"],
            "diag_prompt_tokens": len(diag_ids),
            "original_history_messages": original,
            "retained_history_messages": retained,
            "removed_history_messages": removed,
            "history_was_clipped": removed > 0,
        }
    return confirmed


def load_preflight(config: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]:
    """Validate immutable E00.2 inputs and reconstruct the exact P0/T1-selected histories."""
    labels = _official_emotions()
    prompt_path = _resolve(config["prompt_path"])
    prompt_bytes = prompt_path.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    _validate_diag_prompt(prompt, labels)
    expected_prompt_sha = config.get("prompt_sha256")
    if expected_prompt_sha and prompt_sha != expected_prompt_sha:
        raise ValueError("Pdiag-E1 SHA-256 与配置冻结值不一致")

    baseline_path = _resolve(config["baseline_manifest_path"])
    _require_hash(baseline_path, BASELINE_MANIFEST_SHA256, "E00.2 run manifest")
    baseline = _read_json(baseline_path)
    if baseline.get("run_id") != config["baseline_run_id"] or baseline.get("status") != "real_smoke_scored":
        raise ValueError("E00.2 A0 Greedy 归档 Manifest 的 run_id/status 不符合预期")
    baseline_greedy = baseline["runs"]["greedy"]
    if baseline["source_files"]["smoke_input"]["sha256"] != BASELINE_INPUT_SHA256:
        raise ValueError("E00.2 Smoke input SHA 与本次方案的归档指纹不符")
    if baseline["source_files"]["smoke_references"]["sha256"] != BASELINE_REFERENCE_SHA256:
        raise ValueError("E00.2 References SHA 与本次方案的归档指纹不符")
    if baseline_greedy["raw_predictions_sha256"] != BASELINE_RAW_SHA256:
        raise ValueError("E00.2 Greedy Raw SHA 与本次方案的归档指纹不符")
    weight_sha = baseline_greedy["model_weight_identity"]["aggregate_sha256"]
    if weight_sha != config["expected_model_weight_sha256"] or weight_sha != BASELINE_WEIGHT_SHA256:
        raise ValueError("E00.2 权重 SHA 与本次方案的归档指纹不符")

    input_path = _resolve(config["input_path"])
    input_sha = _require_hash(input_path, BASELINE_INPUT_SHA256, "Smoke input")
    evaluation_manifest_path = _resolve(config["evaluation_manifest_path"])
    evaluation_manifest_sha = _require_hash(
        evaluation_manifest_path,
        baseline["source_files"]["evaluation_manifest"]["sha256"],
        "E00.1 evaluation manifest",
    )
    model_path = _resolve(config["model_path"])
    expected_revision = config["expected_model_revision"]
    if expected_revision not in str(model_path.resolve()):
        raise ValueError(f"本地模型挂载未指向归档 snapshot revision: {model_path}")
    index_path = model_path / "model.safetensors.index.json"
    archived_index_sha = baseline["model_mount"]["weight_index_sha256"]
    _require_hash(index_path, archived_index_sha, "模型权重 index")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError("本地模型挂载缺少 config.json")
    weight_index = _read_json(index_path)
    indexed_shards = sorted(set(weight_index.get("weight_map", {}).values()))
    archived_shards = sorted(baseline["model_mount"]["weight_shards"])
    if indexed_shards != archived_shards:
        raise ValueError("模型权重 index 分片列表与 E00.2 归档不一致")
    for shard_name in indexed_shards:
        shard_path = model_path / shard_name
        if not shard_path.is_file() or shard_path.stat().st_size <= 0:
            raise FileNotFoundError(f"模型权重分片缺失或为空: {shard_name}")

    inference_config_path = _resolve(config["inference_config_path"])
    inference_config = run_inference.load_config(inference_config_path)
    inference_prompt_path = _resolve(inference_config["prompt"]["path"])
    if _require_hash(inference_prompt_path, BASELINE_PROMPT_SHA256, "冻结 P0") != BASELINE_PROMPT_SHA256:
        raise ValueError("冻结 P0 指纹不符")
    tokenizer_config = inference_config["tokenizer"]
    if tokenizer_config.get("expected_signature_sha256") != BASELINE_TOKENIZER_SHA256:
        raise ValueError("Tokenizer 签名预期值与 E00.2 归档不符")
    if tokenizer_config.get("expected_template_sha256") != BASELINE_TEMPLATE_SHA256:
        raise ValueError("Chat Template 指纹预期值与 E00.2 归档不符")
    local_tokenizer = tokenizer
    tokenizer_facts: dict[str, Any]
    if local_tokenizer is None:
        local_tokenizer, tokenizer_facts = run_inference.load_runtime_tokenizer(inference_config, model_path)
    else:
        _, tokenizer_facts = run_inference.load_runtime_tokenizer(inference_config, model_path)
    if tokenizer_facts.get("signature_sha256") != BASELINE_TOKENIZER_SHA256:
        raise ValueError("加载出的 Tokenizer 签名与 E00.2 归档不符")

    input_rows = read_jsonl(input_path)
    inputs = validate_unlabeled_inputs(input_rows)
    if len(inputs) != config["expected_sample_count"]:
        raise ValueError(f"Smoke 样本数必须为 {config['expected_sample_count']}，实际为 {len(inputs)}")
    perf_path = _resolve(baseline_greedy["performance_path"])
    perf = _read_json(perf_path)
    _require_hash(_resolve(baseline_greedy["raw_path"]), BASELINE_RAW_SHA256, "E00.2 Greedy Raw")
    _require_hash(_resolve(baseline_greedy["prediction_path"]), baseline_greedy["submission_sha256"], "E00.2 Greedy Submission")
    _require_hash(perf_path, BASELINE_PERFORMANCE_SHA256, "E00.2 Greedy performance")
    if perf.get("test_file_sha256") != input_sha:
        raise ValueError("E00.2 Greedy performance input SHA 与 Smoke input 不一致")
    if perf.get("model_weight_identity", {}).get("aggregate_sha256") != BASELINE_WEIGHT_SHA256:
        raise ValueError("E00.2 Greedy performance 权重指纹不符")
    perf_rows = perf.get("sample_diagnostics")
    if not isinstance(perf_rows, list) or len(perf_rows) != len(inputs):
        raise ValueError("E00.2 Greedy performance diagnostics 必须恰好有 20 条")
    perf_by_id = _index(perf_rows, "id", "E00.2 Greedy performance")
    _require_ids(set(inputs), perf_by_id, "E00.2 Greedy performance")

    evaluation_manifest = _read_json(evaluation_manifest_path)
    id_map_path = _resolve(config["id_map_path"])
    id_map_fact = evaluation_manifest.get("outputs", {}).get("smoke_id_map.jsonl")
    if not isinstance(id_map_fact, dict):
        raise ValueError("E00.1 Manifest 缺少 smoke_id_map.jsonl 指纹")
    _require_hash(id_map_path, id_map_fact["sha256"], "E00.1 Smoke ID map")
    if id_map_fact["sha256"] != BASELINE_SMOKE_ID_MAP_SHA256:
        raise ValueError("E00.1 Smoke ID map 指纹与当前方案归档不符")
    id_map_rows = read_jsonl(id_map_path)
    id_map_by_id = _index(id_map_rows, "eval_id", "Smoke ID map")
    _require_ids(set(inputs), id_map_by_id, "Smoke ID map")
    sample_by_id = {sample_id: id_map_by_id[sample_id]["sample_id"] for sample_id in inputs}
    d042_manifest_path = _resolve(config["d042_manifest_path"])
    d042_manifest = _read_json(d042_manifest_path)
    d042 = _resolve(config["d042_budget_path"])
    if d042_manifest.get("audit_version") != "D0.4.2-v1":
        raise ValueError("D0.4.2 Manifest 版本不符合已归档审计")
    d042_fact = d042_manifest.get("result_files", {}).get("budget_simulations")
    if not isinstance(d042_fact, dict) or d042_fact.get("sha256") != BASELINE_D042_BUDGET_SHA256:
        raise ValueError("D0.4.2 budget ledger 指纹与当前方案归档不符")
    _require_hash(d042, d042_fact["sha256"], "D0.4.2 budget ledger")
    wanted = set(sample_by_id.values())
    budget_by_sample: dict[str, dict[str, Any]] = {}
    with d042.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("split") == "dev"
                and record.get("max_train_seq_length") == 2048
                and record.get("max_prompt_tokens") == config["prompt_budget"]
                and record.get("sample_id") in wanted
            ):
                sample_ref = record["sample_id"]
                if sample_ref in budget_by_sample:
                    raise ValueError(f"D0.4.2 D=2048/P={config['prompt_budget']} 预算记录重复: {sample_ref}")
                budget_by_sample[sample_ref] = record
    if set(budget_by_sample) != wanted:
        raise ValueError(f"D0.4.2 预算账本 Smoke ID 缺失: {sorted(wanted - set(budget_by_sample))}")

    histories = confirm_fixed_histories(
        input_rows,
        perf_by_id,
        budget_by_sample,
        sample_by_id,
        local_tokenizer,
        inference_prompt_path.read_text(encoding="utf-8"),
        prompt,
        config["prompt_budget"],
    )
    clipped = sum(item["history_was_clipped"] for item in histories.values())
    if clipped != 4:
        raise ValueError(f"归档 E00.2 应恰有 4 条历史被裁剪，实际为 {clipped}")
    submission_rows = read_jsonl(_resolve(baseline_greedy["prediction_path"]))
    baseline_predictions = _index(submission_rows, "id", "E00.2 Greedy Submission")
    _require_ids(set(inputs), baseline_predictions, "E00.2 Greedy Submission")
    labels = set(_official_emotions())
    baseline_emotions = {}
    for sample_id, row in baseline_predictions.items():
        label = row.get("emotion_label")
        if label not in labels and label != "":
            raise ValueError(f"E00.2 Greedy Submission 情绪值异常: {sample_id}")
        baseline_emotions[sample_id] = label if label in labels else None
    return {
        "config": config,
        "labels": _official_emotions(),
        "prompt": prompt,
        "prompt_sha256": prompt_sha,
        "prompt_path": prompt_path,
        "input_path": input_path,
        "input_sha256": input_sha,
        "input_rows": input_rows,
        "histories": histories,
        "baseline": baseline,
        "baseline_path": baseline_path,
        "baseline_manifest_sha256": sha256_file(baseline_path),
        "baseline_predictions": baseline_predictions,
        "baseline_emotions": baseline_emotions,
        "evaluation_manifest_sha256": evaluation_manifest_sha,
        "d042_manifest_sha256": sha256_file(d042_manifest_path),
        "tokenizer_facts": tokenizer_facts,
        "tokenizer": local_tokenizer,
        "model_path": model_path,
        "model_index_sha256": archived_index_sha,
        "inference_config": inference_config,
        "references_expected_sha256": baseline["source_files"]["smoke_references"]["sha256"],
    }


def _outcome(label: str | None, gold: str, labels: set[str]) -> str:
    if label not in labels:
        return "invalid_or_missing"
    return "correct" if label == gold else "valid_wrong"


def score_prediction_rows(
    raw_rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    baseline_predictions: dict[str, dict[str, Any]],
    history_facts: dict[str, dict[str, Any]],
    labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_by_id = _index(raw_rows, "id", "diagnostic Raw")
    refs_by_id = _index(references, "eval_id", "Smoke References")
    ids = set(baseline_predictions)
    _require_ids(ids, raw_by_id, "diagnostic Raw")
    _require_ids(ids, refs_by_id, "Smoke References")
    _require_ids(ids, history_facts, "confirmed history metadata")
    label_set = set(labels)
    gold: list[str] = []
    predicted_for_metrics: list[str] = []
    cases: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    stop_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    paired_counts: Counter[str] = Counter()
    group_rows: dict[str, list[dict[str, Any]]] = {"clipped": [], "not_clipped": []}
    for sample_id in sorted(ids):
        reference = refs_by_id[sample_id]
        target = reference.get("target")
        if not isinstance(target, dict) or target.get("emotion_label") not in label_set:
            raise ValueError(f"Smoke Reference Gold emotion 不符合官方枚举: {sample_id}")
        gold_label = target["emotion_label"]
        raw_row = raw_by_id[sample_id]
        stop_reason = raw_row.get("stop_reason")
        prediction, error_type = validate_single_label(raw_row.get("raw_text"), labels, stop_reason)
        if raw_row.get("status") not in {"success", "completed"} and prediction is None:
            error_type = "generation_failed"
        valid = prediction is not None
        correct = prediction == gold_label
        old_label = baseline_predictions[sample_id].get("emotion_label")
        old_label = old_label if old_label in label_set else None
        old_correct = old_label == gold_label if old_label is not None else None
        facts = history_facts[sample_id]
        stop_counts[str(stop_reason or "missing")] += 1
        if error_type:
            error_counts[error_type] += 1
        if prediction:
            prediction_counts[prediction] += 1
        paired_counts[f"{_outcome(old_label, gold_label, label_set)}->{_outcome(prediction, gold_label, label_set)}"] += 1
        case = {
            "id": sample_id,
            "gold_label": gold_label,
            "predicted_label": prediction,
            "prediction_status": "valid" if valid else "invalid",
            "error_type": error_type,
            "stop_reason": stop_reason,
            "truncated": stop_reason in {"max_new_tokens", "server_finish_reason:length", "server_finish_reason:max_tokens"},
            "correct": bool(correct),
            "a0_greedy_label": old_label,
            "a0_greedy_label_exists": old_label is not None,
            "a0_greedy_correct": old_correct,
            "history_was_clipped": facts["history_was_clipped"],
            "a0_prompt_tokens": facts["a0_prompt_tokens"],
            "diag_prompt_tokens": facts["diag_prompt_tokens"],
            "retained_history_messages": facts["retained_history_messages"],
        }
        cases.append(case)
        group_rows["clipped" if facts["history_was_clipped"] else "not_clipped"].append(case)
        gold.append(gold_label)
        predicted_for_metrics.append(prediction if prediction is not None else "__invalid_or_empty_prediction__")

    metrics = emotion_metrics(gold, predicted_for_metrics, labels)
    valid_count = sum(case["prediction_status"] == "valid" for case in cases)
    correct_count = sum(case["correct"] for case in cases)
    metrics.update({
        "strict_label_valid_count": valid_count,
        "strict_label_valid_rate": valid_count / len(cases) if cases else None,
        "correct_count": correct_count,
        "accuracy": correct_count / len(cases) if cases else None,
        "accuracy_when_strictly_valid": correct_count / valid_count if valid_count else None,
        "invalid_count": len(cases) - valid_count,
        "format_error_counts": dict(sorted(error_counts.items())),
        "stop_reason_counts": dict(sorted(stop_counts.items())),
        "generation_cap_count": sum(case["truncated"] for case in cases),
        "prediction_label_counts": {label: prediction_counts.get(label, 0) for label in labels},
        "a0_greedy_paired_transitions": dict(sorted(paired_counts.items())),
        "history_groups": {
            name: {
                "sample_count": len(group),
                "strict_label_valid_count": sum(row["prediction_status"] == "valid" for row in group),
                "strict_label_valid_rate": (
                    sum(row["prediction_status"] == "valid" for row in group) / len(group) if group else None
                ),
                "correct_count": sum(row["correct"] for row in group),
                "accuracy": sum(row["correct"] for row in group) / len(group) if group else None,
            }
            for name, group in group_rows.items()
        },
    })
    return metrics, cases


def _train_majority(config: dict[str, Any], labels: list[str]) -> tuple[str, Counter[str], str]:
    train_path = _resolve(config["train_path"])
    d03_path = _resolve(config["d03_manifest_path"])
    d03 = _read_json(d03_path)
    train_fact = next((item for item in d03.get("outputs", []) if item.get("path") == config["train_path"]), None)
    if not train_fact:
        raise ValueError("D0.3 Manifest 缺少 Train 输出指纹")
    train_sha = _require_hash(train_path, train_fact["sha256"], "D0.3 Train")
    counts = Counter(row["target"]["emotion_label"] for row in read_jsonl(train_path))
    if any(label not in labels for label in counts):
        raise ValueError("D0.3 Train 存在官方 Schema 外的情绪标签")
    majority = max(labels, key=lambda label: counts.get(label, 0))
    if majority != config["majority_reference_label"]:
        raise ValueError(f"D0.3 Train 多数类不是预先记录的 {config['majority_reference_label']}: {majority}")
    return majority, counts, train_sha


def _render_report(metrics: dict[str, Any], run_manifest: dict[str, Any], majority_accuracy: int | None) -> str:
    a0_valid_count = sum(
        count for outcome, count in metrics["a0_greedy_paired_transitions"].items()
        if outcome.split("->", 1)[0] != "invalid_or_missing"
    )
    a0_correct_count = sum(
        count for outcome, count in metrics["a0_greedy_paired_transitions"].items()
        if outcome.split("->", 1)[0] == "correct"
    )
    correct_delta = metrics["correct_count"] - a0_correct_count
    if abs(correct_delta) <= 2:
        decision = "`inconclusive_small_n`：相对 A0 Greedy 的正确数变化不超过 2 条，不据此判断方案优劣。"
    elif metrics["strict_label_valid_count"] > a0_valid_count and metrics["correct_count"] > a0_correct_count:
        decision = "合法单标签数和正确数都高于 A0 Greedy；这提示单字段任务值得进一步诊断，但不能区分 Prompt、输出格式与移除其他字段的贡献。若仍需定位原因，可另设单字段最小 JSON 版本。"
    elif metrics["strict_label_valid_count"] > a0_valid_count:
        decision = "合法率高于 A0 Greedy，但正确数没有明确改善；后续优先核查标签语义、类别支持和 E01 SFT，不先修改 A0 Parser。"
    else:
        decision = "严格合法率没有高于 A0 Greedy；先检查 Raw、EOS/64 Token 上限及 Chat Template 记录，只有发现可复现实现错误才调整代码。"
    if metrics["generation_cap_count"] >= 2:
        decision += " 本轮有多条样本触及生成上限，分类差异还与截断混杂。"
    lines = [
        "# E00.2-D 单字段情绪识别诊断报告",
        "",
        f"- Run ID：`{run_manifest['run_id']}`；运行状态：`{run_manifest['status']}`。",
        f"- 数据：20 条 E00.1 Dev Smoke；输入 SHA-256：`{run_manifest['input_sha256']}`。",
        f"- 模型权重 SHA-256：`{run_manifest.get('model_weight_sha256') or '未加载'}`；Pdiag-E1 SHA-256：`{run_manifest['prompt_sha256']}`。",
        f"- 严格合法单标签：{metrics['strict_label_valid_count']}/{metrics['sample_count']}（{metrics['strict_label_valid_rate']:.3f}）。",
        f"- Accuracy：{metrics['correct_count']}/{metrics['sample_count']}（{metrics['accuracy']:.3f}）；合法输出子集准确率：{metrics['accuracy_when_strictly_valid']}。",
        f"- 固定 16 类 Macro-F1：{metrics['macro_f1']:.6f}；生成达到 64 Token 上限：{metrics['generation_cap_count']}。",
        f"- Train 多数类基线：`anxiety`，本 Smoke 正确数：{majority_accuracy}/20。",
        f"- A0 Greedy 对齐参照：严格有效情绪标签 {a0_valid_count}/20，正确 {a0_correct_count}/20。",
        f"- A0 Greedy 同 ID 配对转移：`{json.dumps(metrics['a0_greedy_paired_transitions'], ensure_ascii=False, sort_keys=True)}`。",
        f"- 原 A0 历史裁剪分组：`{json.dumps(metrics['history_groups'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 标签支持与预测分布",
        "",
        "| 情绪 | Gold Support | 预测数 | TP | FP | FN | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, fact in metrics["per_label"].items():
        lines.append(
            f"| `{label}` | {fact['support']} | {metrics['prediction_label_counts'][label]} | "
            f"{fact['tp']} | {fact['fp']} | {fact['fn']} | {fact['f1']:.4f} |"
        )
    lines.extend([
        "",
        "## 严格格式错误与停止原因",
        "",
        f"- 格式错误：`{json.dumps(metrics['format_error_counts'], ensure_ascii=False, sort_keys=True)}`。",
        f"- 停止原因：`{json.dumps(metrics['stop_reason_counts'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 解释边界与下一步",
        "",
        "20 条是分层 Smoke，样本小且不代表总体分布；不据此作模型排名或因果结论。Pdiag-E1 同时改变了任务说明和输出契约，因此结果不能区分去掉画像/回复、去掉 JSON 与 Prompt 文案的影响。",
        "",
        decision,
        "",
        "后续实验仅作建议，本脚本不会自动启动训练、Prompt 迭代或新的模型运行。",
        "",
    ])
    return "\n".join(lines)


def _effective_snapshot(config: dict[str, Any], context: dict[str, Any]) -> str:
    snapshot = {
        **config,
        "resolved": {
            "input_sha256": context["input_sha256"],
            "references_expected_sha256": context["references_expected_sha256"],
            "baseline_manifest_sha256": context["baseline_manifest_sha256"],
            "evaluation_manifest_sha256": context["evaluation_manifest_sha256"],
            "d042_manifest_sha256": context["d042_manifest_sha256"],
            "model_mount": str(context["model_path"]),
            "model_index_sha256": context["model_index_sha256"],
            "prompt_sha256": context["prompt_sha256"],
            "tokenizer_signature_sha256": context["tokenizer_facts"]["signature_sha256"],
            "chat_template_sha256": context["tokenizer_facts"]["template_sha256"],
        },
    }
    return yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False)


def _safe_run_dir(path_value: str | Path, run_id: str, root: Path | None = None) -> Path:
    root = root or RUNS_ROOT
    path = _resolve(path_value)
    expected = (root / run_id).resolve()
    if path != expected:
        raise ValueError(f"诊断输出必须严格位于 {expected}")
    return path


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def run_diagnostic(config_path: Path, run_id: str, output_dir: Path) -> int:
    if not re.fullmatch(r"E00\.2-D-emotion-[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id 必须使用 E00.2-D-emotion- 前缀并只含安全字符")
    output_dir = _safe_run_dir(output_dir, run_id)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    config = _read_yaml(config_path)
    tokenizer = None
    context = load_preflight(config, tokenizer)
    _validate_diag_prompt(context["prompt"], context["labels"])
    output_dir.mkdir(parents=True)
    (output_dir / "effective_config.yaml").write_text(_effective_snapshot(config, context), encoding="utf-8")
    sample_metadata = {
        sample_id: {key: value for key, value in facts.items() if key != "selected_history"}
        for sample_id, facts in context["histories"].items()
    }
    base_manifest: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "run_id": run_id,
        "status": "preflight_passed",
        "input_path": str(context["input_path"]),
        "input_sha256": context["input_sha256"],
        "input_rows": len(context["input_rows"]),
        "references_path": str(_resolve(config["references_path"])),
        "references_sha256_expected": context["references_expected_sha256"],
        "baseline_run_id": config["baseline_run_id"],
        "baseline_manifest_sha256": context["baseline_manifest_sha256"],
        "baseline_greedy_raw_sha256": BASELINE_RAW_SHA256,
        "baseline_greedy_weight_sha256": BASELINE_WEIGHT_SHA256,
        "baseline_p0_sha256": BASELINE_PROMPT_SHA256,
        "prompt_version": config["prompt_version"],
        "prompt_sha256": context["prompt_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "tokenizer": context["tokenizer_facts"],
        "model_mount": str(context["model_path"]),
        "model_index_sha256": context["model_index_sha256"],
        "expected_model_weight_sha256": config["expected_model_weight_sha256"],
        "generation": config["generation"],
        "history_policy": config["history_policy"],
        "prompt_budget": config["prompt_budget"],
        "confirmed_history_summary": {
            "sample_count": len(sample_metadata),
            "clipped_count": sum(item["history_was_clipped"] for item in sample_metadata.values()),
            "samples": sample_metadata,
        },
    }
    if not _cuda_available():
        base_manifest.update({
            "status": "blocked_in_codex_env",
            "blocked_reason": "torch.cuda.is_available() is false; real NF4 model inference was not attempted",
            "cuda_available": False,
            "model_weight_sha256": None,
        })
        _write_json(output_dir / "manifest.json", base_manifest)
        (output_dir / "runner.log").write_text(
            "BLOCKED_IN_CODEX_ENV: torch.cuda.is_available() is false; no model was loaded and no predictions were produced.\n",
            encoding="utf-8",
        )
        return 2

    try:
        import torch
        from transformers import AutoModelForCausalLM

        model_config = copy.deepcopy(context["inference_config"])
        model_config["model_path"] = str(context["model_path"])
        model_config["model_id"] = config["model_reference"]
        model_config["model_revision"] = config["expected_model_revision"]
        model_config["quantization"] = {
            "enabled": True,
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "use_double_quant": True,
        }
        run_inference.require_cuda_for_quantization(model_config, torch)
        weight_identity = run_inference._weight_identity(context["model_path"], model_config)
        validate_model_identity(weight_identity, config["expected_model_weight_sha256"])
        quantization_config = run_inference.build_quantization_config(model_config, torch)
        model = AutoModelForCausalLM.from_pretrained(
            context["model_path"],
            trust_remote_code=False,
            local_files_only=True,
            device_map="auto",
            torch_dtype=model_config.get("torch_dtype", "auto"),
            quantization_config=quantization_config,
        )
        model.eval()
        quant_facts = run_inference.inspect_quantization_runtime(model, model_config, torch)
        validate_quantization_runtime(quant_facts, torch.cuda.is_available())
        rows: list[dict[str, Any]] = []
        for sample in context["input_rows"]:
            sample_id = sample["id"]
            facts = context["histories"][sample_id]
            prompt_ids = encode_inference_prompt(
                context["tokenizer"], build_messages(facts["selected_history"], context["prompt"])
            )
            try:
                run_inference.synchronize(torch)
                started = time.perf_counter()
                raw_text, details = run_inference.generate_transformers_sample(
                    model,
                    torch,
                    context["tokenizer"],
                    prompt_ids,
                    config["generation"],
                )
                run_inference.synchronize(torch)
                rows.append({
                    "id": sample_id,
                    "raw_text": raw_text,
                    "status": "success",
                    "prompt_tokens": len(prompt_ids),
                    **details,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error_type": None,
                })
            except Exception as exc:  # Keep the input ID and preserve the fixed denominator.
                rows.append({
                    "id": sample_id,
                    "raw_text": None,
                    "status": "generation_failed",
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": 0,
                    "stop_reason": "generation_failed",
                    "latency_ms": None,
                    "error_type": f"{type(exc).__name__}: {exc}",
                })
        _write_jsonl(output_dir / "raw_emotion_predictions.jsonl", rows)
        status = "real_run_completed" if len(rows) == len(context["input_rows"]) else "incomplete"
        base_manifest.update({
            "status": status,
            "cuda_available": True,
            "model_weight_sha256": weight_identity["aggregate_sha256"],
            "model_weight_identity": weight_identity,
            "quantization": quant_facts,
            "raw_predictions_sha256": sha256_file(output_dir / "raw_emotion_predictions.jsonl"),
            "raw_predictions_rows": len(rows),
            "generation_failure_count": sum(row["status"] != "success" for row in rows),
        })
        _write_json(output_dir / "manifest.json", base_manifest)
        (output_dir / "runner.log").write_text(
            f"{status}: rows={len(rows)} failures={base_manifest['generation_failure_count']}\n",
            encoding="utf-8",
        )
        return 0 if status == "real_run_completed" else 2
    except Exception as exc:
        base_manifest.update({
            "status": "failed_before_generation",
            "cuda_available": True,
            "failure": f"{type(exc).__name__}: {exc}",
        })
        _write_json(output_dir / "manifest.json", base_manifest)
        (output_dir / "runner.log").write_text(f"FAILED: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 2


def score_run(run_dir: Path) -> int:
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(RUNS_ROOT.resolve()):
        raise ValueError(f"诊断目录必须位于 {RUNS_ROOT}")
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "real_run_completed":
        raise ValueError(f"只有完整真实运行可评分，当前状态为 {manifest.get('status')}")
    config = _read_yaml(run_dir / "effective_config.yaml")
    # The effective snapshot nests runtime facts, but its root configuration remains the source of truth.
    original_config = {key: value for key, value in config.items() if key != "resolved"}
    if run_dir.name != manifest.get("run_id"):
        raise ValueError("诊断目录名与 Manifest run_id 不一致")
    context = load_preflight(original_config)
    if context["prompt_sha256"] != manifest["prompt_sha256"]:
        raise ValueError("评分时 Pdiag-E1 指纹与运行 Manifest 不符")
    if context["input_sha256"] != manifest["input_sha256"]:
        raise ValueError("评分时 Smoke input 指纹与运行 Manifest 不符")
    if context["baseline_manifest_sha256"] != manifest["baseline_manifest_sha256"]:
        raise ValueError("评分时 E00.2 Manifest 指纹与运行记录不一致")
    if manifest.get("model_weight_sha256") != original_config["expected_model_weight_sha256"]:
        raise ValueError("诊断模型权重 SHA 与 E00.2 归档不一致")
    raw_path = run_dir / "raw_emotion_predictions.jsonl"
    raw_sha = _require_hash(raw_path, manifest["raw_predictions_sha256"], "诊断 Raw")
    raw_rows = read_jsonl(raw_path)
    refs_path = _resolve(original_config["references_path"])
    _require_hash(refs_path, manifest["references_sha256_expected"], "Smoke References")
    references = read_jsonl(refs_path)
    histories = {
        sample_id: {key: value for key, value in facts.items() if key != "selected_history"}
        for sample_id, facts in context["histories"].items()
    }
    stored_histories = manifest["confirmed_history_summary"]["samples"]
    if histories != stored_histories:
        raise ValueError("评分时重建的 A0 历史/Prompt Token 元数据与运行 Manifest 不一致")
    metrics, cases = score_prediction_rows(
        raw_rows, references, context["baseline_predictions"], histories, _official_emotions()
    )
    majority, train_counts, train_sha = _train_majority(original_config, _official_emotions())
    majority_count = sum(row["target"]["emotion_label"] == majority for row in references)
    if majority_count != 2:
        raise ValueError(f"预先记录的 Train 多数类 {majority} 在该 Smoke 应为 2/20，实际为 {majority_count}/{len(references)}")
    metrics["train_majority_baseline"] = {
        "status": "computed_from_D0.3_Train_only",
        "label": majority,
        "train_sha256": train_sha,
        "train_sample_count": sum(train_counts.values()),
        "train_label_counts": {label: train_counts.get(label, 0) for label in _official_emotions()},
        "smoke_correct_count": majority_count,
        "smoke_accuracy": majority_count / len(references),
    }
    metrics["run_id"] = manifest["run_id"]
    metrics["execution_scope"] = "diagnostic_only_not_official_baseline"
    manifest["score_status"] = "scored"
    manifest["score_artifacts"] = {
        "raw_predictions_sha256": raw_sha,
        "metrics_sha256": None,
        "per_case_sha256": None,
        "report_sha256": None,
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_jsonl(run_dir / "per_case.jsonl", cases)
    report_text = _render_report(metrics, manifest, majority_count)
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")
    manifest["score_artifacts"].update({
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        "per_case_sha256": sha256_file(run_dir / "per_case.jsonl"),
        "report_sha256": sha256_file(run_dir / "report.md"),
    })
    _write_json(manifest_path, manifest)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run Pdiag-E1 without reading References/Gold")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    score = subparsers.add_parser("score", help="score a completed run; this command reads References/Gold")
    score.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        return run_diagnostic(args.config.resolve(), args.run_id, args.output_dir)
    return score_run(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
