#!/usr/bin/env python3
"""Full-Dev inference and paired E03-A1 versus E01-B comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e01_b_evaluate as e01
from src.evaluation import read_jsonl, sha256_file, schema_enums, load_submission_schema
from src.training.training_sft import canonical_sha256


_ORIGINAL_E01_CODE_IDENTITY = e01._code_identity()
_ORIGINAL_E01_TRAINING_RUN = e01._training_run
_ORIGINAL_E01_SNAPSHOTS = e01._snapshots
_ORIGINAL_E01_CONFIG = e01._config
_PRE_FIX_SCORE_SHA256 = "c8b7d92c22c579cda47df1030647a87ba9ebcad30c08dc57de4b4e71b62d3808"
_PRE_RECOVERY_E03_EVALUATOR_SHA256 = "421df95b58a197f9325b6f344be7881c68c4ecd4351461fd329dc1d8295cb3ba"
_PRE_FAILURE_CATEGORY_RUNNER_SHA256 = "59082f5f4d7a996dad7b2c7935b9a0b44432297ce5c795432ae903b08d286630"


def _e03_code_identity() -> dict[str, Any]:
    identity = copy.deepcopy(_ORIGINAL_E01_CODE_IDENTITY)
    identity["files"]["scripts/e03_a1_evaluate.py"] = sha256_file(ROOT / "scripts/e03_a1_evaluate.py")
    return identity


def _training_run(path: Path, scope: str, expected_base: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = path.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training Run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E03-A1":
        return _ORIGINAL_E01_TRAINING_RUN(path, scope, expected_base)
    if scope != "full-dev" or not root.is_relative_to((ROOT / "outputs/training").resolve()):
        raise ValueError("E03-A1 generation evaluation accepts only Full-stage Runs under outputs/training")
    snapshot_path = root / "config_snapshot.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if canonical_sha256(snapshot) != manifest.get("config_sha256"):
        raise ValueError("E03-A1 frozen training config identity changed")
    if (
        manifest.get("run_id") != root.name
        or manifest.get("status") != "passed"
        or manifest.get("training_scope") != "full_contrast"
        or manifest.get("train_rows") != 16192
        or manifest.get("optimizer_steps") != 4048
        or manifest.get("loss_variant") not in {"profile_x2", "label_value_x2"}
        or manifest.get("init_adapter") is not None
        or manifest.get("base_weights_unchanged") is not True
        or manifest.get("base_weight_sha256") != expected_base
        or manifest.get("reload_check_status") != "passed"
    ):
        raise ValueError("E03-A1 evaluation requires a passed, independently initialized Full Run with verified reload/Base identity")
    short_gate = manifest.get("short_stage_gate") or {}
    if short_gate.get("status") != "passed" or short_gate.get("candidate_variant") != manifest.get("loss_variant"):
        raise ValueError("E03-A1 Full Run has no matching passed short-stage loss gate")
    identity, adapter_config = e01.runner.adapter_weight_identity(root / "adapter")
    if any(identity.get(key) != (manifest.get("adapter_identity") or {}).get(key) for key in ("weight_sha256", "config_sha256")):
        raise ValueError("E03-A1 Adapter identity differs from the completed training Run")
    e01.runner.validate_adapter_base_reference(adapter_config, "Qwen/Qwen3-1.7B", (ROOT / "participant/model").resolve())
    return {
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "experiment_id": "E03-A1",
        "training_scope": "full_contrast",
        "loss_variant": manifest["loss_variant"],
        "loss_mapping_sha256": manifest.get("loss_mapping_sha256"),
    }, identity


def _config(scope: str) -> dict[str, Any]:
    config = _ORIGINAL_E01_CONFIG(scope)
    if scope == "full-dev":
        config["execution"]["output_root"] = "outputs/evaluation/e03_a1/full"
    return config


def _snapshots(config: dict[str, Any], scope: str, model: str, training_root: Path | None, hardware_label: str):
    inference_config, evaluation_config, training = _ORIGINAL_E01_SNAPSHOTS(config, scope, model, training_root, hardware_label)
    if training and training.get("experiment_id") == "E03-A1":
        inference_config["experiment_id"] = "E03-A1"
        evaluation_config["experiment_id"] = "E03-A1"
    return inference_config, evaluation_config, training


def _current_run_identity(training_root: Path, run_id: str, hardware_label: str) -> dict[str, Any]:
    config = e01._config("full-dev")
    dataset = e01._dataset(config, "full-dev")
    inference_config, evaluation_config, training = e01._snapshots(
        config, "full-dev", "adapter", training_root, hardware_label
    )
    inference_bytes = (json.dumps(inference_config, ensure_ascii=False, indent=2) + "\n").encode()
    evaluation_bytes = yaml.safe_dump(evaluation_config, allow_unicode=True, sort_keys=False).encode()
    return {
        "run_id": run_id,
        "scope": "full-dev",
        "model": "adapter",
        "dataset": dataset,
        "training_run": training,
        "source_code": _e03_code_identity(),
        "inference_config_sha256": hashlib.sha256(inference_bytes).hexdigest(),
        "evaluation_config_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
    }


def _is_known_pre_fix_identity(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_files = previous.get("files")
    current_files = current.get("files")
    if previous.get("packages") != current.get("packages") or not isinstance(previous_files, dict):
        return False
    if set(previous_files) != set(current_files or {}):
        return False
    allowed_previous = {
        "src/evaluation/score.py": _PRE_FIX_SCORE_SHA256,
        "scripts/e03_a1_evaluate.py": _PRE_RECOVERY_E03_EVALUATOR_SHA256,
    }
    differences = {name for name in previous_files if previous_files[name] != current_files[name]}
    return bool(differences) and all(
        name in allowed_previous and previous_files[name] == allowed_previous[name]
        for name in differences
    )


def _recover_score_only_resume(
    training_root: Path, run_id: str, hardware_label: str
) -> None:
    config = e01._config("full-dev")
    root = e01._resolve(config["execution"]["output_root"]) / run_id
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return
    record = e01._json(manifest_path)
    if record.get("status") != "failed":
        return
    error = record.get("error")
    if not isinstance(error, str) or not error.startswith(
        "ValueError: Raw prediction 字段不符合 allowlist:"
    ):
        return

    current_identity = _current_run_identity(training_root, run_id, hardware_label)
    if any(
        record.get(key) != value
        for key, value in current_identity.items()
        if key != "source_code"
    ):
        raise ValueError("score-only recovery rejected: evaluation, data or Adapter identity changed")
    if not _is_known_pre_fix_identity(record.get("source_code", {}), current_identity["source_code"]):
        raise ValueError("score-only recovery rejected: source-code change is not the known scorer fix")

    report = e01._json(root / "runner/performance_report.json")
    raw_path = root / "raw_predictions.jsonl"
    submission_path = root / "runner/submission.jsonl"
    if (
        report.get("complete") is not True
        or report.get("run_id") != run_id
        or report.get("samples") != 1833
        or report.get("raw_predictions_rows") != 1833
        or report.get("raw_predictions_sha256") != sha256_file(raw_path)
        or report.get("submission_sha256") != sha256_file(submission_path)
    ):
        raise ValueError("score-only recovery rejected: complete inference artifacts or hashes are missing")
    raw_rows = read_jsonl(raw_path)
    expected_fields = {
        "id", "raw_text", "status", "prompt_tokens", "generated_tokens", "stop_reason",
        "latency_ms", "removed_history_messages", "backend", "error_type", "failure_category",
    }
    raw_by_id = e01._unique_by_id(raw_rows, "id", "Raw predictions")
    if len(raw_by_id) != 1833 or any(set(row) != expected_fields for row in raw_rows):
        raise ValueError("score-only recovery rejected: raw prediction rows do not match the expected new schema")
    dataset_root = Path(current_identity["dataset"]["root"])
    input_ids = {row["id"] for row in read_jsonl(dataset_root / current_identity["dataset"]["input"])}
    if set(raw_by_id) != input_ids:
        raise ValueError("score-only recovery rejected: raw prediction IDs differ from the frozen Dev set")

    record["scoring_recovery"] = {
        "reason": "rescore after accepting optional failure_category in raw telemetry",
        "inference_artifacts_reused": True,
        "previous_error": error,
        "previous_source_code": record["source_code"],
        "validated_raw_predictions_sha256": report["raw_predictions_sha256"],
        "validated_submission_sha256": report["submission_sha256"],
        "recovered_at_utc": e01._now(),
    }
    record["source_code"] = current_identity["source_code"]
    record.pop("error", None)
    runner = e01.runner
    runner._write_json_atomic(manifest_path, record)


def _install_e03_hooks() -> None:
    e01._code_identity = _e03_code_identity
    e01._training_run = _training_run
    e01._config = _config
    e01._snapshots = _snapshots


def _load_scored(
    root: Path,
    *,
    expected_code: dict[str, Any],
    legacy_source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    record = e01._json(root / "run_manifest.json")
    if record.get("status") != "scored" or record.get("model") != "adapter" or record.get("scope") != "full-dev":
        raise ValueError("comparison requires completed Full-Dev Adapter evaluation Runs")
    actual_code = record.get("source_code") or {}
    if actual_code != expected_code and not _matches_legacy_source(
        actual_code, expected_code, legacy_source_files or {}
    ):
        raise ValueError("evaluation source-code identity differs from the recorded Run")
    for name, expected in record.get("artifacts", {}).items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"evaluation artifact hash changed: {name}")
    for name, key in (("inference_config.json", "inference_config_sha256"), ("evaluation_config.yaml", "evaluation_config_sha256")):
        if sha256_file(root / name) != record.get(key):
            raise ValueError("frozen evaluation config snapshot changed")
    return record


def _matches_legacy_source(
    actual: dict[str, Any], expected: dict[str, Any], legacy_files: dict[str, str]
) -> bool:
    actual_files, expected_files = actual.get("files"), expected.get("files")
    if (
        actual.get("packages") != expected.get("packages")
        or not isinstance(actual_files, dict)
        or not isinstance(expected_files, dict)
        or set(actual_files) != set(expected_files)
    ):
        return False
    differences = {name for name in actual_files if actual_files[name] != expected_files[name]}
    return bool(differences) and all(
        name in legacy_files and actual_files[name] == legacy_files[name]
        for name in differences
    )


def _compare_configs(baseline_root: Path, candidate_root: Path) -> None:
    for name in ("inference_config.json", "evaluation_config.yaml"):
        load = (lambda path: json.loads(path.read_text(encoding="utf-8"))) if name.endswith("json") else (lambda path: yaml.safe_load(path.read_text(encoding="utf-8")))
        baseline = e01._comparable(load(baseline_root / name))
        candidate = e01._comparable(load(candidate_root / name))
        if name == "inference_config.json":
            baseline.pop("hardware_label", None)
            candidate.pop("hardware_label", None)
        else:
            baseline.get("execution", {}).pop("output_root", None)
            candidate.get("execution", {}).pop("output_root", None)
        if baseline != candidate:
            raise ValueError("E01-B and E03-A1 must use identical Dev, prompt, tokenizer, scoring and generation settings")


def _retention_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    b_profile, c_profile = baseline["profile"], candidate["profile"]
    b_emotion, c_emotion = baseline["emotion"], candidate["emotion"]
    b_raw, c_raw = baseline["raw_diagnostics"], candidate["raw_diagnostics"]
    b_response, c_response = baseline["response"], candidate["response"]
    micro_delta = c_profile["overall_33_label_micro_f1"] - b_profile["overall_33_label_micro_f1"]
    macro_delta = c_profile["overall_33_label_macro_f1"] - b_profile["overall_33_label_macro_f1"]
    baseline_fp, candidate_fp = int(b_profile["overall_33_label_fp"]), int(c_profile["overall_33_label_fp"])
    fp_gate = candidate_fp <= baseline_fp * 1.10 if baseline_fp else candidate_fp == 0

    def relative_change(before: float, after: float) -> float:
        if before == 0:
            return 0.0 if after == 0 else 1.0 if after > 0 else -1.0
        return (after - before) / abs(before)

    bleu_before = float(b_response["local_corpus_bleu4"]["score_percent"])
    bleu_after = float(c_response["local_corpus_bleu4"]["score_percent"])
    rouge_before = float(b_response["local_rouge_l_macro_f1"]["value"])
    rouge_after = float(c_response["local_rouge_l_macro_f1"]["value"])
    bleu_change = relative_change(bleu_before, bleu_after)
    rouge_change = relative_change(rouge_before, rouge_after)
    checks = {
        "profile_micro_or_macro_f1_improves_by_0_01": micro_delta >= 0.01 or macro_delta >= 0.01,
        "profile_false_positives_increase_at_most_10_percent": fp_gate,
        "emotion_accuracy_drop_at_most_0_01": c_emotion["accuracy"] - b_emotion["accuracy"] >= -0.01,
        "emotion_macro_f1_drop_at_most_0_01": c_emotion["macro_f1"] - b_emotion["macro_f1"] >= -0.01,
        "raw_a0_schema_drop_at_most_0_01": c_raw["raw_schema_valid_rate"] - b_raw["raw_schema_valid_rate"] >= -0.01,
        "strict_json_drop_at_most_0_01": c_raw["raw_json_parse_rate"] - b_raw["raw_json_parse_rate"] >= -0.01,
        "response_bleu4_relative_drop_at_most_5_percent": bleu_change >= -0.05,
        "response_rouge_l_relative_drop_at_most_5_percent": rouge_change >= -0.05,
    }
    return {
        "status": "passed" if all(checks.values()) else "not_passed",
        "checks": checks,
        "profile_micro_f1_delta": micro_delta,
        "profile_macro_f1_delta": macro_delta,
        "profile_false_positives": {"baseline": baseline_fp, "candidate": candidate_fp},
        "emotion_accuracy_delta": c_emotion["accuracy"] - b_emotion["accuracy"],
        "emotion_macro_f1_delta": c_emotion["macro_f1"] - b_emotion["macro_f1"],
        "raw_a0_schema_delta": c_raw["raw_schema_valid_rate"] - b_raw["raw_schema_valid_rate"],
        "strict_json_delta": c_raw["raw_json_parse_rate"] - b_raw["raw_json_parse_rate"],
        "response_bleu4_relative_change": bleu_change,
        "response_rouge_l_relative_change": rouge_change,
    }


def compare(baseline_root: Path, candidate_root: Path, output_dir: Path) -> dict[str, Any]:
    baseline_root, candidate_root, output_dir = baseline_root.resolve(), candidate_root.resolve(), output_dir.resolve()
    if not output_dir.is_relative_to((ROOT / "outputs/evaluation/e03_a1").resolve()):
        raise ValueError("E03-A1 comparison output must stay under outputs/evaluation/e03_a1")
    baseline = _load_scored(
        baseline_root,
        expected_code=_ORIGINAL_E01_CODE_IDENTITY,
        legacy_source_files={
            "src/evaluation/score.py": _PRE_FIX_SCORE_SHA256,
            "participant/run_inference.py": _PRE_FAILURE_CATEGORY_RUNNER_SHA256,
        },
    )
    candidate = _load_scored(candidate_root, expected_code=_e03_code_identity())
    if baseline["dataset"] != candidate["dataset"] or baseline["dataset"].get("sample_count") != 1833:
        raise ValueError("paired E03-A1 comparison requires the same complete 1,833-row Dev set")
    if (baseline.get("training_run") or {}).get("experiment_id") != "E01-B":
        raise ValueError("baseline evaluation must use the completed E01-B Full Adapter")
    if (candidate.get("training_run") or {}).get("experiment_id") != "E03-A1":
        raise ValueError("candidate evaluation must use an E03-A1 Full Adapter")
    _compare_configs(baseline_root, candidate_root)

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "baseline_run_root": str(baseline_root),
        "candidate_run_root": str(candidate_root),
        "dataset": baseline["dataset"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        baseline_metrics = e01._score(baseline_root, baseline, output_dir / "baseline_scored")
        candidate_metrics = e01._score(candidate_root, candidate, output_dir / "candidate_scored")
        source = Path(baseline["dataset"]["root"])
        references = e01._unique_by_id(read_jsonl(source / baseline["dataset"]["references"]), "eval_id", "References")
        id_maps = e01._unique_by_id(read_jsonl(source / baseline["dataset"]["id_map"]), "eval_id", "ID map")
        predictions = {
            "baseline": e01._unique_by_id(read_jsonl(baseline_root / "runner/submission.jsonl"), "id", "E01-B predictions"),
            "candidate": e01._unique_by_id(read_jsonl(candidate_root / "runner/submission.jsonl"), "id", "E03-A1 predictions"),
        }
        raw = {
            "baseline": e01._unique_by_id(read_jsonl(baseline_root / "raw_predictions.jsonl"), "id", "E01-B raw predictions"),
            "candidate": e01._unique_by_id(read_jsonl(candidate_root / "raw_predictions.jsonl"), "id", "E03-A1 raw predictions"),
        }
        if any(set(rows) != set(references) for rows in (*predictions.values(), *raw.values())) or set(id_maps) != set(references):
            raise ValueError("paired comparison found missing, duplicated or mismatched Dev IDs")
        enums = schema_enums(load_submission_schema(ROOT))
        cases = []
        paired_cases = []
        for eval_id, reference in references.items():
            baseline_case = e01._case(predictions["baseline"][eval_id], raw["baseline"][eval_id], reference["target"], enums)
            candidate_case = e01._case(predictions["candidate"][eval_id], raw["candidate"][eval_id], reference["target"], enums)
            cases.append({"id": eval_id, "e01_b_baseline": baseline_case, "e03_a1_candidate": candidate_case})
            paired_cases.append({"id": eval_id, "base": baseline_case, "adapter": candidate_case})
        groups = {
            eval_id: id_maps[eval_id].get("conversation_id") or id_maps[eval_id]["sample_id"].rsplit(":", 1)[0]
            for eval_id in references
        }
        retention = _retention_gate(baseline_metrics, candidate_metrics)
        summary = {
            "status": "completed",
            "candidate_retention_status": retention["status"],
            "scope": "full-dev",
            "metrics_scope": "local_proxy",
            "dataset": baseline["dataset"],
            "baseline_training_run": baseline["training_run"],
            "candidate_training_run": candidate["training_run"],
            "paired": e01.paired_statistics(paired_cases, groups),
            "metrics": {"e01_b_baseline": baseline_metrics, "e03_a1_candidate": candidate_metrics},
            "diagnostics": {
                "e01_b_baseline": e01._extra_summary(paired_cases, "base"),
                "e03_a1_candidate": e01._extra_summary(paired_cases, "adapter"),
            },
            "retention_gate": retention,
            "limitations": [
                "Local proxy metrics are not official competition points.",
                "The comparison uses the same 1,833 Dev IDs; repeated Dev-based selection can overfit this set.",
                "A retained candidate still requires a separate project decision before replacing E01-B.",
                "BERTScore and official performance were not run.",
            ],
        }
        e01.runner._write_json_atomic(output_dir / "comparison_summary.json", summary)
        (output_dir / "comparison_cases.jsonl").write_bytes(e01.jsonl_bytes(cases))
        rows = [
            "# E03-A1 与 E01-B 完整 Dev 配对比较", "",
            f"样本：{len(cases)}；候选：{candidate['training_run']['loss_variant']}；范围：本地代理指标。", "",
            "| 指标 | E01-B 标准 Loss | E03-A1 候选 | 差值 |", "|---|---:|---:|---:|",
        ]
        metric_rows = (
            ("情绪 Accuracy", "emotion", "accuracy"),
            ("情绪 Macro-F1", "emotion", "macro_f1"),
            ("画像 33 标签 Micro-F1", "profile", "overall_33_label_micro_f1"),
            ("画像 33 标签 Macro-F1", "profile", "overall_33_label_macro_f1"),
            ("画像 FP 数", "profile", "overall_33_label_fp"),
            ("Raw A0 Schema", "raw_diagnostics", "raw_schema_valid_rate"),
            ("严格 JSON", "raw_diagnostics", "raw_json_parse_rate"),
        )
        for title, section, key in metric_rows:
            before = baseline_metrics[section][key]
            after = candidate_metrics[section][key]
            rows.append(f"| {title} | {before} | {after} | {after - before:+.6f} |")
        bleu_before = baseline_metrics["response"]["local_corpus_bleu4"]["score_percent"]
        bleu_after = candidate_metrics["response"]["local_corpus_bleu4"]["score_percent"]
        rouge_before = baseline_metrics["response"]["local_rouge_l_macro_f1"]["value"]
        rouge_after = candidate_metrics["response"]["local_rouge_l_macro_f1"]["value"]
        rows.extend([
            f"| 回复 BLEU-4 | {bleu_before:.6f} | {bleu_after:.6f} | {bleu_after - bleu_before:+.6f} |",
            f"| 回复 ROUGE-L | {rouge_before:.6f} | {rouge_after:.6f} | {rouge_after - rouge_before:+.6f} |",
            "",
            f"保留门槛：**{retention['status']}**；逐项判断见 comparison_summary.json。",
            f"配对情绪 Accuracy 差值：{summary['paired']['emotion_accuracy_delta']:.6f}；会话簇 Bootstrap 95% 区间：{summary['paired']['emotion_delta_95_percent_interval']}。",
            "",
            "官方成绩未运行。画像指标同时报告 FP、逐维非空召回与 Exact；不得只看全空预测带来的 Exact。",
        ])
        (output_dir / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest.update(status="completed", candidate_retention_status=retention["status"], comparison_summary_sha256=sha256_file(output_dir / "comparison_summary.json"))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        e01.runner._write_json_atomic(manifest_path, manifest)
        raise
    e01.runner._write_json_atomic(manifest_path, manifest)
    return {"status": manifest["status"], "candidate_retention_status": manifest["candidate_retention_status"], "output_dir": str(output_dir)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "compare"), required=True)
    parser.add_argument("--training-run-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--hardware-label", default="local-user-gpu")
    parser.add_argument("--allow-full-dev", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-run-root", type=Path)
    parser.add_argument("--candidate-run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _install_e03_hooks()
    if args.mode == "run":
        if not args.training_run_root or not args.run_id:
            raise ValueError("run mode requires --training-run-root and --run-id")
        if args.resume:
            _recover_score_only_resume(
                args.training_run_root, args.run_id, args.hardware_label
            )
        result = e01.execute(
            "full-dev", "adapter", args.run_id,
            training_root=args.training_run_root,
            hardware_label=args.hardware_label,
            allow_full_dev=args.allow_full_dev,
            resume=args.resume,
        )
    else:
        if not args.baseline_run_root or not args.candidate_run_root or not args.output_dir:
            raise ValueError("compare mode requires --baseline-run-root, --candidate-run-root and --output-dir")
        result = compare(args.baseline_run_root, args.candidate_run_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"scored", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
