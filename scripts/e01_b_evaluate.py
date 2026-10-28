#!/usr/bin/env python3
"""Explicit, resumable E01-B Dev inference and offline paired comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as runner
from src.evaluation import jsonl_bytes, read_jsonl, schema_enums, load_submission_schema, sha256_file, strict_json_loads
from src.evaluation.metrics import raw_diagnostics
from src.evaluation.prepare_dev import adapter_config_identity, source_identity_sha256
from src.evaluation.score import score_files, _assert_file_matches_manifest, _assert_target_free_inference_input, _unique_by_id, _validate_d03_manifest

DIMENSIONS = ("personality_traits", "interests", "style")
SCORER_FILES = ("src/evaluation/score.py", "src/evaluation/metrics.py", "src/evaluation/__init__.py", "src/evaluation/prepare_dev.py")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _code_identity() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "sacrebleu", "jsonschema", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "files": {name: sha256_file(ROOT / name) for name in (*SCORER_FILES, "participant/run_inference.py", "src/shared/prompting.py", "scripts/e01_b_evaluate.py")},
        "packages": packages,
    }


def _config(scope: str) -> dict[str, Any]:
    name = "e01_b_pilot.yaml" if scope == "pilot-dev" else "e01_b_full.yaml"
    config = yaml.safe_load((ROOT / "configs/evaluation" / name).read_text(encoding="utf-8"))
    baseline = yaml.safe_load((ROOT / "configs/evaluation/e00.yaml").read_text(encoding="utf-8"))
    for key in ("protocol", "generation", "scoring", "quantization"):
        if config[key] != baseline[key]:
            raise ValueError(f"E01-B evaluation must preserve the E00 {key} contract")
    if config["data"]["expected_dev_count"] != 1833 or config["generation"] != {"max_new_tokens": 1200, "do_sample": False}:
        raise ValueError("E01-B requires the full D0.3 Dev identity and P1280/G1200 Greedy protocol")
    return config


def _dataset(config: dict[str, Any], scope: str) -> dict[str, Any]:
    root = _resolve(config["data"]["output_dir"])
    manifest = _json(root / "manifest.json")
    if manifest.get("dataset_scope") != "D0.3_dev":
        raise ValueError("E01-B evaluation requires real D0.3 Dev, not synthetic fixtures")
    _validate_d03_manifest(manifest, root / "manifest.json")
    adapter, adapter_sha = adapter_config_identity(config)
    if manifest.get("adapter_config") != adapter or manifest.get("adapter_config_sha256") != adapter_sha:
        raise ValueError("evaluation dataset adapter conditions differ from the frozen config")
    if manifest.get("source_identity_sha256") != source_identity_sha256(manifest["source"]):
        raise ValueError("evaluation source identity hash is invalid")
    smoke = scope == "pilot-dev"
    names = (
        ("smoke_input.jsonl", "smoke_references.jsonl", "smoke_id_map.jsonl")
        if smoke else ("inference_input.jsonl", "references.jsonl", "id_map.jsonl")
    )
    for name in names:
        _assert_file_matches_manifest(root / name, manifest, name)
    inputs = read_jsonl(root / names[0])
    mapping = _unique_by_id(inputs, "id", "inference input")
    count = 160 if smoke else 1833
    if len(mapping) != count:
        raise ValueError(f"{scope} requires exactly {count} IDs")
    if smoke:
        selection = manifest.get("smoke_selection", {})
        if set(selection.get("eval_ids", [])) != set(mapping) or selection.get("actual_count") != count or selection.get("requested_count") != count:
            raise ValueError("Pilot Dev membership differs from its frozen Smoke selection")
    _assert_target_free_inference_input(root / names[0], set(mapping))
    id_map = _unique_by_id(read_jsonl(root / names[2]), "eval_id", "ID map")
    if set(id_map) != set(mapping):
        raise ValueError("evaluation input and ID map sets differ")
    return {
        "root": str(root), "scope": scope, "sample_count": count,
        "input": names[0], "references": names[1], "id_map": names[2],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        **{name + "_sha256": sha256_file(root / name) for name in names},
        "ids_sha256": _canonical(list(mapping)),
    }


def _training_run(path: Path, scope: str, expected_base: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = path.resolve()
    if not root.is_relative_to((ROOT / "outputs/training").resolve()):
        raise ValueError("Adapter training Run must be under outputs/training")
    manifest = _json(root / "manifest.json")
    expected_experiment, training_scope, count = ("E01-B-PILOT", "pilot", 512) if scope == "pilot-dev" else ("E01-B", "full_train", 16192)
    if (
        manifest.get("run_id") != root.name or manifest.get("status") != "passed"
        or manifest.get("experiment_id") != expected_experiment or manifest.get("training_scope") != training_scope
        or manifest.get("train_rows") != count or manifest.get("init_adapter") is not None
        or manifest.get("base_weights_unchanged") is not True
        or manifest.get("base_weight_sha256") != expected_base
        or manifest.get("reload_check_status") != "passed"
    ):
        raise ValueError("evaluation requires a completed, independently initialized matching B Run with verified reload/Base identity")
    if sha256_file(root / "config_snapshot.yaml") != manifest.get("config_sha256"):
        raise ValueError("training config snapshot hash changed")
    identity, adapter_config = runner.adapter_weight_identity(root / "adapter")
    if any(identity.get(key) != (manifest.get("adapter_identity") or {}).get(key) for key in ("weight_sha256", "config_sha256")):
        raise ValueError("Adapter weight/config identity differs from the completed training Run")
    runner.validate_adapter_base_reference(adapter_config, "Qwen/Qwen3-1.7B", (ROOT / "participant/model").resolve())
    return {
        "root": str(root), "manifest_sha256": sha256_file(root / "manifest.json"),
        "experiment_id": expected_experiment, "training_scope": training_scope,
    }, identity


def _snapshots(config: dict[str, Any], scope: str, model: str, training_root: Path | None, hardware_label: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    inference_config = _json(ROOT / "participant/configs/inference_config.json")
    protocol = config["protocol"]
    if (
        inference_config["prompt"]["sha256"] != protocol["prompt_sha256"]
        or inference_config["context"]["max_prompt_tokens"] != protocol["max_prompt_tokens"]
        or inference_config["tokenizer"]["expected_signature_sha256"] != protocol["tokenizer_signature_sha256"]
        or inference_config["tokenizer"]["expected_template_sha256"] != protocol["chat_template_sha256"]
        or inference_config["tokenizer"]["revision"] != protocol["tokenizer_revision"]
    ):
        raise ValueError("runner source differs from the frozen P0/F2/T1/Tokenizer evaluation protocol")
    evaluation_config = copy.deepcopy(config)
    experiment = "E00.3" if model == "base" and scope == "full-dev" else "E01-B" if scope == "full-dev" else "E01-B-PILOT-EVAL"
    expected_base = config["execution"]["expected_base_weight_sha256"]
    inference_config.update(
        backend="transformers", experiment_id=experiment,
        model_revision=config["model"]["weight_revision"],
        model_path=str((ROOT / "participant/model").resolve()),
        hardware_label=hardware_label,
        expected_weight_sha256=expected_base,
        generation=copy.deepcopy(config["generation"]),
    )
    inference_config["quantization"]["enabled"] = True
    inference_config.pop("sampling", None)
    evaluation_config["experiment_id"] = experiment
    training = None
    adapter: dict[str, Any] = {"enabled": False}
    if model == "adapter":
        if training_root is None:
            raise ValueError("Adapter evaluation requires --training-run-root")
        training, identity = _training_run(training_root, scope, expected_base)
        adapter = {
            "enabled": True, "path": str(training_root.resolve() / "adapter"),
            "expected_base_weight_sha256": expected_base,
            "expected_adapter_weight_sha256": identity["weight_sha256"],
            "expected_adapter_config_sha256": identity["config_sha256"],
        }
    elif training_root is not None:
        raise ValueError("Base evaluation must not receive an Adapter training Run")
    inference_config["adapter"] = adapter
    evaluation_config["model"]["adapter"] = {key: value for key, value in adapter.items() if key != "path"}
    return inference_config, evaluation_config, training


def _invoke_runner(root: Path, record: dict[str, Any]) -> int:
    env = {key: value for key, value in os.environ.items() if not key.startswith("EVALUATION_")}
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "EVALUATION_INFERENCE_CONFIG": str(root / "inference_config.json"),
        "EVALUATION_EVALUATION_CONFIG": str(root / "evaluation_config.yaml"),
        "EVALUATION_RUN_ID": record["run_id"],
        "EVALUATION_RAW_OUTPUT_PATH": str(root / "raw_predictions.jsonl"),
        "EVALUATION_RESUME": "1",
    })
    command = [sys.executable, str(ROOT / "participant/run_inference.py"), str(Path(record["dataset"]["root"]) / record["dataset"]["input"]), str(root / "runner")]
    with (root / "runner.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def _score(root: Path, record: dict[str, Any], output: Path) -> dict[str, Any]:
    dataset = record["dataset"]
    dataset_root = Path(dataset["root"])
    config = yaml.safe_load((root / "evaluation_config.yaml").read_text(encoding="utf-8"))
    if _dataset(config, record["scope"]) != dataset:
        raise ValueError("recorded evaluation source/Reference/input identity changed")
    metrics = score_files(
        config=config, config_sha256=sha256_file(root / "evaluation_config.yaml"),
        references_path=dataset_root / dataset["references"],
        submission_path=root / "runner/submission.jsonl", raw_path=root / "raw_predictions.jsonl",
        output_dir=output, manifest_path=dataset_root / "manifest.json",
        id_map_path=dataset_root / dataset["id_map"], run_metadata_path=root / "runner/performance_report.json",
    )
    expected_identity = "verified_base_and_adapter" if record["model"] == "adapter" else "verified"
    if (
        metrics.get("sample_count") != dataset["sample_count"]
        or metrics.get("run_id") != record["run_id"]
        or metrics.get("runtime_compatibility", {}).get("status") != "verified"
        or metrics.get("model_identity_verification", {}).get("status") != expected_identity
        or metrics.get("model_identity_verification", {}).get("observed_weight_hash") != config["execution"]["expected_base_weight_sha256"]
        or metrics.get("execution_mode") != "model_inference"
    ):
        raise ValueError("B evaluation completeness/runtime/Base/Adapter identity checks failed")
    return metrics


def execute(scope: str, model: str, run_id: str, *, training_root: Path | None, hardware_label: str, allow_full_dev: bool, resume: bool = False, render_only: bool = False) -> dict[str, Any]:
    if scope == "full-dev" and not (allow_full_dev or render_only):
        raise ValueError("Full Dev inference requires the separate explicit --allow-full-dev resource permission")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}", run_id):
        raise ValueError("--run-id must contain 3-96 ASCII letters, digits, dots, underscores or hyphens")
    config = _config(scope)
    dataset = _dataset(config, scope)
    inference_config, evaluation_config, training = _snapshots(config, scope, model, training_root, hardware_label)
    root = _resolve(config["execution"]["output_root"]) / run_id
    if not root.is_relative_to((ROOT / "outputs/evaluation").resolve()):
        raise ValueError("evaluation outputs must remain under outputs/evaluation")
    inference_bytes = (json.dumps(inference_config, ensure_ascii=False, indent=2) + "\n").encode()
    evaluation_bytes = yaml.safe_dump(evaluation_config, allow_unicode=True, sort_keys=False).encode()
    identity = {
        "run_id": run_id, "scope": scope, "model": model, "dataset": dataset,
        "training_run": training, "source_code": _code_identity(),
        "inference_config_sha256": hashlib.sha256(inference_bytes).hexdigest(),
        "evaluation_config_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
    }
    manifest_path = root / "run_manifest.json"
    if resume:
        record = _json(manifest_path)
        if record.get("status") == "scored":
            raise FileExistsError("scored evaluation Run cannot be resumed or overwritten")
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("resume evaluation identity/config/data/code/Adapter changed")
        for name, key in (("inference_config.json", "inference_config_sha256"), ("evaluation_config.yaml", "evaluation_config_sha256")):
            if sha256_file(root / name) != record[key]:
                raise ValueError("frozen evaluation snapshot hash changed")
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root / "inference_config.json").write_bytes(inference_bytes)
        (root / "evaluation_config.yaml").write_bytes(evaluation_bytes)
        record = {**identity, "status": "prepared", "created_at_utc": _now(), "references_read_by_generator": False}
        runner._write_json_atomic(manifest_path, record)
    if render_only:
        return {"run_root": str(root), "status": record["status"]}
    try:
        if not (root / "runner/performance_report.json").is_file():
            record.update(status="running", started_at_utc=_now())
            runner._write_json_atomic(manifest_path, record)
            return_code = _invoke_runner(root, record)
            if return_code:
                record.update(status="inference_failed", return_code=return_code)
                runner._write_json_atomic(manifest_path, record)
                return {"run_root": str(root), "status": record["status"]}
        record.update(status="inference_completed")
        runner._write_json_atomic(manifest_path, record)
        _score(root, record, root / "scored")
        paths = ("runner/submission.jsonl", "runner/performance_report.json", "raw_predictions.jsonl", "scored/metrics.json")
        record.update(status="scored", ended_at_utc=_now(), artifacts={name: sha256_file(root / name) for name in paths})
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
        runner._write_json_atomic(manifest_path, record)
        raise
    runner._write_json_atomic(manifest_path, record)
    return {"run_root": str(root), "status": record["status"]}


def _load_completed(root: Path, model: str) -> dict[str, Any]:
    root = root.resolve()
    record = _json(root / "run_manifest.json")
    if record.get("status") != "scored" or record.get("model") != model:
        raise ValueError("comparison requires completed, correctly labeled Base and Adapter evaluation Runs")
    if record.get("source_code") != _code_identity():
        raise ValueError("comparison scorer/runner code or dependencies changed since inference")
    for name, expected in record["artifacts"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"evaluation artifact hash changed: {name}")
    for name, key in (("inference_config.json", "inference_config_sha256"), ("evaluation_config.yaml", "evaluation_config_sha256")):
        if sha256_file(root / name) != record[key]:
            raise ValueError("evaluation config snapshot changed")
    return record


def _comparable(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("experiment_id", None)
    result.pop("adapter", None)
    if "model" in result:
        result["model"].pop("adapter", None)
    return result


def _case(prediction: dict[str, Any], raw: dict[str, Any], target: dict[str, Any], enums: dict[str, list[str]]) -> dict[str, Any]:
    diagnostics = raw_diagnostics([raw], enums, [prediction], [True], generation_cap=1200)
    try:
        obj = strict_json_loads(raw["raw_text"])
    except (ValueError, TypeError):
        obj = None
    profile_counts = {}
    for dim in DIMENSIONS:
        gold, predicted = set(target["user_profile"][dim]), set(prediction["user_profile"][dim])
        profile_counts[dim] = {"tp": len(gold & predicted), "fp": len(predicted - gold), "fn": len(gold - predicted), "gold_nonempty": bool(gold), "pred_nonempty": bool(predicted), "nonempty_gold_hit": bool(gold & predicted), "exact": gold == predicted}
    illegal_profile_label = False
    raw_profile = obj.get("user_profile") if isinstance(obj, dict) else None
    if isinstance(raw_profile, dict):
        illegal_profile_label = any(isinstance(raw_profile.get(dim), list) and any(isinstance(label, str) and label not in enums[dim] for label in raw_profile[dim]) for dim in DIMENSIONS)
    return {
        "emotion_correct": prediction["emotion_label"] == target["emotion_label"],
        "emotion_error_kind": "correct" if prediction["emotion_label"] == target["emotion_label"] else "generation_format_failure" if not diagnostics["raw_json_parse_rate"] else "legal_label_wrong_class" if isinstance(obj, dict) and obj.get("emotion_label") in enums["emotion_label"] else "invalid_emotion_label",
        "raw_json_valid": bool(diagnostics["raw_json_parse_rate"]),
        "raw_schema_valid": bool(diagnostics["raw_schema_valid_rate"]),
        "raw_errors": list(diagnostics["error_case_counts"]),
        "raw_missing_fields": sorted({"emotion_label", "user_profile", "response_text", "memory_refs"} - set(obj)) if isinstance(obj, dict) else [],
        "raw_profile_illegal_label": illegal_profile_label,
        "profile": profile_counts,
        "stop_reason": raw.get("stop_reason"), "generated_tokens": raw.get("generated_tokens"),
        "latency_ms": raw.get("latency_ms"),
    }


def paired_statistics(cases: list[dict[str, Any]], groups: dict[str, str], *, seed: int = 42, bootstrap_samples: int = 2000) -> dict[str, Any]:
    if not cases or bootstrap_samples < 1 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("paired statistics require unique aligned IDs and a positive bootstrap count")
    transitions: Counter[str] = Counter()
    clusters: dict[str, list[int]] = defaultdict(list)
    for case in cases:
        base, adapter = case["base"]["emotion_correct"], case["adapter"]["emotion_correct"]
        transitions[f"base_{'correct' if base else 'wrong'}_adapter_{'correct' if adapter else 'wrong'}"] += 1
        clusters[groups[case["id"]]].append(int(adapter) - int(base))
    cluster_values = list(clusters.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        resampled = [cluster_values[rng.randrange(len(cluster_values))] for _ in cluster_values]
        estimates.append(sum(sum(cluster) for cluster in resampled) / sum(map(len, resampled)))
    estimates.sort()
    n = len(cases)
    return {
        "sample_count": n, "emotion_transitions": dict(sorted(transitions.items())),
        "emotion_accuracy_delta": sum(int(case["adapter"]["emotion_correct"]) - int(case["base"]["emotion_correct"]) for case in cases) / n,
        "emotion_delta_95_percent_interval": [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]],
        "interval_method": "paired_conversation_cluster_percentile_bootstrap_v1", "cluster_count": len(clusters), "seed": seed, "bootstrap_samples": bootstrap_samples,
    }


def _extra_summary(cases: list[dict[str, Any]], model: str) -> dict[str, Any]:
    result: dict[str, Any] = {"emotion_error_counts": dict(Counter(case[model]["emotion_error_kind"] for case in cases)), "profile": {}}
    n = len(cases)
    tokens = [case[model]["generated_tokens"] for case in cases if isinstance(case[model].get("generated_tokens"), int)]
    latencies = sorted(float(case[model]["latency_ms"]) for case in cases if isinstance(case[model].get("latency_ms"), (int, float)))
    result["generation"] = {"total_tokens": sum(tokens), "mean_tokens": sum(tokens) / len(tokens) if tokens else None, "min_tokens": min(tokens) if tokens else None, "max_tokens": max(tokens) if tokens else None, "stop_counts": dict(Counter(str(case[model]["stop_reason"]) for case in cases))}
    result["raw_missing_field_counts"] = dict(Counter(field for case in cases for field in case[model]["raw_missing_fields"]))
    result["raw_profile_illegal_label_count"] = sum(case[model]["raw_profile_illegal_label"] for case in cases)
    result["latency_ms"] = {"count": len(latencies), "mean": sum(latencies) / len(latencies) if latencies else None, "p50": latencies[(len(latencies) - 1) // 2] if latencies else None, "p95": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None, "max": max(latencies) if latencies else None}
    for dim in DIMENSIONS:
        values = [case[model]["profile"][dim] for case in cases]
        nonempty = sum(row["gold_nonempty"] for row in values)
        result["profile"][dim] = {
            "gold_nonempty_count": nonempty,
            "gold_nonempty_rate": nonempty / n,
            "pred_nonempty_count": sum(row["pred_nonempty"] for row in values),
            "pred_nonempty_rate": sum(row["pred_nonempty"] for row in values) / n,
            "nonempty_gold_sample_recall": sum(row["nonempty_gold_hit"] for row in values) / nonempty if nonempty else 0.0,
            "false_empty_count": sum(row["gold_nonempty"] and not row["pred_nonempty"] for row in values),
            "missing_label_count": sum(row["fn"] for row in values),
        }
    return result


def _hardware_summary(root: Path) -> dict[str, Any]:
    report = _json(root / "runner/performance_report.json")
    return {key: report.get(key) for key in ("hardware_label", "runtime_environment", "timing_scope", "model_load_seconds", "cuda_memory_after_model_load", "sample_timing", "resumption", "rounds", "complete")}


def compare(base_root: Path, adapter_root: Path, output_dir: Path) -> dict[str, Any]:
    base_root, adapter_root, output_dir = base_root.resolve(), adapter_root.resolve(), output_dir.resolve()
    if not output_dir.is_relative_to((ROOT / "outputs/evaluation").resolve()):
        raise ValueError("comparison output must remain under outputs/evaluation")
    base, adapter = _load_completed(base_root, "base"), _load_completed(adapter_root, "adapter")
    if base["dataset"] != adapter["dataset"] or base["scope"] != adapter["scope"]:
        raise ValueError("paired comparison requires the identical complete Dev ID set, Reference hash and scope")
    for filename in ("inference_config.json", "evaluation_config.yaml"):
        loader = _json if filename.endswith("json") else lambda path: yaml.safe_load(path.read_text(encoding="utf-8"))
        if _comparable(loader(base_root / filename)) != _comparable(loader(adapter_root / filename)):
            raise ValueError("Base/Adapter evaluation protocol, scoring, generation, hardware or model settings differ")
    base_hardware, adapter_hardware = _hardware_summary(base_root), _hardware_summary(adapter_root)
    if any(base_hardware[key] != adapter_hardware[key] for key in ("runtime_environment", "hardware_label", "timing_scope")):
        raise ValueError("Base/Adapter actual runtime environment or timing scope differs")
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {"status": "validating", "created_at_utc": _now(), "base_run_root": str(base_root), "adapter_run_root": str(adapter_root)}
    runner._write_json_atomic(output_dir / "manifest.json", record)
    try:
        metrics = {"base": _score(base_root, base, output_dir / "base_scored"), "adapter": _score(adapter_root, adapter, output_dir / "adapter_scored")}
        dataset = base["dataset"]
        source = Path(dataset["root"])
        references = _unique_by_id(read_jsonl(source / dataset["references"]), "eval_id", "References")
        id_maps = _unique_by_id(read_jsonl(source / dataset["id_map"]), "eval_id", "ID map")
        predictions = {name: _unique_by_id(read_jsonl(path / "runner/submission.jsonl"), "id", name) for name, path in (("base", base_root), ("adapter", adapter_root))}
        raw = {name: _unique_by_id(read_jsonl(path / "raw_predictions.jsonl"), "id", name) for name, path in (("base", base_root), ("adapter", adapter_root))}
        if any(set(rows) != set(references) for rows in (*predictions.values(), *raw.values())) or len(references) != dataset["sample_count"]:
            raise ValueError("partial, duplicate or mismatched IDs cannot be compared as complete Dev")
        enums = schema_enums(load_submission_schema(ROOT))
        cases = [{"id": eval_id, **{name: _case(predictions[name][eval_id], raw[name][eval_id], reference["target"], enums) for name in ("base", "adapter")}} for eval_id, reference in references.items()]
        groups = {eval_id: id_maps[eval_id].get("conversation_id") or id_maps[eval_id]["sample_id"].rsplit(":", 1)[0] for eval_id in references}
        summary = {
            "status": "completed", "scope": base["scope"], "metrics_scope": "local_proxy",
            "dataset": dataset, "paired": paired_statistics(cases, groups),
            "metrics": metrics,
            "diagnostics": {name: _extra_summary(cases, name) for name in ("base", "adapter")},
            "hardware_and_timing": {"base": base_hardware, "adapter": adapter_hardware},
            "official_performance": "not_run", "bert_score": metrics["adapter"]["response"]["bert_score"]["status"],
            "limitations": ["Local proxy metrics are not official competition points.", "Profile Exact Match must be interpreted together with non-empty Gold recall and the empty-profile rates.", "Pilot Dev results are engineering diagnostics." ] if base["scope"] == "pilot-dev" else ["Local proxy metrics are not official competition points.", "Repeated Dev-driven selection can overfit the development set.", "Resumed inference combines timing from multiple model loads; official performance remains not_run."],
        }
        summary["metric_deltas_adapter_minus_base"] = {
            f"{section}.{key}": metrics["adapter"][section][key] - metrics["base"][section][key]
            for section, key in (("emotion", "accuracy"), ("emotion", "macro_f1"), ("profile", "overall_33_label_micro_f1"), ("profile", "overall_33_label_macro_f1"), ("profile", "all_three_dimensions_exact_match"), ("raw_diagnostics", "raw_json_parse_rate"), ("raw_diagnostics", "raw_schema_valid_rate"))
        }
        summary["metric_deltas_adapter_minus_base"].update({
            "response.bleu4_percent": metrics["adapter"]["response"]["local_corpus_bleu4"]["score_percent"] - metrics["base"]["response"]["local_corpus_bleu4"]["score_percent"],
            "response.rouge_l": metrics["adapter"]["response"]["local_rouge_l_macro_f1"]["value"] - metrics["base"]["response"]["local_rouge_l_macro_f1"]["value"],
        })
        runner._write_json_atomic(output_dir / "comparison_summary.json", summary)
        (output_dir / "comparison_cases.jsonl").write_bytes(jsonl_bytes(cases))
        rows = ["# E01-B 配对 Dev 评测", "", f"范围：{base['scope']}；相同 ID：{len(cases)}；所有指标均为 local_proxy。", "", "| 指标 | Base | Adapter |", "|---|---:|---:|"]
        for title, section, key in (("Emotion Accuracy", "emotion", "accuracy"), ("Emotion Macro-F1", "emotion", "macro_f1"), ("Profile Micro-F1", "profile", "overall_33_label_micro_f1"), ("Profile Macro-F1", "profile", "overall_33_label_macro_f1"), ("Profile Exact Match", "profile", "all_three_dimensions_exact_match"), ("Raw strict JSON", "raw_diagnostics", "raw_json_parse_rate"), ("Raw A0 Schema", "raw_diagnostics", "raw_schema_valid_rate"), ("Submission Schema", "raw_diagnostics", "submission_schema_valid_rate")):
            rows.append(f"| {title} | {metrics['base'][section][key]:.6f} | {metrics['adapter'][section][key]:.6f} |")
        rows.extend(["", f"配对情绪 Accuracy 差值：{summary['paired']['emotion_accuracy_delta']:.6f}；会话簇 Bootstrap 95% 区间：{summary['paired']['emotion_delta_95_percent_interval']}。", "", "逐类混淆矩阵、33 标签/逐维指标、非空画像 Recall、Raw 错误分布、回复 BLEU-4/ROUGE-L 与计时元数据见 comparison_summary.json；逐 ID 差异不含原始对话。", "", "BERTScore 按配置 not_run；官方性能口径 not_run。空画像可抬高 Exact Match，须同时检查非空 Gold 的 Recall。"])
        (output_dir / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        record.update(status="completed", ended_at_utc=_now(), summary_sha256=sha256_file(output_dir / "comparison_summary.json"))
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        runner._write_json_atomic(output_dir / "manifest.json", record)
        raise
    runner._write_json_atomic(output_dir / "manifest.json", record)
    return {"status": "completed", "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot-dev", "full-dev", "compare"), required=True)
    parser.add_argument("--model", choices=("base", "adapter"))
    parser.add_argument("--run-id")
    parser.add_argument("--training-run-root", type=Path)
    parser.add_argument("--hardware-label", default="local-user-gpu")
    parser.add_argument("--allow-full-dev", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-run-root", type=Path)
    parser.add_argument("--adapter-run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.scope == "compare":
        if not all((args.base_run_root, args.adapter_run_root, args.output_dir)):
            parser.error("compare requires --base-run-root, --adapter-run-root and --output-dir")
        result = compare(args.base_run_root, args.adapter_run_root, args.output_dir)
    else:
        if not args.model or not args.run_id:
            parser.error("inference requires --model and --run-id")
        result = execute(args.scope, args.model, args.run_id, training_root=args.training_run_root, hardware_label=args.hardware_label, allow_full_dev=args.allow_full_dev, resume=args.resume, render_only=args.render_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"prepared", "scored", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
