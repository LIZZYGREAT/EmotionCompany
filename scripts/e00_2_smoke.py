#!/usr/bin/env python3
"""Render and optionally execute the E00.2 real-model smoke comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import load_submission_schema, schema_enums, strict_json_loads  # noqa: E402


PROFILE_PATH = ROOT / "configs/evaluation/e00_2_profiles.yaml"
OUTPUT_ROOT = ROOT / "outputs/evaluation/e00_2"
RUNNER_ENV_OVERRIDES = {
    "EVALUATION_BACKEND", "EVALUATION_BASE_URL", "EVALUATION_API_MODEL", "EVALUATION_API_KEY",
    "EVALUATION_EXPERIMENT_ID", "EVALUATION_MODEL_PATH", "EVALUATION_HARDWARE_LABEL",
    "EVALUATION_TOKENIZER_PATH", "EVALUATION_ENABLE_NF4", "EVALUATION_MAX_PROMPT_TOKENS",
    "EVALUATION_PROMPT_BUDGET_PROFILE", "EVALUATION_SERVING_CONTEXT_LIMIT",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id):
        raise ValueError("run-id 仅允许 1-100 位字母、数字、点、下划线和连字符，且必须以字母或数字开头")
    return run_id


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"期望 JSON object: {path}")
    return value


def _write_snapshot(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖配置快照: {path}")
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def render_profiles(run_id: str, *, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    root = output_root.resolve() / run_id
    if root.exists():
        raise FileExistsError(f"运行目录已存在，拒绝覆盖或混合结果: {root}")
    profile_cfg = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    runner_source = ROOT / profile_cfg["source_files"]["inference_config"]
    evaluation_source = ROOT / profile_cfg["source_files"]["evaluation_config"]
    runner_base = _load_json(runner_source)
    evaluation_base = yaml.safe_load(evaluation_source.read_text(encoding="utf-8"))
    if not isinstance(evaluation_base, dict):
        raise ValueError("E00.1 evaluation YAML 根节点必须为 mapping")
    if runner_base.get("model_id") != profile_cfg["model"]["model_id"]:
        raise ValueError("E00.2 档案模型 ID 与基础 runner 配置不符")
    if evaluation_base.get("model", {}).get("model_id") != profile_cfg["model"]["model_id"]:
        raise ValueError("E00.2 档案模型 ID 与 E00.1 评测配置不符")
    if evaluation_base.get("protocol", {}).get("max_prompt_tokens") != profile_cfg["context"]["max_prompt_tokens"]:
        raise ValueError("E00.2 P 值与 E00.1 冻结 Smoke 协议不一致")

    root.mkdir(parents=True)
    source_facts = {
        name: {"path": str((ROOT / relative).resolve()), "sha256": _sha256_file(ROOT / relative)}
        for name, relative in profile_cfg["source_files"].items()
    }
    rendered: dict[str, Any] = {}
    expected_quantization = {
        key: profile_cfg["quantization"][key]
        for key in ("load_in_4bit", "quant_type", "compute_dtype", "use_double_quant")
    }
    for profile_name, definition in profile_cfg["profiles"].items():
        runner_config = copy.deepcopy(runner_base)
        runner_config["experiment_id"] = profile_cfg["experiment_id"]
        runner_config["model_id"] = profile_cfg["model"]["model_id"]
        runner_config["model_revision"] = profile_cfg["model"]["weight_revision"]
        runner_config["model_path"] = "./model"
        runner_config["hardware_label"] = "user-wsl-single-gpu-candidate"
        runner_config["quantization"] = {
            **expected_quantization,
            "enabled": profile_cfg["quantization"]["enabled"],
        }
        runner_config["context"]["max_prompt_tokens"] = profile_cfg["context"]["max_prompt_tokens"]
        runner_config["context"]["budget_profile"] = profile_cfg["context"]["budget_profile"]
        runner_config["generation"] = {
            **profile_cfg["generation_common"],
            **definition["generation"],
        }
        if runner_config["generation"]["do_sample"]:
            runner_config["sampling"] = {"base_seed": profile_cfg["sampling"]["base_seed"]}
        else:
            runner_config.pop("sampling", None)

        evaluation_config = copy.deepcopy(evaluation_base)
        evaluation_config["experiment_id"] = profile_cfg["experiment_id"]
        evaluation_config["model"] = copy.deepcopy(profile_cfg["model"])
        evaluation_config["quantization"]["planned"] = expected_quantization
        evaluation_config["quantization"]["observed_runtime"] = None
        evaluation_config["generation"] = copy.deepcopy(runner_config["generation"])
        if runner_config["generation"] != evaluation_config["generation"]:
            raise AssertionError(f"{profile_name}: runner/evaluation generation 不一致")
        if evaluation_config["data"] != evaluation_base["data"]:
            raise AssertionError("渲染改变了 E00.1 数据适配配置")
        if evaluation_config["protocol"] != evaluation_base["protocol"]:
            raise AssertionError("渲染改变了 E00.1 冻结协议")

        profile_dir = root / profile_name
        runner_path = profile_dir / "inference_config.json"
        evaluation_path = profile_dir / "evaluation.yaml"
        runner_sha = _write_snapshot(runner_path, _json_bytes(runner_config))
        evaluation_bytes = yaml.safe_dump(
            evaluation_config, allow_unicode=True, sort_keys=False, default_flow_style=False
        ).encode("utf-8")
        evaluation_sha = _write_snapshot(evaluation_path, evaluation_bytes)
        rendered[profile_name] = {
            "profile": profile_name,
            "run_id": f"e00_2_{definition['run_suffix']}_{run_id}",
            "runner_config_path": str(runner_path.resolve()),
            "runner_config_sha256": runner_sha,
            "evaluation_config_path": str(evaluation_path.resolve()),
            "evaluation_config_sha256": evaluation_sha,
            "generation": runner_config["generation"],
            "sampling_base_seed": runner_config.get("sampling", {}).get("base_seed"),
        }

    manifest = {
        "status": "code_ready",
        "experiment_id": profile_cfg["experiment_id"],
        "run_id": run_id,
        "gpu_status": "not_checked_render_only",
        "real_smoke_status": "not_run",
        "source_files": source_facts,
        "profile_config_path": str(PROFILE_PATH.resolve()),
        "profile_config_sha256": _sha256_file(PROFILE_PATH),
        "environment_override_names": [],
        "profiles": rendered,
    }
    (root / "run_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _check_local_model_mount(expected_revision: str) -> Path:
    model_path = ROOT / "participant/model"
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地模型挂载缺失或悬空: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError("participant/model/config.json 缺失")
    if (model_path / "adapter_config.json").exists():
        raise ValueError("participant/model 含 adapter_config.json；E00.2 要求原始基础模型")
    model_config = _load_json(model_path / "config.json")
    if model_config.get("model_type") != "qwen3":
        raise ValueError(f"基础模型 config.model_type 不符: {model_config.get('model_type')!r}")
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError("participant/model/model.safetensors.index.json 缺失")
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("模型 weight index 缺少有效 weight_map")
    shards = sorted(set(weight_map.values()))
    for shard_name in shards:
        if not isinstance(shard_name, str):
            raise ValueError("模型 weight index 含非字符串分片路径")
        shard = model_path / shard_name
        if Path(shard_name).is_absolute() or ".." in Path(shard_name).parts:
            raise ValueError(f"模型分片路径越界: {shard_name}")
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise FileNotFoundError(f"模型分片缺失或为空: {shard_name}")
    resolved = model_path.resolve()
    if resolved.name != expected_revision:
        raise ValueError(
            f"模型 snapshot revision 与 E00.2 profile 不符: expected={expected_revision}, actual={resolved.name}"
        )
    return resolved


def _runner_environment(profile: dict[str, Any], run_id: str, raw_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in RUNNER_ENV_OVERRIDES:
        env.pop(name, None)
    env.update({
        "EVALUATION_INFERENCE_CONFIG": profile["runner_config_path"],
        "EVALUATION_EVALUATION_CONFIG": profile["evaluation_config_path"],
        "EVALUATION_RUN_ID": run_id,
        "EVALUATION_EXPERIMENT_ID": "E00.2",
        "EVALUATION_RAW_OUTPUT_PATH": str(raw_path.resolve()),
    })
    return env


def _run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"命令失败 (exit={completed.returncode}): {' '.join(command)}\n"
            f"日志: {log_path}\n{tail}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_preflight(result_dir: Path, raw_path: Path) -> dict[str, Any]:
    performance = _load_json(result_dir / "performance_report.json")
    submission = _read_jsonl(result_dir / "submission.jsonl")
    raw = _read_jsonl(raw_path)
    quantization = performance.get("quantization", {})
    identity = performance.get("model_weight_identity", {})
    if performance.get("backend") != "transformers":
        raise ValueError("单样本预检 backend 不是 transformers")
    if performance.get("samples") != 1 or len(submission) != 1 or len(raw) != 1:
        raise ValueError("单样本预检必须恰好产生一行 Submission 和 Raw")
    if not isinstance(raw[0].get("raw_text"), str):
        raise ValueError("单样本预检缺少真实模型 Raw 文本")
    if (
        quantization.get("real_nf4_verified") is not True
        or quantization.get("linear4bit_module_count", 0) <= 0
        or quantization.get("offload_devices")
    ):
        raise ValueError(f"单样本预检未通过真实 NF4 验收: {quantization}")
    if identity.get("identity_status") != "observed" or not identity.get("observed_weight_hash"):
        raise ValueError("单样本预检未得到实际权重 hash")
    return {
        "run_id": performance.get("run_id"),
        "model_weight_hash": identity["observed_weight_hash"],
        "quantization": quantization,
        "cuda_memory_after_model_load": performance.get("cuda_memory_after_model_load"),
        "model_load_seconds": performance.get("model_load_seconds"),
    }


def _score_profile(
    *, profile: dict[str, Any], profile_name: str, run_dir: Path,
    smoke_input: Path, smoke_references: Path, evaluation_manifest: Path,
) -> dict[str, Any]:
    prediction_dir = run_dir / "runner"
    scoring_dir = run_dir / "scored"
    raw_path = run_dir / "raw_predictions.jsonl"
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖 profile 输出目录: {run_dir}")
    run_dir.mkdir(parents=True)
    run_id = profile["run_id"]
    run_env = _runner_environment(profile, run_id, raw_path)
    _run_logged(
        [sys.executable, str(ROOT / "participant/run_inference.py"), str(smoke_input), str(prediction_dir)],
        cwd=ROOT,
        env=run_env,
        log_path=run_dir / "runner.log",
    )
    _run_logged(
        [
            sys.executable, "-m", "src.evaluation.score",
            "--config", profile["evaluation_config_path"],
            "--references", str(smoke_references),
            "--submission", str(prediction_dir / "submission.jsonl"),
            "--raw", str(raw_path),
            "--manifest", str(evaluation_manifest),
            "--output-dir", str(scoring_dir),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        log_path=run_dir / "scorer.log",
    )
    performance = _load_json(prediction_dir / "performance_report.json")
    metrics = _load_json(scoring_dir / "metrics.json")
    input_rows = _read_jsonl(smoke_input)
    predictions = _read_jsonl(prediction_dir / "submission.jsonl")
    raw_rows = _read_jsonl(raw_path)
    expected_ids = [row["id"] for row in input_rows]
    if performance.get("backend") != "transformers":
        raise ValueError(f"{profile_name}: 正式 Smoke backend 不是 transformers")
    if [row.get("id") for row in predictions] != expected_ids:
        raise ValueError(f"{profile_name}: Submission ID/顺序与 Smoke 输入不一致")
    if [row.get("id") for row in raw_rows] != expected_ids:
        raise ValueError(f"{profile_name}: Raw ID/顺序与 Smoke 输入不一致")
    if performance.get("samples") != 20 or metrics.get("sample_count") != 20:
        raise ValueError(f"{profile_name}: 正式 Smoke 必须恰好评分 20 条")
    quantization = performance.get("quantization", {})
    if quantization.get("real_nf4_verified") is not True or quantization.get("offload_devices"):
        raise ValueError(f"{profile_name}: NF4/GPU 实测证据无效")
    if metrics.get("runtime_compatibility", {}).get("status") != "verified":
        raise ValueError(f"{profile_name}: E00 scorer runtime compatibility 未通过")
    return {
        "profile": profile_name,
        "run_id": performance["run_id"],
        "runner_config_sha256": profile["runner_config_sha256"],
        "evaluation_config_sha256": profile["evaluation_config_sha256"],
        "input_sha256": performance["test_file_sha256"],
        "submission_sha256": performance["submission_sha256"],
        "raw_predictions_sha256": performance["raw_predictions_sha256"],
        "model_weight_identity": performance["model_weight_identity"],
        "quantization": quantization,
        "model_load_seconds": performance.get("model_load_seconds"),
        "cuda_memory_after_model_load": performance.get("cuda_memory_after_model_load"),
        "sample_timing": performance.get("sample_timing"),
        "metrics": metrics,
        "prediction_path": str((prediction_dir / "submission.jsonl").resolve()),
        "raw_path": str(raw_path.resolve()),
        "performance_path": str((prediction_dir / "performance_report.json").resolve()),
        "metrics_path": str((scoring_dir / "metrics.json").resolve()),
    }


def _write_comparison(root: Path, runs: dict[str, dict[str, Any]], smoke_references: Path) -> None:
    greedy = runs["greedy"]
    sampling = runs["sampling"]
    greedy_predictions = {row["id"]: row for row in _read_jsonl(Path(greedy["prediction_path"]))}
    sampling_predictions = {row["id"]: row for row in _read_jsonl(Path(sampling["prediction_path"]))}
    greedy_raw = {row["id"]: row for row in _read_jsonl(Path(greedy["raw_path"]))}
    sampling_raw = {row["id"]: row for row in _read_jsonl(Path(sampling["raw_path"]))}
    references = {row["eval_id"]: row["target"] for row in _read_jsonl(smoke_references)}
    enums = schema_enums(load_submission_schema(ROOT))
    if not (set(greedy_predictions) == set(sampling_predictions) == set(references)):
        raise ValueError("两组结果与 Smoke References ID 集合不一致")
    cases = []
    for sample_id in references:
        target = references[sample_id]
        case: dict[str, Any] = {"id": sample_id}
        for label, predictions, raw_rows in (
            ("greedy", greedy_predictions, greedy_raw),
            ("sampling", sampling_predictions, sampling_raw),
        ):
            prediction = predictions[sample_id]
            raw_text = raw_rows[sample_id].get("raw_text")
            try:
                raw_object = strict_json_loads(raw_text) if isinstance(raw_text, str) else None
            except (json.JSONDecodeError, ValueError, TypeError):
                raw_object = None
            dimensions = ("personality_traits", "interests", "style")
            raw_profile = raw_object.get("user_profile") if isinstance(raw_object, dict) else None
            raw_profile_valid = isinstance(raw_profile, dict) and set(raw_profile) == set(dimensions)
            if raw_profile_valid:
                raw_profile_valid = all(
                    isinstance(raw_profile[dimension], list)
                    and all(isinstance(label, str) for label in raw_profile[dimension])
                    and len(raw_profile[dimension]) == len(set(raw_profile[dimension]))
                    and set(raw_profile[dimension]) <= set(enums[dimension])
                    for dimension in dimensions
                )
            raw_schema_valid = bool(
                isinstance(raw_object, dict)
                and set(raw_object) == {"emotion_label", "user_profile", "response_text", "memory_refs"}
                and raw_object.get("emotion_label") in enums["emotion_label"]
                and raw_profile_valid
                and isinstance(raw_object.get("response_text"), str)
                and bool(raw_object.get("response_text", "").strip())
                and raw_object.get("memory_refs") == []
            )
            case[label] = {
                "status": raw_rows[sample_id].get("status"),
                "emotion_correct": prediction.get("emotion_label") == target.get("emotion_label"),
                "profile_missing_labels": sum(
                    len(set(target["user_profile"][dim]) - set(prediction["user_profile"][dim]))
                    for dim in dimensions
                ),
                "profile_false_positive_labels": sum(
                    len(set(prediction["user_profile"][dim]) - set(target["user_profile"][dim]))
                    for dim in dimensions
                ),
                "response_characters": len(prediction.get("response_text", "")),
                "stop_reason": raw_rows[sample_id].get("stop_reason"),
                "generated_tokens": raw_rows[sample_id].get("generated_tokens"),
                "raw_json_valid": isinstance(raw_object, dict),
                "raw_schema_valid": raw_schema_valid,
                "submission_schema_valid": True,
            }
        cases.append(case)
    (root / "comparison_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in cases),
        encoding="utf-8",
    )
    _write_snapshot(root / "comparison_summary.json", _json_bytes({
        "sample_count": len(cases),
        "interpretation": "20-row D0.3 Smoke is exploratory; this comparison does not establish a generally superior decoding strategy.",
        "greedy": {
            "emotion": greedy["metrics"]["emotion"],
            "profile": greedy["metrics"]["profile"],
            "response": greedy["metrics"]["response"],
            "raw_diagnostics": greedy["metrics"]["raw_diagnostics"],
        },
        "sampling": {
            "emotion": sampling["metrics"]["emotion"],
            "profile": sampling["metrics"]["profile"],
            "response": sampling["metrics"]["response"],
            "raw_diagnostics": sampling["metrics"]["raw_diagnostics"],
        },
    }))


def execute(run_id: str) -> int:
    manifest = render_profiles(run_id)
    run_root = OUTPUT_ROOT / run_id
    manifest_path = run_root / "run_manifest.json"
    try:
        import torch
    except ImportError as exc:
        manifest.update({
            "status": "code_ready",
            "gpu_status": "gpu_not_run",
            "real_smoke_status": "real_smoke_not_run",
            "blocker": f"PyTorch import failed: {type(exc).__name__}",
        })
        manifest_path.write_bytes(_json_bytes(manifest))
        print(f"BLOCKED_ENVIRONMENT: torch import failed; details: {manifest_path}")
        return 2
    manifest["environment"] = {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not torch.cuda.is_available():
        manifest.update({
            "status": "code_ready",
            "gpu_status": "gpu_not_run",
            "real_smoke_status": "real_smoke_not_run",
            "blocker": "BLOCKED_IN_CODEX_ENV: torch.cuda.is_available() is false; no CPU fallback was attempted",
        })
        manifest_path.write_bytes(_json_bytes(manifest))
        print(f"BLOCKED_IN_CODEX_ENV: CUDA unavailable; rendered configs: {run_root}")
        return 2

    manifest["gpu_status"] = "cuda_available_run_attempted"
    manifest["stage"] = "model_mount_and_preflight"
    manifest_path.write_bytes(_json_bytes(manifest))
    try:
        profile_cfg = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
        model_mount = _check_local_model_mount(profile_cfg["model"]["weight_revision"])
        manifest["model_mount"] = {
            "path": str(model_mount),
            "weight_index_sha256": _sha256_file(model_mount / "model.safetensors.index.json"),
            "weight_shards": sorted(set(_load_json(model_mount / "model.safetensors.index.json")["weight_map"].values())),
        }
        smoke_input = ROOT / profile_cfg["source_files"]["smoke_input"]
        smoke_references = ROOT / profile_cfg["source_files"]["smoke_references"]
        evaluation_manifest = ROOT / profile_cfg["source_files"]["evaluation_manifest"]
        input_rows = _read_jsonl(smoke_input)
        if len(input_rows) != 20:
            raise ValueError(f"E00.1 Smoke 输入预期 20 行，实际 {len(input_rows)}")

        greedy_profile = manifest["profiles"]["greedy"]
        preflight_dir = run_root / "preflight"
        preflight_input = preflight_dir / "input.jsonl"
        preflight_input.parent.mkdir(parents=True)
        preflight_input.write_text(json.dumps(input_rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
        preflight_result = preflight_dir / "runner"
        preflight_raw = preflight_dir / "raw_predictions.jsonl"
        preflight_id = f"e00_2_preflight_{run_id}"
        _run_logged(
            [sys.executable, str(ROOT / "participant/run_inference.py"), str(preflight_input), str(preflight_result)],
            cwd=ROOT,
            env=_runner_environment(greedy_profile, preflight_id, preflight_raw),
            log_path=preflight_dir / "runner.log",
        )
        manifest["preflight"] = _assert_preflight(preflight_result, preflight_raw)
        manifest["stage"] = "single_sample_preflight_passed"
        manifest["real_smoke_status"] = "preflight_passed_smoke_not_started"
        manifest_path.write_bytes(_json_bytes(manifest))

        runs: dict[str, dict[str, Any]] = {}
        for profile_name in ("greedy", "sampling"):
            profile = manifest["profiles"][profile_name]
            run_dir = run_root / profile["run_id"]
            runs[profile_name] = _score_profile(
                profile=profile,
                profile_name=profile_name,
                run_dir=run_dir,
                smoke_input=smoke_input,
                smoke_references=smoke_references,
                evaluation_manifest=evaluation_manifest,
            )
            manifest["stage"] = f"{profile_name}_scored"
            manifest.setdefault("runs", {})[profile_name] = {
                key: value for key, value in runs[profile_name].items() if key not in {"metrics"}
            }
            manifest_path.write_bytes(_json_bytes(manifest))

        hashes = {
            runs[name]["model_weight_identity"].get("observed_weight_hash")
            for name in ("greedy", "sampling")
        }
        if len(hashes) != 1 or None in hashes:
            raise ValueError("Greedy/Sampling 实际权重 hash 不一致")
        if runs["greedy"]["input_sha256"] != runs["sampling"]["input_sha256"]:
            raise ValueError("Greedy/Sampling Smoke 输入 hash 不一致")
        _write_comparison(run_root, runs, smoke_references)
        manifest["comparison"] = {
            "sample_count": 20,
            "input_sha256": runs["greedy"]["input_sha256"],
            "model_weight_sha256": next(iter(hashes)),
            "case_table": str((run_root / "comparison_cases.jsonl").resolve()),
            "summary": str((run_root / "comparison_summary.json").resolve()),
            "conclusion": "candidate comparison only; no decoding strategy frozen from 20 rows",
        }
        manifest.update({"status": "real_smoke_scored", "gpu_status": "gpu_run_passed", "real_smoke_status": "20_rows_both_profiles_scored"})
        manifest_path.write_bytes(_json_bytes(manifest))
        print(f"PASS: E00.2 real smoke complete: {run_root}")
        return 0
    except Exception as exc:  # save a local audit trail before stopping
        manifest.update({
            "status": "blocked_run_failed",
            "stage": manifest.get("stage", "preflight_or_smoke"),
            "gpu_status": manifest.get("gpu_status", "cuda_available_run_attempted"),
            "real_smoke_status": "failed_or_incomplete",
            "error": f"{type(exc).__name__}: {exc}",
        })
        manifest_path.write_bytes(_json_bytes(manifest))
        print(f"BLOCKED: {type(exc).__name__}: {exc}\nAudit manifest: {manifest_path}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染或执行 E00.2 真实模型 Smoke 与解码对照")
    parser.add_argument("--run-id", required=True, help="新的唯一 run ID；输出目录不得已存在")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true", help="只渲染配置快照，不加载模型（默认）")
    mode.add_argument("--execute", action="store_true", help="单样本预检通过后执行两组 20 条真实 Smoke 与评分")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = validate_run_id(args.run_id)
    if args.execute:
        raise SystemExit(execute(run_id))
    manifest = render_profiles(run_id)
    print(f"PASS: code_ready profile snapshots: {OUTPUT_ROOT / run_id}")
    print(f"runner/evaluation config hashes recorded for: {', '.join(manifest['profiles'])}")


if __name__ == "__main__":
    main()
