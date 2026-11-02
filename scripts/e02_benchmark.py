#!/usr/bin/env python3
"""Run the fixed E02.2-B benchmark through the official participant entry point.

The runner validates immutable candidate/config identities, then delegates all
model requests to ``participant/start.sh`` and ``participant/run_inference.py``.
Generated text stays in ignored outputs and is excluded from this script's
summary and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
E02_OUTPUT_ROOT = (WORKSPACE_ROOT / "outputs/evaluation/e02").resolve()
DEFAULT_CONFIG = WORKSPACE_ROOT / "configs/deployment/e02_vllm_benchmark.json"
EXPECTED_SAMPLE_COUNT = 100
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from participant import run_inference as runner  # noqa: E402
from scripts import e02_vllm_probe as probe  # noqa: E402
from src.evaluation import read_jsonl, sha256_file, strict_json_loads  # noqa: E402
from src.evaluation.metrics import raw_diagnostics  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 根节点必须为 JSON object")
    return value


def validate_benchmark_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _read_json(config_path, "E02.2-B 配置")
    smoke_config = _read_json(WORKSPACE_ROOT / "configs/deployment/e02_vllm_smoke.json", "E02.1 Smoke 基线配置")
    shared_frozen_fields = (
        "backend", "model_path", "model_id", "model_revision", "trust_remote_code",
        "device_map", "torch_dtype", "quantization", "base_url", "api_model", "api_key",
        "request_timeout_seconds", "api_options", "openai_compatible", "prompt", "tokenizer",
        "generation", "hardware_label", "adapter", "deployment",
    )
    changed_frozen_fields = [
        name for name in shared_frozen_fields if config.get(name) != smoke_config.get(name)
    ]
    if changed_frozen_fields:
        raise ValueError(f"E02.2-B 必须沿用 E02.1 冻结推理身份/请求设置: {changed_frozen_fields}")
    checks = {
        "backend": config.get("backend") == "openai_compatible",
        "request_count": config.get("performance_rounds") == EXPECTED_SAMPLE_COUNT,
        "greedy": config.get("generation", {}).get("do_sample") is False,
        "prompt_cap": config.get("context", {}).get("max_prompt_tokens") == 1280,
        "generation_cap": config.get("generation", {}).get("max_new_tokens") == 1200,
        "serving_context": config.get("context", {}).get("serving_context_limit") == 2560,
        "api_model": config.get("api_model") == "e01-full",
        "adapter_enabled": config.get("adapter", {}).get("enabled") is True,
        "expected_adapter_route": config.get("adapter", {}).get("expected_server_model_id") == "e01-full",
        "vllm_version": config.get("deployment", {}).get("vllm", {}).get("expected_version") == "0.11.1",
        "dtype": config.get("deployment", {}).get("vllm", {}).get("dtype") == "bfloat16",
        "max_num_seqs": config.get("deployment", {}).get("vllm", {}).get("max_num_seqs") == 1,
        "max_num_batched_tokens": config.get("deployment", {}).get("vllm", {}).get("max_num_batched_tokens") == 2560,
        "gpu_memory_utilization": config.get("deployment", {}).get("vllm", {}).get("gpu_memory_utilization") == 0.8,
        "quantization_disabled": config.get("quantization", {}).get("enabled") is False,
        "independent_experiment_id": config.get("experiment_id") != smoke_config.get("experiment_id"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"E02.2-B 配置不符合冻结基线: {failed}")
    if config.get("adapter", {}).get("expected_rank") != 8:
        raise ValueError("E02.2-B Full B Adapter rank 必须为 8")
    return config


def validate_candidate(selection_root: Path, config_path: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    selection_root = selection_root.expanduser().resolve()
    manifest_path = selection_root / "selection_manifest.json"
    manifest = _read_json(manifest_path, "Candidate Selection Manifest")
    if manifest.get("schema_version") != "e02_vllm_benchmark_selection_v2":
        raise ValueError("Candidate Run 缺少经过 Smoke ID 映射修复的 v2 Manifest")
    if manifest.get("status") != "passed_candidate_audit":
        raise ValueError("Candidate Run 未通过固定候选审计")
    candidate_facts = manifest.get("candidate_input", {})
    candidate_path = selection_root / "candidate_input.jsonl"
    expected_path = candidate_facts.get("path")
    if not isinstance(expected_path, str) or Path(expected_path).expanduser().resolve() != candidate_path:
        raise ValueError("Candidate Manifest 输入路径与候选 Run 路径不一致")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate 输入缺失: {candidate_path}")
    candidate_sha = sha256_file(candidate_path)
    if candidate_sha != candidate_facts.get("sha256"):
        raise ValueError("Candidate 输入 SHA-256 与 Selection Manifest 不一致")
    rows = read_jsonl(candidate_path)
    if len(rows) != EXPECTED_SAMPLE_COUNT or candidate_facts.get("rows") != EXPECTED_SAMPLE_COUNT:
        raise ValueError("E02.2-B 固定候选输入必须恰好为 100 条")
    ids: list[str] = []
    for index, row in enumerate(rows, 1):
        if set(row) != {"id", "history"} or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Candidate 输入第 {index} 行必须恰好包含 id 和 history")
        if not isinstance(row.get("history"), list) or not row["history"]:
            raise ValueError(f"Candidate 输入第 {index} 行 history 必须为非空数组")
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate 输入含重复 ID")
    if any(not re.fullmatch(r"test_[0-9]{6}_t[0-9]+", sample_id) for sample_id in ids):
        raise ValueError("Candidate 输入 ID 必须符合赛事 Submission Schema")
    selected_facts = manifest.get("samples", [])
    if len(selected_facts) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Candidate Manifest 缺少 100 条来源 sample_id 映射记录")
    if [item.get("id") for item in selected_facts] != ids:
        raise ValueError("Candidate Manifest 的本地请求 ID 顺序与候选输入不一致")
    source_sample_ids = [item.get("source_sample_id") for item in selected_facts]
    if any(not isinstance(value, str) or not value for value in source_sample_ids) or len(set(source_sample_ids)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Candidate Manifest 的 D0.3 source_sample_id 缺失或重复")
    if candidate_facts.get("smoke_overlap_count") != 0:
        raise ValueError("Candidate Manifest 仍报告与历史 Smoke 有重叠")
    selection = manifest.get("selection", {})
    if selection.get("selection_rule_sha256") is None:
        raise ValueError("Candidate Manifest 缺少选择规则指纹")
    source = manifest.get("source", {})
    if not source.get("id_map_sha256") or source.get("mapped_sample_id_count") != source.get("smoke_eval_id_count"):
        raise ValueError("Candidate Manifest 缺少完整 Smoke eval_id → sample_id 映射")
    if set(source_sample_ids) & set(source.get("excluded_smoke_sample_id_overlap", [])):
        raise ValueError("Candidate D0.3 source_sample_id 与映射后的历史 Smoke 样本重叠")
    config_sha = sha256_file(config_path)
    if manifest.get("protocol", {}).get("config_sha256") != config_sha:
        raise ValueError("候选选择与正式基准必须使用同一份独立 E02.2-B 配置")
    return manifest, candidate_path, rows


def _distribution(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def summarize_runner_outputs(
    *,
    run_root: Path,
    expected_ids: list[str],
    expected_input_sha256: str,
    expected_config_sha256: str,
    expected_run_id: str,
    config: dict[str, Any],
    process_elapsed_seconds: float,
    gpu_memory: dict[str, Any],
) -> dict[str, Any]:
    result_dir = run_root / "runner"
    report_path = result_dir / "performance_report.json"
    raw_path = run_root / "raw_predictions.jsonl"
    submission_path = result_dir / "submission.jsonl"
    if not report_path.is_file() or not raw_path.is_file() or not submission_path.is_file():
        raise FileNotFoundError("官方 Runner 未生成完整 performance_report.json、Raw 或 submission.jsonl")
    runner_report = _read_json(report_path, "Runner Performance Report")
    expected_count = len(expected_ids)
    if (
        runner_report.get("complete") is not True
        or runner_report.get("samples") != expected_count
        or runner_report.get("submission_rows") != expected_count
        or runner_report.get("required_rounds") != expected_count
        or runner_report.get("rounds") != expected_count
    ):
        raise ValueError("官方 Runner Performance Report 未证明恰好完成 100 条")
    if (
        runner_report.get("run_id") != expected_run_id
        or runner_report.get("test_file_sha256") != expected_input_sha256
        or runner_report.get("configuration", {}).get("inference_config_sha256") != expected_config_sha256
        or runner_report.get("backend") != config.get("backend")
        or runner_report.get("api_model") != config.get("api_model")
        or runner_report.get("generation_config") != config.get("generation")
        or runner_report.get("submission_schema_validated") is not True
        or runner_report.get("resumption", {}).get("performance_is_single_uninterrupted_run") is not True
    ):
        raise ValueError("官方 Runner Report 的输入、配置、运行身份或请求设置与固定 Benchmark 不一致")
    raw_rows = read_jsonl(raw_path)
    submission_rows = read_jsonl(submission_path)
    raw_ids = [row.get("id") for row in raw_rows]
    submission_ids = [row.get("id") for row in submission_rows]
    if raw_ids != expected_ids or submission_ids != expected_ids:
        raise ValueError("Raw/Submission 的 ID 顺序或集合与固定 100 条候选输入不一致")
    if len(raw_rows) != expected_count or len(submission_rows) != expected_count:
        raise ValueError("Raw/Submission 行数与固定候选输入不一致")

    schema = runner.load_submission_schema()
    submission_validity: list[bool] = []
    for row in submission_rows:
        try:
            runner.validate_submission_prediction(row, schema)
        except Exception:
            submission_validity.append(False)
        else:
            submission_validity.append(True)
    enums = runner.submission_enums(schema)
    raw_facts = raw_diagnostics(
        raw_rows,
        enums,
        submission_rows,
        submission_validity,
        generation_cap=int(config["generation"]["max_new_tokens"]),
    )

    latencies = [float(row["latency_ms"]) for row in raw_rows if _valid_number(row.get("latency_ms"))]
    token_values = [int(row["generated_tokens"]) for row in raw_rows if _valid_number(row.get("generated_tokens"))]
    total_tokens = sum(token_values)
    summed_request_seconds = sum(latencies) / 1000.0
    statuses = Counter(str(row.get("status", "unknown")) for row in raw_rows)
    error_categories = Counter(
        str(row.get("failure_category", "other"))
        for row in raw_rows
        if row.get("status") == "generation_failure"
    )
    error_types = Counter(
        str(row.get("error_type", "unknown"))
        for row in raw_rows
        if row.get("status") == "generation_failure"
    )
    generation_cap = int(config["generation"]["max_new_tokens"])
    cap_ids = [
        row.get("id") for row in raw_rows
        if (
            _valid_number(row.get("generated_tokens"))
            and int(row["generated_tokens"]) >= generation_cap
        ) or str(row.get("stop_reason", "")).endswith((":length", ":max_tokens"))
    ]
    stop_counts = Counter(str(row.get("stop_reason", "unknown")) for row in raw_rows)
    normal_stop_count = sum(
        1 for row in raw_rows
        if row.get("stop_reason") in {"server_finish_reason:stop", "server_finish_reason:eos"}
    )
    length_stop_count = sum(
        1 for row in raw_rows
        if str(row.get("stop_reason", "")).endswith((":length", ":max_tokens"))
    )
    submission_schema_count = sum(submission_validity)
    raw_json_count = int(round(raw_facts["raw_json_parse_rate"] * expected_count))
    raw_a0_count = int(round(raw_facts["raw_schema_valid_rate"] * expected_count))
    submission_bytes_sha = sha256_file(submission_path)
    raw_bytes_sha = sha256_file(raw_path)
    if runner_report.get("submission_sha256") != submission_bytes_sha or runner_report.get("raw_predictions_sha256") != raw_bytes_sha:
        raise ValueError("Runner Report 中的 Raw/Submission SHA-256 与实际文件不一致")
    latency_ms = _distribution(latencies)
    token_distribution = _distribution(token_values)
    log_path = run_root / "vllm.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    log_lines = log_text.splitlines()
    kv_tokens: list[int] = []
    for line in log_lines:
        match = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens?", line, re.IGNORECASE)
        if match:
            kv_tokens.append(int(match.group(1).replace(",", "")))
    preemption_lines = sum(bool(re.search(r"preempt|recompute", line, re.IGNORECASE)) for line in log_lines)
    oom_log_lines = sum(bool(re.search(r"out of memory|cuda.*oom|oom.*cuda", line, re.IGNORECASE)) for line in log_lines)
    timeout_count = error_categories.get("timeout", 0)
    oom_count = error_categories.get("oom", 0)
    lifecycle_events: list[dict[str, Any]] = []
    lifecycle_path = run_root / "vllm_lifecycle.jsonl"
    if lifecycle_path.is_file():
        lifecycle_events = read_jsonl(lifecycle_path)
    events = {row.get("event"): row.get("epoch_seconds") for row in lifecycle_events}
    serve_start = events.get("serve_process_started")
    service_ready = events.get("health_ready")
    service_startup_seconds = (
        max(0.0, float(service_ready) - float(serve_start))
        if _valid_number(serve_start) and _valid_number(service_ready) else None
    )

    return {
        "schema_version": "e02_vllm_benchmark_summary_v1",
        "status": "completed",
        "created_at_utc": _utc_now(),
        "expected_samples": expected_count,
        "completed_samples": len(raw_rows),
        "runner": {
            "status": runner_report.get("complete"),
            "success_count": runner_report.get("success_count"),
            "status_counts": dict(sorted(statuses.items())),
            "runner_failure_counts": runner_report.get("failure_counts"),
            "performance_report_sha256": sha256_file(report_path),
            "input_sha256": expected_input_sha256,
            "inference_config_sha256": expected_config_sha256,
            "raw_predictions_sha256": raw_bytes_sha,
            "submission_sha256": submission_bytes_sha,
            "submission_schema_valid_count": submission_schema_count,
            "raw_strict_json_object_count": raw_json_count,
            "raw_a0_schema_valid_count": raw_a0_count,
            "raw_a0_definitions": raw_facts.get("definitions"),
            "protocol": {
                "prompt": runner_report.get("protocol", {}).get("prompt"),
                "tokenizer": runner_report.get("protocol", {}).get("tokenizer"),
                "budget": runner_report.get("budget"),
                "generation_config": runner_report.get("generation_config"),
            },
            "model_identity": {
                "model_weight_identity": runner_report.get("model_weight_identity"),
                "adapter_identity": runner_report.get("adapter_identity"),
            },
            "vllm": {
                "version": runner_report.get("api_server_runtime", {}).get("vllm_version"),
                "advertised_model_ids": runner_report.get("api_server_runtime", {}).get("advertised_model_ids"),
                "launch_configuration": runner_report.get("api_server_runtime", {}).get("launch_configuration"),
                "server_weight_identity": runner_report.get("api_server_runtime", {}).get("server_weight_identity"),
            },
        },
        "latency": {
            "unit": "ms",
            "distribution": latency_ms,
            "scope": "Runner per-sample timer: local prompt selection/tokenization + HTTP request/server generation + response decode + JSON parse/normalization + Submission Schema validation; excludes server/model startup and Runner tokenizer initialization",
            "all_rows_have_latency": len(latencies) == expected_count,
        },
        "generation": {
            "token_distribution": token_distribution,
            "total_generated_tokens": total_tokens,
            "tokens_per_second": (total_tokens / summed_request_seconds) if summed_request_seconds > 0 else None,
            "requests_per_second": (expected_count / summed_request_seconds) if summed_request_seconds > 0 else None,
            "tokens_per_second_definition": "sum of observed completion tokens divided by the sum of per-request latency intervals; includes failed/parse-invalid requests in elapsed time and excludes service startup and inter-request process gaps",
            "requests_per_second_definition": "completed request records divided by the sum of per-request latency intervals; includes failed/parse-invalid requests and excludes service startup and inter-request process gaps",
            "whole_process_tokens_per_second": (total_tokens / process_elapsed_seconds) if process_elapsed_seconds > 0 else None,
            "whole_process_requests_per_second": (expected_count / process_elapsed_seconds) if process_elapsed_seconds > 0 else None,
            "whole_process_throughput_definition": "observed completion tokens or completed request records divided by Runner process wall time; this interval includes start.sh preflight, vLLM startup/model load, requests, and service teardown",
            "request_seconds_sum": summed_request_seconds,
            "generation_cap": generation_cap,
            "token_cap_hit_count": len(cap_ids),
            "token_cap_hit_ids": cap_ids,
            "token_cap_hit_rate": len(cap_ids) / expected_count,
            "normal_stop_count": normal_stop_count,
            "normal_stop_rate": normal_stop_count / expected_count,
            "length_stop_count": length_stop_count,
            "length_stop_rate": length_stop_count / expected_count,
            "normal_stop_definition": "vLLM finish_reason=stop/eos; treated as a normal EOS stop observation",
            "stop_reason_counts": dict(sorted(stop_counts.items())),
        },
        "failures": {
            "non_success_status_count": expected_count - statuses.get("success", 0),
            "generation_failure_count": statuses.get("generation_failure", 0),
            "timeout_count": timeout_count,
            "oom_count": oom_count,
            "oom_observed_in_server_log": oom_log_lines > 0,
            "failure_category_counts": dict(sorted(error_categories.items())),
            "failure_error_type_counts": dict(sorted(error_types.items())),
            "oom_log_line_count": oom_log_lines,
            "failure_cases": [
                {
                    "id": row.get("id"),
                    "status": row.get("status"),
                    "failure_category": row.get("failure_category"),
                    "error_type": row.get("error_type"),
                    "stop_reason": row.get("stop_reason"),
                }
                for row in raw_rows if row.get("status") != "success"
            ],
        },
        "runtime_observations": {
            "process_elapsed_seconds": process_elapsed_seconds,
            "vllm_serve_to_health_ready_seconds": service_startup_seconds,
            "vllm_serve_to_health_ready_boundary": "vLLM child process launch to start.sh health endpoint readiness; includes model weight loading and other server initialization, not isolated model-load time",
            "model_load_time_seconds": None,
            "model_load_time_status": "not_separately_observable; included in vLLM serve-to-health-ready interval",
            "first_request_cold_start_effect_possible": True,
            "warmup_requests": 0,
            "gpu_memory": gpu_memory,
            "kv_cache": {
                "gpu_kv_cache_token_capacity_observations": kv_tokens,
                "status": "observed_in_vllm_log" if kv_tokens else "not_observed_in_vllm_log",
                "source": "vllm.log keyword/number scan; no configuration-derived estimate substituted",
            },
            "preemption_or_recompute_log_line_count": preemption_lines,
            "preemption_or_recompute_source": "vllm.log keyword scan; zero means no matching line was observed",
            "transformers_comparison_caveat": "Historical Full B Transformers runs used NF4; this E02.2-B vLLM service uses BF16. Latency differences cannot be attributed to the inference framework alone.",
        },
    }


class GpuMemorySampler:
    """Low-rate nvidia-smi sampler; samples whole-device usage, not vLLM-only."""

    def __init__(self, provider: Callable[[], dict[str, Any]] | None = None, interval_seconds: float = 1.0):
        self.provider = provider or probe.nvidia_gpu_facts
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._count = 0
        self._errors = 0
        self._first: dict[str, Any] | None = None
        self._last: dict[str, Any] | None = None
        self._peak_by_index: dict[str, dict[str, Any]] = {}

    def _record(self) -> None:
        try:
            facts = self.provider()
            if not isinstance(facts, dict) or facts.get("status") != "observed":
                raise ValueError("GPU provider returned no observed facts")
        except Exception:
            with self._lock:
                self._errors += 1
            return
        devices = facts.get("devices", [])
        with self._lock:
            self._count += 1
            if self._first is None:
                self._first = facts
            self._last = facts
            for index, device in enumerate(devices):
                key = str(index)
                used_value = device.get("memory_used_mib_at_observation")
                total_value = device.get("memory_total_mib")
                try:
                    used = int(used_value)
                    total = int(total_value)
                except (TypeError, ValueError):
                    continue
                old = self._peak_by_index.get(key)
                if old is None or used > old["peak_memory_used_mib_observed"]:
                    self._peak_by_index[key] = {
                        "index": index,
                        "name": device.get("name"),
                        "memory_total_mib": total,
                        "peak_memory_used_mib_observed": used,
                    }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="e02-gpu-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=25)
        with self._lock:
            return {
                "source": "nvidia-smi",
                "scope": "whole-device memory usage; includes other processes and is a periodic observed peak, not a hardware instantaneous maximum",
                "sampling_interval_seconds": self.interval_seconds,
                "sample_count": self._count,
                "query_error_count": self._errors,
                "first_observation": self._first,
                "last_observation": self._last,
                "peak_observed_by_device": [self._peak_by_index[key] for key in sorted(self._peak_by_index, key=int)],
                "peak_observability": "periodic nvidia-smi samples during preflight/startup/inference/teardown; shorter peaks can be missed",
            }


def _completed_sample_count(run_root: Path) -> int:
    path = run_root / "runner/run_progress.json"
    if path.is_file():
        try:
            value = _read_json(path, "Runner progress")
            count = value.get("completed_samples")
            return count if isinstance(count, int) and not isinstance(count, bool) else 0
        except Exception:
            return 0
    raw_path = run_root / "raw_predictions.jsonl"
    if raw_path.is_file():
        return sum(1 for line in raw_path.open(encoding="utf-8") if line.strip())
    return 0


def run_benchmark(
    *,
    selection_root: Path,
    config_path: Path,
    run_root: Path,
    start_script: Path,
    command_runner: Callable[..., Any] | None = None,
    gpu_memory_provider: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selection_root = selection_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    start_script = start_script.expanduser().resolve()
    try:
        run_root.relative_to(E02_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Benchmark Run 必须位于 ignored 目录 {E02_OUTPUT_ROOT}") from exc
    if not _RUN_ID_RE.fullmatch(run_root.name):
        raise ValueError("Run ID 仅允许字母数字、点、下划线和连字符")
    if run_root.exists():
        raise FileExistsError(f"拒绝覆盖已有 E02.2-B Benchmark Run: {run_root}")
    if run_root == selection_root:
        raise ValueError("Benchmark Run 必须与固定 Candidate Run 分开")
    if not start_script.is_file():
        raise FileNotFoundError(f"participant/start.sh 不存在: {start_script}")

    config = validate_benchmark_config(config_path)
    selection_manifest, candidate_path, candidate_rows = validate_candidate(selection_root, config_path)
    candidate_sha = sha256_file(candidate_path)
    config_sha = sha256_file(config_path)
    selection_manifest_path = selection_root / "selection_manifest.json"
    selection_manifest_sha = sha256_file(selection_manifest_path)
    code_hashes = {
        relative: sha256_file(WORKSPACE_ROOT / relative)
        for relative in (
            "participant/start.sh",
            "participant/run_inference.py",
            "scripts/e02_vllm_probe.py",
            "src/shared/prompting.py",
            "src/evaluation/metrics.py",
            "participant/templates/submission_schema.json",
        )
    }

    run_root.mkdir(parents=True, exist_ok=False)
    snapshot_input = run_root / "test_input.jsonl"
    snapshot_config = run_root / "inference_config.json"
    snapshot_input.write_bytes(candidate_path.read_bytes())
    snapshot_config.write_bytes(config_path.read_bytes())
    raw_path = run_root / "raw_predictions.jsonl"
    result_dir = run_root / "runner"
    timing_path = run_root / "vllm_lifecycle.jsonl"
    log_path = run_root / "vllm.log"
    initial_manifest: dict[str, Any] = {
        "schema_version": "e02_vllm_benchmark_run_v1",
        "run_id": run_root.name,
        "status": "running",
        "created_at_utc": _utc_now(),
        "candidate_run_id": selection_manifest.get("run_id"),
        "identity": {
            "candidate_input_sha256": candidate_sha,
            "candidate_row_count": len(candidate_rows),
            "selection_manifest_sha256": selection_manifest_sha,
            "selection_rule_sha256": selection_manifest.get("selection", {}).get("selection_rule_sha256"),
            "source_input_sha256": selection_manifest.get("source", {}).get("raw_input_sha256"),
            "smoke_input_sha256": selection_manifest.get("source", {}).get("excluded_smoke_input_sha256"),
            "id_map_sha256": selection_manifest.get("source", {}).get("id_map_sha256"),
            "inference_config_sha256": config_sha,
            "selector_script_sha256": selection_manifest.get("protocol", {}).get("selector_script_sha256"),
            "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
            "execution_code_sha256": code_hashes,
            "prompt_sha256": config["prompt"]["sha256"],
            "submission_schema_sha256": code_hashes["participant/templates/submission_schema.json"],
        },
        "paths": {
            "candidate_run_root": str(selection_root),
            "test_input_snapshot": str(snapshot_input),
            "inference_config_snapshot": str(snapshot_config),
            "runner_result_dir": str(result_dir),
            "raw_predictions": str(raw_path),
            "vllm_log": str(log_path),
            "vllm_lifecycle": str(timing_path),
        },
        "execution": {
            "entrypoint": str(start_script),
            "official_arguments": [str(snapshot_input), str(result_dir)],
            "backend": config["backend"],
            "api_model": config["api_model"],
            "vllm_expected_version": config["deployment"]["vllm"]["expected_version"],
            "gpu_memory_utilization": config["deployment"]["vllm"]["gpu_memory_utilization"],
        },
    }
    manifest_path = run_root / "run_manifest.json"
    _write_json_exclusive(manifest_path, initial_manifest)

    def finalize(status: str, *, return_code: int | None, started_epoch: float, details: dict[str, Any] | None = None) -> dict[str, Any]:
        current = _read_json(manifest_path, "Benchmark Manifest")
        current.update({
            "status": status,
            "finished_at_utc": _utc_now(),
            "return_code": return_code,
            "completed_samples": _completed_sample_count(run_root),
            "wall_elapsed_seconds": max(0.0, time.time() - started_epoch),
        })
        if details:
            current.update(details)
        _write_json_atomic(manifest_path, current)
        return current

    if sha256_file(candidate_path) != candidate_sha or sha256_file(snapshot_input) != candidate_sha:
        return finalize(
            "identity_changed_before_launch",
            return_code=None,
            started_epoch=time.time(),
            details={"failure_type": "InputIdentityMismatch", "failure_summary": "candidate input changed while the immutable run snapshot was being created"},
        )
    if sha256_file(snapshot_config) != config_sha:
        return finalize(
            "identity_changed_before_launch",
            return_code=None,
            started_epoch=time.time(),
            details={"failure_type": "ConfigIdentityMismatch", "failure_summary": "benchmark config snapshot differs from the validated config"},
        )

    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("EVALUATION_"):
            environment.pop(name, None)
    environment.update({
        "EVALUATION_BACKEND": "openai_compatible",
        "EVALUATION_INFERENCE_CONFIG": str(snapshot_config),
        "EVALUATION_RUN_ID": run_root.name,
        "EVALUATION_EXPERIMENT_ID": config["experiment_id"],
        "EVALUATION_RAW_OUTPUT_PATH": str(raw_path),
        "EVALUATION_RESUME": "true",
        "VLLM_HOST": "127.0.0.1",
        "VLLM_PORT": "8000",
        "VLLM_API_KEY": "",
        "VLLM_TENSOR_PARALLEL_SIZE": "1",
        "VLLM_MODEL_DIR": config["model_path"],
        "VLLM_SERVED_MODEL_NAME": "model",
        "VLLM_LORA_ENABLED": "true",
        "VLLM_LORA_NAME": config["adapter"]["expected_server_model_id"],
        "VLLM_LORA_PATH": config["adapter"]["path"],
        "VLLM_MAX_LORA_RANK": str(config["adapter"]["expected_rank"]),
        "VLLM_LOG_FILE": str(log_path),
        "E02_VLLM_TIMING_FILE": str(timing_path),
    })
    environment.pop("VLLM_GPU_MEMORY_UTILIZATION", None)
    environment.pop("E02_VLLM_RUN_ROOT", None)
    command = ["bash", str(start_script), str(snapshot_input), str(result_dir)]
    initial_manifest["execution"]["command"] = command
    initial_manifest["execution"]["resume_journal_enabled"] = True
    _write_json_atomic(manifest_path, initial_manifest)
    started_epoch = time.time()
    started_monotonic = time.monotonic()
    sampler = GpuMemorySampler(provider=gpu_memory_provider)
    sampler.start()
    return_code: int | None = None
    interrupted = False
    launch_error: Exception | None = None
    invoke = command_runner or subprocess.run
    try:
        result = invoke(command, cwd=WORKSPACE_ROOT, env=environment, check=False)
        return_code = getattr(result, "returncode", 0)
    except KeyboardInterrupt:
        interrupted = True
        return_code = 130
    except OSError as exc:
        launch_error = exc
        return_code = 127
    finally:
        gpu_memory = sampler.stop()
    process_elapsed = max(0.0, time.monotonic() - started_monotonic)

    if interrupted or return_code in {130, -2, -15}:
        return finalize(
            "interrupted",
            return_code=return_code,
            started_epoch=started_epoch,
            details={"failure_type": "KeyboardInterrupt", "gpu_memory": gpu_memory},
        )
    if launch_error is not None:
        return finalize(
            "launch_failed",
            return_code=return_code,
            started_epoch=started_epoch,
            details={"failure_type": type(launch_error).__name__, "gpu_memory": gpu_memory},
        )
    if return_code != 0:
        partial = _completed_sample_count(run_root)
        return finalize(
            "process_failed_with_partial_output" if partial else "process_failed",
            return_code=return_code,
            started_epoch=started_epoch,
            details={"gpu_memory": gpu_memory},
        )

    try:
        summary = summarize_runner_outputs(
            run_root=run_root,
            expected_ids=[row["id"] for row in candidate_rows],
            expected_input_sha256=candidate_sha,
            expected_config_sha256=config_sha,
            expected_run_id=run_root.name,
            config=config,
            process_elapsed_seconds=process_elapsed,
            gpu_memory=gpu_memory,
        )
    except Exception as exc:
        return finalize(
            "output_validation_failed",
            return_code=return_code,
            started_epoch=started_epoch,
            details={"failure_type": type(exc).__name__, "failure_summary": str(exc), "gpu_memory": gpu_memory},
        )
    summary_path = run_root / "performance_summary.json"
    _write_json_exclusive(summary_path, summary)
    current = _read_json(manifest_path, "Benchmark Manifest")
    has_request_failures = summary["failures"]["non_success_status_count"] > 0
    current.update({
        "status": "completed_with_request_failures" if has_request_failures else "completed",
        "finished_at_utc": _utc_now(),
        "return_code": return_code,
        "completed_samples": summary["completed_samples"],
        "wall_elapsed_seconds": max(0.0, time.time() - started_epoch),
        "performance_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "request_failures": summary["failures"]["non_success_status_count"],
            "raw_strict_json_count": summary["runner"]["raw_strict_json_object_count"],
            "raw_a0_valid_count": summary["runner"]["raw_a0_schema_valid_count"],
            "submission_schema_valid_count": summary["runner"]["submission_schema_valid_count"],
        },
        "gpu_memory": gpu_memory,
    })
    _write_json_atomic(manifest_path, current)
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["run"])
    parser.add_argument("--selection-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--start-script", type=Path, default=WORKSPACE_ROOT / "participant/start.sh")
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(
            selection_root=args.selection_root,
            config_path=args.config,
            run_root=args.run_root,
            start_script=args.start_script,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed_preflight", "failure_type": type(exc).__name__, "failure_summary": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "completed_samples": result.get("completed_samples", 0),
        "run_manifest": str(args.run_root.expanduser().resolve() / "run_manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else (130 if result["status"] == "interrupted" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
