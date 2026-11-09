#!/usr/bin/env python3
"""Official dual-Adapter inference entry point used by participant/start.sh."""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.final_dual_protocol import (  # noqa: E402
    PromptBuildError,
    assemble_submission,
    build_emotion_response_prompt,
    build_profile_prompt,
)
from src.inference.final_dual_vllm import DualLoRAVLLMService  # noqa: E402

SCHEMA_PATH = ROOT / "participant/templates/submission_schema.json"
ROUTES = (
    ("profile", "profile"),
    ("emotion_response", "emotion-response"),
)
def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _jsonl_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def read_test_samples(path: Path) -> list[dict[str, Any]]:
    """Read required id/history fields and exclude unrelated metadata from prompts."""
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or "id" not in row or "history" not in row:
                raise ValueError(f"{path}:{line_number}: each row must provide id and history")
            sample_id = row["id"]
            if not isinstance(sample_id, str):
                raise ValueError(f"{path}:{line_number}: id must be a string")

            raw_history = row["history"]
            if not isinstance(raw_history, list):
                history: list[dict[str, Any]] = []
            else:
                history = []
                for message in raw_history:
                    if not isinstance(message, dict):
                        history.append({})
                        continue
                    clean_message = {
                        key: message[key]
                        for key in ("role", "content")
                        if key in message
                    }
                    turn_id = message.get("turn_id")
                    if type(turn_id) is int and turn_id >= 0:
                        clean_message["turn_id"] = turn_id
                    history.append(clean_message)
            samples.append({"id": sample_id, "history": history})
    if not samples:
        raise ValueError(f"TEST_FILE contains no JSON records: {path}")
    return samples


def load_runtime_assets(config: dict[str, Any]) -> tuple[Any, dict[str, str], dict[str, Path]]:
    """Load only the local model inputs needed to run inference."""
    base_path = resolve_path(config.get("model_path", ""))
    if not base_path.is_dir() or not (base_path / "config.json").is_file():
        raise FileNotFoundError(f"Base model directory or config.json not found: {base_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_path), local_files_only=True, trust_remote_code=False
    )

    prompt_names = ("selection", "profile", "profile_history_budget", "emotion_response")
    prompt_records = config.get("prompts")
    if not isinstance(prompt_records, dict):
        raise ValueError("deployment config must define the four prompts")
    prompts: dict[str, str] = {}
    for name in prompt_names:
        record = prompt_records.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"deployment config is missing the {name} Prompt path")
        path = resolve_path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{name} Prompt not found: {path}")
        prompts[name] = path.read_text(encoding="utf-8")

    configured_adapters = config.get("adapters")
    if not isinstance(configured_adapters, dict):
        raise ValueError("deployment config must define profile and emotion_response Adapter paths")
    adapter_paths: dict[str, Path] = {}
    for key in ("profile", "emotion_response"):
        record = configured_adapters.get(key)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"deployment config is missing the {key} Adapter path")
        path = resolve_path(record["path"])
        if not path.is_dir():
            raise FileNotFoundError(f"{key} Adapter directory not found: {path}")
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (path / filename).is_file():
                raise FileNotFoundError(f"{key} Adapter is missing {filename}: {path}")
        adapter_paths[key] = path

    return tokenizer, prompts, adapter_paths


def _temporary_file(directory: Path, prefix: str, suffix: str = ".tmp") -> tuple[int, Path]:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.fchmod(descriptor, 0o644)
    return descriptor, Path(name)


def _write_report_temp(directory: Path, report: dict[str, Any]) -> Path:
    descriptor, path = _temporary_file(directory, ".performance_report.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]


def _load_schema() -> dict[str, Any]:
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"official submission schema not found: {SCHEMA_PATH}")
    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("official submission schema must be a JSON object")
    return value


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} TEST_FILE RESULT_DIR", file=sys.stderr)
        return 2

    submission_temp: Path | None = None
    report_temp: Path | None = None
    server_log: Path | None = None
    service: DualLoRAVLLMService | None = None
    runtime: Any = None
    try:
        test_file = Path(sys.argv[1]).expanduser().resolve()
        result_dir = Path(sys.argv[2]).expanduser().resolve()
        if not test_file.is_file() or test_file.suffix.lower() != ".jsonl":
            raise FileNotFoundError(f"TEST_FILE must be an existing .jsonl file: {test_file}")
        if result_dir.exists() and not result_dir.is_dir():
            raise NotADirectoryError(f"RESULT_DIR exists and is not a directory: {result_dir}")
        result_dir.mkdir(parents=True, exist_ok=True)

        config_value = os.environ.get(
            "EVALUATION_INFERENCE_CONFIG", "configs/deployment/final_dual_v1_2.json"
        )
        config_path = resolve_path(config_value)
        if not config_path.is_file():
            raise FileNotFoundError(f"deployment config not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("mode") != "dual_adapter":
            raise ValueError("deployment config mode must be dual_adapter")
        backend = os.environ.get("EVALUATION_BACKEND", config.get("backend", "transformers"))
        backend = str(backend).strip().lower()
        if backend not in {"transformers", "vllm"}:
            raise ValueError("dual_adapter backend must be transformers or vllm")

        samples = read_test_samples(test_file)
        tokenizer, prompts, adapter_paths = load_runtime_assets(config)
        schema = _load_schema()
        prompt_cap = int(config.get("context", {}).get("max_prompt_tokens", 1280))
        generation_config = config.get("generation", {})
        eos_token_id = int(generation_config.get("eos_token_id", getattr(tokenizer, "eos_token_id", 0)))
        pad_token_id = int(generation_config.get("pad_token_id", getattr(tokenizer, "pad_token_id", eos_token_id)))
        for route, _alias in ROUTES:
            if int(generation_config.get(route, {}).get("max_new_tokens", 0)) <= 0:
                raise ValueError(f"generation.{route}.max_new_tokens must be positive")

        init_started = time.perf_counter()
        service_facts: dict[str, Any] | None = None
        torch_module: Any = None
        if backend == "vllm":
            descriptor, log_name = tempfile.mkstemp(prefix="final-dual-vllm-", suffix=".log")
            os.close(descriptor)
            server_log = Path(log_name)
            service = DualLoRAVLLMService(
                config=config,
                root=ROOT,
                log_path=server_log,
                vllm_bin=os.environ.get("VLLM_BIN", "vllm"),
                startup_timeout_seconds=float(os.environ.get("VLLM_START_TIMEOUT", "900")),
            )
            service_facts = service.start()
            initialization_seconds = float(service_facts.get("startup_seconds", time.perf_counter() - init_started))
        else:
            import torch as torch_module
            from src.inference.final_dual_runtime import DualAdapterRuntime

            runtime = DualAdapterRuntime.load(
                base_model_path=resolve_path(config["model_path"]),
                tokenizer=tokenizer,
                adapters={
                    "profile": adapter_paths["profile"],
                    "emotion-response": adapter_paths["emotion_response"],
                },
                torch_module=torch_module,
            )
            initialization_seconds = time.perf_counter() - init_started

        submission_descriptor, submission_temp = _temporary_file(result_dir, ".submission.")
        latencies_ms: list[float] = []
        fallback_counts = {"profile": 0, "emotion_label": 0, "response_text": 0}
        fallback_sample_ids: set[str] = set()
        submission_rows = 0
        try:
            with os.fdopen(submission_descriptor, "w", encoding="utf-8", newline="\n") as submission_handle:
                for index, sample in enumerate(samples, 1):
                    sample_started = time.perf_counter()
                    prepared: dict[str, dict[str, Any] | None] = {
                        "profile": None,
                        "emotion_response": None,
                    }
                    generated: dict[str, str | None] = {
                        "profile": None,
                        "emotion_response": None,
                    }
                    try:
                        prepared["profile"] = build_profile_prompt(
                            sample,
                            tokenizer,
                            selection_prompt=prompts["selection"],
                            profile_prompt=prompts["profile"],
                            profile_budget_prompt=prompts["profile_history_budget"],
                            prompt_cap=prompt_cap,
                            include_diagnostics=False,
                        )
                    except PromptBuildError as exc:
                        print(f"WARNING: {sample['id']} profile prompt unavailable; using field fallback: {exc}", file=sys.stderr)
                    try:
                        prepared["emotion_response"] = build_emotion_response_prompt(
                            sample,
                            tokenizer,
                            selection_prompt=prompts["selection"],
                            emotion_response_prompt=prompts["emotion_response"],
                            prompt_cap=prompt_cap,
                            include_diagnostics=False,
                        )
                    except PromptBuildError as exc:
                        print(f"WARNING: {sample['id']} emotion-response prompt unavailable; using field fallback: {exc}", file=sys.stderr)

                    for route, model_alias in ROUTES:
                        prompt = prepared[route]
                        if prompt is None:
                            continue
                        max_new_tokens = int(generation_config[route]["max_new_tokens"])
                        if backend == "vllm":
                            if service is None:
                                raise RuntimeError("vLLM service is not initialized")
                            response = service.chat_completion(
                                model_alias=model_alias,
                                messages=prompt["messages"],
                                max_tokens=max_new_tokens,
                            )
                        else:
                            if runtime is None:
                                raise RuntimeError("Transformers runtime is not initialized")
                            adapter_name = "profile" if route == "profile" else "emotion-response"
                            response = runtime.generate(
                                adapter_name,
                                prompt["prompt_ids"],
                                max_new_tokens=max_new_tokens,
                                eos_token_id=eos_token_id,
                                pad_token_id=pad_token_id,
                            )
                        text = response.get("text") if isinstance(response, dict) else None
                        generated[route] = text if isinstance(text, str) else ""

                    submission, diagnostics = assemble_submission(
                        sample["id"],
                        generated["profile"],
                        generated["emotion_response"],
                        schema,
                        validate_schema=False,
                    )
                    row_errors = {
                        "profile": diagnostics.get("profile_parse_error"),
                        "emotion_label": diagnostics.get("emotion_parse_error"),
                        "response_text": diagnostics.get("response_parse_error"),
                    }
                    for field, error in row_errors.items():
                        if error is not None:
                            fallback_counts[field] += 1
                            fallback_sample_ids.add(sample["id"])

                    submission_handle.write(_jsonl_line(submission))
                    submission_rows += 1
                    elapsed_ms = (time.perf_counter() - sample_started) * 1000
                    if index <= 100:
                        latencies_ms.append(elapsed_ms)
                    print(f"[{index}/{len(samples)}] {sample['id']}: {elapsed_ms:.3f} ms", flush=True)
                submission_handle.flush()
                os.fsync(submission_handle.fileno())
        except Exception:
            if submission_temp is not None:
                submission_temp.unlink(missing_ok=True)
            raise

        if service is not None:
            service.close()
            service = None
        if runtime is not None:
            runtime.close()
            runtime = None

        report = {
            "version": "final-dual-official-run-v2",
            "status": "passed",
            "complete": submission_rows == len(samples),
            "backend": backend,
            "samples": len(samples),
            "submission_rows": submission_rows,
            "initialization_seconds": round(initialization_seconds, 6),
            "field_fallback_count": fallback_counts,
            "field_fallback_sample_ids": sorted(fallback_sample_ids),
            "performance": {
                "measurement_limit": 100,
                "measured_samples": len(latencies_ms),
                "latencies_ms": [round(value, 6) for value in latencies_ms],
                "average_latency_ms": round(statistics.fmean(latencies_ms), 6) if latencies_ms else None,
                "median_latency_ms": round(statistics.median(latencies_ms), 6) if latencies_ms else None,
                "p95_latency_ms": round(_percentile95(latencies_ms), 6) if latencies_ms else None,
            },
            "server": {"ready": service_facts is not None, "startup_seconds": service_facts.get("startup_seconds") if service_facts else None},
            "outputs": {
                "submission.jsonl": {"path": "submission.jsonl", "rows": submission_rows},
                "performance_report.json": {"path": "performance_report.json"},
            },
        }
        if not report["complete"]:
            raise OSError(f"wrote {submission_rows} outputs for {len(samples)} input records")

        report_temp = _write_report_temp(result_dir, report)
        os.replace(submission_temp, result_dir / "submission.jsonl")
        submission_temp = None
        os.replace(report_temp, result_dir / "performance_report.json")
        report_temp = None
        print(f"PASS: {result_dir / 'submission.jsonl'}", flush=True)
        print(f"PASS: {result_dir / 'performance_report.json'}", flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if service is not None:
            try:
                service.close()
            except Exception as exc:
                print(f"WARNING: could not stop vLLM cleanly: {exc}", file=sys.stderr, flush=True)
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                print(f"WARNING: could not close Transformers runtime cleanly: {exc}", file=sys.stderr, flush=True)
        for path in (submission_temp, report_temp, server_log):
            if path is not None:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
