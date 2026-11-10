#!/usr/bin/env python3
"""Official dual-Adapter inference entry point used by participant/start.sh."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.final_dual_protocol import (  # noqa: E402
    PromptBuildError,
    assemble_submission,
    build_emotion_response_prompt,
    build_profile_prompt,
)
from src.inference.final_dual_vllm import DualLoRAVLLMService, VLLMServiceError  # noqa: E402

SCHEMA_PATH = ROOT / "participant/templates/submission_schema.json"
ROUTES = (
    ("profile", "profile"),
    ("emotion_response", "emotion-response"),
)
MEASUREMENT_LIMIT = 100


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = _temporary_file(path.parent, f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_test_samples(path: Path) -> list[dict[str, Any]]:
    """Read required id/history fields and exclude unrelated metadata from prompts."""
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            sample_id = row.get("id")
            if not isinstance(sample_id, str):
                raise ValueError(f"{path}:{line_number}: id must be a string")
            if sample_id in seen_ids:
                print(
                    f"WARNING: duplicate test id {sample_id!r} at line {line_number}; "
                    "the platform may reject duplicate output IDs",
                    file=sys.stderr,
                    flush=True,
                )
            seen_ids.add(sample_id)

            raw_history = row.get("history")
            if not isinstance(raw_history, list):
                raise ValueError(f"{path}:{line_number}: history must be an array")
            history: list[dict[str, Any]] = []
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


def _normalize_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend in {"openai_compatible", "openai-compatible", "openai", "vllm"}:
        return "vllm"
    if backend == "transformers":
        return "transformers"
    raise ValueError("dual_adapter backend must be transformers or openai_compatible/vllm")


def _runtime_diagnostics(config_path: Path, backend: str) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "backend": backend,
        "config_path": str(config_path),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["torch_cuda"] = torch.version.cuda
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            facts["device_name"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            facts["compute_capability"] = f"{capability[0]}.{capability[1]}"
    except Exception as exc:
        facts["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import importlib.metadata

        facts["vllm"] = importlib.metadata.version("vllm")
    except Exception as exc:
        facts["vllm_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import importlib.metadata

        facts["transformers"] = importlib.metadata.version("transformers")
    except Exception as exc:
        facts["transformers_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _valid_report_latencies(latencies: list[float]) -> list[float]:
    return [round(value, 6) for value in latencies]


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} TEST_FILE RESULT_DIR", file=sys.stderr)
        return 2

    submission_temp: Path | None = None
    report_temp: Path | None = None
    server_log: Path | None = None
    service: DualLoRAVLLMService | None = None
    runtime: Any = None
    result_dir: Path | None = None
    failure_error: str | None = None
    degraded_reason: str | None = None
    service_unavailable = False
    clean_success = False
    diagnostics: dict[str, Any] = {}
    schema_invalid_count = 0

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
        backend = _normalize_backend(
            os.environ.get("EVALUATION_BACKEND", config.get("backend", "transformers"))
        )
        fallback_enabled = bool(
            config.get("fallback", {}).get("degraded_output_on_runtime_failure", False)
            if isinstance(config.get("fallback"), dict)
            else False
        )
        diagnostics = _runtime_diagnostics(config_path, backend)

        samples = read_test_samples(test_file)
        test_file_sha256 = sha256_file(test_file)

        schema: dict[str, Any] | None
        try:
            schema = _load_schema()
        except Exception as exc:
            schema = None
            degraded_reason = f"submission_schema_unavailable: {type(exc).__name__}: {exc}"
            print(f"WARNING: {degraded_reason}", file=sys.stderr, flush=True)

        if backend == "transformers":
            import importlib.util

            missing = [
                name
                for name in ("torch", "peft", "accelerate", "bitsandbytes")
                if importlib.util.find_spec(name) is None
            ]
            if missing:
                print(
                    "WARNING: transformers backend dependencies are missing "
                    f"({', '.join(missing)}); falling back to vLLM",
                    file=sys.stderr,
                    flush=True,
                )
                backend = "vllm"
                diagnostics["backend"] = backend
            else:
                try:
                    import torch as torch_module  # noqa: F401
                    from src.inference.final_dual_runtime import DualAdapterRuntime  # noqa: F401
                except Exception as exc:
                    print(
                        "WARNING: transformers backend is unavailable in this image "
                        f"({type(exc).__name__}: {exc}); falling back to vLLM",
                        file=sys.stderr,
                        flush=True,
                    )
                    backend = "vllm"
                    diagnostics["backend"] = backend

        generation_config = config.get("generation", {})
        eos_token_id = int(generation_config.get("eos_token_id", 0))
        pad_token_id = int(generation_config.get("pad_token_id", eos_token_id))
        for route, _alias in ROUTES:
            if int(generation_config.get(route, {}).get("max_new_tokens", 0)) <= 0:
                raise ValueError(f"generation.{route}.max_new_tokens must be positive")

        tokenizer: Any = None
        prompts: dict[str, str] = {}
        adapter_paths: dict[str, Path] = {}
        assets_error: Exception | None = None
        try:
            tokenizer, prompts, adapter_paths = load_runtime_assets(config)
            eos_token_id = int(
                generation_config.get("eos_token_id", getattr(tokenizer, "eos_token_id", 0))
            )
            pad_token_id = int(
                generation_config.get("pad_token_id", getattr(tokenizer, "pad_token_id", eos_token_id))
            )
        except Exception as exc:
            assets_error = exc
            print(
                f"WARNING: runtime assets unavailable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not fallback_enabled:
                raise
            degraded_reason = degraded_reason or f"runtime_assets_unavailable: {type(exc).__name__}: {exc}"

        if degraded_reason is not None and not fallback_enabled:
            raise RuntimeError(degraded_reason)

        service_facts: dict[str, Any] | None = None
        init_started = time.perf_counter()
        if assets_error is None and degraded_reason is None and backend == "vllm":
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
            try:
                service_facts = service.start()
            except Exception as exc:
                print(
                    f"WARNING: vLLM could not start: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if not fallback_enabled:
                    raise
                first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                degraded_reason = f"vllm_start_failed: {first_line[:800]}"
                service = None
        elif assets_error is None and degraded_reason is None and backend == "transformers":
            import torch as torch_module  # noqa: F811
            from src.inference.final_dual_runtime import DualAdapterRuntime

            try:
                runtime = DualAdapterRuntime.load(
                    base_model_path=resolve_path(config["model_path"]),
                    tokenizer=tokenizer,
                    adapters={
                        "profile": adapter_paths["profile"],
                        "emotion-response": adapter_paths["emotion_response"],
                    },
                    torch_module=torch_module,
                )
            except Exception as exc:
                print(
                    f"WARNING: transformers runtime could not load: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if not fallback_enabled:
                    raise
                degraded_reason = f"transformers_load_failed: {type(exc).__name__}: {exc}"
                runtime = None
        initialization_seconds = (
            float(service_facts.get("startup_seconds", time.perf_counter() - init_started))
            if service_facts
            else time.perf_counter() - init_started
        )

        submission_descriptor, submission_temp = _temporary_file(result_dir, ".submission.")
        latencies_ms: list[float] = []
        fallback_counts = {"profile": 0, "emotion_label": 0, "response_text": 0}
        fallback_sample_ids: set[str] = set()
        submission_rows = 0
        schema_valid_count = 0
        schema_invalid_count = 0
        schema_error_samples: list[str] = []
        validator = Draft202012Validator(schema) if schema is not None else None

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
                    if degraded_reason is None and not service_unavailable:
                        for route, builder in (
                            (
                                "profile",
                                lambda: build_profile_prompt(
                                    sample,
                                    tokenizer,
                                    selection_prompt=prompts["selection"],
                                    profile_prompt=prompts["profile"],
                                    profile_budget_prompt=prompts["profile_history_budget"],
                                    prompt_cap=int(config.get("context", {}).get("max_prompt_tokens", 1280)),
                                    include_diagnostics=False,
                                ),
                            ),
                            (
                                "emotion_response",
                                lambda: build_emotion_response_prompt(
                                    sample,
                                    tokenizer,
                                    selection_prompt=prompts["selection"],
                                    emotion_response_prompt=prompts["emotion_response"],
                                    prompt_cap=int(config.get("context", {}).get("max_prompt_tokens", 1280)),
                                    include_diagnostics=False,
                                ),
                            ),
                        ):
                            try:
                                prepared[route] = builder()
                            except PromptBuildError as exc:
                                print(
                                    f"WARNING: {sample['id']} {route} prompt unavailable; "
                                    f"using field fallback: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            except Exception as exc:
                                print(
                                    f"WARNING: {sample['id']} {route} prompt construction failed; "
                                    f"using field fallback: {type(exc).__name__}: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )

                    for route, model_alias in ROUTES:
                        prompt = prepared[route]
                        if prompt is None or degraded_reason is not None or service_unavailable:
                            continue
                        max_new_tokens = int(generation_config[route]["max_new_tokens"])
                        try:
                            if backend == "vllm":
                                if service is None:
                                    raise VLLMServiceError("vLLM service is not initialized")
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
                        except Exception as exc:
                            print(
                                f"WARNING: {sample['id']} {route} request failed; using field fallback: "
                                f"{type(exc).__name__}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            generated[route] = None
                            if backend == "vllm":
                                unavailable = service is None or not service.is_alive()
                                if not unavailable:
                                    try:
                                        service.list_models(timeout=5)
                                    except Exception:
                                        unavailable = True
                                if unavailable:
                                    service_unavailable = True
                                    if not fallback_enabled:
                                        raise RuntimeError(
                                            "vLLM service became unavailable during inference"
                                        ) from exc
                                    print(
                                        "WARNING: vLLM became unavailable; writing field fallbacks "
                                        "for the remaining samples",
                                        file=sys.stderr,
                                        flush=True,
                                    )

                    submission, diagnostics_row = assemble_submission(
                        sample["id"],
                        generated["profile"],
                        generated["emotion_response"],
                        schema if schema is not None else {},
                    )
                    row_errors = {
                        "profile": diagnostics_row.get("profile_parse_error"),
                        "emotion_label": diagnostics_row.get("emotion_parse_error"),
                        "response_text": diagnostics_row.get("response_parse_error"),
                    }
                    for field, error in row_errors.items():
                        if error is not None:
                            fallback_counts[field] += 1
                            fallback_sample_ids.add(sample["id"])

                    if validator is not None:
                        errors = sorted(validator.iter_errors(submission), key=lambda item: list(item.path))
                        if errors:
                            schema_invalid_count += 1
                            if schema_invalid_count <= 5:
                                schema_error_samples.append(sample["id"])
                                print(
                                    f"WARNING: {sample['id']} submission does not satisfy the official "
                                    f"Schema: {errors[0].message}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                        else:
                            schema_valid_count += 1

                    submission_handle.write(_jsonl_line(submission))
                    submission_rows += 1
                    elapsed_ms = (time.perf_counter() - sample_started) * 1000
                    if index <= MEASUREMENT_LIMIT:
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

        submission_sha256 = sha256_file(submission_temp)
        latencies_rounded = _valid_report_latencies(latencies_ms)
        average_latency = round(statistics.fmean(latencies_ms), 6) if latencies_ms else None
        median_latency = round(statistics.median(latencies_ms), 6) if latencies_ms else None
        p95_latency = round(_percentile95(latencies_ms), 6) if latencies_ms else None
        minimum_latency = round(min(latencies_ms), 6) if latencies_ms else None
        maximum_latency = round(max(latencies_ms), 6) if latencies_ms else None
        hardware_label = os.environ.get("EVALUATION_HARDWARE_LABEL") or socket.gethostname()
        parse_failure_count = sum(fallback_counts.values())
        status = "degraded" if (degraded_reason is not None or service_unavailable) else "passed"

        report = {
            "version": "final-dual-official-run-v3",
            "status": status,
            "inference_success": status == "passed",
            "degraded_reason": degraded_reason,
            "service_unavailable": service_unavailable,
            "complete": submission_rows == len(samples),
            "backend": backend,
            "official_backend": "openai_compatible" if backend == "vllm" else "transformers",
            "samples": len(samples),
            "submission_rows": submission_rows,
            "initialization_seconds": round(initialization_seconds, 6),
            "field_fallback_count": fallback_counts,
            "field_fallback_sample_ids": sorted(fallback_sample_ids),
            "required_rounds": MEASUREMENT_LIMIT,
            "rounds": len(latencies_ms),
            "average_latency_ms": average_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "min_latency_ms": minimum_latency,
            "max_latency_ms": maximum_latency,
            "latencies_ms": latencies_rounded,
            "hardware_label": hardware_label,
            "test_file_sha256": test_file_sha256,
            "submission_sha256": submission_sha256,
            "parse_failure_count": parse_failure_count,
            "parse_failure_ids": sorted(fallback_sample_ids),
            "timing_scope": (
                "per-sample prompt construction through two serial generations, parsing, and "
                "output append; server/model initialization excluded"
            ),
            "model_id": config.get("model_id"),
            "model_revision": config.get("model_revision"),
            "generation_config": generation_config,
            "submission_schema": {
                "validated": validator is not None,
                "valid_count": schema_valid_count,
                "invalid_count": schema_invalid_count,
                "invalid_sample_ids_first": schema_error_samples,
            },
            "runtime_environment": diagnostics,
            "performance": {
                "measurement_limit": MEASUREMENT_LIMIT,
                "measured_samples": len(latencies_ms),
                "latencies_ms": latencies_rounded,
                "average_latency_ms": average_latency,
                "median_latency_ms": median_latency,
                "p95_latency_ms": p95_latency,
            },
            "server": {
                "ready": service_facts is not None,
                "startup_seconds": service_facts.get("startup_seconds") if service_facts else None,
                "attempts": service_facts.get("attempts") if service_facts else None,
                "port": service_facts.get("port") if service_facts else None,
                "gpu_memory_utilization": (
                    service_facts.get("gpu_memory_utilization") if service_facts else None
                ),
                "enforce_eager": service_facts.get("enforce_eager") if service_facts else None,
                "models": service_facts.get("models") if service_facts else None,
            },
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
        clean_success = status == "passed"
        print(f"PASS: {result_dir / 'submission.jsonl'}", flush=True)
        print(f"PASS: {result_dir / 'performance_report.json'}", flush=True)
        return 0
    except Exception as exc:
        failure_error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {failure_error}", file=sys.stderr, flush=True)
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
                print(
                    f"WARNING: could not close Transformers runtime cleanly: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if result_dir is not None and (failure_error is not None or degraded_reason is not None or service_unavailable):
            log_text = ""
            if server_log is not None and server_log.is_file():
                try:
                    log_text = server_log.read_text(encoding="utf-8", errors="replace")
                    if log_text:
                        shutil.copyfile(server_log, result_dir / "vllm_server.log")
                        print(
                            f"INFO: vLLM log preserved at {result_dir / 'vllm_server.log'}",
                            file=sys.stderr,
                            flush=True,
                        )
                except OSError:
                    pass
            failure = {
                "status": "failed" if failure_error is not None else "degraded",
                "error": failure_error,
                "degraded_reason": degraded_reason,
                "service_unavailable": service_unavailable,
                "diagnostics": diagnostics,
                "vllm_log_tail": log_text[-8000:],
            }
            try:
                _write_json_atomic(result_dir / "runtime_failure.json", failure)
                print(
                    f"INFO: runtime diagnostics preserved at {result_dir / 'runtime_failure.json'}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
        for path in (submission_temp, report_temp):
            if path is not None:
                path.unlink(missing_ok=True)
        if server_log is not None and server_log.exists():
            if clean_success:
                server_log.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
