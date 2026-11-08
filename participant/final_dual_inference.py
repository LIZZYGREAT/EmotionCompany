#!/usr/bin/env python3
"""Official dual-Adapter entry point used by participant/start.sh."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.final_dual_protocol import (  # noqa: E402
    assemble_submission,
    build_emotion_response_prompt,
    build_profile_prompt,
)
from src.inference.final_dual_vllm import (  # noqa: E402
    DualLoRAVLLMService,
    GpuMemoryMonitor,
    gpu_snapshot,
)

OFFICIAL_ID = re.compile(r"^test_[0-9]{6}_t[0-9]+$")
ROUTE_ORDER = ("profile", "emotion_response")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _execution_git_head() -> str | None:
    declared = os.environ.get("EVALUATION_GIT_HEAD", "").strip()
    if declared:
        return declared
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    manifest_path = ROOT / "package_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = manifest.get("source_commit")
        return value if isinstance(value, str) and value else None
    return None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _jsonl_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _read_jsonl_prefix(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    rows: list[dict[str, Any]] = []
    lines: list[bytes] = []
    if not path.is_file():
        return rows, lines
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            if not isinstance(row, dict):
                break
            rows.append(row)
            lines.append(line)
    return rows, lines


def read_test_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or set(row) != {"id", "history"}:
                raise ValueError(f"{path}:{line_number}: each row must contain only id and history")
            sample_id, history = row["id"], row["history"]
            if not isinstance(sample_id, str) or not OFFICIAL_ID.fullmatch(sample_id) or sample_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate official test ID")
            if not isinstance(history, list) or not history:
                raise ValueError(f"{path}:{line_number}: history must be a non-empty array")
            for message in history:
                if not isinstance(message, dict) or set(message) - {"role", "content", "turn_id"}:
                    raise ValueError(f"{path}:{line_number}: history message contains unsupported fields")
                if message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str):
                    raise ValueError(f"{path}:{line_number}: history message must have role and string content")
                if "turn_id" in message and (type(message["turn_id"]) is not int or message["turn_id"] < 0):
                    raise ValueError(f"{path}:{line_number}: turn_id must be a non-negative integer")
            if history[-1].get("role") != "user":
                raise ValueError(f"{path}:{line_number}: history must end with the current user message")
            samples.append({"id": sample_id, "history": history})
            seen.add(sample_id)
    if not samples:
        raise ValueError(f"test file is empty: {path}")
    return samples


def _resume_identity(
    *,
    result_dir: Path,
    test_file: Path,
    config_path: Path,
    lock_path: Path,
    samples: list[dict[str, Any]],
    backend: str,
    resume_enabled: bool,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    submission_path = result_dir / "submission.jsonl"
    raw_path = result_dir / "raw_outputs.jsonl"
    report_path = result_dir / "performance_report.json"
    progress_path = result_dir / "run_progress.json"
    if report_path.exists():
        raise FileExistsError("performance_report.json exists; a completed run cannot be overwritten or resumed")
    previous_progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
    requested_run_id = os.environ.get("EVALUATION_RUN_ID", "").strip()
    run_id = requested_run_id or (previous_progress or {}).get("run_id") or uuid.uuid4().hex
    identity = {
        "run_id": run_id,
        "test_file_sha256": sha256_file(test_file),
        "effective_config_sha256": sha256_file(config_path),
        "model_lock_sha256": sha256_file(lock_path),
        "backend": backend,
        "sample_count": len(samples),
        "sample_ids_sha256": hashlib.sha256("\n".join(sample["id"] for sample in samples).encode("utf-8")).hexdigest(),
        "submission_path": str(submission_path.resolve()),
        "raw_output_path": str(raw_path.resolve()),
    }
    if previous_progress is not None and not resume_enabled:
        raise FileExistsError("an interrupted dual-Adapter run exists; set EVALUATION_RESUME=true to continue")
    if not resume_enabled:
        nonempty = [path.name for path in (submission_path, raw_path) if path.exists() and path.stat().st_size]
        if nonempty:
            raise FileExistsError(f"refusing to overwrite existing outputs without resume: {nonempty}")
        completed = 0
        submissions: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []
        submission_bytes = raw_bytes = b""
        progress = {
            **identity,
            "status": "running",
            "attempt": 1,
            "completed_samples": 0,
            "submission_prefix_sha256": hashlib.sha256(b"").hexdigest(),
            "raw_prefix_sha256": hashlib.sha256(b"").hexdigest(),
        }
    else:
        if previous_progress is None:
            nonempty = [path.name for path in (submission_path, raw_path) if path.exists() and path.stat().st_size]
            if nonempty:
                raise FileExistsError(f"existing outputs have no resumable progress record: {nonempty}")
            completed = 0
            submissions, raw_rows = [], []
            submission_bytes = raw_bytes = b""
            progress = {
                **identity,
                "status": "running",
                "attempt": 1,
                "completed_samples": 0,
                "submission_prefix_sha256": hashlib.sha256(b"").hexdigest(),
                "raw_prefix_sha256": hashlib.sha256(b"").hexdigest(),
            }
        else:
            if any(previous_progress.get(key) != value for key, value in identity.items()):
                raise ValueError("resume input, model lock, backend, ordered IDs, Run ID, or output path changed")
            completed = previous_progress.get("completed_samples")
            if type(completed) is not int or not 0 <= completed <= len(samples):
                raise ValueError("resume completed_samples is invalid")
            submissions, submission_lines = _read_jsonl_prefix(submission_path)
            raw_rows, raw_lines = _read_jsonl_prefix(raw_path)
            expected_ids = [sample["id"] for sample in samples[:completed]]
            if [row.get("id") for row in submissions[:completed]] != expected_ids:
                raise ValueError("committed submission prefix IDs differ from the input")
            if [row.get("id") for row in raw_rows[:completed]] != expected_ids:
                raise ValueError("committed raw output prefix IDs differ from the input")
            if len(submission_lines) < completed or len(raw_lines) < completed:
                raise ValueError("resume output files are shorter than committed progress")
            submission_bytes = b"".join(submission_lines[:completed])
            raw_bytes = b"".join(raw_lines[:completed])
            if previous_progress.get("submission_prefix_sha256") != hashlib.sha256(submission_bytes).hexdigest():
                raise ValueError("committed submission prefix hash changed")
            if previous_progress.get("raw_prefix_sha256") != hashlib.sha256(raw_bytes).hexdigest():
                raise ValueError("committed raw output prefix hash changed")
            # Trim any one-file-ahead append left by a process interruption.
            submission_path.write_bytes(submission_bytes)
            raw_path.write_bytes(raw_bytes)
            progress = {**previous_progress, "status": "running", "attempt": int(previous_progress.get("attempt", 1)) + 1}
    result_dir.mkdir(parents=True, exist_ok=True)
    submission_path.write_bytes(submission_bytes)
    raw_path.write_bytes(raw_bytes)
    _atomic_json(progress_path, {**progress, "updated_at_epoch_seconds": time.time()})
    return completed, submissions[:completed], raw_rows[:completed], progress


def _verify_assets(config: dict[str, Any], config_path: Path, lock_path: Path) -> tuple[Any, dict[str, Any]]:
    from participant import run_inference as runner

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("deployment_config", {}).get("sha256") != sha256_file(config_path):
        raise RuntimeError("deployment config SHA-256 differs from model_lock.json")
    if lock.get("deployment_config", {}).get("path") != str(config_path.relative_to(ROOT)):
        raise RuntimeError("deployment config path differs from model_lock.json")
    base_path = resolve_path(config["model_path"])
    base_config = {"model_id": config["model_id"], "model_revision": config["model_revision"]}
    base_identity = runner._weight_identity(base_path, base_config)
    expected_base_sha = config["expected_base_weight_sha256"]
    locked_base_identity = lock.get("base", {}).get("observed", {})
    if base_identity.get("observed_weight_hash") != expected_base_sha or locked_base_identity.get("observed_weight_hash") != expected_base_sha:
        raise RuntimeError("local Base weight identity does not match the dual-Adapter lock")
    tokenizer_config = {**config["tokenizer"], "local_path": str(base_path), "local_files_only": True}
    tokenizer, tokenizer_identity = runner.load_runtime_tokenizer({
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "tokenizer": tokenizer_config,
    })
    if tokenizer_identity.get("signature_sha256") != config["tokenizer"]["expected_signature_sha256"]:
        raise RuntimeError("runtime Tokenizer signature differs from the dual-Adapter lock")
    if tokenizer_identity.get("template_sha256") != config["tokenizer"]["expected_template_sha256"]:
        raise RuntimeError("runtime Chat Template differs from the dual-Adapter lock")
    prompts = {}
    for name, record in config["prompts"].items():
        path = resolve_path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"locked Prompt SHA-256 mismatch: {name}")
        prompts[name] = path.read_text(encoding="utf-8")
    adapters = {}
    deployment = lock["deployment_adapters"]
    for key in ("profile", "emotion_response"):
        record = deployment[key]
        adapter_path = resolve_path(record["path"])
        weights_path = adapter_path / "adapter_model.safetensors"
        config_file = adapter_path / "adapter_config.json"
        if sha256_file(weights_path) != record["weight_sha256"] or sha256_file(config_file) != record["config_sha256"]:
            raise RuntimeError(f"deployed {key} Adapter SHA-256 differs from model_lock.json")
        adapter_config = json.loads(config_file.read_text(encoding="utf-8"))
        if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
            raise RuntimeError(f"deployed {key} adapter_config is not causal-LM LoRA")
        if adapter_config.get("r", 0) > config["deployment"]["vllm"]["max_lora_rank"]:
            raise RuntimeError(f"deployed {key} Adapter exceeds configured max_lora_rank")
        adapters[key] = {
            **record,
            "resolved_path": str(adapter_path),
            "adapter_config": adapter_config,
        }
    identity = {
        "base": base_identity,
        "tokenizer": tokenizer_identity,
        "prompts": {name: {"path": config["prompts"][name]["path"], "sha256": config["prompts"][name]["sha256"]} for name in prompts},
        "adapters": adapters,
        "model_lock": {"path": str(lock_path.relative_to(ROOT)), "sha256": sha256_file(lock_path)},
    }
    return tokenizer, identity


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, int(0.95 * len(values) + 0.999999) - 1)]


def _generation_metadata(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {key: value for key, value in result.items() if key != "text"}


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} TEST_FILE RESULT_DIR", file=sys.stderr)
        return 2
    test_file = Path(sys.argv[1]).expanduser().resolve()
    result_dir = Path(sys.argv[2]).expanduser().resolve()
    if not test_file.is_file() or test_file.suffix.lower() != ".jsonl":
        raise ValueError(f"TEST_FILE must be an existing .jsonl file: {test_file}")
    if result_dir.exists() and not result_dir.is_dir():
        raise ValueError(f"RESULT_DIR exists and is not a directory: {result_dir}")
    config_value = os.environ.get("EVALUATION_INFERENCE_CONFIG", "configs/deployment/final_dual.yaml")
    config_path = resolve_path(config_value)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "dual_adapter":
        raise ValueError("deployment config mode must be dual_adapter")
    backend = os.environ.get("EVALUATION_BACKEND", config.get("backend", "transformers")).strip().lower()
    if backend not in {"transformers", "vllm"}:
        raise ValueError("dual_adapter backend must be transformers or vllm")
    lock_path = resolve_path(config["model_lock_path"])
    if not lock_path.is_file():
        raise FileNotFoundError(f"model lock not found: {lock_path}")
    samples = read_test_samples(test_file)
    result_dir.mkdir(parents=True, exist_ok=True)
    resume_value = os.environ.get("EVALUATION_RESUME", "").strip().lower()
    if resume_value not in {"", "0", "false", "no", "1", "true", "yes"}:
        raise ValueError("EVALUATION_RESUME must be true or false")
    resume_enabled = resume_value in {"1", "true", "yes"}
    tokenizer, asset_identity = _verify_assets(config, config_path, lock_path)
    completed, prior_predictions, prior_raw, progress = _resume_identity(
        result_dir=result_dir,
        test_file=test_file,
        config_path=config_path,
        lock_path=lock_path,
        samples=samples,
        backend=backend,
        resume_enabled=resume_enabled,
    )
    submission_path = result_dir / "submission.jsonl"
    raw_path = result_dir / "raw_outputs.jsonl"
    progress_path = result_dir / "run_progress.json"
    schema = json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8"))
    if len(prior_predictions) != completed or len(prior_raw) != completed:
        raise ValueError("resume output prefixes do not match their committed progress count")
    for prior_submission in prior_predictions:
        Draft202012Validator(schema).validate(prior_submission)
    prompts = {key: resolve_path(value["path"]).read_text(encoding="utf-8") for key, value in config["prompts"].items()}
    generation_config = config["generation"]
    model_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    adapter_paths = {
        "profile": resolve_path(model_lock["deployment_adapters"]["profile"]["path"]),
        "emotion-response": resolve_path(model_lock["deployment_adapters"]["emotion_response"]["path"]),
    }
    service = None
    runtime = None
    torch = None
    monitor = GpuMemoryMonitor(interval_seconds=1.0)
    monitor.start()
    init_seconds: float | None = None
    service_facts: dict[str, Any] | None = None
    tokenizer_facts = asset_identity["tokenizer"]
    server_alive_before_shutdown = backend != "vllm"
    route_failures: dict[str, list[dict[str, str]]] = {"profile": [], "emotion_response": []}
    cap_hits = {"profile": 0, "emotion_response": 0}
    route_exact_matches = {"profile": [], "emotion_response": []}
    route_attempts = {"profile": 0, "emotion_response": 0}
    route_latencies = {"profile": [], "emotion_response": []}
    format_success = {"profile": 0, "emotion_response": 0}
    schema_valid_count = 0
    sample_diagnostics: list[dict[str, Any]] = []
    perf_latencies_ms = [
        float(row["sample_end_to_end_ms"])
        for index, row in enumerate(prior_raw, 1)
        if index <= 100 and isinstance(row.get("sample_end_to_end_ms"), (int, float))
    ]
    for index, row in enumerate(prior_raw, 1):
        failures = row.get("route_failures", {})
        for route in ROUTE_ORDER:
            route_attempts[route] += int(row.get("route_attempts", {}).get(route, 0))
            if route in failures:
                route_failures[route].append({"id": row["id"], "error": str(failures[route])})
            route_result = row.get(f"{route}_generation")
            if isinstance(route_result, dict):
                if route_result.get("generation_seconds") is not None:
                    route_latencies[route].append(float(route_result["generation_seconds"]))
                if backend == "vllm":
                    route_exact_matches[route].append(route_result.get("prompt_token_ids_exact_match") is True)
                if route_result.get("finish_reason") == "length" or route_result.get("cap_hit") is True:
                    cap_hits[route] += 1
        diag = row.get("diagnostics", {})
        format_success["profile"] += diag.get("profile_parse_error") is None
        format_success["emotion_response"] += all(diag.get(name) is None for name in ("emotion_parse_error", "response_parse_error", "emotion_response_format_error"))
        sample_diagnostics.append({"id": row["id"], "resumed": True, "route_failures": failures, "diagnostics": diag, "sample_end_to_end_ms": row.get("sample_end_to_end_ms")})
        schema_valid_count += 1

    try:
        if backend == "vllm":
            service = DualLoRAVLLMService(
                config=config,
                model_lock=model_lock,
                root=ROOT,
                log_path=result_dir / "vllm_server.log",
                vllm_bin=os.environ.get("VLLM_BIN", "vllm"),
                startup_timeout_seconds=float(os.environ.get("VLLM_START_TIMEOUT", "900")),
            )
            service_facts = service.start()
            server_models = service.require_adapters()
            service_facts["registered_model_ids"] = server_models
            init_seconds = service_facts["startup_seconds"]
        else:
            import torch as torch_module
            from src.inference.final_dual_runtime import DualAdapterRuntime

            torch = torch_module
            started = time.perf_counter()
            runtime = DualAdapterRuntime.load(
                base_model_path=resolve_path(config["model_path"]),
                tokenizer=tokenizer,
                adapters={"profile": adapter_paths["profile"], "emotion-response": adapter_paths["emotion-response"]},
                torch_module=torch,
            )
            init_seconds = time.perf_counter() - started

        for index, sample in enumerate(samples[completed:], completed + 1):
            started = time.perf_counter()
            prepared: dict[str, dict[str, Any] | None] = {"profile": None, "emotion_response": None}
            generated: dict[str, dict[str, Any] | None] = {"profile": None, "emotion_response": None}
            attempted: dict[str, dict[str, Any] | None] = {"profile": None, "emotion_response": None}
            sample_route_attempts = {"profile": 0, "emotion_response": 0}
            generation_failures: dict[str, str] = {}
            for route in ROUTE_ORDER:
                try:
                    if route == "profile":
                        prepared[route] = build_profile_prompt(
                            sample, tokenizer,
                            selection_prompt=prompts["selection"],
                            profile_prompt=prompts["profile"],
                            profile_budget_prompt=prompts["profile_history_budget"],
                            prompt_cap=int(config["context"]["max_prompt_tokens"]),
                        )
                    else:
                        prepared[route] = build_emotion_response_prompt(
                            sample, tokenizer,
                            selection_prompt=prompts["selection"],
                            emotion_response_prompt=prompts["emotion_response"],
                            prompt_cap=int(config["context"]["max_prompt_tokens"]),
                        )
                except Exception as exc:
                    generation_failures[route] = f"prompt:{type(exc).__name__}: {exc}"

            for route in ROUTE_ORDER:
                prompt = prepared[route]
                if prompt is None:
                    continue
                adapter_name = "profile" if route == "profile" else "emotion-response"
                generation = generation_config[route]
                try:
                    if backend == "vllm":
                        if service is None or not service.is_alive():
                            raise RuntimeError("vLLM server is not alive")
                        route_attempts[route] += 1
                        sample_route_attempts[route] += 1
                        result = service.chat_completion(
                            model_alias=adapter_name,
                            messages=prompt["messages"],
                            expected_prompt_token_ids=prompt["prompt_ids"],
                            max_tokens=int(generation["max_new_tokens"]),
                        )
                        route_exact_matches[route].append(result["prompt_token_ids_exact_match"] is True)
                        if result["generation_seconds"] is not None:
                            route_latencies[route].append(float(result["generation_seconds"]))
                        attempted[route] = result
                        if not result["prompt_token_ids_exact_match"]:
                            generation_failures[route] = "server_prompt_token_ids_mismatch"
                            result["discarded_due_to_prompt_mismatch"] = True
                            generated[route] = None
                        else:
                            generated[route] = result
                        if result["finish_reason"] == "length" or result["usage_completion_tokens"] >= int(generation["max_new_tokens"]):
                            cap_hits[route] += 1
                            result["cap_hit"] = True
                    else:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        route_attempts[route] += 1
                        sample_route_attempts[route] += 1
                        route_started = time.perf_counter()
                        result = runtime.generate(
                            adapter_name,
                            prompt["prompt_ids"],
                            max_new_tokens=int(generation["max_new_tokens"]),
                            eos_token_id=int(generation_config["eos_token_id"]),
                            pad_token_id=int(generation_config["pad_token_id"]),
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        result["generation_seconds"] = time.perf_counter() - route_started
                        attempted[route] = result
                        route_latencies[route].append(float(result["generation_seconds"]))
                        generated[route] = result
                except Exception as exc:
                    generation_failures[route] = f"{type(exc).__name__}: {exc}"
                if route in generation_failures:
                    route_failures[route].append({"id": sample["id"], "error": generation_failures[route]})

            profile_result, er_result = generated["profile"], generated["emotion_response"]
            submission, diagnostics = assemble_submission(
                sample["id"],
                None if profile_result is None else profile_result["text"],
                None if er_result is None else er_result["text"],
                schema,
            )
            Draft202012Validator(schema).validate(submission)
            schema_valid_count += 1
            format_success["profile"] += diagnostics["profile_parse_error"] is None
            format_success["emotion_response"] += all(diagnostics[name] is None for name in ("emotion_parse_error", "response_parse_error", "emotion_response_format_error"))
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if index <= 100:
                perf_latencies_ms.append(elapsed_ms)
            raw_row = {
                "id": sample["id"],
                "profile_raw": None if profile_result is None else profile_result["text"],
                "emotion_response_raw": None if er_result is None else er_result["text"],
                "profile_attempt_raw": None if attempted["profile"] is None else attempted["profile"].get("text"),
                "emotion_response_attempt_raw": None if attempted["emotion_response"] is None else attempted["emotion_response"].get("text"),
                "profile_generation": _generation_metadata(profile_result or attempted["profile"]),
                "emotion_response_generation": _generation_metadata(er_result or attempted["emotion_response"]),
                "profile_prompt": None if prepared["profile"] is None else {key: prepared["profile"][key] for key in ("prompt_tokens", "prompt_token_ids_sha256", "selected_message_ids", "selected_history_sha256")},
                "emotion_response_prompt": None if prepared["emotion_response"] is None else {key: prepared["emotion_response"][key] for key in ("prompt_tokens", "prompt_token_ids_sha256", "selected_message_ids", "selected_history_sha256")},
                "route_failures": generation_failures,
                "route_attempts": sample_route_attempts,
                "diagnostics": diagnostics,
                "sample_end_to_end_ms": elapsed_ms,
                "backend": backend,
            }
            submission_line = _jsonl_line(submission)
            raw_line = _jsonl_line(raw_row)
            with submission_path.open("ab") as handle:
                handle.write(submission_line)
                handle.flush()
                os.fsync(handle.fileno())
            with raw_path.open("ab") as handle:
                handle.write(raw_line)
                handle.flush()
                os.fsync(handle.fileno())
            prior_predictions.append(submission)
            prior_raw.append(raw_row)
            sample_diagnostics.append({"id": sample["id"], "resumed": False, "route_failures": generation_failures, "diagnostics": diagnostics, "sample_end_to_end_ms": elapsed_ms})
            _atomic_json(progress_path, {
                **progress,
                "status": "running",
                "completed_samples": index,
                "last_completed_id": sample["id"],
                "submission_prefix_sha256": sha256_file(submission_path),
                "raw_prefix_sha256": sha256_file(raw_path),
                "updated_at_epoch_seconds": time.time(),
            })
            print(f"[{index}/{len(samples)}] {sample['id']}: {elapsed_ms:.3f} ms", flush=True)
    finally:
        server_alive_before_shutdown = service.is_alive() if service is not None else True
        shutdown_returncode = service.close() if service is not None else None
        if runtime is not None:
            runtime.close()
        gpu_memory = monitor.stop()

    input_ids = [sample["id"] for sample in samples]
    output_ids = [row.get("id") for row in prior_predictions]
    output_rows_match = output_ids == input_ids and len(set(output_ids)) == len(output_ids)
    format_threshold = int(len(samples) * 0.95 + 0.999999)
    exact_match_counts = {
        route: sum(
            isinstance(row.get(f"{route}_generation"), dict)
            and row[f"{route}_generation"].get("prompt_token_ids_exact_match") is True
            for row in prior_raw
        )
        for route in ROUTE_ORDER
    }
    prompt_id_gate = backend != "vllm" or all(exact_match_counts[route] == len(samples) for route in ROUTE_ORDER)
    aliases_gate = backend != "vllm" or (
        service_facts is not None
        and {"profile", "emotion-response"}.issubset(set(service_facts.get("registered_model_ids", [])))
    )
    generation_error_free = all(not values for values in route_failures.values())
    cap_gate = all(count == 0 for count in cap_hits.values())
    gate_passed = (
        len(prior_predictions) == len(samples)
        and output_rows_match
        and schema_valid_count == len(samples)
        and format_success["profile"] >= format_threshold
        and format_success["emotion_response"] >= format_threshold
        and generation_error_free
        and cap_gate
        and prompt_id_gate
        and aliases_gate
        and server_alive_before_shutdown
    )
    server_log = result_dir / "vllm_server.log"
    report = {
        "version": "final-dual-official-run-v1",
        "status": "passed" if gate_passed else "failed",
        "complete": len(prior_predictions) == len(samples),
        "run_id": progress["run_id"],
        "backend": backend,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _execution_git_head(),
        "runtime_environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "configuration": {"path": _display_path(config_path), "sha256": sha256_file(config_path), "model_lock_path": _display_path(lock_path), "model_lock_sha256": sha256_file(lock_path)},
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "model_weight_identity": asset_identity["base"],
        "tokenizer_identity": tokenizer_facts,
        "adapter_identities": asset_identity["adapters"],
        "prompt_identities": asset_identity["prompts"],
        "initialization_seconds": init_seconds,
        "server": {
            "runtime": service_facts,
            "registered_model_ids": [] if service_facts is None else service_facts.get("registered_model_ids", []),
            "requested_alias_count": {"profile": route_attempts["profile"], "emotion-response": route_attempts["emotion_response"]},
            "alive_before_intentional_shutdown": server_alive_before_shutdown,
            "intentional_shutdown_returncode": shutdown_returncode,
            "log_path": str(server_log) if server_log.is_file() else None,
            "log_sha256": sha256_file(server_log) if server_log.is_file() else None,
        },
        "gpu_memory": gpu_memory,
        "performance": {
            "measurement_scope": "per-sample prompt construction through two serial generations, parse, official Schema validation, and output append; server/model initialization excluded",
            "measured_samples": len(perf_latencies_ms),
            "measurement_limit": 100,
            "latencies_ms": [round(value, 6) for value in perf_latencies_ms],
            "average_latency_ms": statistics.fmean(perf_latencies_ms) if perf_latencies_ms else None,
            "median_latency_ms": statistics.median(perf_latencies_ms) if perf_latencies_ms else None,
            "p95_latency_ms": _percentile95(perf_latencies_ms),
            "per_route_generation_seconds": {
                route: {"count": len(values), "median": statistics.median(values) if values else None, "p95": _percentile95(values)}
                for route, values in route_latencies.items()
            },
        },
        "resumption": {"enabled": resume_enabled, "reused_samples": completed, "attempt": progress.get("attempt", 1), "performance_is_single_uninterrupted_run": completed == 0},
        "samples": len(samples),
        "submission_rows": len(prior_predictions),
        "submission_schema_validated_count": schema_valid_count,
        "submission_schema_validated": schema_valid_count == len(samples),
        "format_success_count": format_success,
        "format_threshold_per_route": format_threshold,
        "route_failure_count": {route: len(values) for route, values in route_failures.items()},
        "route_failures": route_failures,
        "generation_cap_hits": cap_hits,
        "prompt_token_ids_exact_match_count": exact_match_counts if backend == "vllm" else None,
        "output_ids_match_input_order": output_rows_match,
        "output_ids_sha256": hashlib.sha256("\n".join(output_ids).encode("utf-8")).hexdigest(),
        "gate_thresholds": {"profile_format": format_threshold, "emotion_response_format": format_threshold, "schema": len(samples), "prompt_token_ids": "exact for every vLLM request", "generation_cap_hits": 0},
        "sample_diagnostics": sample_diagnostics,
        "outputs": {
            "submission.jsonl": {"path": str(submission_path), "sha256": sha256_file(submission_path), "rows": len(prior_predictions)},
            "raw_outputs.jsonl": {"path": str(raw_path), "sha256": sha256_file(raw_path), "rows": len(prior_raw)},
            "performance_report.json": {"path": str(result_dir / "performance_report.json")},
        },
    }
    report_path = result_dir / "performance_report.json"
    _atomic_json(report_path, report)
    _atomic_json(progress_path, {
        **progress,
        "status": "complete" if report["complete"] else "failed",
        "completed_samples": len(prior_predictions),
        "submission_prefix_sha256": sha256_file(submission_path),
        "raw_prefix_sha256": sha256_file(raw_path),
        "updated_at_epoch_seconds": time.time(),
    })
    print(f"{'PASS' if gate_passed else 'FAIL'}: {submission_path}", flush=True)
    print(f"{'PASS' if gate_passed else 'FAIL'}: {report_path}", flush=True)
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
