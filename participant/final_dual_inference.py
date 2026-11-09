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
    parse_emotion_response_output,
)
from src.inference.final_dual_vllm import (  # noqa: E402
    DualLoRAVLLMService,
    GpuMemoryMonitor,
    gpu_snapshot,
)

OFFICIAL_ID = re.compile(r"^test_[0-9]{6}_t[0-9]+$")
ROUTE_ORDER = ("profile", "emotion_response")
PROGRESS_CHECKPOINT_INTERVAL = 10
RUNTIME_CODE_FILES = (
    Path("participant/start.sh"),
    Path("participant/final_dual_inference.py"),
    Path("participant/run_inference.py"),
    Path("src/inference/final_dual_protocol.py"),
    Path("src/inference/final_dual_runtime.py"),
    Path("src/inference/final_dual_vllm.py"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_code_hashes(root: Path = ROOT) -> dict[str, str]:
    return {path.as_posix(): sha256_file(root / path) for path in RUNTIME_CODE_FILES}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _relative_result_path(path: Path, result_dir: Path) -> str:
    return path.resolve().relative_to(result_dir.resolve()).as_posix()


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


def _sha256_state(path: Path) -> Any:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest


def _sync_checkpoint(
    *,
    progress_path: Path,
    progress: dict[str, Any],
    completed: int,
    last_completed_id: str | None,
    submission_handle: Any,
    raw_handle: Any,
    submission_digest: Any,
    raw_digest: Any,
) -> dict[str, Any]:
    for handle in (submission_handle, raw_handle):
        handle.flush()
        os.fsync(handle.fileno())
    updated = {
        **progress,
        "status": "running",
        "completed_samples": completed,
        "last_completed_id": last_completed_id,
        "submission_prefix_sha256": submission_digest.hexdigest(),
        "raw_prefix_sha256": raw_digest.hexdigest(),
        "updated_at_epoch_seconds": time.time(),
    }
    _atomic_json(progress_path, updated)
    return updated


def _new_transaction_dir(result_dir: Path) -> Path:
    transaction_root = result_dir / ".emotionalcompany_runs"
    transaction_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=transaction_root))


def _find_resumable_transaction_dir(result_dir: Path, requested_run_id: str) -> Path | None:
    transaction_root = result_dir / ".emotionalcompany_runs"
    if not transaction_root.is_dir():
        return None
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for progress_path in transaction_root.glob("*/run_progress.json"):
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if progress.get("status") != "running":
            continue
        if requested_run_id and progress.get("run_id") != requested_run_id:
            continue
        candidates.append((progress_path.parent, progress))
    if len(candidates) > 1:
        raise ValueError("multiple interrupted runs exist; set EVALUATION_RUN_ID to select one")
    if not candidates:
        return None
    run_dir, progress = candidates[0]
    if not requested_run_id and isinstance(progress.get("run_id"), str):
        os.environ["EVALUATION_RUN_ID"] = progress["run_id"]
    return run_dir


def _publish_official_outputs(run_dir: Path, result_dir: Path) -> None:
    """Atomically replace only the two official files after a complete successful run."""
    result_dir.mkdir(parents=True, exist_ok=True)
    required = ("submission.jsonl", "performance_report.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot publish incomplete official output set: {missing}")
    for name in required:
        os.replace(run_dir / name, result_dir / name)
    try:
        descriptor = os.open(result_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _verify_runtime_code_identity(config: dict[str, Any], lock: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any] | None:
    runtime_behavior = config.get("runtime_behavior")
    if runtime_behavior is None:
        return None
    if (
        runtime_behavior.get("version") != "E03-v1.2-readme-v2-contract"
        or runtime_behavior.get("max_emotion_response_retries") != 1
        or runtime_behavior.get("retry_only_on_finish_reason") != "length"
    ):
        raise RuntimeError("v1.2 runtime behavior config must lock exactly one retry on finish_reason=length")
    actual_code_hashes = runtime_code_hashes(root)
    locked_code_identity = lock.get("runtime_code_identity", {})
    if locked_code_identity.get("version") != runtime_behavior["version"]:
        raise RuntimeError("runtime code identity version differs from the v1.2 behavior config")
    if locked_code_identity.get("file_sha256") != actual_code_hashes:
        raise RuntimeError("runtime inference code hashes differ from the v1.2 model lock")
    execution_head = _execution_git_head()
    if locked_code_identity.get("git_head") and execution_head and locked_code_identity["git_head"] != execution_head:
        raise RuntimeError("runtime Git source identity differs from the v1.2 model lock")
    return {
        "version": runtime_behavior["version"],
        "git_head": locked_code_identity.get("git_head"),
        "file_sha256": actual_code_hashes,
    }


def _verify_assets(config: dict[str, Any], config_path: Path, lock_path: Path) -> tuple[Any, dict[str, Any]]:
    from participant import run_inference as runner

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("deployment_config", {}).get("sha256") != sha256_file(config_path):
        raise RuntimeError("deployment config SHA-256 differs from model_lock.json")
    if lock.get("deployment_config", {}).get("path") != str(config_path.relative_to(ROOT)):
        raise RuntimeError("deployment config path differs from model_lock.json")
    runtime_code_identity = _verify_runtime_code_identity(config, lock)
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
        "runtime_code_identity": runtime_code_identity,
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


def _repeated_response_fragment(text: str) -> dict[str, Any] | None:
    """Identify long, exact repeated fragments without changing model text."""
    if not text or len(text) > 20_000:
        return None
    match = re.search(r"(.{20,120}?)(?:\1){2,}", text, flags=re.DOTALL)
    if match is None:
        return None
    fragment = match.group(1)
    repeated = match.group(0)
    if len(fragment) < 20 or len(repeated) < 90 or len(repeated) / max(1, len(text)) < 0.3:
        return None
    return {
        "kind": "repeated_fragment",
        "fragment_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
        "fragment_chars": len(fragment),
        "repeated_chars": len(repeated),
    }


def _er_attempt_evidence(
    result: dict[str, Any] | None,
    *,
    sample_id: str,
    attempt_index: int,
    schema: dict[str, Any],
    profile_raw: str | None,
    route_error: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if result is None:
        return ({
            "attempt_index": attempt_index,
            "attempt_status": "route_error",
            "route_error": route_error,
            "raw_text": None,
            "raw_text_sha256": None,
            "parse_diagnostics": None,
            "severe_text_anomalies": [],
            "response_valid": False,
        }, False)

    raw_text = result.get("text")
    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if isinstance(raw_text, str) else None
    finish_reason = result.get("finish_reason")
    parsed = parse_emotion_response_output(raw_text, schema)
    anomalies = []
    repeated = _repeated_response_fragment(parsed["response_text"])
    if repeated is not None:
        anomalies.append(repeated)
    valid = (
        finish_reason in {"stop", "length"}
        and parsed["emotion_error"] is None
        and parsed["response_error"] is None
        and parsed["format_error"] is None
        and not anomalies
    )
    if valid:
        # Validate the official final row shape as well as the ER-specific parser.
        try:
            assemble_submission(sample_id, profile_raw, raw_text, schema)
        except Exception:
            valid = False
    attempt_status = "valid_candidate" if valid else "invalid_output"
    if finish_reason == "length" and valid:
        attempt_status = "length_limited_candidate"
    elif finish_reason == "stop" and valid:
        attempt_status = "completed_stop_candidate"
    evidence = {
        "attempt_index": attempt_index,
        "attempt_status": attempt_status,
        "request_id": result.get("response_id"),
        "finish_reason": finish_reason,
        "completion_tokens": result.get("usage_completion_tokens"),
        "generation_seconds": result.get("generation_seconds"),
        "prompt_token_ids_exact_match": result.get("prompt_token_ids_exact_match"),
        "prompt_token_count_expected": result.get("prompt_token_count_expected"),
        "prompt_token_count_observed": result.get("prompt_token_count_observed"),
        "prompt_token_ids_expected_sha256": result.get("prompt_token_ids_expected_sha256"),
        "prompt_token_ids_observed_sha256": result.get("prompt_token_ids_observed_sha256"),
        "completion_tokens_at_budget": (
            finish_reason == "stop" and result.get("usage_completion_tokens") == result.get("requested_max_tokens")
        ),
        "raw_text": raw_text,
        "raw_text_sha256": text_hash,
        "parse_diagnostics": {
            "emotion_error": parsed["emotion_error"],
            "response_error": parsed["response_error"],
            "format_error": parsed["format_error"],
        },
        "partial_fields_validity": {
            "emotion_label": parsed["emotion_error"] is None and bool(parsed["emotion_label"]),
            "response_text": parsed["response_error"] is None and bool(parsed["response_text"].strip()),
        },
        "severe_text_anomalies": anomalies,
        "response_valid": valid,
    }
    return evidence, valid


def _best_partial_er_candidate(
    candidates: list[tuple[int, dict[str, Any] | None]], schema: dict[str, Any]
) -> tuple[int | None, dict[str, Any] | None]:
    usable: list[tuple[int, int, dict[str, Any]]] = []
    for attempt_index, result in candidates:
        if result is None or not isinstance(result.get("text"), str):
            continue
        parsed = parse_emotion_response_output(result["text"], schema)
        if parsed["format_error"] is not None or _repeated_response_fragment(parsed["response_text"]):
            continue
        score = int(parsed["emotion_error"] is None and bool(parsed["emotion_label"]))
        score += int(parsed["response_error"] is None and bool(parsed["response_text"].strip()))
        if score:
            usable.append((score, attempt_index, result))
    if not usable:
        return None, None
    # Preserve as many independently valid fields as possible; for ties prefer the later attempt.
    _, attempt_index, result = max(usable, key=lambda item: (item[0], item[1]))
    return attempt_index, result


def _recover_emotion_response(
    initial: dict[str, Any] | None,
    retry: dict[str, Any] | None,
    *,
    sample_id: str,
    retry_attempted: bool,
    schema: dict[str, Any],
    profile_raw: str | None,
    initial_error: str | None = None,
    retry_error: str | None = None,
    protocol_mismatch: bool = False,
) -> dict[str, Any]:
    """Select at most one retry result while retaining the evidence for both attempts."""
    records: list[dict[str, Any]] = []
    initial_record, initial_valid = _er_attempt_evidence(
        initial, sample_id=sample_id, attempt_index=1, schema=schema, profile_raw=profile_raw, route_error=initial_error
    )
    records.append(initial_record)
    retry_valid = False
    if retry_attempted or retry_error is not None:
        retry_record, retry_valid = _er_attempt_evidence(
            retry, sample_id=sample_id, attempt_index=2, schema=schema, profile_raw=profile_raw, route_error=retry_error
        )
        records.append(retry_record)

    initial_cap = initial is not None and initial.get("finish_reason") == "length"
    counters = {
        "initial_cap_hit": int(initial_cap),
        "retry_attempts": int(retry_attempted),
        "retry_resolved": 0,
        "unresolved_cap_hit": 0,
        "degraded_response": False,
    }
    if protocol_mismatch:
        return {
            "selected": None,
            "outcome": "asset_or_protocol_mismatch",
            "attempts": records,
            "selected_attempt_index": None,
            "counters": {**counters, "degraded_response": True},
        }

    if initial is None:
        return {
            "selected": None,
            "outcome": "route_error" if initial_error else "invalid_output",
            "attempts": records,
            "selected_attempt_index": None,
            "counters": {**counters, "degraded_response": True},
        }

    initial_finish = initial.get("finish_reason")
    if initial_cap:
        if retry is not None and retry.get("finish_reason") == "stop" and retry_valid:
            counters["retry_resolved"] = 1
            return {
                "selected": retry,
                "outcome": "completed_stop",
                "attempts": records,
                "selected_attempt_index": 2,
                "counters": counters,
            }
        selected = initial if initial_valid else retry if retry_valid else None
        if selected is not None:
            selected_index = 1 if selected is initial else 2
            counters["unresolved_cap_hit"] = 1
            counters["degraded_response"] = True
            return {
                "selected": selected,
                "outcome": "length_limited_unresolved",
                "attempts": records,
                "selected_attempt_index": selected_index,
                "counters": counters,
            }
        counters["unresolved_cap_hit"] = 1
        counters["degraded_response"] = True
        partial_index, partial_result = _best_partial_er_candidate(
            [(1, initial), (2, retry)], schema
        )
        return {
            "selected": partial_result,
            "outcome": "route_error" if retry_error and not initial_valid else "invalid_output",
            "attempts": records,
            "selected_attempt_index": partial_index,
            "counters": counters,
        }

    if initial_finish == "stop" and initial_valid:
        return {
            "selected": initial,
            "outcome": "completed_stop",
            "attempts": records,
            "selected_attempt_index": 1,
            "counters": counters,
        }
    partial_index, partial_result = _best_partial_er_candidate([(1, initial)], schema)
    return {
        "selected": partial_result,
        "outcome": "route_error" if initial_error else "invalid_output",
        "attempts": records,
        "selected_attempt_index": partial_index,
        "counters": {**counters, "degraded_response": True},
    }


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
    official_result_dir = result_dir
    resume_value = os.environ.get("EVALUATION_RESUME", "").strip().lower()
    if resume_value not in {"", "0", "false", "no", "1", "true", "yes"}:
        raise ValueError("EVALUATION_RESUME must be true or false")
    resume_enabled = resume_value in {"1", "true", "yes"}
    tokenizer, asset_identity = _verify_assets(config, config_path, lock_path)
    transactional_outputs = True
    if resume_enabled and (result_dir / "run_progress.json").is_file():
        # Preserve the original explicit local-resume path for runs created before v1.2.
        transactional_outputs = False
    elif resume_enabled:
        existing_run_dir = _find_resumable_transaction_dir(
            result_dir, os.environ.get("EVALUATION_RUN_ID", "").strip()
        )
        result_dir = existing_run_dir or _new_transaction_dir(result_dir)
    else:
        result_dir = _new_transaction_dir(result_dir)
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
    route_response_counts = {"profile": 0, "emotion_response": 0}
    prepared_prompt_counts = {"profile": 0, "emotion_response": 0}
    route_latencies = {"profile": [], "emotion_response": []}
    initial_cap_hits = {"profile": 0, "emotion_response": 0}
    retry_attempts = {"profile": 0, "emotion_response": 0}
    retry_resolved = {"profile": 0, "emotion_response": 0}
    unresolved_cap_hits = {"profile": 0, "emotion_response": 0}
    degraded_sample_ids: list[str] = []
    hard_failures: list[dict[str, Any]] = []
    format_success = {"profile": 0, "emotion_response": 0}
    schema_valid_count = 0
    sample_diagnostics: list[dict[str, Any]] = []
    perf_latencies_ms = [
        float(row["sample_end_to_end_ms"])
        for index, row in enumerate(prior_raw, 1)
        if index <= 100 and isinstance(row.get("sample_end_to_end_ms"), (int, float))
    ]
    for index, row in enumerate(prior_raw, 1):
        for route, prompt_key in (("profile", "profile_prompt"), ("emotion_response", "emotion_response_prompt")):
            if prompt_key not in row or row.get(prompt_key) is not None:
                prepared_prompt_counts[route] += 1
        failures = row.get("route_failures", {})
        for route in ROUTE_ORDER:
            route_attempts[route] += int(row.get("route_attempts", {}).get(route, 0))
            if route in failures:
                route_failures[route].append({"id": row["id"], "error": str(failures[route])})
            attempt_records = row.get("generation_attempts", {}).get(route)
            if isinstance(attempt_records, list):
                for attempt_record in attempt_records:
                    if attempt_record.get("attempt_status") == "response":
                        route_response_counts[route] += 1
                    generation_record = attempt_record.get("generation", {})
                    if generation_record.get("generation_seconds") is not None:
                        route_latencies[route].append(float(generation_record["generation_seconds"]))
                    if backend == "vllm" and attempt_record.get("attempt_status") != "route_error":
                        route_exact_matches[route].append(
                            generation_record.get("prompt_token_ids_exact_match") is True
                        )
                    if generation_record.get("finish_reason") == "length":
                        cap_hits[route] += 1
                        if route == "profile":
                            initial_cap_hits[route] += 1
                            unresolved_cap_hits[route] += 1
                            degraded_sample_ids.append(row["id"])
                if route == "emotion_response":
                    recovery = row.get("emotion_response_recovery", {})
                    recovery_counters = recovery.get("counters", {})
                    initial_cap_hits[route] += int(recovery_counters.get("initial_cap_hit", 0))
                    retry_attempts[route] += int(recovery_counters.get("retry_attempts", 0))
                    retry_resolved[route] += int(recovery_counters.get("retry_resolved", 0))
                    unresolved_cap_hits[route] += int(recovery_counters.get("unresolved_cap_hit", 0))
                    if recovery_counters.get("degraded_response") is True:
                        degraded_sample_ids.append(row["id"])
                    for failure in row.get("hard_failures", []):
                        hard_failures.append({"id": row["id"], **failure})
            else:
                # v1 compatibility: older raw rows had a single generation record per route.
                route_result = row.get(f"{route}_generation")
                if isinstance(route_result, dict):
                    route_response_counts[route] += 1
                    prepared_prompt_counts[route] += 1
                    if route_result.get("generation_seconds") is not None:
                        route_latencies[route].append(float(route_result["generation_seconds"]))
                    if backend == "vllm":
                        route_exact_matches[route].append(route_result.get("prompt_token_ids_exact_match") is True)
                    if route_result.get("finish_reason") == "length" or (
                        route_result.get("finish_reason") is None and route_result.get("cap_hit") is True
                    ):
                        cap_hits[route] += 1
                        initial_cap_hits[route] += 1
                        unresolved_cap_hits[route] += 1
                        degraded_sample_ids.append(row["id"])
        diag = row.get("diagnostics", {})
        format_success["profile"] += diag.get("profile_parse_error") is None
        format_success["emotion_response"] += all(diag.get(name) is None for name in ("emotion_parse_error", "response_parse_error", "emotion_response_format_error"))
        sample_diagnostics.append({"id": row["id"], "resumed": True, "route_failures": failures, "diagnostics": diag, "sample_end_to_end_ms": row.get("sample_end_to_end_ms")})
        schema_valid_count += 1

    submission_digest = _sha256_state(submission_path)
    raw_digest = _sha256_state(raw_path)
    submission_handle = submission_path.open("ab")
    try:
        raw_handle = raw_path.open("ab")
    except Exception:
        submission_handle.close()
        raise
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
            generation_attempt_evidence: dict[str, list[dict[str, Any]]] = {"profile": [], "emotion_response": []}
            sample_route_attempts = {"profile": 0, "emotion_response": 0}
            generation_failures: dict[str, str] = {}
            sample_hard_failures: list[dict[str, Any]] = []
            emotion_recovery: dict[str, Any] | None = None
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
                if prepared[route] is not None:
                    prepared_prompt_counts[route] += 1

            for route in ROUTE_ORDER:
                prompt = prepared[route]
                if prompt is None:
                    continue
                adapter_name = "profile" if route == "profile" else "emotion-response"
                generation = generation_config[route]
                if backend == "vllm":
                    def make_vllm_request(attempt_index: int) -> tuple[dict[str, Any] | None, str | None, bool]:
                        route_attempts[route] += 1
                        sample_route_attempts[route] += 1
                        if service is None or not service.is_alive():
                            error = "RuntimeError: vLLM server is not alive"
                            sample_hard_failures.append({"kind": "service_exit", "route": route, "error": error})
                            generation_attempt_evidence[route].append({
                                "attempt_index": attempt_index,
                                "attempt_status": "service_unavailable",
                                "route_error": error,
                                "generation": {},
                                "raw_text": None,
                                "raw_text_sha256": None,
                            })
                            return None, error, False
                        request_started = time.perf_counter()
                        try:
                            result = service.chat_completion(
                                model_alias=adapter_name,
                                messages=prompt["messages"],
                                expected_prompt_token_ids=prompt["prompt_ids"],
                                max_tokens=int(generation["max_new_tokens"]),
                            )
                        except Exception as exc:
                            elapsed = time.perf_counter() - request_started
                            route_latencies[route].append(elapsed)
                            error = f"{type(exc).__name__}: {exc}"
                            generation_attempt_evidence[route].append({
                                "attempt_index": attempt_index,
                                "attempt_status": "route_error",
                                "route_error": error,
                                "generation": {"generation_seconds": elapsed},
                                "raw_text": None,
                                "raw_text_sha256": None,
                            })
                            if "prompt_token_ids" in str(exc).lower():
                                sample_hard_failures.append({"kind": "asset_or_protocol_mismatch", "route": route, "error": error})
                            if not service.is_alive():
                                sample_hard_failures.append({"kind": "service_exit", "route": route, "error": error})
                            return None, error, False

                        result["requested_max_tokens"] = int(generation["max_new_tokens"])
                        route_exact_matches[route].append(result.get("prompt_token_ids_exact_match") is True)
                        seconds = result.get("generation_seconds")
                        if seconds is not None:
                            route_latencies[route].append(float(seconds))
                        text_value = result.get("text")
                        text_hash = hashlib.sha256(text_value.encode("utf-8")).hexdigest() if isinstance(text_value, str) else None
                        generation_attempt_evidence[route].append({
                            "attempt_index": attempt_index,
                            "attempt_status": "response",
                            "generation": _generation_metadata(result),
                            "raw_text": text_value,
                            "raw_text_sha256": text_hash,
                        })
                        route_response_counts[route] += 1
                        if result.get("finish_reason") == "length":
                            cap_hits[route] += 1
                        if result.get("prompt_token_ids_exact_match") is not True:
                            result["discarded_due_to_prompt_mismatch"] = True
                            sample_hard_failures.append({
                                "kind": "asset_or_protocol_mismatch",
                                "route": route,
                                "error": "server_prompt_token_ids_mismatch",
                            })
                            return result, None, True
                        return result, None, False

                    first_result, first_error, first_mismatch = make_vllm_request(1)
                    attempted[route] = first_result
                    if route == "profile":
                        generated[route] = None if first_mismatch else first_result
                        if first_result is not None and first_result.get("finish_reason") == "length":
                            initial_cap_hits[route] += 1
                            unresolved_cap_hits[route] += 1
                            degraded_sample_ids.append(sample["id"])
                    else:
                        retry_result = None
                        retry_error = None
                        retry_mismatch = False
                        retry_was_attempted = bool(
                            first_result is not None
                            and first_result.get("finish_reason") == "length"
                            and first_result.get("prompt_token_ids_exact_match") is True
                        )
                        if retry_was_attempted:
                            retry_attempts[route] += 1
                            retry_result, retry_error, retry_mismatch = make_vllm_request(2)
                        emotion_recovery = _recover_emotion_response(
                            first_result,
                            retry_result,
                            sample_id=sample["id"],
                            retry_attempted=retry_was_attempted,
                            schema=schema,
                            profile_raw=None if generated["profile"] is None else generated["profile"].get("text"),
                            initial_error=first_error,
                            retry_error=retry_error,
                            protocol_mismatch=first_mismatch or retry_mismatch,
                        )
                        generated[route] = emotion_recovery["selected"]
                        recovery_counters = emotion_recovery["counters"]
                        initial_cap_hits[route] += int(recovery_counters["initial_cap_hit"])
                        retry_resolved[route] += int(recovery_counters["retry_resolved"])
                        unresolved_cap_hits[route] += int(recovery_counters["unresolved_cap_hit"])
                        if recovery_counters["degraded_response"]:
                            degraded_sample_ids.append(sample["id"])
                        if first_mismatch or retry_mismatch:
                            emotion_recovery["outcome"] = "asset_or_protocol_mismatch"
                    if first_error or (route == "emotion_response" and retry_error):
                        generation_failures[route] = first_error or retry_error or "route_error"
                    if first_mismatch or (route == "emotion_response" and retry_mismatch):
                        generation_failures[route] = "server_prompt_token_ids_mismatch"
                else:
                    try:
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
                        result["requested_max_tokens"] = int(generation["max_new_tokens"])
                        attempted[route] = result
                        route_latencies[route].append(float(result["generation_seconds"]))
                        generated[route] = result
                        route_response_counts[route] += 1
                        text_value = result.get("text")
                        generation_attempt_evidence[route].append({
                            "attempt_index": 1,
                            "attempt_status": "response",
                            "generation": _generation_metadata(result),
                            "raw_text": text_value,
                            "raw_text_sha256": hashlib.sha256(text_value.encode("utf-8")).hexdigest() if isinstance(text_value, str) else None,
                        })
                    except Exception as exc:
                        generation_failures[route] = f"{type(exc).__name__}: {exc}"

            if prepared["emotion_response"] is None:
                emotion_recovery = {
                    "selected": None,
                    "outcome": "route_error",
                    "attempts": [],
                    "selected_attempt_index": None,
                    "counters": {
                        "initial_cap_hit": 0,
                        "retry_attempts": 0,
                        "retry_resolved": 0,
                        "unresolved_cap_hit": 0,
                        "degraded_response": True,
                    },
                }
                degraded_sample_ids.append(sample["id"])
            for route in ROUTE_ORDER:
                if route in generation_failures:
                    route_failures[route].append({"id": sample["id"], "error": generation_failures[route]})
            hard_failures.extend({"id": sample["id"], **failure} for failure in sample_hard_failures)

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
            if generation_failures or diagnostics["profile_parse_error"] is not None or any(
                diagnostics[name] is not None
                for name in ("emotion_parse_error", "response_parse_error", "emotion_response_format_error")
            ):
                degraded_sample_ids.append(sample["id"])
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
                "generation_attempts": generation_attempt_evidence,
                "emotion_response_recovery": emotion_recovery,
                "emotion_response_outcome": "asset_or_protocol_mismatch" if any(
                    failure.get("kind") == "asset_or_protocol_mismatch" for failure in sample_hard_failures
                ) else emotion_recovery["outcome"],
                "degraded_response": sample["id"] in degraded_sample_ids,
                "hard_failures": sample_hard_failures,
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
            submission_handle.write(submission_line)
            raw_handle.write(raw_line)
            submission_digest.update(submission_line)
            raw_digest.update(raw_line)
            prior_predictions.append(submission)
            prior_raw.append(raw_row)
            sample_diagnostics.append({"id": sample["id"], "resumed": False, "route_failures": generation_failures, "diagnostics": diagnostics, "sample_end_to_end_ms": elapsed_ms})
            if index % PROGRESS_CHECKPOINT_INTERVAL == 0:
                _sync_checkpoint(
                    progress_path=progress_path,
                    progress=progress,
                    completed=index,
                    last_completed_id=sample["id"],
                    submission_handle=submission_handle,
                    raw_handle=raw_handle,
                    submission_digest=submission_digest,
                    raw_digest=raw_digest,
                )
            print(f"[{index}/{len(samples)}] {sample['id']}: {elapsed_ms:.3f} ms", flush=True)
        _sync_checkpoint(
            progress_path=progress_path,
            progress=progress,
            completed=len(prior_predictions),
            last_completed_id=None if not prior_predictions else str(prior_predictions[-1].get("id")),
            submission_handle=submission_handle,
            raw_handle=raw_handle,
            submission_digest=submission_digest,
            raw_digest=raw_digest,
        )
    finally:
        for handle in (submission_handle, raw_handle):
            if not handle.closed:
                handle.flush()
                handle.close()
        server_alive_before_shutdown = service.is_alive() if service is not None else True
        if backend == "vllm" and not server_alive_before_shutdown and not any(
            failure.get("kind") == "service_exit" for failure in hard_failures
        ):
            hard_failures.append({"kind": "service_exit", "route": None, "error": "vLLM server exited before intentional shutdown"})
        shutdown_returncode = service.close() if service is not None else None
        if runtime is not None:
            runtime.close()
        gpu_memory = monitor.stop()

    input_ids = [sample["id"] for sample in samples]
    output_ids = [row.get("id") for row in prior_predictions]
    output_rows_match = output_ids == input_ids and len(set(output_ids)) == len(output_ids)
    format_threshold = int(len(samples) * 0.95 + 0.999999)
    exact_match_counts = {
        route: sum(value is True for value in route_exact_matches[route])
        for route in ROUTE_ORDER
    }
    prompt_check_denominators = {
        route: len(route_exact_matches[route]) for route in ROUTE_ORDER
    }
    prompt_id_gate = backend != "vllm" or all(
        exact_match_counts[route] == prompt_check_denominators[route]
        for route in ROUTE_ORDER
    )
    aliases_gate = backend != "vllm" or (
        service_facts is not None
        and {"profile", "emotion-response"}.issubset(set(service_facts.get("registered_model_ids", [])))
        and route_attempts["profile"] == prepared_prompt_counts["profile"]
        and route_attempts["emotion_response"] == prepared_prompt_counts["emotion_response"] + retry_attempts["emotion_response"]
    )
    service_health_gate = server_alive_before_shutdown and not any(
        failure.get("kind") == "service_exit" for failure in hard_failures
    )
    identity_gate = not any(
        failure.get("kind") == "asset_or_protocol_mismatch" for failure in hard_failures
    )
    systemic_route_failures = {
        route: {
            "prepared_prompts": prepared_prompt_counts[route],
            "successful_responses": route_response_counts[route],
        }
        for route in ROUTE_ORDER
        if route_response_counts[route] == 0
    }
    route_failure_total = sum(len(values) for values in route_failures.values())
    unique_degraded_ids = sorted(set(degraded_sample_ids))
    inference_failure_reasons = []
    if len(prior_predictions) != len(samples):
        inference_failure_reasons.append("output_row_count_mismatch")
    if not output_rows_match:
        inference_failure_reasons.append("output_ids_missing_duplicate_or_out_of_order")
    if schema_valid_count != len(samples):
        inference_failure_reasons.append("official_schema_validation_incomplete")
    if not prompt_id_gate:
        inference_failure_reasons.append("prompt_token_ids_mismatch")
    if not aliases_gate:
        inference_failure_reasons.append("model_alias_registration_or_request_count_mismatch")
    if not service_health_gate:
        inference_failure_reasons.append("vllm_service_unhealthy")
    if not identity_gate:
        inference_failure_reasons.append("asset_or_protocol_identity_mismatch")
    if systemic_route_failures:
        inference_failure_reasons.append("all_prepared_requests_failed_for_route")
    if hard_failures:
        inference_failure_reasons.append("hard_runtime_failures")
    inference_success = not inference_failure_reasons
    diagnostic_quality_gate = (
        format_success["profile"] >= format_threshold
        and format_success["emotion_response"] >= format_threshold
        and route_failure_total == 0
        and not any(unresolved_cap_hits.values())
        and not unique_degraded_ids
        and prompt_id_gate
        and aliases_gate
        and service_health_gate
        and identity_gate
    )
    legacy_gate_passed = (
        len(prior_predictions) == len(samples)
        and output_rows_match
        and schema_valid_count == len(samples)
        and format_success["profile"] >= format_threshold
        and format_success["emotion_response"] >= format_threshold
        and prompt_id_gate
        and aliases_gate
        and service_health_gate
        and identity_gate
    )
    outcome_counts = {
        name: sum(row.get("emotion_response_outcome") == name for row in prior_raw)
        for name in ("completed_stop", "length_limited_unresolved", "invalid_output", "route_error", "asset_or_protocol_mismatch")
    }
    severe_response_anomaly_count = sum(
        len(attempt.get("severe_text_anomalies", []))
        for row in prior_raw
        for attempt in row.get("emotion_response_recovery", {}).get("attempts", [])
    )
    if retry_attempts["emotion_response"]:
        quality_warnings = [f"ER length retry requests: {retry_attempts['emotion_response']}"]
    else:
        quality_warnings = []
    if any(unresolved_cap_hits.values()):
        quality_warnings.append(f"unresolved generation length cap events: {sum(unresolved_cap_hits.values())}")
    if unique_degraded_ids:
        quality_warnings.append(f"degraded responses: {len(unique_degraded_ids)}")
    if route_failure_total:
        quality_warnings.append(f"route errors: {route_failure_total}")
    for route in ROUTE_ORDER:
        if format_success[route] < format_threshold:
            quality_warnings.append(
                f"{route} format diagnostic below threshold: {format_success[route]}/{len(samples)}"
            )
    if hard_failures:
        quality_warnings.append(f"hard failures: {len(hard_failures)}")
    if not diagnostic_quality_gate:
        quality_warnings.append("diagnostic quality gate not met; inference completeness is reported separately")
    server_log = result_dir / "vllm_server.log"
    final_submission_sha256 = sha256_file(submission_path)
    final_raw_sha256 = sha256_file(raw_path)
    if final_submission_sha256 != submission_digest.hexdigest() or final_raw_sha256 != raw_digest.hexdigest():
        raise OSError("final output SHA-256 differs from the incrementally written checkpoint identity")
    published_submission_path = official_result_dir / "submission.jsonl"
    published_report_path = official_result_dir / "performance_report.json"
    report = {
        "version": "final-dual-official-run-v1.2",
        "runtime_behavior_version": config.get("runtime_behavior", {}).get("version", "E03-v1.2-readme-v2-contract"),
        "status": "passed" if inference_success else "failed",
        "inference_success": inference_success,
        "diagnostic_quality_gate": diagnostic_quality_gate,
        "legacy_combined_gate_passed": legacy_gate_passed,
        "inference_failure_reasons": inference_failure_reasons,
        "complete": len(prior_predictions) == len(samples),
        "quality_warnings": quality_warnings,
        "hard_failures": hard_failures,
        "emotion_response_outcome_count": outcome_counts,
        "emotion_response_severe_anomaly_count": severe_response_anomaly_count,
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
        "runtime_code_identity": asset_identity.get("runtime_code_identity"),
        "initialization_seconds": init_seconds,
        "server": {
            "runtime": service_facts,
            "registered_model_ids": [] if service_facts is None else service_facts.get("registered_model_ids", []),
            "requested_alias_count": {"profile": route_attempts["profile"], "emotion-response": route_attempts["emotion_response"]},
            "alive_before_intentional_shutdown": server_alive_before_shutdown,
            "intentional_shutdown_returncode": shutdown_returncode,
            "log_path": _relative_result_path(server_log, official_result_dir) if server_log.is_file() else None,
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
        "initial_cap_hits": initial_cap_hits,
        "retry_attempts": retry_attempts,
        "retry_resolved": retry_resolved,
        "unresolved_cap_hits": unresolved_cap_hits,
        "degraded_responses": {"count": len(unique_degraded_ids), "sample_ids": unique_degraded_ids},
        "prompt_token_ids_exact_match_count": exact_match_counts if backend == "vllm" else None,
        "prompt_token_ids_check_denominator": prompt_check_denominators if backend == "vllm" else None,
        "prompt_token_ids_unchecked_request_count": {
            route: max(0, route_attempts[route] - prompt_check_denominators[route])
            for route in ROUTE_ORDER
        } if backend == "vllm" else None,
        "output_ids_match_input_order": output_rows_match,
        "output_ids_sha256": hashlib.sha256("\n".join(output_ids).encode("utf-8")).hexdigest(),
        "gate_thresholds": {
            "profile_format": format_threshold,
            "emotion_response_format": format_threshold,
            "schema": len(samples),
            "prompt_token_ids": "exact for every successful vLLM response, including retries",
            "route_alias_requests": "one per prepared sample plus actual ER retries",
            "generation_cap_hits": "reported; unresolved valid ER length limits are quality warnings, not run-level failures",
        },
        "sample_diagnostics": sample_diagnostics,
        "outputs": {
            "submission.jsonl": {"path": "submission.jsonl", "sha256": final_submission_sha256, "rows": len(prior_predictions)},
            "raw_outputs.jsonl": {"path": _relative_result_path(raw_path, official_result_dir), "sha256": final_raw_sha256, "rows": len(prior_raw)},
            "performance_report.json": {"path": "performance_report.json"},
            "run_progress.json": {"path": _relative_result_path(progress_path, official_result_dir)},
        },
        "io_checkpointing": {
            "checkpoint_interval_samples": PROGRESS_CHECKPOINT_INTERVAL,
            "prefix_hashing": "incremental during run; full-file SHA-256 verified once before final report",
        },
        "prepared_prompt_count": prepared_prompt_counts,
        "successful_generation_response_count": route_response_counts,
        "systemic_route_failures": systemic_route_failures,
    }
    report_path = result_dir / "performance_report.json"
    _atomic_json(report_path, report)
    _atomic_json(progress_path, {
        **progress,
        "status": "complete" if report["complete"] else "failed",
        "completed_samples": len(prior_predictions),
        "submission_prefix_sha256": final_submission_sha256,
        "raw_prefix_sha256": final_raw_sha256,
        "updated_at_epoch_seconds": time.time(),
    })
    if inference_success and transactional_outputs:
        _publish_official_outputs(result_dir, official_result_dir)
    result_submission = published_submission_path if inference_success and transactional_outputs else submission_path
    result_report = published_report_path if inference_success and transactional_outputs else report_path
    print(f"{'PASS' if inference_success else 'FAIL'}: {result_submission}", flush=True)
    print(f"{'PASS' if inference_success else 'FAIL'}: {result_report}", flush=True)
    if inference_success and not diagnostic_quality_gate:
        print("WARNING: inference completed with prediction-quality warnings", flush=True)
    return 0 if inference_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
