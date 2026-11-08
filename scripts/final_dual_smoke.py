#!/usr/bin/env python3
"""Prepare a frozen target-free sample and run Transformers dual-Adapter gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_VAL_ROOT = ROOT / "outputs/evaluation/e03_a8_devft_v1/E03-A8-DEVFT-20261108-02-PDEV-1EPOCH"
DEFAULT_TARGET_FREE_VAL = DEFAULT_VAL_ROOT / "val_target_free.jsonl"
DEFAULT_TARGET_FREE_MANIFEST = DEFAULT_VAL_ROOT / "val_input_manifest.json"
EQUIVALENCE_SAMPLES = 5
VLLM_SMOKE_COUNTS = (10, 20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_jsonl_target_free(path: Path, *, allow_test_metadata: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            forbidden = {key for key in row if key == "target" or key.startswith("gold_") or key.startswith("target_")}
            if forbidden and not allow_test_metadata:
                raise ValueError(f"{path}:{line_number}: target fields are forbidden in the Smoke source")
            sample_id, history = row.get("id"), row.get("history")
            if not isinstance(sample_id, str) or not sample_id or not isinstance(history, list) or not history:
                raise ValueError(f"{path}:{line_number}: target-free input requires id and non-empty history")
            if history[-1].get("role") != "user":
                raise ValueError(f"{path}:{line_number}: history must end with the current user message")
            rows.append({"id": sample_id, "history": history})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return rows


def _latin_words(sample: dict[str, Any]) -> int:
    text = " ".join(str(message.get("content", "")) for message in sample["history"])
    return len(re.findall(r"[A-Za-z]{3,}", text))


def _val_sample_turn(sample: dict[str, Any]) -> int:
    turn_id = sample["history"][-1].get("turn_id")
    if isinstance(turn_id, int) and not isinstance(turn_id, bool) and turn_id >= 0:
        return turn_id
    match = re.search(r"(?:pp|t)(\d+)$", sample["id"])
    return int(match.group(1)) if match else 0


def select_val_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    ordered = sorted(rows, key=lambda row: (len(row["history"]), row["id"]))
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()

    def add_spread(candidates: list[dict[str, Any]], amount: int) -> None:
        available = [row for row in candidates if row["id"] not in chosen_ids]
        if not available or amount <= 0:
            return
        if amount >= len(available):
            indexes = list(range(len(available)))
        elif amount == 1:
            indexes = [len(available) // 2]
        else:
            indexes = sorted({round(index * (len(available) - 1) / (amount - 1)) for index in range(amount)})
        for index in indexes:
            row = available[index]
            if row["id"] not in chosen_ids and len(chosen) < count:
                chosen.append(row)
                chosen_ids.add(row["id"])

    english = sorted((row for row in rows if _latin_words(row) >= 2), key=lambda row: (-_latin_words(row), -len(row["history"]), row["id"]))
    add_spread(english, min(2, count))
    long_history = [row for row in ordered if len(row["history"]) >= 8]
    add_spread(long_history, min(6, count - len(chosen)))
    short_history = [row for row in ordered if len(row["history"]) <= 3]
    add_spread(short_history, min(2, count - len(chosen)))
    add_spread(ordered, count - len(chosen))
    return chosen[:count]


def prepare(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_root = resolve_path(Path(config["model_lock_path"]).parent)
    smoke_root = run_root / "transformers_smoke20"
    input_dir = smoke_root / "input"
    input_path = input_dir / "smoke20_input.jsonl"
    id_map_path = input_dir / "id_map.jsonl"
    manifest_path = input_dir / "manifest.json"
    if any(path.exists() for path in (input_path, id_map_path, manifest_path)):
        raise FileExistsError(f"frozen smoke input already exists under {input_dir}")

    test_path = args.test_inference_data.resolve()
    target_free_path = args.target_free_val.resolve()
    target_free_manifest_path = args.target_free_manifest.resolve()
    test_rows = read_jsonl_target_free(test_path, allow_test_metadata=True)
    test_raw_rows = []
    with test_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if "target" in row or any(key.startswith("gold_") for key in row):
                    raise ValueError("test_inference_data.jsonl unexpectedly includes Gold fields")
                test_raw_rows.append(row)
    if not test_rows or len(test_rows) >= 20:
        raise ValueError("this preparation path expects 1-19 existing test_inference_data rows")

    target_free_manifest = json.loads(target_free_manifest_path.read_text(encoding="utf-8"))
    if target_free_manifest.get("status") != "target_free_val_frozen":
        raise ValueError("Val source is not the already frozen target-free input")
    if target_free_manifest.get("target_fields_serialized") is not False or target_free_manifest.get("references_read") is not False:
        raise ValueError("Val target-free manifest does not certify Gold-blind input creation")
    if sha256_file(target_free_path) != target_free_manifest.get("files", {}).get(target_free_path.name):
        raise ValueError("frozen target-free Val file hash differs from its manifest")
    val_rows = read_jsonl_target_free(target_free_path)
    needed = 20 - len(test_rows)
    selected_val = select_val_rows(val_rows, needed)
    if len(selected_val) != needed:
        raise ValueError(f"target-free Val provides only {len(selected_val)} unique rows; {needed} required")

    combined = [(row, "test_inference_data.jsonl") for row in test_rows]
    combined.extend((row, str(target_free_path.relative_to(ROOT))) for row in selected_val)
    input_dir.mkdir(parents=True, exist_ok=True)
    inputs, mappings = [], []
    for index, (source_row, source_name) in enumerate(combined, 1):
        source_id = source_row["id"]
        turn_id = _val_sample_turn(source_row) if source_name != "test_inference_data.jsonl" else int(
            test_raw_rows[index - 1].get("target_user_turn_id", _val_sample_turn(source_row))
            if index <= len(test_rows) else _val_sample_turn(source_row)
        )
        official_id = f"test_{index:06d}_t{turn_id}"
        history = source_row["history"]
        inputs.append({"id": official_id, "history": history})
        mappings.append({
            "test_id": official_id,
            "source": source_name,
            "source_id": source_id,
            "source_turn_id": turn_id,
            "selected_history_sha256": hashlib.sha256(
                json.dumps(history, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })
    if len({row["id"] for row in inputs}) != 20:
        raise ValueError("generated smoke input IDs are not unique")

    def write_lines(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_lines(input_path, inputs)
    write_lines(id_map_path, mappings)
    coverage = {
        "sample_count": len(inputs),
        "history_message_count_min": min(len(row["history"]) for row in inputs),
        "history_message_count_max": max(len(row["history"]) for row in inputs),
        "long_history_count_ge_8": sum(len(row["history"]) >= 8 for row in inputs),
        "english_word_samples": sum(_latin_words(row) >= 2 for row in inputs),
    }
    manifest = {
        "version": "final-dual-smoke20-input-v1",
        "status": "frozen_target_free_input",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_inference_data": {"path": str(test_path.relative_to(ROOT)), "sha256": sha256_file(test_path), "row_count": len(test_rows)},
        "target_free_val": {
            "path": str(target_free_path.relative_to(ROOT)),
            "sha256": sha256_file(target_free_path),
            "source_manifest_sha256": sha256_file(target_free_manifest_path),
            "selected_rows": len(selected_val),
            "source_manifest_status": target_free_manifest["status"],
        },
        "input": {"path": str(input_path.relative_to(ROOT)), "sha256": sha256_file(input_path)},
        "id_map": {"path": str(id_map_path.relative_to(ROOT)), "sha256": sha256_file(id_map_path)},
        "coverage": coverage,
        "gold_read_for_generation_or_selection": False,
        "scoring_planned": False,
        "exposure_note": "Engineering-only Smoke; no quality claim. The Val source may share conversation context with earlier exploration.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "input": manifest["input"], "coverage": coverage}, ensure_ascii=False))


def _load_lock(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    lock_path = resolve_path(config["model_lock_path"])
    if not lock_path.is_file():
        raise FileNotFoundError(f"model lock not found: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return lock, lock_path


def run_transformers(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lock, lock_path = _load_lock(config)
    smoke_root = lock_path.parent / "transformers_smoke20"
    input_path = smoke_root / "input/smoke20_input.jsonl"
    manifest_path = smoke_root / "input/manifest.json"
    run_manifest_path = smoke_root / "manifest.json"
    if not input_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run `prepare` before the Transformers Smoke")
    if run_manifest_path.exists():
        raise FileExistsError(f"Transformers Smoke is already recorded: {run_manifest_path}")
    samples = read_jsonl_target_free(input_path)
    if len(samples) != 20:
        raise ValueError(f"expected exactly 20 frozen smoke rows, got {len(samples)}")
    if sha256_file(input_path) != json.loads(manifest_path.read_text(encoding="utf-8"))["input"]["sha256"]:
        raise ValueError("frozen input SHA-256 changed since preparation")

    from participant import run_inference as runner
    from src.inference.final_dual_protocol import (
        assemble_submission,
        build_emotion_response_prompt,
        build_profile_prompt,
    )
    from src.inference.final_dual_runtime import DualAdapterRuntime, cuda_memory_facts
    import torch

    model_path = resolve_path(config["model_path"])
    tokenizer_config = {**config["tokenizer"], "local_path": "model", "local_files_only": True}
    tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(
        {"model_id": config["model_id"], "model_revision": config["model_revision"], "tokenizer": tokenizer_config}
    )
    prompts = {key: resolve_path(value["path"]).read_text(encoding="utf-8") for key, value in config["prompts"].items()}
    prepared: dict[str, dict[str, Any]] = {}
    preparation_errors: dict[str, dict[str, str]] = {}
    for sample in samples:
        sample_id = sample["id"]
        routes = {}
        for route, builder in (
            ("profile", lambda: build_profile_prompt(
                sample, tokenizer,
                selection_prompt=prompts["selection"],
                profile_prompt=prompts["profile"],
                profile_budget_prompt=prompts["profile_history_budget"],
                prompt_cap=config["context"]["max_prompt_tokens"],
            )),
            ("emotion_response", lambda: build_emotion_response_prompt(
                sample, tokenizer,
                selection_prompt=prompts["selection"],
                emotion_response_prompt=prompts["emotion_response"],
                prompt_cap=config["context"]["max_prompt_tokens"],
            )),
        ):
            try:
                routes[route] = builder()
            except Exception as exc:
                routes[route] = None
                preparation_errors.setdefault(sample_id, {})[route] = f"{type(exc).__name__}: {exc}"
        prepared[sample_id] = routes

    adapter_paths = {
        "profile": resolve_path(lock["deployment_adapters"]["profile"]["path"]),
        "emotion-response": resolve_path(lock["deployment_adapters"]["emotion_response"]["path"]),
    }
    generation = config["generation"]
    eos = int(generation["eos_token_id"])
    pad = int(generation["pad_token_id"])
    baseline: dict[str, dict[str, dict[str, Any]]] = {"profile": {}, "emotion_response": {}}
    load_seconds: dict[str, float] = {}
    gpu_peak: dict[str, Any] = {}
    eq_samples = samples[:EQUIVALENCE_SAMPLES]

    for route, adapter_name, max_tokens in (
        ("profile", "profile", generation["profile"]["max_new_tokens"]),
        ("emotion_response", "emotion-response", generation["emotion_response"]["max_new_tokens"]),
    ):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        runtime = DualAdapterRuntime.load(
            base_model_path=model_path,
            tokenizer=tokenizer,
            adapters={adapter_name: adapter_paths[adapter_name]},
            torch_module=torch,
        )
        load_seconds[route] = time.monotonic() - started
        outputs = []
        try:
            for sample in eq_samples:
                prompt = prepared[sample["id"]][route]
                if prompt is None:
                    continue
                started = time.monotonic()
                result = runtime.generate(
                    adapter_name,
                    prompt["prompt_ids"],
                    max_new_tokens=max_tokens,
                    eos_token_id=eos,
                    pad_token_id=pad,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result["generation_seconds"] = time.monotonic() - started
                baseline[route][sample["id"]] = result
                outputs.append(result)
            gpu_peak[route] = cuda_memory_facts(torch)
        finally:
            runtime.close()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dual_started = time.monotonic()
    runtime = DualAdapterRuntime.load(
        base_model_path=model_path,
        tokenizer=tokenizer,
        adapters={name: adapter_paths[name] for name in ("profile", "emotion-response")},
        torch_module=torch,
    )
    dual_load_seconds = time.monotonic() - dual_started
    dual_results: dict[str, dict[str, Any]] = {}
    repeat_results: dict[str, dict[str, dict[str, Any]]] = {"profile": {}, "emotion_response": {}}
    route_latencies: dict[str, list[float]] = {"profile": [], "emotion_response": []}
    try:
        for sample in samples:
            sample_id = sample["id"]
            dual_results[sample_id] = {}
            for route, adapter_name, max_tokens in (
                ("profile", "profile", generation["profile"]["max_new_tokens"]),
                ("emotion_response", "emotion-response", generation["emotion_response"]["max_new_tokens"]),
            ):
                prompt = prepared[sample_id][route]
                if prompt is None:
                    dual_results[sample_id][route] = None
                    continue
                started = time.monotonic()
                result = runtime.generate(
                    adapter_name,
                    prompt["prompt_ids"],
                    max_new_tokens=max_tokens,
                    eos_token_id=eos,
                    pad_token_id=pad,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result["generation_seconds"] = time.monotonic() - started
                route_latencies[route].append(result["generation_seconds"])
                dual_results[sample_id][route] = result

        # Revisit the equivalence subset in reverse sample order and reverse route order.
        for sample in reversed(eq_samples):
            sample_id = sample["id"]
            repeat_results["emotion_response"][sample_id] = None
            repeat_results["profile"][sample_id] = None
            for route, adapter_name, max_tokens in (
                ("emotion_response", "emotion-response", generation["emotion_response"]["max_new_tokens"]),
                ("profile", "profile", generation["profile"]["max_new_tokens"]),
            ):
                prompt = prepared[sample_id][route]
                if prompt is None:
                    continue
                repeat_results[route][sample_id] = runtime.generate(
                    adapter_name,
                    prompt["prompt_ids"],
                    max_new_tokens=max_tokens,
                    eos_token_id=eos,
                    pad_token_id=pad,
                )
        dual_gpu_peak = cuda_memory_facts(torch)
    finally:
        runtime.close()

    predictions, raw_rows, equivalence = [], [], {"profile": [], "emotion_response": []}
    for route in ("profile", "emotion_response"):
        for sample in eq_samples:
            sample_id = sample["id"]
            base_result = baseline[route].get(sample_id)
            dual_result = dual_results.get(sample_id, {}).get(route)
            repeated = repeat_results[route].get(sample_id)
            equal_baseline = bool(base_result and dual_result and base_result["token_ids"] == dual_result["token_ids"])
            equal_repeat = bool(dual_result and repeated and dual_result["token_ids"] == repeated["token_ids"])
            equivalence[route].append({
                "id": sample_id,
                "single_vs_dual_exact": equal_baseline,
                "dual_repeat_exact": equal_repeat,
                "single_token_ids_sha256": None if not base_result else hashlib.sha256(json.dumps(base_result["token_ids"], separators=(",", ":")).encode()).hexdigest(),
                "dual_token_ids_sha256": None if not dual_result else hashlib.sha256(json.dumps(dual_result["token_ids"], separators=(",", ":")).encode()).hexdigest(),
                "repeat_token_ids_sha256": None if not repeated else hashlib.sha256(json.dumps(repeated["token_ids"], separators=(",", ":")).encode()).hexdigest(),
            })

    profile_success = er_success = schema_success = 0
    for sample in samples:
        sample_id = sample["id"]
        route_results = dual_results[sample_id]
        profile_result = route_results.get("profile")
        er_result = route_results.get("emotion_response")
        submission, diagnostics = assemble_submission(
            sample_id,
            None if profile_result is None else profile_result["text"],
            None if er_result is None else er_result["text"],
            json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8")),
        )
        profile_success += diagnostics["profile_parse_error"] is None
        er_success += all(diagnostics[key] is None for key in (
            "emotion_parse_error", "response_parse_error", "emotion_response_format_error"
        ))
        schema_success += 1
        predictions.append(submission)
        raw_rows.append({
            "id": sample_id,
            "profile_raw": None if profile_result is None else profile_result["text"],
            "emotion_response_raw": None if er_result is None else er_result["text"],
            "profile_token_ids": None if profile_result is None else profile_result["token_ids"],
            "emotion_response_token_ids": None if er_result is None else er_result["token_ids"],
            "profile_generation_seconds": None if profile_result is None else profile_result["generation_seconds"],
            "emotion_response_generation_seconds": None if er_result is None else er_result["generation_seconds"],
            "profile_prompt": prepared[sample_id]["profile"],
            "emotion_response_prompt": prepared[sample_id]["emotion_response"],
            "preparation_errors": preparation_errors.get(sample_id, {}),
            "diagnostics": diagnostics,
        })

    dual_wall_seconds = time.monotonic() - dual_started
    equivalence_counts = {
        route: {
            "single_vs_dual_exact": sum(item["single_vs_dual_exact"] for item in equivalence[route]),
            "dual_repeat_exact": sum(item["dual_repeat_exact"] for item in equivalence[route]),
            "sample_count": len(equivalence[route]),
        }
        for route in equivalence
    }
    gate_passed = (
        len(predictions) == 20
        and schema_success == 20
        and profile_success >= 19
        and er_success >= 19
        and all(value["single_vs_dual_exact"] == EQUIVALENCE_SAMPLES and value["dual_repeat_exact"] == EQUIVALENCE_SAMPLES for value in equivalence_counts.values())
    )
    output_files = {
        "predictions.jsonl": predictions,
        "raw_outputs.jsonl": raw_rows,
    }
    for filename, rows in output_files.items():
        (smoke_root / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    (smoke_root / "equivalence.json").write_text(json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    code_paths = (
        "scripts/final_dual_smoke.py",
        "scripts/final_dual_preflight.py",
        "src/inference/final_dual_protocol.py",
        "src/inference/final_dual_runtime.py",
        "tests/test_final_dual_protocol.py",
        "tests/test_final_dual_runtime.py",
        "configs/deployment/final_dual.yaml",
    )
    execution_git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    execution_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).splitlines()
    execution_code_changes = [line for line in execution_status if any(path in line for path in code_paths)]
    report = {
        "version": "final-dual-transformers-smoke-v1",
        "status": "passed" if gate_passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": execution_git_head,
        "model_lock_git_head": lock["git_head"],
        "execution_code_changes": execution_code_changes,
        "code_files_sha256": {path: sha256_file(ROOT / path) for path in code_paths},
        "model_lock": {"path": str(lock_path.relative_to(ROOT)), "sha256": sha256_file(lock_path)},
        "input_sha256": sha256_file(input_path),
        "sample_count": len(predictions),
        "format_success_count": {"profile": profile_success, "emotion_response": er_success},
        "schema_valid_count": schema_success,
        "equivalence": equivalence_counts,
        "init_seconds": {"single_profile": load_seconds["profile"], "single_emotion_response": load_seconds["emotion_response"], "dual": dual_load_seconds},
        "generation_seconds": {
            route: {
                "count": len(values),
                "median": statistics.median(values) if values else None,
                "p95": sorted(values)[max(0, int(0.95 * len(values) + 0.999) - 1)] if values else None,
            }
            for route, values in route_latencies.items()
        },
        "dual_run_wall_seconds": dual_wall_seconds,
        "gpu_memory": gpu_peak | {"dual": dual_gpu_peak},
        "tokenizer_identity": tokenizer_facts,
        "outputs": {name: sha256_file(smoke_root / name) for name in output_files},
        "failed_samples": [row["id"] for row in raw_rows if any(row["diagnostics"].values())],
        "gate_thresholds": {"profile_format": "19/20", "emotion_response_format": "19/20", "schema": "20/20", "single_dual_token_equivalence": "5/5 per Adapter", "repeat_switch_equivalence": "5/5 per Adapter"},
    }
    run_manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "format_success_count": report["format_success_count"], "schema_valid_count": schema_success, "equivalence": equivalence_counts}, ensure_ascii=False))
    if not gate_passed:
        raise SystemExit(1)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, int(0.95 * len(values) + 0.999999) - 1)]


def _code_identity() -> dict[str, Any]:
    paths = (
        "scripts/final_dual_smoke.py",
        "src/inference/final_dual_protocol.py",
        "src/inference/final_dual_vllm.py",
        "tests/test_final_dual_vllm.py",
        "configs/deployment/final_dual.yaml",
    )
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_files_sha256": {path: sha256_file(ROOT / path) for path in paths if (ROOT / path).is_file()},
    }


def run_vllm(args: argparse.Namespace) -> None:
    """Run a 10-row gate followed by the same frozen 20-row vLLM gate."""
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lock, lock_path = _load_lock(config)
    smoke_root = lock_path.parent / "vllm_smoke"
    input_path = lock_path.parent / "transformers_smoke20/input/smoke20_input.jsonl"
    input_manifest_path = lock_path.parent / "transformers_smoke20/input/manifest.json"
    aggregate_path = smoke_root / "manifest.json"
    if aggregate_path.exists():
        raise FileExistsError(f"vLLM Smoke is already recorded: {aggregate_path}")
    if not input_path.is_file() or not input_manifest_path.is_file():
        raise FileNotFoundError("run `prepare` before the vLLM Smoke")
    samples = read_jsonl_target_free(input_path)
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if len(samples) != 20 or sha256_file(input_path) != input_manifest.get("input", {}).get("sha256"):
        raise ValueError("vLLM Smoke requires the unchanged frozen 20-row input")

    from participant import run_inference as runner
    from src.inference.final_dual_protocol import (
        assemble_submission,
        build_emotion_response_prompt,
        build_profile_prompt,
    )
    from src.inference.final_dual_vllm import DualLoRAVLLMService, GpuMemoryMonitor, gpu_snapshot

    tokenizer_config = {**config["tokenizer"], "local_path": "model", "local_files_only": True}
    tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(
        {"model_id": config["model_id"], "model_revision": config["model_revision"], "tokenizer": tokenizer_config}
    )
    prompts = {key: resolve_path(value["path"]).read_text(encoding="utf-8") for key, value in config["prompts"].items()}
    schema = json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8"))
    routes = (
        ("profile", "profile", "profile", "profile", config["generation"]["profile"]),
        ("emotion_response", "emotion-response", "emotion_response", "emotion_response", config["generation"]["emotion_response"]),
    )
    service = DualLoRAVLLMService(
        config=config,
        model_lock=lock,
        root=ROOT,
        log_path=smoke_root / "vllm_server.log",
        vllm_bin=args.vllm_bin,
        startup_timeout_seconds=args.startup_timeout,
    )
    monitor = GpuMemoryMonitor(interval_seconds=1.0)
    monitor.start()
    stages: list[dict[str, Any]] = []
    service_start: dict[str, Any] | None = None
    aliases: list[str] = []
    service_alive_before_stop = False
    shutdown_returncode: int | None = None
    failure: str | None = None
    stage_directories: list[Path] = []
    try:
        service_start = service.start()
        aliases = service.require_adapters()
        for sample_count in VLLM_SMOKE_COUNTS:
            stage_name = f"smoke{sample_count}"
            stage_root = smoke_root / stage_name
            if stage_root.exists():
                raise FileExistsError(f"refusing to overwrite vLLM evidence: {stage_root}")
            stage_root.mkdir(parents=True, exist_ok=False)
            stage_directories.append(stage_root)
            stage_samples = samples[:sample_count]
            predictions: list[dict[str, Any]] = []
            raw_rows: list[dict[str, Any]] = []
            end_to_end_seconds: list[float] = []
            route_seconds: dict[str, list[float]] = {"profile": [], "emotion_response": []}
            prompt_exact: dict[str, list[bool]] = {"profile": [], "emotion_response": []}
            format_success = {"profile": 0, "emotion_response": 0}
            cap_hits = {"profile": 0, "emotion_response": 0}
            route_attempts = {"profile": 0, "emotion_response": 0}
            route_errors: dict[str, list[dict[str, str]]] = {"profile": [], "emotion_response": []}
            schema_success = 0
            stage_failure: str | None = None

            for sample in stage_samples:
                sample_id = sample["id"]
                sample_started = time.monotonic()
                prepared: dict[str, dict[str, Any] | None] = {"profile": None, "emotion_response": None}
                generated: dict[str, dict[str, Any] | None] = {"profile": None, "emotion_response": None}
                preparation_errors: dict[str, str] = {}
                route_error_facts: dict[str, str] = {}
                for route, _, prompt_key, _, _ in routes:
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
                        preparation_errors[route] = f"{type(exc).__name__}: {exc}"

                for route, alias, _, _, generation in routes:
                    prompt = prepared[route]
                    if prompt is None:
                        route_error_facts[route] = preparation_errors[route]
                        continue
                    route_attempts[route] += 1
                    try:
                        response = service.chat_completion(
                            model_alias=alias,
                            messages=prompt["messages"],
                            expected_prompt_token_ids=prompt["prompt_ids"],
                            max_tokens=int(generation["max_new_tokens"]),
                        )
                        generated[route] = response
                        route_seconds[route].append(float(response["generation_seconds"]))
                        prompt_exact[route].append(bool(response["prompt_token_ids_exact_match"]))
                        cap_hits[route] += response["finish_reason"] == "length" or response["usage_completion_tokens"] >= int(generation["max_new_tokens"])
                        if not response["prompt_token_ids_exact_match"]:
                            route_error_facts[route] = "server_prompt_token_ids_mismatch"
                        if response["requested_model_alias"] != alias:
                            route_error_facts[route] = "requested_model_alias_mismatch"
                    except Exception as exc:
                        route_error_facts[route] = f"{type(exc).__name__}: {exc}"
                        route_errors[route].append({"id": sample_id, "error": route_error_facts[route]})
                        if not service.is_alive():
                            stage_failure = f"vLLM server exited while generating {sample_id}: {route_error_facts[route]}"
                            break

                profile_result = generated["profile"]
                er_result = generated["emotion_response"]
                submission, diagnostics = assemble_submission(
                    sample_id,
                    None if profile_result is None else profile_result["text"],
                    None if er_result is None else er_result["text"],
                    schema,
                )
                format_success["profile"] += diagnostics["profile_parse_error"] is None
                format_success["emotion_response"] += all(diagnostics[key] is None for key in (
                    "emotion_parse_error", "response_parse_error", "emotion_response_format_error"
                ))
                schema_success += 1
                predictions.append(submission)
                sample_seconds = time.monotonic() - sample_started
                end_to_end_seconds.append(sample_seconds)
                raw_rows.append({
                    "id": sample_id,
                    "profile_raw": None if profile_result is None else profile_result["text"],
                    "emotion_response_raw": None if er_result is None else er_result["text"],
                    "profile_generation": None if profile_result is None else {key: value for key, value in profile_result.items() if key != "text"},
                    "emotion_response_generation": None if er_result is None else {key: value for key, value in er_result.items() if key != "text"},
                    "profile_prompt": None if prepared["profile"] is None else {key: prepared["profile"][key] for key in ("prompt_tokens", "prompt_token_ids_sha256", "selected_message_ids", "selected_history_sha256")},
                    "emotion_response_prompt": None if prepared["emotion_response"] is None else {key: prepared["emotion_response"][key] for key in ("prompt_tokens", "prompt_token_ids_sha256", "selected_message_ids", "selected_history_sha256")},
                    "preparation_errors": preparation_errors,
                    "route_errors": route_error_facts,
                    "sample_end_to_end_seconds": sample_seconds,
                    "diagnostics": diagnostics,
                })
                if stage_failure:
                    break

            _write_jsonl(stage_root / "predictions.jsonl", predictions)
            _write_jsonl(stage_root / "raw_outputs.jsonl", raw_rows)
            route_prompt_gates = {
                route: len(values) == sample_count and all(values)
                for route, values in prompt_exact.items()
            }
            format_threshold = int(sample_count * 0.95 + 0.999999)
            stage_gate = (
                stage_failure is None
                and len(predictions) == sample_count
                and schema_success == sample_count
                and format_success["profile"] >= format_threshold
                and format_success["emotion_response"] >= format_threshold
                and all(route_prompt_gates.values())
                and all(count == sample_count for count in route_attempts.values())
                and all(count == 0 for count in cap_hits.values())
                and all(not errors for errors in route_errors.values())
            )
            stage_report = {
                "version": "final-dual-vllm-smoke-v1",
                "status": "passed" if stage_gate else "failed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sample_count": len(predictions),
                "requested_sample_count": sample_count,
                "input_sha256": sha256_file(input_path),
                "input_ids": [sample["id"] for sample in stage_samples],
                "registered_model_ids": aliases,
                "route_request_count": route_attempts,
                "format_success_count": format_success,
                "format_threshold_per_route": format_threshold,
                "schema_valid_count": schema_success,
                "prompt_token_ids_exact_match_count": {route: sum(values) for route, values in prompt_exact.items()},
                "prompt_token_ids_route_gate": route_prompt_gates,
                "generation_cap_hits": cap_hits,
                "route_failures": route_errors,
                "route_generation_seconds": {
                    route: {"count": len(values), "median": statistics.median(values) if values else None, "p95": _p95(values)}
                    for route, values in route_seconds.items()
                },
                "dual_end_to_end_seconds": {
                    "count": len(end_to_end_seconds),
                    "median": statistics.median(end_to_end_seconds) if end_to_end_seconds else None,
                    "p95": _p95(end_to_end_seconds),
                    "scope": "from per-sample prompt construction through both serial completions and official output assembly",
                },
                "stage_failure": stage_failure,
                "stage_gate_passed": stage_gate,
                "outputs": {
                    "predictions.jsonl": sha256_file(stage_root / "predictions.jsonl"),
                    "raw_outputs.jsonl": sha256_file(stage_root / "raw_outputs.jsonl"),
                },
                "gpu_memory_snapshot_after_batch": gpu_snapshot(),
            }
            (stage_root / "manifest.json").write_text(json.dumps(stage_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            stages.append({"name": stage_name, **stage_report})
            print(json.dumps({"stage": stage_name, "status": stage_report["status"], "format_success_count": format_success, "schema_valid_count": schema_success, "prompt_token_ids_exact_match_count": stage_report["prompt_token_ids_exact_match_count"]}, ensure_ascii=False))
            if not stage_gate:
                break

        service_alive_before_stop = service.is_alive()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"status": "failed", "error": failure}, ensure_ascii=False), file=sys.stderr)
    finally:
        shutdown_returncode = service.close()
        gpu_memory = monitor.stop()

    gate_passed = len(stages) == 2 and all(stage.get("status") == "passed" for stage in stages) and service_alive_before_stop
    log_path = smoke_root / "vllm_server.log"
    report = {
        "version": "final-dual-vllm-smoke-v1",
        "status": "passed" if gate_passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_identity": _code_identity(),
        "config": {"path": str(args.config.relative_to(ROOT)), "sha256": sha256_file(args.config)},
        "model_lock": {"path": str(lock_path.relative_to(ROOT)), "sha256": sha256_file(lock_path)},
        "input": {"path": str(input_path.relative_to(ROOT)), "sha256": sha256_file(input_path), "sample_count": len(samples)},
        "tokenizer_identity": tokenizer_facts,
        "service": service_start,
        "registered_model_ids": aliases,
        "service_alive_before_intentional_shutdown": service_alive_before_stop,
        "intentional_shutdown_returncode": shutdown_returncode,
        "gpu_memory_monitor": gpu_memory,
        "server_log": {"path": str(log_path.relative_to(ROOT)), "sha256": sha256_file(log_path) if log_path.is_file() else None},
        "stages": stages,
        "failure": failure,
        "gate_thresholds": {
            "format": "ceil(95% of each stage row count) per route",
            "schema": "100%",
            "prompt_token_ids": "exact equality for both routes on every request",
            "registered_aliases": ["profile", "emotion-response"],
            "generation_cap_hits": 0,
            "one_service_for_10_then_20": True,
        },
    }
    smoke_root.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "stages": [{"name": item["name"], "status": item["status"]} for item in stages], "gpu_memory_monitor": gpu_memory}, ensure_ascii=False))
    if not gate_passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, default=ROOT / "configs/deployment/final_dual.yaml")
    prep.add_argument("--test-inference-data", type=Path, default=ROOT / "test_inference_data.jsonl")
    prep.add_argument("--target-free-val", type=Path, default=DEFAULT_TARGET_FREE_VAL)
    prep.add_argument("--target-free-manifest", type=Path, default=DEFAULT_TARGET_FREE_MANIFEST)
    prep.set_defaults(func=prepare)
    run = sub.add_parser("run-transformers")
    run.add_argument("--config", type=Path, default=ROOT / "configs/deployment/final_dual.yaml")
    run.set_defaults(func=run_transformers)
    run_vllm_parser = sub.add_parser("run-vllm")
    run_vllm_parser.add_argument("--config", type=Path, default=ROOT / "configs/deployment/final_dual.yaml")
    run_vllm_parser.add_argument("--vllm-bin", default="vllm")
    run_vllm_parser.add_argument("--startup-timeout", type=float, default=900.0)
    run_vllm_parser.set_defaults(func=run_vllm)
    args = parser.parse_args()
    args.config = args.config.resolve()
    if hasattr(args, "test_inference_data"):
        args.test_inference_data = args.test_inference_data.resolve()
        args.target_free_val = args.target_free_val.resolve()
        args.target_free_manifest = args.target_free_manifest.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
