#!/usr/bin/env python3
"""Prepare a frozen target-free sample and run Transformers dual-Adapter gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
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
            torch.cuda.reset_peak_memory_stats(0)
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
        torch.cuda.reset_peak_memory_stats(0)
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
    report = {
        "version": "final-dual-transformers-smoke-v1",
        "status": "passed" if gate_passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": lock["git_head"],
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
    args = parser.parse_args()
    args.config = args.config.resolve()
    if hasattr(args, "test_inference_data"):
        args.test_inference_data = args.test_inference_data.resolve()
        args.target_free_val = args.target_free_val.resolve()
        args.target_free_manifest = args.target_free_manifest.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
