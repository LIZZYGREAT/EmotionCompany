#!/usr/bin/env python3
"""Generate target-free E01-B Val predictions, then compare them to A8 P4504."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from src.evaluation import load_submission_schema, read_jsonl, schema_enums, sha256_file
from src.evaluation.e01_b_final_val import (
    paired_conversation_bootstrap,
    prefix_token_ids_match,
    profile_hit_transitions,
    select_stable_ids,
)
from src.evaluation.metrics import emotion_metrics, profile_metrics, raw_diagnostics, response_metrics
from src.evaluation.profile_prefix import ProfilePrefixStopper, parse_profile_prefix


DEFAULT_CONFIG = ROOT / "configs/evaluation/e01_b_final_val.yaml"
BASE_SHA256 = "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
PROFILE_DIMS = ("personality_traits", "interests", "style")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path, *, allow_external_symlink: bool = False) -> Path:
    path = Path(value).expanduser()
    lexical = Path(os.path.abspath(path if path.is_absolute() else ROOT / path))
    if not lexical.is_relative_to(ROOT.resolve()):
        raise ValueError(f"configured path escapes the repository: {value}")
    resolved = lexical.resolve()
    if not allow_external_symlink and not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"configured path resolves outside the repository: {value}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"{path}:{line_number}: blank or incomplete JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _unique_by(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{label} has a missing or duplicate {field}")
        result[value] = row
    return result


def _runner_config(adapter_path: Path, adapter_sha256: str, adapter_config_sha256: str, generation: dict[str, Any]) -> dict[str, Any]:
    value = _read_json(ROOT / "participant/configs/inference_config.json")
    base_path = _resolve("participant/model", allow_external_symlink=True)
    value["model_path"] = str(base_path)
    value["format_version"] = "F2"
    value["expected_weight_sha256"] = BASE_SHA256
    value["quantization"]["enabled"] = True
    value["context"]["max_prompt_tokens"] = int(generation["max_prompt_tokens"])
    value["generation"] = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    value["experiment_id"] = "E01-B-FINAL-VAL"
    value["adapter"] = {
        "enabled": True,
        "path": str(adapter_path),
        "expected_base_weight_sha256": BASE_SHA256,
        "expected_adapter_weight_sha256": adapter_sha256,
        "expected_adapter_config_sha256": adapter_config_sha256,
    }
    return value


def _invoke(model: Any, tokenizer: Any, prompt_ids: list[int], generation: dict[str, Any], criteria: Any | None = None) -> tuple[list[int], str, str, float]:
    import torch

    from transformers import StoppingCriteriaList

    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    options = dict(generation)
    if criteria is not None:
        options["stopping_criteria"] = StoppingCriteriaList([criteria])
    inference.synchronize(torch)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **options)
    inference.synchronize(torch)
    elapsed = time.monotonic() - started
    token_ids = [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop = (
        "eos" if eos_id is not None and eos_id in token_ids
        else "max_new_tokens" if len(token_ids) >= int(generation["max_new_tokens"])
        else "unknown"
    )
    return token_ids, text, stop, elapsed


def _generate_prefix(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    generation: dict[str, Any],
    enums: dict[str, list[str]],
) -> dict[str, Any]:
    from transformers import StoppingCriteria

    stopper = ProfilePrefixStopper(
        lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True),
        enums,
        stop_on_rejection=True,
    )

    class PrefixCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0, len(prompt_ids):].detach().cpu().tolist()
            return stopper.inspect(generated)

    token_ids, text, stop_reason, seconds = _invoke(
        model, tokenizer, prompt_ids, generation, PrefixCriteria(),
    )
    result = stopper.result or stopper.finish(stop_reason, token_ids)
    return {
        "token_ids": token_ids,
        "text": text,
        "result": result,
        "stop_reason": stopper.stop_reason or stop_reason,
        "seconds": seconds,
        "generated_tokens": len(token_ids),
    }


def _profile_result(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "reason": result.reason,
        "emotion_label": result.emotion_label if result.valid else "",
        "user_profile": {
            dimension: list((result.user_profile or {}).get(dimension, []))
            for dimension in PROFILE_DIMS
        },
    }


def _parse_full(sample_id: str, raw_text: str, enums: dict[str, list[str]]) -> tuple[dict[str, Any], bool, bool]:
    return inference.parse_prediction(sample_id, raw_text, enums)


def _full_profile_row(sample_id: str, cache: dict[str, Any], prompt_tokens: int, *, reused: bool) -> dict[str, Any]:
    return {
        "id": sample_id,
        "raw_text": cache["text"],
        "generated_tokens": len(cache["token_ids"]),
        "stop_reason": cache["stop_reason"],
        "generation_seconds": cache["seconds"],
        "prompt_tokens": prompt_tokens,
        "reused_from_full_generation": reused,
    }


def _generate(config_path: Path, config: dict[str, Any], run_id: str) -> tuple[Path, dict[str, Any]]:
    import torch
    from jsonschema import Draft202012Validator
    from scripts.e03_profile_fast_eval import _load_model

    training_config = _config(_resolve(config["sources"]["training_config"]))
    training_root = _resolve(config["sources"]["candidate_run_root"])
    training_manifest = _read_json(training_root / "manifest.json")
    training_summary = _read_json(training_root / "training_summary.json")
    if (
        training_manifest.get("status") != "completed"
        or training_manifest.get("optimizer_steps") != 4504
        or training_summary.get("status") != "passed"
        or training_summary.get("global_step") != 4504
        or training_summary.get("rows_consumed") != 1821
        or training_summary.get("base_weights_unchanged") is not True
        or training_summary.get("independent_reload", {}).get("status") != "passed"
    ):
        raise ValueError("E01-B Train+Dev candidate is not a completed, independently reloaded Step 4504 Run")
    adapter_path = training_root / "adapter"
    adapter_identity, _ = inference.adapter_weight_identity(adapter_path)
    if adapter_identity.get("weight_sha256") != training_summary.get("final_adapter_weight_sha256"):
        raise ValueError("E01-B final Adapter weight identity differs from its training summary")

    a8_eval_root = _resolve(config["sources"]["a8_evaluation_root"])
    input_manifest = _read_json(a8_eval_root / "val_input_manifest.json")
    generation_manifest = _read_json(a8_eval_root / "generation_manifest.json")
    scoring_manifest = _read_json(a8_eval_root / "scoring_manifest.json")
    target_free_path = _resolve(config["sources"]["target_free_input"])
    id_map_path = _resolve(config["sources"]["id_map"])
    if (
        input_manifest.get("status") != "target_free_val_frozen"
        or generation_manifest.get("status") != "predictions_frozen"
        or scoring_manifest.get("status") != "scored_after_prediction_freeze"
    ):
        raise ValueError("A8 target-free Val source or prediction freeze is incomplete")
    for filename, expected in input_manifest.get("files", {}).items():
        if sha256_file(a8_eval_root / filename) != expected:
            raise ValueError(f"frozen A8 Val input changed: {filename}")
    if sha256_file(target_free_path) != input_manifest["files"].get("val_target_free.jsonl"):
        raise ValueError("configured target-free input differs from its A8 Manifest")
    if sha256_file(id_map_path) != input_manifest["files"].get("val_id_map.jsonl"):
        raise ValueError("configured Val ID map differs from its A8 Manifest")
    if sha256_file(a8_eval_root / "predictions.jsonl") != generation_manifest.get("prediction_sha256"):
        raise ValueError("A8 predictions changed after their freeze")
    target_free_rows = _read_jsonl(target_free_path)
    id_map_rows = _read_jsonl(id_map_path)
    if len(target_free_rows) != 2354 or len(id_map_rows) != 2354:
        raise ValueError("frozen target-free Val input and ID map must contain 2,354 rows")
    input_by_id = _unique_by(target_free_rows, "id", "target-free Val")
    map_by_id = _unique_by(id_map_rows, "eval_id", "Val ID map")
    if set(input_by_id) != set(map_by_id):
        raise ValueError("target-free Val input and ID map cover different sample IDs")
    for row in target_free_rows:
        if set(row) != {"id", "history"} or not isinstance(row.get("history"), list) or not row["history"]:
            raise ValueError("target-free inference input contains Gold or lacks history")

    eval_root = _resolve(config["output_root"]) / run_id
    if eval_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing Val evaluation Run: {eval_root}")
    eval_root.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "version": "E01-B-Final-Val-run-v1",
        "run_id": run_id,
        "experiment_id": "E01-B-FINAL-VAL",
        "status": "generating",
        "stage": "prefix_equivalence_preflight",
        "created_at_utc": _now(),
        "candidate_run_manifest_sha256": sha256_file(training_root / "manifest.json"),
        "candidate_training_summary_sha256": sha256_file(training_root / "training_summary.json"),
        "candidate_adapter_weight_sha256": adapter_identity["weight_sha256"],
        "base_weight_sha256_expected": BASE_SHA256,
        "target_free_input_sha256": sha256_file(target_free_path),
        "target_free_input_count": len(target_free_rows),
        "target_free_input_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in target_free_rows).encode("utf-8")).hexdigest(),
        "val_id_map_sha256": sha256_file(id_map_path),
        "val_id_map_count": len(id_map_rows),
        "official_val_gold_read": False,
    }
    _write_json(eval_root / "manifest.json", run_manifest, exclusive=True)
    equivalence_ids = select_stable_ids(
        list(input_by_id),
        count=int(config["generation"]["equivalence_count"]),
        seed=int(config["generation"]["equivalence_seed"]),
        purpose=str(config["generation"]["prefix_selection_purpose"]),
    )
    full_json_ids = select_stable_ids(
        list(input_by_id),
        count=int(config["generation"]["full_json_count"]),
        seed=int(config["generation"]["full_json_seed"]),
        purpose=str(config["generation"]["full_json_selection_purpose"]),
    )
    _write_json(eval_root / "fixed_full_json_ids.json", {
        "selection_method": "sha256_ranked_id_selection_v1",
        "purpose": config["generation"]["full_json_selection_purpose"],
        "seed": config["generation"]["full_json_seed"],
        "count": len(full_json_ids),
        "ids": full_json_ids,
        "ids_sha256": hashlib.sha256("\n".join(full_json_ids).encode("utf-8")).hexdigest(),
    }, exclusive=True)
    _write_json(eval_root / "prefix_equivalence_ids.json", {
        "selection_method": "sha256_ranked_id_selection_v1",
        "purpose": config["generation"]["prefix_selection_purpose"],
        "seed": config["generation"]["equivalence_seed"],
        "count": len(equivalence_ids),
        "ids": equivalence_ids,
        "ids_sha256": hashlib.sha256("\n".join(equivalence_ids).encode("utf-8")).hexdigest(),
    }, exclusive=True)

    runner_config = _runner_config(
        adapter_path,
        adapter_identity["weight_sha256"],
        adapter_identity["config_sha256"],
        config["generation"],
    )
    prompt_text, submission_schema, enums, prompt_facts = inference.load_runtime_protocol(runner_config)
    if prompt_facts.get("sha256") != training_config["protocol"]["prompt_sha256"]:
        raise ValueError("E01-B Val evaluation Prompt does not match the approved P0 hash")
    model_load_started = time.monotonic()
    model, tokenizer, model_config, model_facts = _load_model(runner_config, inference, torch)
    model_load_seconds = time.monotonic() - model_load_started
    generation = runner_config["generation"]
    full_cache: dict[str, dict[str, Any]] = {}
    prefix_cache: dict[str, dict[str, Any]] = {}
    fallback_raw_rows: list[dict[str, Any]] = []
    equivalence_path = eval_root / "prefix_equivalence.jsonl.tmp"
    profile_temp = eval_root / "profile_predictions.jsonl.tmp"
    full_temp = eval_root / "full_json_predictions.jsonl.tmp"
    equivalence_handle = equivalence_path.open("x", encoding="utf-8", newline="\n")
    profile_handle = profile_temp.open("x", encoding="utf-8", newline="\n")
    equivalence_rows: list[dict[str, Any]] = []
    prefix_generation_times: list[float] = []
    full_generation_times: list[float] = []
    full_json_generation_times: list[float] = []
    prefix_fallback_count = 0
    try:
        for index, sample_id in enumerate(equivalence_ids, 1):
            sample = input_by_id[sample_id]
            selection, _messages = inference.prepare_sample_messages(
                sample, prompt_text, tokenizer, int(config["generation"]["max_prompt_tokens"]),
            )
            if selection.get("floor_exceeds_cap"):
                raise ValueError(f"fixed prefix equivalence Prompt exceeds P=1280 for ID {sample_id}")
            prompt_ids = list(selection["prompt_ids"])
            full_ids, full_text, full_stop, full_seconds = _invoke(model, tokenizer, prompt_ids, generation)
            full_generation_times.append(full_seconds)
            full_cache[sample_id] = {
                "token_ids": full_ids,
                "text": full_text,
                "stop_reason": full_stop,
                "seconds": full_seconds,
                "prompt_ids": prompt_ids,
                "prompt_tokens": len(prompt_ids),
            }
            full_result = parse_profile_prefix(full_text, enums)
            prefix = _generate_prefix(model, tokenizer, prompt_ids, generation, enums)
            prefix_generation_times.append(prefix["seconds"])
            prefix_cache[sample_id] = prefix
            token_match = prefix_token_ids_match(full_ids, prefix["token_ids"])
            labels_match = bool(
                full_result.valid
                and prefix["result"].valid
                and full_result.emotion_label == prefix["result"].emotion_label
                and dict(full_result.user_profile or {}) == dict(prefix["result"].user_profile or {})
            )
            row = {
                "id": sample_id,
                "full_status": full_result.status,
                "full_reason": full_result.reason,
                "prefix_status": prefix["result"].status,
                "prefix_reason": prefix["result"].reason,
                "full_token_count": len(full_ids),
                "prefix_token_count": len(prefix["token_ids"]),
                "token_prefix_exact_match": token_match,
                "parsed_labels_exact_match": labels_match,
                "equivalence_passed": token_match and labels_match,
                "full_generated_token_ids": full_ids,
                "prefix_generated_token_ids": prefix["token_ids"],
                "full_generation_seconds": full_seconds,
                "prefix_generation_seconds": prefix["seconds"],
            }
            equivalence_rows.append(row)
            _append_jsonl(equivalence_handle, row)
            print(f"[E01-B prefix proof] {index}/{len(equivalence_ids)} pass={row['equivalence_passed']}", flush=True)
        equivalence_passed = len(equivalence_rows) == len(equivalence_ids) and all(row["equivalence_passed"] for row in equivalence_rows)
        route = "profile_prefix" if equivalence_passed else "full_json_fallback_all_val"
        run_manifest.update({
            "stage": "full_val_generation",
            "prefix_equivalence_passed": equivalence_passed,
            "prefix_equivalence_count": len(equivalence_rows),
            "prefix_equivalence_pass_count": sum(row["equivalence_passed"] for row in equivalence_rows),
            "primary_prediction_method": route,
            "model_load_seconds": model_load_seconds,
            "prompt_facts": prompt_facts,
            "model_facts": model_facts,
            "updated_at_utc": _now(),
        })
        _write_json(eval_root / "manifest.json", run_manifest)

        if route == "profile_prefix":
            for sample_id in equivalence_ids:
                prefix = prefix_cache[sample_id]
                parsed = _profile_result(prefix["result"])
                row = {
                    "id": sample_id,
                    **parsed,
                    "generated_tokens": prefix["generated_tokens"],
                    "stop_reason": prefix["stop_reason"],
                    "generation_seconds": prefix["seconds"],
                    "prompt_tokens": full_cache[sample_id]["prompt_tokens"],
                    "fallback": False,
                }
                _append_jsonl(profile_handle, row)
                prefix_cache[sample_id]["prediction_row"] = row
            for index, sample_id in enumerate(input_by_id, 1):
                if sample_id in prefix_cache:
                    continue
                sample = input_by_id[sample_id]
                selection, _messages = inference.prepare_sample_messages(
                    sample, prompt_text, tokenizer, int(config["generation"]["max_prompt_tokens"]),
                )
                prompt_ids = list(selection["prompt_ids"])
                if selection.get("floor_exceeds_cap"):
                    prefix_fallback_count += 1
                    prediction = {"emotion_label": "", "user_profile": {dimension: [] for dimension in PROFILE_DIMS}}
                    row = {
                        "id": sample_id,
                        "status": "prompt_floor_exceeds_cap",
                        "reason": "prompt_floor_exceeds_cap",
                        **prediction,
                        "generated_tokens": 0,
                        "stop_reason": "not_run",
                        "generation_seconds": 0.0,
                        "prompt_tokens": int(selection.get("prompt_tokens", 0)),
                        "fallback": False,
                    }
                    _append_jsonl(profile_handle, row)
                    continue
                prefix = _generate_prefix(model, tokenizer, prompt_ids, generation, enums)
                prefix_generation_times.append(prefix["seconds"])
                if prefix["result"].valid:
                    parsed = _profile_result(prefix["result"])
                    row = {
                        "id": sample_id,
                        **parsed,
                        "generated_tokens": prefix["generated_tokens"],
                        "stop_reason": prefix["stop_reason"],
                        "generation_seconds": prefix["seconds"],
                        "prompt_tokens": len(prompt_ids),
                        "fallback": False,
                    }
                else:
                    prefix_fallback_count += 1
                    full_ids, full_text, full_stop, full_seconds = _invoke(model, tokenizer, prompt_ids, generation)
                    full_generation_times.append(full_seconds)
                    full_cache[sample_id] = {
                        "token_ids": full_ids,
                        "text": full_text,
                        "stop_reason": full_stop,
                        "seconds": full_seconds,
                        "prompt_ids": prompt_ids,
                        "prompt_tokens": len(prompt_ids),
                    }
                    normalized, parse_failed, invalid_label = _parse_full(sample_id, full_text, enums)
                    row = {
                        "id": sample_id,
                        "status": "full_fallback_parse_failure" if parse_failed else "full_fallback_invalid_label" if invalid_label else "full_fallback_valid",
                        "reason": prefix["result"].reason,
                        "emotion_label": normalized["emotion_label"],
                        "user_profile": {dimension: list(normalized["user_profile"][dimension]) for dimension in PROFILE_DIMS},
                        "generated_tokens": len(full_ids),
                        "stop_reason": full_stop,
                        "generation_seconds": full_seconds,
                        "prefix_generation_seconds": prefix["seconds"],
                        "prompt_tokens": len(prompt_ids),
                        "fallback": True,
                    }
                    fallback_raw_rows.append({
                        "id": sample_id,
                        "raw_text": full_text,
                        "status": row["status"],
                        "generated_tokens": len(full_ids),
                        "stop_reason": full_stop,
                        "generation_seconds": full_seconds,
                        "prompt_tokens": len(prompt_ids),
                    })
                prefix_cache[sample_id] = {"prediction_row": row, **prefix}
                _append_jsonl(profile_handle, row)
                if index % 100 == 0:
                    print(f"[E01-B Val prefix] {index}/{len(input_by_id)}", flush=True)
        else:
            # A failed 32-row equivalence gate disables prefix scoring for the whole Val.
            for sample_id in input_by_id:
                if sample_id not in full_cache:
                    sample = input_by_id[sample_id]
                    selection, _messages = inference.prepare_sample_messages(
                        sample, prompt_text, tokenizer, int(config["generation"]["max_prompt_tokens"]),
                    )
                    if selection.get("floor_exceeds_cap"):
                        cache = {"token_ids": [], "text": "", "stop_reason": "prompt_floor_exceeds_cap", "seconds": 0.0, "prompt_ids": [], "prompt_tokens": int(selection.get("prompt_tokens", 0))}
                    else:
                        token_ids, text, stop_reason, seconds = _invoke(model, tokenizer, list(selection["prompt_ids"]), generation)
                        cache = {"token_ids": token_ids, "text": text, "stop_reason": stop_reason, "seconds": seconds, "prompt_ids": list(selection["prompt_ids"]), "prompt_tokens": len(selection["prompt_ids"])}
                        full_generation_times.append(seconds)
                    full_cache[sample_id] = cache
                cache = full_cache[sample_id]
                normalized, parse_failed, invalid_label = _parse_full(sample_id, cache["text"], enums)
                row = {
                    "id": sample_id,
                    "status": "full_json_parse_failure" if parse_failed else "full_json_invalid_label" if invalid_label else "full_json_valid",
                    "reason": None,
                    "emotion_label": normalized["emotion_label"],
                    "user_profile": {dimension: list(normalized["user_profile"][dimension]) for dimension in PROFILE_DIMS},
                    "generated_tokens": len(cache["token_ids"]),
                    "stop_reason": cache["stop_reason"],
                    "generation_seconds": cache["seconds"],
                    "prompt_tokens": cache["prompt_tokens"],
                    "fallback": True,
                }
                prefix_cache[sample_id] = {"prediction_row": row}
                _append_jsonl(profile_handle, row)
                fallback_raw_rows.append({
                    "id": sample_id,
                    "raw_text": cache["text"],
                    "status": row["status"],
                    "generated_tokens": len(cache["token_ids"]),
                    "stop_reason": cache["stop_reason"],
                    "generation_seconds": cache["seconds"],
                    "prompt_tokens": cache["prompt_tokens"],
                })
                if len(prefix_cache) % 50 == 0:
                    print(f"[E01-B Val full fallback] {len(prefix_cache)}/{len(input_by_id)}", flush=True)

        with full_temp.open("x", encoding="utf-8", newline="\n") as full_handle:
            for index, sample_id in enumerate(full_json_ids, 1):
                sample = input_by_id[sample_id]
                cache = full_cache.get(sample_id)
                reused = cache is not None
                if cache is None:
                    selection, _messages = inference.prepare_sample_messages(
                        sample, prompt_text, tokenizer, int(config["generation"]["max_prompt_tokens"]),
                    )
                    if selection.get("floor_exceeds_cap"):
                        cache = {"token_ids": [], "text": "", "stop_reason": "prompt_floor_exceeds_cap", "seconds": 0.0, "prompt_ids": list(selection.get("prompt_ids", [])), "prompt_tokens": int(selection.get("prompt_tokens", 0))}
                    else:
                        token_ids, text, stop_reason, seconds = _invoke(model, tokenizer, list(selection["prompt_ids"]), generation)
                        cache = {"token_ids": token_ids, "text": text, "stop_reason": stop_reason, "seconds": seconds, "prompt_ids": list(selection["prompt_ids"]), "prompt_tokens": len(selection["prompt_ids"])}
                        full_json_generation_times.append(seconds)
                    full_cache[sample_id] = cache
                normalized, parse_failed, invalid_label = _parse_full(sample_id, cache["text"], enums)
                record = {
                    "id": sample_id,
                    "raw_text": cache["text"],
                    "normalized_prediction": normalized,
                    "parse_failed": parse_failed,
                    "invalid_label": invalid_label,
                    "generated_tokens": len(cache["token_ids"]),
                    "stop_reason": cache["stop_reason"],
                    "generation_seconds": cache["seconds"],
                    "prompt_tokens": cache["prompt_tokens"],
                    "reused_from_prior_full_generation": reused,
                }
                _append_jsonl(full_handle, record)
                if index % 40 == 0:
                    print(f"[E01-B full JSON 160] {index}/{len(full_json_ids)}", flush=True)
        equivalence_handle.close()
        profile_handle.close()
        for temporary, final in (
            (equivalence_path, eval_root / "prefix_equivalence.jsonl"),
            (profile_temp, eval_root / "profile_predictions.jsonl"),
            (full_temp, eval_root / "full_json_predictions.jsonl"),
        ):
            os.replace(temporary, final)
        if fallback_raw_rows:
            with (eval_root / "fallback_full_outputs.jsonl").open("x", encoding="utf-8", newline="\n") as fallback_handle:
                for row in fallback_raw_rows:
                    _append_jsonl(fallback_handle, row)

        profile_rows = _read_jsonl(eval_root / "profile_predictions.jsonl")
        full_rows = _read_jsonl(eval_root / "full_json_predictions.jsonl")
        if len(profile_rows) != 2354 or set(_unique_by(profile_rows, "id", "E01-B Val profile predictions")) != set(input_by_id):
            raise RuntimeError("E01-B Val profile predictions do not cover all 2,354 target-free IDs")
        if len(full_rows) != 160 or len(_unique_by(full_rows, "id", "fixed full JSON cohort")) != 160:
            raise RuntimeError("fixed full JSON generation did not cover exactly 160 unique IDs")
        inference.synchronize(torch)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        generation_manifest = {
            "version": "E01-B-Final-Val-generation-v1",
            "run_id": run_id,
            "status": "predictions_frozen",
            "created_at_utc": _now(),
            "candidate_run_manifest_sha256": sha256_file(training_root / "manifest.json"),
            "candidate_adapter_weight_sha256": adapter_identity["weight_sha256"],
            "base_weight_sha256": BASE_SHA256,
            "target_free_input_sha256": sha256_file(target_free_path),
            "target_free_id_map_sha256": sha256_file(id_map_path),
            "prediction_method": route,
            "prefix_equivalence_count": len(equivalence_rows),
            "prefix_equivalence_pass_count": sum(row["equivalence_passed"] for row in equivalence_rows),
            "prefix_equivalence_sha256": sha256_file(eval_root / "prefix_equivalence.jsonl"),
            "profile_prediction_count": len(profile_rows),
            "profile_prediction_sha256": sha256_file(eval_root / "profile_predictions.jsonl"),
            "prefix_or_full_fallback_count": prefix_fallback_count if route == "profile_prefix" else 2354,
            "fallback_full_outputs_sha256": sha256_file(eval_root / "fallback_full_outputs.jsonl") if (eval_root / "fallback_full_outputs.jsonl").exists() else None,
            "full_json_count": len(full_rows),
            "full_json_ids_sha256": hashlib.sha256("\n".join(full_json_ids).encode("utf-8")).hexdigest(),
            "full_json_prediction_sha256": sha256_file(eval_root / "full_json_predictions.jsonl"),
            "model_load_seconds": model_load_seconds,
            "prefix_generation_seconds": {
                "count": len(prefix_generation_times),
                "total": sum(prefix_generation_times),
                "mean": sum(prefix_generation_times) / len(prefix_generation_times) if prefix_generation_times else 0.0,
            },
            "full_generation_seconds": {
                "count": len(full_generation_times) + len(full_json_generation_times),
                "total": sum(full_generation_times) + sum(full_json_generation_times),
                "mean": (sum(full_generation_times) + sum(full_json_generation_times)) / (len(full_generation_times) + len(full_json_generation_times)) if full_generation_times or full_json_generation_times else 0.0,
            },
            "official_val_gold_read": False,
            "references_read": False,
        }
        _write_json(eval_root / "generation_manifest.json", generation_manifest, exclusive=True)
        run_manifest.update({
            "status": "predictions_frozen",
            "stage": "generation_complete_awaiting_scoring",
            "generation_manifest_sha256": sha256_file(eval_root / "generation_manifest.json"),
            "prediction_method": route,
            "official_val_gold_read": False,
            "updated_at_utc": _now(),
        })
        _write_json(eval_root / "manifest.json", run_manifest)
        return eval_root, generation_manifest
    except BaseException as exc:
        if not equivalence_handle.closed:
            equivalence_handle.close()
        if not profile_handle.closed:
            profile_handle.close()
        run_manifest.update({
            "status": "generation_failed",
            "stage": "generation_failed",
            "stop_reason": type(exc).__name__,
            "error_summary": str(exc)[:1200],
            "completed_equivalence_rows": len(equivalence_rows),
            "updated_at_utc": _now(),
        })
        _write_json(eval_root / "manifest.json", run_manifest)
        raise


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _score(config: dict[str, Any], eval_root: Path) -> dict[str, Any]:
    generation_manifest = _read_json(eval_root / "generation_manifest.json")
    run_manifest = _read_json(eval_root / "manifest.json")
    if generation_manifest.get("status") != "predictions_frozen" or run_manifest.get("status") != "predictions_frozen":
        raise ValueError("scoring requires a frozen complete prediction set")
    for filename, expected in (
        ("prefix_equivalence.jsonl", generation_manifest["prefix_equivalence_sha256"]),
        ("profile_predictions.jsonl", generation_manifest["profile_prediction_sha256"]),
        ("full_json_predictions.jsonl", generation_manifest["full_json_prediction_sha256"]),
    ):
        if sha256_file(eval_root / filename) != expected:
            raise ValueError(f"frozen prediction artifact changed before scoring: {filename}")

    a8_eval_root = _resolve(config["sources"]["a8_evaluation_root"])
    a8_scoring_manifest = _read_json(a8_eval_root / "scoring_manifest.json")
    a8_generation_manifest = _read_json(a8_eval_root / "generation_manifest.json")
    a8_predictions = _read_jsonl(a8_eval_root / "predictions.jsonl")
    if (
        a8_scoring_manifest.get("status") != "scored_after_prediction_freeze"
        or a8_generation_manifest.get("status") != "predictions_frozen"
        or sha256_file(a8_eval_root / "predictions.jsonl") != a8_scoring_manifest.get("prediction_sha256")
    ):
        raise ValueError("A8 P4504 predictions are not frozen and scored under their recorded identity")

    target_free_path = _resolve(config["sources"]["target_free_input"])
    id_map_path = _resolve(config["sources"]["id_map"])
    target_free_rows = _read_jsonl(target_free_path)
    id_map_rows = _read_jsonl(id_map_path)
    target_free = _unique_by(target_free_rows, "id", "target-free Val")
    id_map = _unique_by(id_map_rows, "eval_id", "Val ID map")
    if set(target_free) != set(id_map) or len(target_free) != 2354:
        raise ValueError("target-free Val / map are not the fixed 2,354-row cohort")

    # Gold is opened only after the E01-B and A8 prediction files have frozen hashes.
    d03_path = _resolve(config["sources"]["d03_manifest"])
    d03_manifest = _read_json(d03_path)
    if sha256_file(d03_path) != _read_json(_resolve(config["sources"]["a8_evaluation_root"]) / "val_input_manifest.json")["source"]["d03_manifest_sha256"]:
        raise ValueError("frozen A8 input source D0.3 Manifest hash changed")
    normalized_val_path = _resolve(config["sources"]["normalized_val"])
    val_fact = next((row for row in d03_manifest.get("outputs", []) if row.get("path") == "data/processed/v1/val.jsonl"), None)
    if not val_fact or val_fact.get("records") != 2354 or sha256_file(normalized_val_path) != val_fact.get("sha256"):
        raise ValueError("normalized Val Gold differs from the locked D0.3 Manifest")
    gold_rows = _read_jsonl(normalized_val_path)
    gold_by_source_id = _unique_by(gold_rows, "sample_id", "D0.3 Val Gold")
    if len(gold_by_source_id) != 2354:
        raise ValueError("D0.3 Val Gold has an unexpected number of unique rows")
    gold_by_eval_id: dict[str, dict[str, Any]] = {}
    conversation_by_id: dict[str, str] = {}
    for eval_id, mapping in id_map.items():
        sample_id = mapping.get("sample_id")
        gold = gold_by_source_id.get(sample_id)
        if gold is None or gold.get("conversation_id") != mapping.get("conversation_id"):
            raise ValueError("Val ID map does not join exactly to D0.3 Gold by sample/conversation ID")
        if target_free[eval_id].get("id") != eval_id:
            raise ValueError("target-free inference ID changed after prediction freeze")
        gold_by_eval_id[eval_id] = gold
        conversation_by_id[eval_id] = str(mapping["conversation_id"])

    schema = load_submission_schema(ROOT)
    enums = schema_enums(schema)
    profile_rows = _read_jsonl(eval_root / "profile_predictions.jsonl")
    profile_by_id = _unique_by(profile_rows, "id", "E01-B profile predictions")
    a8_by_id = _unique_by(a8_predictions, "id", "A8 P4504 predictions")
    if set(profile_by_id) != set(target_free) or set(a8_by_id) != set(target_free):
        raise ValueError("E01-B/A8 predictions are not paired on the same 2,354 Val IDs")

    def clean_profile(row: Mapping[str, Any], key: str = "user_profile") -> dict[str, list[str]]:
        value = row.get(key)
        result = {}
        for dimension in PROFILE_DIMS:
            labels = value.get(dimension) if isinstance(value, Mapping) else None
            allowed = set(enums[dimension])
            valid = (
                isinstance(labels, list)
                and all(isinstance(label, str) for label in labels)
                and len(labels) == len(set(labels))
                and set(labels) <= allowed
            )
            result[dimension] = list(labels) if valid else []
        return result

    gold_profiles: dict[str, dict[str, list[str]]] = {}
    e01_profiles: dict[str, dict[str, list[str]]] = {}
    a8_profiles: dict[str, dict[str, list[str]]] = {}
    e01_emotions: dict[str, str] = {}
    gold_emotions: dict[str, str] = {}
    for eval_id in target_free:
        gold = gold_by_eval_id[eval_id]["target"]
        gold_profiles[eval_id] = {dim: list(gold["user_profile"][dim]) for dim in PROFILE_DIMS}
        gold_emotions[eval_id] = str(gold["emotion_label"])
        e01_profiles[eval_id] = clean_profile(profile_by_id[eval_id])
        e01_emotions[eval_id] = str(profile_by_id[eval_id].get("emotion_label", ""))
        a8_row = a8_by_id[eval_id]
        parsed = a8_row.get("parsed") if a8_row.get("status") == "valid" else None
        a8_profiles[eval_id] = clean_profile(parsed if isinstance(parsed, Mapping) else {})

    ordered_ids = [row["id"] for row in target_free_rows]
    gold_profile_list = [gold_profiles[value] for value in ordered_ids]
    e01_profile_list = [e01_profiles[value] for value in ordered_ids]
    a8_profile_list = [a8_profiles[value] for value in ordered_ids]
    e01_profile_metrics = profile_metrics(gold_profile_list, e01_profile_list, enums)
    a8_profile_metrics = profile_metrics(gold_profile_list, a8_profile_list, enums)
    e01_emotion_metrics = emotion_metrics(
        [gold_emotions[value] for value in ordered_ids],
        [e01_emotions[value] for value in ordered_ids],
        enums["emotion_label"],
    )
    historical_scoring = _read_json(a8_eval_root / "scoring.json")
    historical_a8_profile = historical_scoring["metrics"]["full_val_2354"]["profile_33_label"]
    for key in (
        "overall_33_label_micro_f1",
        "overall_33_label_micro_precision",
        "overall_33_label_micro_recall",
        "overall_33_label_tp",
        "overall_33_label_fp",
        "overall_33_label_fn",
    ):
        if a8_profile_metrics[key] != historical_a8_profile[key]:
            raise ValueError(f"A8 P4504 unified scoring recheck differs from its frozen score: {key}")

    transitions = profile_hit_transitions(gold_profiles, e01_profiles, a8_profiles)
    bootstrap = paired_conversation_bootstrap(
        gold_profiles,
        e01_profiles,
        a8_profiles,
        conversation_by_id,
        enums,
        replicates=int(config["scoring"]["bootstrap_replicates"]),
        seed=int(config["scoring"]["bootstrap_seed"]),
    )

    full_rows = _read_jsonl(eval_root / "full_json_predictions.jsonl")
    full_by_id = _unique_by(full_rows, "id", "fixed full JSON predictions")
    fixed_ids = _read_json(eval_root / "fixed_full_json_ids.json")["ids"]
    if len(fixed_ids) != 160 or set(fixed_ids) != set(full_by_id):
        raise ValueError("fixed full JSON IDs and predictions do not match the 160-row cohort")
    validator = Draft202012Validator(schema)
    normalized_rows = [full_by_id[sample_id]["normalized_prediction"] for sample_id in fixed_ids]
    full_profile_gold = [gold_profiles[sample_id] for sample_id in fixed_ids]
    full_profile_pred = [clean_profile(full_by_id[sample_id]["normalized_prediction"]) for sample_id in fixed_ids]
    full_emotion_gold = [gold_emotions[sample_id] for sample_id in fixed_ids]
    full_emotion_pred = [str(full_by_id[sample_id]["normalized_prediction"].get("emotion_label", "")) for sample_id in fixed_ids]
    full_targets = [gold_by_eval_id[sample_id]["target"] for sample_id in fixed_ids]
    scoring_config = _config(_resolve(config["scoring"]["response_metrics_config"]))["scoring"]
    response_result = response_metrics(
        [target["response_text"] for target in full_targets],
        [str(row.get("response_text", "")) for row in normalized_rows],
        scoring_config,
        workspace_root=ROOT,
    )
    submission_validity = [not list(validator.iter_errors(row)) for row in normalized_rows]
    raw_records = []
    per_sample_latencies = []
    generation_tokens = []
    for sample_id in fixed_ids:
        row = full_by_id[sample_id]
        raw_records.append({
            "id": sample_id,
            "raw_text": row.get("raw_text"),
            "status": "parse_failure" if row.get("parse_failed") else "invalid_label" if row.get("invalid_label") else "success",
            "generated_tokens": row.get("generated_tokens"),
            "stop_reason": row.get("stop_reason"),
        })
        per_sample_latencies.append(float(row.get("generation_seconds", 0.0)))
        generation_tokens.append(int(row.get("generated_tokens", 0)))
    full_raw_metrics = raw_diagnostics(
        raw_records,
        enums,
        normalized_rows,
        submission_validity,
        generation_cap=1200,
    )
    full_json_result = {
        "sample_count": len(fixed_ids),
        "fixed_ids_sha256": _read_json(eval_root / "fixed_full_json_ids.json")["ids_sha256"],
        "strict_json_and_raw_a0": full_raw_metrics,
        "profile_metrics": profile_metrics(full_profile_gold, full_profile_pred, enums),
        "emotion_metrics": emotion_metrics(full_emotion_gold, full_emotion_pred, enums["emotion_label"]),
        "response_metrics": response_result,
        "latency_seconds": {
            "mean": sum(per_sample_latencies) / len(per_sample_latencies),
            "p50": _percentile(per_sample_latencies, 0.50),
            "p95": _percentile(per_sample_latencies, 0.95),
            "total": sum(per_sample_latencies),
        },
        "generated_tokens": {
            "mean": sum(generation_tokens) / len(generation_tokens),
            "total": sum(generation_tokens),
            "max": max(generation_tokens, default=0),
        },
        "metrics_scope": "local_proxy",
    }
    a8_prediction_rows = list(a8_by_id.values())
    a8_seconds = [float(row.get("generation_seconds", 0.0)) for row in a8_prediction_rows]
    a8_tokens = [int(row.get("generated_token_count", 0)) for row in a8_prediction_rows]
    generation_info = {
        "e01_b": generation_manifest,
        "a8_p4504_native_generation": {
            "prediction_count": len(a8_seconds),
            "generation_seconds_total": sum(a8_seconds),
            "generation_seconds_mean": sum(a8_seconds) / len(a8_seconds),
            "generated_tokens_total": sum(a8_tokens),
            "generated_tokens_mean": sum(a8_tokens) / len(a8_tokens),
            "comparison_note": "different native Prompt/task; timings are descriptive, not a controlled latency benchmark",
        },
    }
    result = {
        "version": "E01-B-Final-Val-score-v1",
        "run_id": run_manifest["run_id"],
        "status": "completed_local_proxy",
        "created_at_utc": _now(),
        "prediction_generation_sha256": sha256_file(eval_root / "generation_manifest.json"),
        "references_read_after_prediction_freeze": True,
        "d03_manifest_sha256": sha256_file(d03_path),
        "normalized_val_sha256": sha256_file(normalized_val_path),
        "val_count": 2354,
        "val_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest(),
        "metrics_scope": "local_proxy; same fixed D0.3 Val IDs, not official Test performance",
        "task_protocol_note": "E01-B uses P0/F2/T1 and predicts emotion plus complete JSON; A8 P4504 uses its native three-line profile prompt. This compares complete candidate systems, not isolated weights.",
        "history_exposure_note": "A8 used D0.3 Dev for training; 209 Val points were previously explored by A2, and remaining points can share conversation context.",
        "e01_b_profile_2354": e01_profile_metrics,
        "a8_p4504_profile_2354": a8_profile_metrics,
        "a8_historical_score_recheck": {
            "status": "passed",
            "frozen_historical_micro_f1": historical_a8_profile["overall_33_label_micro_f1"],
            "recomputed_micro_f1": a8_profile_metrics["overall_33_label_micro_f1"],
            "frozen_prediction_sha256": sha256_file(a8_eval_root / "predictions.jsonl"),
        },
        "e01_b_emotion_2354": e01_emotion_metrics,
        "paired_profile_hit_transitions": transitions,
        "conversation_cluster_bootstrap": bootstrap,
        "full_json_160": full_json_result,
        "generation_cost": generation_info,
        "model_selection": "both E01-B Train+Dev 4504 and A8 P4504 are retained for user selection; no single-metric automatic deletion",
    }
    score_path = eval_root / "comparison_report.json"
    _write_json(score_path, result, exclusive=True)
    score_manifest = {
        "version": "E01-B-Final-Val-score-manifest-v1",
        "run_id": run_manifest["run_id"],
        "status": "scored_after_prediction_freeze",
        "generation_manifest_sha256": sha256_file(eval_root / "generation_manifest.json"),
        "profile_prediction_sha256": sha256_file(eval_root / "profile_predictions.jsonl"),
        "full_json_prediction_sha256": sha256_file(eval_root / "full_json_predictions.jsonl"),
        "a8_prediction_sha256": sha256_file(a8_eval_root / "predictions.jsonl"),
        "normalized_val_gold_sha256": sha256_file(normalized_val_path),
        "d03_manifest_sha256": sha256_file(d03_path),
        "scoring_sha256": sha256_file(score_path),
        "references_read_after_prediction_freeze": True,
        "created_at_utc": _now(),
    }
    _write_json(eval_root / "scoring_manifest.json", score_manifest, exclusive=True)
    run_manifest.update({
        "status": "completed",
        "stage": "same_val_comparison_scored",
        "scoring_manifest_sha256": sha256_file(eval_root / "scoring_manifest.json"),
        "comparison_report_sha256": sha256_file(score_path),
        "official_val_gold_read": True,
        "updated_at_utc": _now(),
    })
    _write_json(eval_root / "manifest.json", run_manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--phase", choices=("run", "generate", "score"), default="run")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _config(config_path)
    run_id = args.run_id or config["run_id"]
    if args.phase == "score":
        eval_root = _resolve(config["output_root"]) / run_id
        result = _score(config, eval_root)
    else:
        eval_root, generation_manifest = _generate(config_path, config, run_id)
        result = generation_manifest if args.phase == "generate" else _score(config, eval_root)
    print(json.dumps({"run_root": str(eval_root.relative_to(ROOT)), "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
