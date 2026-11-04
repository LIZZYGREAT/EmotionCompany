#!/usr/bin/env python3
"""CPU audit and isolated, budgeted Greedy comparison for profile-only output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import read_jsonl
from src.evaluation.metrics import profile_metrics
from src.evaluation.profile_fast_eval import (
    compare_generated_prefix,
    profile_nonempty_recall,
    select_unique_conversation_samples,
)
from src.evaluation.profile_prefix import ProfilePrefixStopper, parse_profile_prefix
from src.evaluation.source_integrity import (
    audit_prediction_point_sources,
    build_source_manifest,
    unique_index,
    validate_target_free_inputs,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _read_selected_jsonl_lines(path: Path, selected_lines: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number not in selected_lines:
                continue
            if not line.strip():
                raise ValueError(f"selected source line {line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"selected source line {line_number} is not an object")
            rows[line_number] = value
    missing = selected_lines - set(rows)
    if missing:
        raise ValueError("one or more selected source lines are missing")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "E03-profile-fast-v1":
        raise ValueError("unsupported E03 profile fast config")
    if value.get("protocol", {}).get("format_version") != "F2" or value.get("protocol", {}).get("truncation_version") != "T1":
        raise ValueError("profile fast evaluation must use the locked F2/T1 protocol")
    if value.get("generation", {}).get("do_sample") is not False:
        raise ValueError("profile fast evaluation requires Greedy generation")
    budget = int(value.get("gpu_budget", {}).get("max_seconds", 0))
    if not 1 <= budget <= 900:
        raise ValueError("GPU budget must be between 1 and 900 seconds")
    return value


def _load_enums() -> dict[str, list[str]]:
    from src.evaluation import load_submission_schema, schema_enums

    return schema_enums(load_submission_schema(ROOT))


def _output_path(config: dict[str, Any], run_id: str, override: str | None) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
        raise ValueError("run-id must contain only letters, digits, dot, underscore, and dash")
    output_root = _resolve(override or config["output_root"]) / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root / "profile_fast_metrics.json"


def parse_existing_raw(
    config: dict[str, Any],
    raw_path: Path,
    output_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    enums = _load_enums()
    rows = _read_jsonl(raw_path)
    raw_index = unique_index(rows, "id", "Raw predictions")
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in raw_index.values():
        raw_text = row.get("raw_text")
        if not isinstance(raw_text, str):
            counts["invalid"] += 1
            reasons["missing_raw_text"] += 1
            continue
        result = parse_profile_prefix(raw_text, enums)
        counts[result.status] += 1
        reasons[result.reason] += 1
    manifest_path = _resolve(config["sources"]["dataset_manifest_path"])
    source = {
        "raw_path": str(raw_path),
        "raw_sha256": _sha256_file(raw_path),
        "raw_rows": len(rows),
        "unique_raw_ids": len(raw_index),
        "source_identity_path": str(manifest_path),
        "source_identity_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else None,
        "parser_source_sha256": _sha256_file(Path(__file__).resolve()),
        "fast_eval_config_sha256": _sha256_file(config_path),
    }
    report = {
        "status": "cpu_raw_parse_completed",
        "created_at_utc": _now(),
        "evaluation_scope": "prefix_parse_only_no_early_stop_equivalence_claim",
        "source_manifest": source,
        "sample_count": len(rows),
        "parse_status_counts": {key: counts.get(key, 0) for key in ("valid", "invalid", "incomplete", "unsupported_order")},
        "parse_reason_counts": dict(sorted(reasons.items())),
        "valid_parse_rate": counts.get("valid", 0) / len(rows) if rows else 0.0,
        "gpu_used": False,
        "references_read": False,
    }
    _write_json(output_path, report)
    return report


def audit_selected_source_history(
    config: dict[str, Any],
    selected_inputs: list[dict[str, Any]],
    selected_id_maps: list[dict[str, Any]],
) -> dict[str, Any]:
    source_cfg = config["sources"]
    normalized_path = _resolve(source_cfg["normalized_dev_path"])
    raw_source_path = _resolve(source_cfg["raw_source_path"])
    normalized_rows = _read_jsonl(normalized_path)
    normalized_index = unique_index(normalized_rows, "sample_id", "normalized Dev")
    id_map = unique_index(selected_id_maps, "eval_id", "selected ID map")
    selected_sample_ids: set[str] = set()
    for row in selected_inputs:
        mapping = id_map.get(row["id"])
        sample_id = mapping.get("sample_id") if mapping else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("selected ID map has a missing sample ID")
        selected_sample_ids.add(sample_id)
    if len(selected_sample_ids) != len(selected_inputs):
        raise ValueError("selected ID map has duplicate sample mappings")
    try:
        selected_normalized = [normalized_index[sample_id] for sample_id in sorted(selected_sample_ids)]
    except KeyError as exc:
        raise ValueError("a selected sample ID is absent from normalized Dev") from exc
    normalized_by_id = {str(row["sample_id"]): row for row in selected_normalized}
    for eval_id, mapping in id_map.items():
        source_row = normalized_by_id.get(str(mapping.get("sample_id")))
        if source_row is None or mapping.get("conversation_id") != source_row.get("conversation_id"):
            raise ValueError("selected sample conversation ID differs from normalized Dev")
    selected_source_lines = {
        row.get("source", {}).get("source_line")
        for row in selected_normalized
        if isinstance(row.get("source"), dict)
    }
    if any(isinstance(line, bool) or not isinstance(line, int) or line < 1 for line in selected_source_lines):
        raise ValueError("selected normalized Dev row has an invalid source line")
    raw_source_rows = _read_selected_jsonl_lines(raw_source_path, selected_source_lines)
    history_audit = audit_prediction_point_sources(selected_normalized, raw_source_rows)
    dataset_manifest = _resolve(source_cfg["dataset_manifest_path"])
    if not dataset_manifest.is_file():
        raise FileNotFoundError(f"D0.3 dataset manifest is missing: {dataset_manifest}")
    dataset_manifest_value = _read_json(dataset_manifest)
    if dataset_manifest_value.get("pipeline_version") != "D0.3-v1":
        raise ValueError("unsupported D0.3 data pipeline version")
    train_fact = dataset_manifest_value.get("inputs", {}).get("train", {})
    dev_outputs = [
        row for row in dataset_manifest_value.get("outputs", [])
        if row.get("path") == "data/processed/v1/dev.jsonl"
    ]
    raw_sha = _sha256_file(raw_source_path)
    normalized_sha = _sha256_file(normalized_path)
    if train_fact.get("sha256_after") != raw_sha or train_fact.get("sha256_before") != raw_sha:
        raise ValueError("raw Train SHA differs from the D0.3 source Manifest")
    if (
        len(dev_outputs) != 1
        or dev_outputs[0].get("sha256") != normalized_sha
        or dev_outputs[0].get("records") != len(normalized_rows)
    ):
        raise ValueError("normalized Dev SHA/count differs from the D0.3 output Manifest")
    return {
        **history_audit,
        "normalized_dev_path": str(normalized_path),
        "normalized_dev_sha256": normalized_sha,
        "raw_source_path": str(raw_source_path),
        "raw_source_sha256": raw_sha,
        "dataset_manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest),
    }


def _enrich_id_map_conversations(
    config: dict[str, Any],
    id_map_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows = _read_jsonl(_resolve(config["sources"]["normalized_dev_path"]))
    normalized_index = unique_index(normalized_rows, "sample_id", "normalized Dev")
    enriched: list[dict[str, Any]] = []
    for row in id_map_rows:
        sample_id = row.get("sample_id")
        normalized = normalized_index.get(sample_id) if isinstance(sample_id, str) else None
        if normalized is None:
            raise ValueError("ID map sample ID is absent from normalized Dev")
        conversation_id = normalized.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("normalized Dev row has no conversation_id")
        enriched.append({**row, "conversation_id": conversation_id})
    return enriched


def run_source_audit(
    config: dict[str, Any],
    output_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    input_rows = _read_jsonl(_resolve(config["sources"]["input_path"]))
    id_map_rows = _enrich_id_map_conversations(
        config, _read_jsonl(_resolve(config["sources"]["id_map_path"]))
    )
    selected = select_unique_conversation_samples(
        input_rows, id_map_rows,
        count=int(config["selection"]["count"]),
        seed=int(config["selection"]["seed"]),
    )
    input_index = validate_target_free_inputs(input_rows)
    map_index = unique_index(id_map_rows, "eval_id", "ID map")
    selected_maps = [dict(map_index[row["id"]]) for row in selected]
    try:
        history_audit = audit_selected_source_history(config, selected, selected_maps)
    except (KeyError, OSError, ValueError) as exc:
        report = {
            "status": "preflight_failed",
            "created_at_utc": _now(),
            "reason": str(exc),
            "gpu_used": False,
            "generation_started": False,
        }
        _write_json(output_path, report)
        return report
    selected_rows = [input_index[row["id"]] for row in selected]
    source_manifest = build_source_manifest(
        source_path=history_audit["raw_source_path"],
        source_sha256=history_audit["raw_source_sha256"],
        input_rows=selected_rows,
        id_map_rows=selected_maps,
        prompt_version=config["protocol"]["prompt_version"],
        prompt_sha256=config["protocol"]["prompt_sha256"],
        format_version=config["protocol"]["format_version"],
        truncation_version=config["protocol"]["truncation_version"],
        exporter_source_sha256=_sha256_file(Path(__file__).resolve()),
        selection_method=config["selection"]["method"],
        selection_seed=int(config["selection"]["seed"]),
    )
    source_manifest["history_audit"] = history_audit
    source_manifest["normalized_dev_path"] = history_audit["normalized_dev_path"]
    source_manifest["normalized_dev_sha256"] = history_audit["normalized_dev_sha256"]
    source_manifest["generation_input_path"] = str(_resolve(config["sources"]["input_path"]))
    source_manifest["generation_input_sha256"] = _sha256_file(_resolve(config["sources"]["input_path"]))
    source_manifest["id_map_path"] = str(_resolve(config["sources"]["id_map_path"]))
    source_manifest["id_map_sha256"] = _sha256_file(_resolve(config["sources"]["id_map_path"]))
    report = {
        "status": "source_preflight_passed",
        "created_at_utc": _now(),
        "fast_eval_config_sha256": _sha256_file(config_path),
        "source_manifest": source_manifest,
        "selected_points": len(selected),
        "selected_conversations": len(selected_maps),
        "gpu_used": False,
        "generation_started": False,
    }
    _write_json(output_path, report)
    return report


class _BudgetCriterion:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.exhausted = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        if time.monotonic() >= self.deadline:
            self.exhausted = True
            return True
        return False


def _load_model(config: dict[str, Any], runner: Any, torch: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    local_model = runner.model_path(config)
    runner.require_cuda_for_quantization(config, torch)
    trust_remote_code = bool(config.get("trust_remote_code", False))
    model_config = AutoConfig.from_pretrained(
        local_model, trust_remote_code=trust_remote_code, local_files_only=True
    )
    model_loader = (
        AutoModelForImageTextToText
        if model_config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        else AutoModelForCausalLM
    )
    tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(config, local_model)
    weight_identity = runner._weight_identity(local_model, config)
    expected_base_sha = config.get("expected_weight_sha256")
    if expected_base_sha and weight_identity.get("observed_weight_hash") != expected_base_sha:
        raise ValueError("Base weight identity differs from the frozen inference config")
    adapter_config = runner.validate_adapter_config(config.get("adapter"), "transformers")
    adapter_identity: dict[str, Any] = {"enabled": False}
    adapter_path: Path | None = None
    if adapter_config["enabled"]:
        if weight_identity.get("identity_status") != "observed":
            raise ValueError("Base weight identity is not observed")
        if weight_identity.get("observed_weight_hash") != adapter_config["expected_base_weight_sha256"]:
            raise ValueError("Adapter is bound to another Base weight identity")
        adapter_path = Path(adapter_config["path"]).expanduser()
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
        adapter_path = adapter_path.resolve()
        adapter_identity, adapter_metadata = runner.adapter_weight_identity(adapter_path)
        if adapter_identity["weight_sha256"] != adapter_config["expected_adapter_weight_sha256"]:
            raise ValueError("Adapter SHA differs from the frozen inference config")
        if (
            adapter_config.get("expected_adapter_config_sha256")
            and adapter_identity["config_sha256"] != adapter_config["expected_adapter_config_sha256"]
        ):
            raise ValueError("Adapter config SHA differs from the frozen inference config")
        runner.validate_adapter_base_reference(adapter_metadata, config.get("model_id", ""), local_model)

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": True,
        "device_map": config.get("device_map", "auto"),
        "torch_dtype": config.get("torch_dtype", "auto"),
    }
    quantization_config = runner.build_quantization_config(config, torch)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = model_loader.from_pretrained(local_model, **load_kwargs)
    if adapter_path is not None:
        model = runner.load_peft_adapter(
            model, adapter_path, prepare_quantized_base=quantization_config is not None,
            is_trainable=False, local_files_only=True,
        )
    model.eval()
    quantization_facts = runner.inspect_quantization_runtime(model, config, torch)
    if quantization_config is not None and not quantization_facts["real_nf4_verified"]:
        raise ValueError("runtime did not pass NF4/BF16/CUDA/no-offload verification")
    facts = {
        "model_path_name": local_model.name,
        "model_type": model_config.model_type,
        "model_weight_identity_sha256": weight_identity.get("observed_weight_hash"),
        "adapter_weight_sha256": adapter_identity.get("weight_sha256"),
        "adapter_config_sha256": adapter_identity.get("config_sha256"),
        "tokenizer_facts": tokenizer_facts,
        "quantization": quantization_facts,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    return model, tokenizer, model_config, facts


def run_gpu_comparison(
    config: dict[str, Any],
    output_path: Path,
    run_config_path: Path | None = None,
    fast_config_path: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    from participant import run_inference as runner

    input_path = _resolve(config["sources"]["input_path"])
    id_map_path = _resolve(config["sources"]["id_map_path"])
    references_path = _resolve(config["sources"]["references_path"])
    inputs = _read_jsonl(input_path)
    id_maps = _enrich_id_map_conversations(config, _read_jsonl(id_map_path))
    input_index = validate_target_free_inputs(inputs)
    map_index = unique_index(id_maps, "eval_id", "ID map")
    if set(input_index) != set(map_index):
        raise ValueError("input and ID map IDs differ")
    count = int(config["selection"]["count"])
    selected = select_unique_conversation_samples(
        inputs, id_maps, count=count, seed=int(config["selection"]["seed"]),
    )
    selected_ids = {row["id"] for row in selected}
    selected_maps = [dict(map_index[eval_id]) for eval_id in sorted(selected_ids)]
    history_audit = audit_selected_source_history(config, selected, selected_maps)
    selected_input_rows = [input_index[eval_id] for eval_id in sorted(selected_ids)]
    source_manifest = build_source_manifest(
        source_path=history_audit["raw_source_path"],
        source_sha256=history_audit["raw_source_sha256"],
        input_rows=selected_input_rows,
        id_map_rows=selected_maps,
        prompt_version=config["protocol"]["prompt_version"],
        prompt_sha256=config["protocol"]["prompt_sha256"],
        format_version=config["protocol"]["format_version"],
        truncation_version=config["protocol"]["truncation_version"],
        exporter_source_sha256=_sha256_file(Path(__file__).resolve()),
        selection_method=config["selection"]["method"],
        selection_seed=int(config["selection"]["seed"]),
    )
    source_manifest["history_audit"] = history_audit
    source_manifest["normalized_dev_path"] = history_audit["normalized_dev_path"]
    source_manifest["normalized_dev_sha256"] = history_audit["normalized_dev_sha256"]
    source_manifest["generation_input_path"] = str(input_path)
    source_manifest["generation_input_sha256"] = _sha256_file(input_path)
    source_manifest["id_map_path"] = str(id_map_path)
    source_manifest["id_map_sha256"] = _sha256_file(id_map_path)
    inference_path = _resolve(run_config_path or config["sources"]["inference_config_path"])
    inference_config = _read_json(inference_path)
    inference_config["generation"] = {
        **inference_config["generation"],
        **config["generation"],
    }
    if inference_config.get("backend") != "transformers":
        raise ValueError("fast profile comparison requires the frozen Transformers runner config")
    prompt_text, _submission_schema, enums, prompt_facts = runner.load_runtime_protocol(inference_config)
    if prompt_facts.get("sha256") != config["protocol"]["prompt_sha256"]:
        raise ValueError("P0 prompt SHA differs from the fast-eval config")
    if inference_config["generation"].get("do_sample") is not False:
        raise ValueError("the paired run must use Greedy decoding")

    budget_seconds = int(config["gpu_budget"]["max_seconds"])
    started = time.monotonic()
    deadline = started + budget_seconds
    if not torch.cuda.is_available():
        report = {
            "status": "blocked_environment",
            "created_at_utc": _now(),
            "reason": "cuda_unavailable",
            "gpu_used": False,
            "sample_count": count,
            "budget_seconds": budget_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "source_manifest": source_manifest,
        }
        _write_json(output_path, report)
        return report

    model, tokenizer, _model_config, model_facts = _load_model(inference_config, runner, torch)
    load_seconds = time.monotonic() - started
    if time.monotonic() >= deadline:
        report = {
            "status": "budget_exhausted",
            "created_at_utc": _now(),
            "reason": "model_load_exhausted_budget",
            "gpu_used": True,
            "model_load_seconds": load_seconds,
            "budget_seconds": budget_seconds,
            "source_manifest": source_manifest,
        }
        _write_json(output_path, report)
        return report

    class BudgetCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.inner = _BudgetCriterion(deadline)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return self.inner(input_ids, scores, **kwargs)

    full_seconds: list[float] = []
    fast_seconds: list[float] = []
    full_peak_bytes = 0
    fast_peak_bytes = 0
    case_status: Counter[str] = Counter()
    comparisons: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    full_prefix_predictions: dict[str, dict[str, Any]] = {}
    budget_exhausted = False
    generation = dict(inference_config["generation"])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def invoke(prompt_ids: list[int], criteria: Any) -> tuple[list[int], str, bool]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.get_input_embeddings().weight.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stopping_criteria=criteria,
                **generation,
            )
        generated = [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        exhausted = any(getattr(item, "inner", None) and item.inner.exhausted for item in criteria)
        return generated, text, exhausted

    for sample in selected:
        if time.monotonic() >= deadline:
            budget_exhausted = True
            break
        selection, _messages = runner.prepare_sample_messages(
            sample, prompt_text, tokenizer, inference_config["context"]["max_prompt_tokens"]
        )
        if selection.get("floor_exceeds_cap"):
            case_status["invalid_prompt_floor_exceeds_cap"] += 1
            continue
        prompt_ids = list(selection["prompt_ids"])
        runner.synchronize(torch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        full_budget = BudgetCriteria()
        full_ids, full_text, full_exhausted = invoke(prompt_ids, StoppingCriteriaList([full_budget]))
        runner.synchronize(torch)
        full_seconds.append(time.monotonic() - before)
        full_peak_bytes = max(full_peak_bytes, int(torch.cuda.max_memory_allocated()))
        if full_exhausted:
            case_status["budget_exhausted_during_full"] += 1
            budget_exhausted = True
            break
        full_prefix = parse_profile_prefix(full_text, enums)
        if not full_prefix.valid:
            case_status[f"full_{full_prefix.status}"] += 1
        if time.monotonic() >= deadline:
            budget_exhausted = True
            break

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        prefix_stopper = ProfilePrefixStopper(
            lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True),
            enums,
            deadline=deadline,
        )

        class PrefixCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                generated = input_ids[0, len(prompt_ids):].detach().cpu().tolist()
                return prefix_stopper.inspect(generated)

        prefix_criteria = PrefixCriteria()
        fast_ids, _fast_text, fast_exhausted = invoke(prompt_ids, StoppingCriteriaList([prefix_criteria]))
        runner.synchronize(torch)
        fast_seconds.append(time.monotonic() - before)
        fast_peak_bytes = max(fast_peak_bytes, int(torch.cuda.max_memory_allocated()))
        if fast_exhausted or prefix_stopper.stop_reason == "budget_exhausted":
            case_status["budget_exhausted_during_fast"] += 1
            budget_exhausted = True
            break
        stop_reason = (
            "profile_prefix_closed"
            if prefix_stopper.result is not None and prefix_stopper.result.valid
            else "eos"
            if eos_token_id is not None and eos_token_id in fast_ids
            else "max_new_tokens"
            if len(fast_ids) >= int(generation["max_new_tokens"])
            else "unknown"
        )
        parsed_fast = prefix_stopper.result or prefix_stopper.finish(stop_reason, fast_ids)
        if not parsed_fast.valid:
            case_status[f"fast_{parsed_fast.status}:{parsed_fast.reason}"] += 1
            continue
        token_ids_for_prefix = prefix_stopper.generated_token_ids or tuple(fast_ids)
        comparison = compare_generated_prefix(
            full_ids,
            token_ids_for_prefix,
            full_prefix.emotion_label,
            parsed_fast.emotion_label,
            full_prefix.user_profile,
            parsed_fast.user_profile,
        )
        if not full_prefix.valid:
            comparison["emotion_and_profile_match"] = False
        comparison.update({
            "fast_status": parsed_fast.status,
            "full_status": full_prefix.status,
            "stop_reason": prefix_stopper.stop_reason or stop_reason,
        })
        comparisons.append(comparison)
        case_status["valid"] += 1
        if not comparison["token_prefix_match"] or not comparison["emotion_and_profile_match"]:
            case_status["mismatch"] += 1
            continue
        predictions[sample["id"]] = {
            "emotion_label": parsed_fast.emotion_label,
            "user_profile": dict(parsed_fast.user_profile or {}),
        }
        full_prefix_predictions[sample["id"]] = {
            "emotion_label": full_prefix.emotion_label,
            "user_profile": dict(full_prefix.user_profile or {}),
        }

    # References are opened only after all generation has finished.
    references = _read_jsonl(references_path)
    reference_index = unique_index(references, "eval_id", "References")
    profile_gold: list[dict[str, list[str]]] = []
    profile_predicted: list[dict[str, list[str]]] = []
    emotion_equal_count = 0
    for sample_id in sorted(predictions):
        if sample_id not in reference_index:
            raise ValueError(f"missing Reference for selected eval ID {sample_id}")
        target = reference_index[sample_id].get("target")
        if not isinstance(target, dict):
            raise ValueError(f"invalid Reference target for {sample_id}")
        gold_profile = target.get("user_profile")
        prediction = predictions[sample_id]
        if not isinstance(gold_profile, dict):
            raise ValueError(f"invalid Reference profile for {sample_id}")
        profile_gold.append(gold_profile)
        profile_predicted.append(prediction["user_profile"])
        emotion_equal_count += target.get("emotion_label") == prediction["emotion_label"]
    metrics = profile_metrics(profile_gold, profile_predicted, enums) if profile_gold else None
    recalls = profile_nonempty_recall(profile_gold, profile_predicted) if profile_gold else None
    source_manifest["references_sha256"] = _sha256_file(references_path)
    valid_count = case_status.get("valid", 0)
    total_selected = count
    valid_coverage = valid_count / total_selected if total_selected else 0.0
    full_total = sum(full_seconds)
    fast_total = sum(fast_seconds)
    saving = 1.0 - fast_total / full_total if full_total else None
    exact_prefix_count = sum(bool(row["token_prefix_match"]) for row in comparisons)
    exact_profile_count = sum(bool(row["emotion_and_profile_match"]) for row in comparisons)
    gates = config["adoption_gates"]
    if budget_exhausted:
        status = "budget_exhausted"
    elif case_status.get("mismatch", 0):
        status = "not_adopted"
    elif valid_coverage < float(gates["minimum_valid_coverage"]):
        status = "not_adopted"
    elif saving is None or saving < float(gates["minimum_generation_saving"]):
        status = "not_adopted"
    elif gates.get("require_fast_peak_memory_not_higher") and fast_peak_bytes > full_peak_bytes:
        status = "not_adopted"
    elif len(comparisons) != total_selected:
        status = "not_adopted"
    else:
        status = "passed"

    report = {
        "status": status,
        "created_at_utc": _now(),
        "evaluation_scope": "fixed_dev_unique_conversation_full_vs_profile_prefix_greedy",
        "run_identity": {
            "config_sha256": _sha256_file(fast_config_path or _resolve("configs/evaluation/e03_profile_fast.yaml")),
            "inference_config_path": str(inference_path),
            "inference_config_sha256": _sha256_file(inference_path),
            "source_code_sha256": _sha256_file(Path(__file__).resolve()),
            "prompt_facts": prompt_facts,
            "model_facts": model_facts,
        },
        "source_manifest": source_manifest,
        "sample_count": total_selected,
        "completed_full_count": len(full_seconds),
        "completed_fast_count": len(fast_seconds),
        "valid_fast_count": valid_count,
        "valid_fast_coverage": valid_coverage,
        "exactly_matched_scoring_count": len(predictions),
        "invalid_or_incomplete_count": total_selected - valid_count,
        "case_status_counts": dict(sorted(case_status.items())),
        "consistency": {
            "compared_valid_count": len(comparisons),
            "token_prefix_match_count": exact_prefix_count,
            "emotion_and_profile_match_count": exact_profile_count,
            "all_valid_token_prefixes_match": exact_prefix_count == len(comparisons) and len(comparisons) == valid_count,
            "all_valid_profiles_match": exact_profile_count == len(comparisons) and len(comparisons) == valid_count,
        },
        "profile_fast_metrics": metrics,
        "emotion_accuracy_on_valid_fast_rows": emotion_equal_count / len(predictions) if predictions else None,
        "three_dimension_nonempty_recall": recalls,
        "timing": {
            "budget_seconds_including_model_load": budget_seconds,
            "elapsed_seconds_including_model_load": round(time.monotonic() - started, 6),
            "model_load_seconds": round(load_seconds, 6),
            "full_generation_seconds": round(full_total, 6),
            "fast_generation_seconds": round(fast_total, 6),
            "generation_time_saving_rate": saving,
            "full_generation_sample_seconds": full_seconds,
            "fast_generation_sample_seconds": fast_seconds,
            "full_peak_memory_bytes": full_peak_bytes,
            "fast_peak_memory_bytes": fast_peak_bytes,
        },
        "fallback_used": False,
        "gpu_used": True,
        "references_read_by_generator": False,
        "adoption_gates": gates,
    }
    _write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("parse-raw", "audit-source", "gpu-compare"), required=True)
    parser.add_argument("--config", default="configs/evaluation/e03_profile_fast.yaml")
    parser.add_argument("--raw", help="completed Raw predictions JSONL for CPU-only prefix audit")
    parser.add_argument("--run-config", help="frozen Transformers inference_config.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_path = _output_path(config, args.run_id, args.output_root)
    if args.mode == "parse-raw":
        if not args.raw:
            parser.error("--raw is required for parse-raw")
        report = parse_existing_raw(config, _resolve(args.raw), output_path, config_path)
    elif args.mode == "audit-source":
        report = run_source_audit(config, output_path, config_path)
    else:
        report = run_gpu_comparison(
            config,
            output_path,
            _resolve(args.run_config) if args.run_config else None,
            config_path,
        )
    print(json.dumps({
        "status": report.get("status"),
        "output": str(output_path),
        "sample_count": report.get("sample_count"),
        "valid_fast_count": report.get("valid_fast_count"),
        "valid_parse_rate": report.get("valid_parse_rate"),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") in {
        "cpu_raw_parse_completed", "source_preflight_passed", "passed",
        "not_adopted", "budget_exhausted", "blocked_environment",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
