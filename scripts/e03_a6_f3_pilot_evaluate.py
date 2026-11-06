#!/usr/bin/env python3
"""Target-free F3/F4 Fast and full-generation Smoke evaluation for the A6 Pilot."""

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
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import read_jsonl
from src.evaluation.f3_pilot import (
    F3FastPrefixStopper,
    derive_smoke_selection,
    score_fast_cases,
    score_smoke_cases,
    selected_ids_sha256,
    validate_reference_coverage,
)
from src.evaluation.f4_pilot import F4FastPrefixStopper
from src.evaluation.metrics import corpus_bleu4_char, profile_metrics, rouge_l_f1
from src.evaluation.profile_prefix import parse_profile_prefix
from src.evaluation.source_integrity import unique_index, validate_target_free_inputs
from src.shared.f3_protocol import F3ParseError, parse_f3
from src.shared.f4_protocol import F4ParseError, parse_f4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
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
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _append_jsonl_fsynced(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("unsupported E03-A6 Pilot evaluation config")
    format_version = value.get("protocol", {}).get("format_version")
    if format_version not in {"F3", "F4"} or value.get("version") != f"E03-A6-{format_version}-pilot-eval-v1":
        raise ValueError("Pilot evaluation config must be version-locked to F3 or F4")
    if value.get("protocol", {}).get("truncation_version") != "T1":
        raise ValueError("Pilot evaluation must retain T1 history selection")
    if value.get("generation", {}).get("do_sample") is not False:
        raise ValueError("Pilot evaluation requires Greedy generation")
    if int(value.get("sources", {}).get("expected_count", 0)) != 160:
        raise ValueError("Pilot Fast cohort must remain the frozen 160-row set")
    if int(value.get("selection", {}).get("smoke_count", 0)) != 32:
        raise ValueError("Pilot full-generation Smoke must remain 32 rows")
    return value


def _validate_training_config(training_config: dict[str, Any]) -> None:
    from src.training.e03_a6_contract import (
        validate_expanded_pilot_config,
        validate_expanded_f4_pilot_config,
        validate_pilot_config,
    )

    if training_config.get("protocol", {}).get("format_version") == "F4":
        validate_expanded_f4_pilot_config(training_config)
    elif training_config.get("training_scope") == "supplemental_pilot":
        validate_expanded_pilot_config(training_config)
    else:
        validate_pilot_config(training_config)


def _load_assets(config: dict[str, Any]) -> dict[str, Any]:
    source = config["sources"]
    input_path = _resolve(source["input_path"])
    id_map_path = _resolve(source["id_map_path"])
    historical_raw_path = _resolve(source["historical_f2_raw_path"])
    input_rows = _read_jsonl(input_path)
    input_index = validate_target_free_inputs(input_rows)
    id_map_rows = _read_jsonl(id_map_path)
    id_map = unique_index(id_map_rows, "eval_id", "Pilot ID map")
    if set(input_index) != set(id_map):
        raise ValueError("Pilot target-free input and ID map IDs differ")
    ids = [row["id"] for row in input_rows]
    if len(ids) != int(source["expected_count"]):
        raise ValueError("Pilot Fast source row count differs from the frozen 160 rows")
    if selected_ids_sha256(ids) != source["expected_ids_sha256"]:
        raise ValueError("Pilot Fast ID-set SHA differs from its frozen value")
    for key, path in (("expected_input_sha256", input_path), ("expected_id_map_sha256", id_map_path)):
        if _sha256(path) != source[key]:
            raise ValueError(f"Pilot source hash changed: {path.name}")
    if _sha256(historical_raw_path) != source["expected_historical_f2_raw_sha256"]:
        raise ValueError("historical E01-B Raw F2 predictions changed")
    historical_raw = _read_jsonl(historical_raw_path)
    historical_index = unique_index(historical_raw, "id", "historical F2 Raw predictions")
    if set(historical_index) != set(input_index):
        raise ValueError("historical E01-B Raw F2 prediction IDs differ from frozen Pilot cohort")

    selection_path = _resolve(config["selection"]["smoke_ids_path"])
    smoke_selection = _read_json(selection_path)
    if smoke_selection.get("source_ids_sha256") != source["expected_ids_sha256"]:
        raise ValueError("frozen Smoke selection belongs to a different 160-row source cohort")
    smoke_ids = smoke_selection.get("selected_ids")
    if not isinstance(smoke_ids, list) or len(smoke_ids) != int(config["selection"]["smoke_count"]):
        raise ValueError("frozen Smoke selection count changed")
    if selected_ids_sha256(smoke_ids) != config["selection"]["expected_smoke_ids_sha256"]:
        raise ValueError("frozen Smoke ID-set SHA differs from the Pilot evaluation config")
    if smoke_selection.get("selected_ids_sha256") != config["selection"]["expected_smoke_ids_sha256"]:
        raise ValueError("frozen Smoke IDs file has a stale self-hash")
    if not set(smoke_ids) <= set(input_index):
        raise ValueError("frozen Smoke selection contains an ID outside the 160-row cohort")
    sample_id_by_eval_id = {eval_id: str(row.get("sample_id", "")) for eval_id, row in id_map.items()}
    derived = derive_smoke_selection(
        ids,
        sample_id_by_eval_id,
        count=int(config["selection"]["smoke_count"]),
        seed=int(smoke_selection["seed"]),
    )
    if set(derived) != set(smoke_ids):
        raise ValueError("frozen Smoke selection does not match its target-free hash-ranking rule")

    run_root = _resolve(config["training"]["run_root"])
    training_manifest = _read_json(run_root / "manifest.json")
    reload_check = _read_json(run_root / "reload_check_standalone.json")
    expected_training = config["training"]
    if (
        training_manifest.get("run_id") != expected_training["run_id"]
        or training_manifest.get("experiment_id") != expected_training["expected_experiment_id"]
        or training_manifest.get("training_scope") != expected_training["expected_training_scope"]
        or training_manifest.get("status") != expected_training["expected_status"]
        or training_manifest.get("stage") != expected_training["expected_stage"]
        or training_manifest.get("optimizer_steps") != expected_training["expected_optimizer_steps"]
        or training_manifest.get("base_weight_sha256_after_training") != expected_training["expected_base_weight_sha256"]
        or training_manifest.get("full_training_status") != "not_authorized_not_run"
    ):
        raise ValueError("Pilot training manifest has not passed its frozen training boundary")
    if (
        reload_check.get("status") != "passed"
        or reload_check.get("base_weights_unchanged") is not True
        or reload_check.get("base_weight_sha256_after") != expected_training["expected_base_weight_sha256"]
    ):
        raise ValueError("Pilot Adapter independent reload-check did not pass")
    adapter_path = run_root / "adapter"
    base_path = _resolve("participant/model")
    from participant import run_inference as runner

    adapter_identity, adapter_metadata = runner.adapter_weight_identity(adapter_path)
    if adapter_identity["weight_sha256"] != expected_training["expected_adapter_weight_sha256"]:
        raise ValueError("Pilot Adapter weight identity changed")
    if adapter_identity["config_sha256"] != expected_training["expected_adapter_config_sha256"]:
        raise ValueError("Pilot Adapter config identity changed")
    if training_manifest.get("adapter_weight_sha256") != adapter_identity["weight_sha256"]:
        raise ValueError("Pilot Adapter SHA differs from the completed training manifest")
    if training_manifest.get("adapter_config_sha256") != adapter_identity["config_sha256"]:
        raise ValueError("Pilot Adapter config SHA differs from the completed training manifest")
    if adapter_metadata.get("r") != 8 or adapter_metadata.get("lora_alpha") != 16:
        raise ValueError("Pilot Adapter LoRA rank or alpha changed")
    runner.validate_adapter_base_reference(adapter_metadata, "Qwen/Qwen3-1.7B", base_path)

    run_config = _read_json(_resolve(expected_training["inference_config_path"]))
    run_config["quantization"] = {
        "enabled": True,
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quant": True,
    }
    run_config["adapter"] = {
        "enabled": True,
        "path": str(adapter_path.resolve()),
        "expected_base_weight_sha256": expected_training["expected_base_weight_sha256"],
        "expected_adapter_weight_sha256": adapter_identity["weight_sha256"],
        "expected_adapter_config_sha256": adapter_identity["config_sha256"],
        "expected_rank": 8,
        "expected_target_modules": list(adapter_metadata["target_modules"]),
    }
    run_config["generation"] = {
        "max_new_tokens": int(config["generation"]["full_max_new_tokens"]),
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    from participant import run_inference as runner

    prompt_text, schema, enums, prompt_facts = runner.load_runtime_protocol(run_config)
    selection_prompt_text, selection_facts = runner.load_f3_selection_prompt(run_config)
    format_version = config["protocol"]["format_version"]
    if prompt_facts.get("format_version") != format_version or selection_facts.get("version") != "P0":
        raise ValueError(f"Pilot {format_version} prompt or P0/T1 history-selection identity changed")
    if run_config.get("context", {}).get("max_prompt_tokens") != int(config["protocol"]["max_prompt_tokens"]):
        raise ValueError(f"Pilot {format_version} max prompt budget differs from the frozen 1280-token limit")
    if prompt_facts.get("tokenizer_id") != run_config["tokenizer"].get("model_id"):
        raise ValueError("Pilot F3 tokenizer identity changed")

    return {
        "input_path": input_path,
        "input_rows": input_rows,
        "input_index": input_index,
        "id_map_path": id_map_path,
        "id_map": id_map,
        "historical_raw_path": historical_raw_path,
        "historical_raw": historical_raw,
        "smoke_ids": smoke_ids,
        "training_manifest": training_manifest,
        "reload_check": reload_check,
        "run_root": run_root,
        "adapter_path": adapter_path,
        "adapter_identity": adapter_identity,
        "run_config": run_config,
        "prompt_text": prompt_text,
        "selection_prompt_text": selection_prompt_text,
        "schema": schema,
        "enums": enums,
        "prompt_facts": prompt_facts,
        "selection_facts": selection_facts,
    }


def _runtime_identity(config_path: Path, config: dict[str, Any], assets: dict[str, Any], mode: str) -> dict[str, Any]:
    format_key = config["protocol"]["format_version"].lower()
    serialized_run_config = json.dumps(
        assets["run_config"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "mode": mode,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        f"{format_key}_eval_helpers_sha256": _sha256(ROOT / f"src/evaluation/{format_key}_pilot.py")
        if config["protocol"]["format_version"] == "F4"
        else _sha256(ROOT / "src/evaluation/f3_pilot.py"),
        f"{format_key}_protocol_sha256": _sha256(ROOT / "src/shared/f4_protocol.py")
        if config["protocol"]["format_version"] == "F4"
        else _sha256(ROOT / "src/shared/f3_protocol.py"),
        "participant_runner_sha256": _sha256(ROOT / "participant/run_inference.py"),
        "training_manifest_sha256": _sha256(assets["run_root"] / "manifest.json"),
        "reload_check_sha256": _sha256(assets["run_root"] / "reload_check_standalone.json"),
        "input_sha256": _sha256(assets["input_path"]),
        "id_map_sha256": _sha256(assets["id_map_path"]),
        "historical_f2_raw_sha256": _sha256(assets["historical_raw_path"]),
        "cohort_ids_sha256": config["sources"]["expected_ids_sha256"],
        "adapter_weight_sha256": assets["adapter_identity"]["weight_sha256"],
        "adapter_config_sha256": assets["adapter_identity"]["config_sha256"],
        "runtime_inference_config_sha256": hashlib.sha256(serialized_run_config).hexdigest(),
        "base_weight_sha256": config["training"]["expected_base_weight_sha256"],
        "prompt_facts": assets["prompt_facts"],
        "selection_facts": assets["selection_facts"],
    }


def _load_model(run_config: dict[str, Any], torch: Any) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from participant import run_inference as runner

    local_model = runner.model_path(run_config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("F3 Pilot evaluation requires CUDA with BF16 support")
    weight_identity = runner._weight_identity(local_model, run_config)
    expected_base_sha = run_config["adapter"]["expected_base_weight_sha256"]
    if weight_identity.get("observed_weight_hash") != expected_base_sha:
        raise ValueError("Base SHA differs from the Pilot training identity")
    model_config = AutoConfig.from_pretrained(
        local_model, trust_remote_code=bool(run_config.get("trust_remote_code", False)),
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model,
        trust_remote_code=bool(run_config.get("trust_remote_code", False)),
        local_files_only=True,
        device_map=run_config.get("device_map", "auto"),
        torch_dtype=run_config.get("torch_dtype", "auto"),
        quantization_config=runner.build_quantization_config(run_config, torch),
    )
    model = runner.load_peft_adapter(
        model,
        run_config["adapter"]["path"],
        prepare_quantized_base=True,
        is_trainable=False,
        local_files_only=True,
        autocast_adapter_dtype=runner.adapter_autocast_dtype_for_format(run_config.get("format_version", "F2")),
    )
    model.eval()
    tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(run_config, local_model)
    quantization_facts = runner.inspect_quantization_runtime(model, run_config, torch)
    if not quantization_facts.get("real_nf4_verified"):
        raise RuntimeError("Pilot evaluation did not load real NF4/BF16/CUDA without offload")
    adapter_dtypes = sorted({str(p.dtype) for name, p in model.named_parameters() if ".lora_A." in name or ".lora_B." in name})
    if adapter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"Pilot Adapter dtype changed during evaluation: {adapter_dtypes}")
    return model, tokenizer, {
        "base_weight_sha256": weight_identity.get("observed_weight_hash"),
        "adapter_weight_sha256": run_config["adapter"]["expected_adapter_weight_sha256"],
        "adapter_config_sha256": run_config["adapter"].get("expected_adapter_config_sha256"),
        "tokenizer_facts": tokenizer_facts,
        "quantization": quantization_facts,
        "adapter_dtypes": adapter_dtypes,
        "model_type": model_config.model_type,
        "cuda_device": torch.cuda.get_device_name(0),
    }


def _run_generate(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    stopper: Any | None = None,
) -> tuple[list[int], str, float, bool]:
    from transformers import StoppingCriteria, StoppingCriteriaList
    from participant import run_inference as runner

    input_ids = torch.tensor(
        [prompt_ids], dtype=torch.long,
        device=model.get_input_embeddings().weight.device,
    )
    attention_mask = torch.ones_like(input_ids)
    generation: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    criteria = None
    if stopper is not None:
        class PrefixCriteria(StoppingCriteria):
            def __call__(self, sequences: Any, scores: Any, **kwargs: Any) -> bool:
                generated = sequences[0, len(prompt_ids):].detach().cpu().tolist()
                return stopper.inspect(generated)

        criteria = StoppingCriteriaList([PrefixCriteria()])
    runner.synchronize(torch)
    before = time.monotonic()
    with torch.inference_mode():
        generation_kwargs: dict[str, Any] = {}
        if criteria is not None:
            generation_kwargs["stopping_criteria"] = criteria
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs, **generation)
    runner.synchronize(torch)
    elapsed = time.monotonic() - before
    generated_ids = [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_generated = eos_id is not None and int(eos_id) in generated_ids
    return generated_ids, decoded, elapsed, bool(eos_generated)


def _model_runtime_config(run_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "inference_config.json"
    _write_json_atomic(config_path, run_config)
    return {"path": str(config_path), "sha256": _sha256(config_path)}


def _prepare_run_directory(output_dir: Path, identity: dict[str, Any], ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    cases_path = output_dir / "cases.jsonl"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("run_identity") != identity:
            raise ValueError("existing Pilot evaluation artifacts have a different frozen run identity")
    else:
        if cases_path.exists():
            raise ValueError("Pilot evaluation journal exists without its identity manifest")
        manifest = {
            "status": "running",
            "created_at_utc": _now(),
            "run_identity": identity,
            "expected_case_count": len(ids),
            "completed_case_count": 0,
            "references_read_by_generator": False,
            "output_contract": "append_and_fsync_each_complete_case; atomic_manifest_replace",
        }
        _write_json_atomic(manifest_path, manifest)
    completed = _read_jsonl(cases_path) if cases_path.exists() else []
    recorded_completed = int(manifest.get("completed_case_count", 0))
    if recorded_completed > len(completed):
        raise ValueError("Pilot evaluation manifest claims cases that are missing from its durable journal")
    if len(completed) > len(ids):
        raise ValueError("existing Pilot evaluation has more rows than the frozen cohort")
    for index, row in enumerate(completed):
        if row.get("id") != ids[index]:
            raise ValueError("existing Pilot evaluation rows are not a frozen-cohort prefix")
    if not cases_path.exists():
        cases_path.touch(exist_ok=False)
    manifest.update(status="running", completed_case_count=len(completed), resumed_case_count=len(completed))
    _write_json_atomic(manifest_path, manifest)
    return manifest, completed


def _record_failure(output_dir: Path, manifest: dict[str, Any], error: BaseException, current_id: str | None) -> None:
    manifest.update(
        status="failed_resumable",
        failed_at_utc=_now(),
        failed_case_id=current_id,
        error=f"{type(error).__name__}: {error}",
    )
    _write_json_atomic(output_dir / "run_manifest.json", manifest)


def _read_references_after_generation(config: dict[str, Any], ids: list[str]) -> dict[str, dict[str, Any]]:
    path = _resolve(config["sources"]["references_path"])
    if _sha256(path) != config["sources"]["expected_references_sha256"]:
        raise ValueError("Pilot References SHA differs from the frozen target-free cohort")
    rows = _read_jsonl(path)
    reference_index = unique_index(rows, "eval_id", "Pilot References")
    validate_reference_coverage(list(reference_index), ids)
    return reference_index


def _historical_f2_score(assets: dict[str, Any], reference_index: dict[str, dict[str, Any]], enums: dict[str, list[str]]) -> dict[str, Any]:
    ids = [row["id"] for row in assets["input_rows"]]
    raw_index = unique_index(assets["historical_raw"], "id", "historical F2 Raw predictions")
    predictions: list[dict[str, list[str]]] = []
    statuses: Counter[str] = Counter()
    gold: list[dict[str, list[str]]] = []
    for eval_id in ids:
        parsed = parse_profile_prefix(str(raw_index[eval_id].get("raw_text", "")), enums)
        statuses[parsed.status] += 1
        predictions.append(dict(parsed.user_profile) if parsed.valid and parsed.user_profile else {
            "personality_traits": [], "interests": [], "style": [],
        })
        gold.append(reference_index[eval_id]["target"]["user_profile"])
    return {
        "run_id": "E01-B-PILOT-EVAL-20261028-101136",
        "evaluation_layer": "Raw F2 generated profile prefix; not Submission output",
        "raw_sha256": _sha256(assets["historical_raw_path"]),
        "status_counts": dict(sorted(statuses.items())),
        "fixed_denominator_profile_metrics": profile_metrics(gold, predictions, enums),
    }


def _score_fast(
    config: dict[str, Any], assets: dict[str, Any], cases: list[dict[str, Any]], output_dir: Path,
) -> dict[str, Any]:
    ids = [row["id"] for row in assets["input_rows"]]
    reference_index = _read_references_after_generation(config, ids)
    targets = [reference_index[eval_id]["target"] for eval_id in ids]
    score = score_fast_cases(targets, cases, assets["enums"], config["pilot_quality_gates"]["fast"])
    if config["protocol"]["format_version"] == "F4":
        score["gate_values"]["strict_f4_parse_rate"] = score["gate_values"].pop("strict_f3_parse_rate")
        score["gate_checks"]["strict_f4_parse_rate"] = score["gate_checks"].pop("strict_f3_parse_rate")
    history = _historical_f2_score(assets, reference_index, assets["enums"])
    format_key = config["protocol"]["format_version"].lower()
    result = {
        "status": "complete",
        "stage": f"{config['protocol']['format_version']} Fast-only Dev 160",
        "created_at_utc": _now(),
        "cohort_count": len(ids),
        "cohort_ids_sha256": config["sources"]["expected_ids_sha256"],
        "references_read_by_generator": False,
        format_key: score,
        "historical_f2_raw_prefix": history,
        "generation_efficiency": {
            f"{format_key}_fast_generated_tokens_total": sum(int(case.get("generated_token_count", 0)) for case in cases),
            f"{format_key}_fast_mean_generated_tokens": (
                sum(int(case.get("generated_token_count", 0)) for case in cases) / len(cases)
                if cases else 0.0
            ),
            "f2_raw_full_generated_tokens_total": sum(
                int(row.get("generated_tokens", 0)) for row in assets["historical_raw"]
            ),
            "f2_raw_full_generation_seconds_total": sum(
                float(row.get("latency_ms", 0.0)) / 1000 for row in assets["historical_raw"]
            ),
        },
        f"{format_key}_vs_f2_same_raw_layer": {
            "profile_micro_f1_delta": (
                score["fixed_denominator_profile_metrics"]["overall_33_label_micro_f1"]
                - history["fixed_denominator_profile_metrics"]["overall_33_label_micro_f1"]
            ),
            "tp_delta": score["fixed_denominator_profile_metrics"]["overall_33_label_tp"]
            - history["fixed_denominator_profile_metrics"]["overall_33_label_tp"],
            "fp_delta": score["fixed_denominator_profile_metrics"]["overall_33_label_fp"]
            - history["fixed_denominator_profile_metrics"]["overall_33_label_fp"],
            "fn_delta": score["fixed_denominator_profile_metrics"]["overall_33_label_fn"]
            - history["fixed_denominator_profile_metrics"]["overall_33_label_fn"],
        },
    }
    _write_json_atomic(output_dir / "summary.json", result)
    return result


def _score_smoke(
    config: dict[str, Any], assets: dict[str, Any], cases: list[dict[str, Any]], output_dir: Path,
) -> dict[str, Any]:
    ids = [row["id"] for row in cases]
    reference_index = _read_references_after_generation(config, ids)
    smoke_gate = config["pilot_quality_gates"]["full_generation_smoke"]
    score = score_smoke_cases(cases, smoke_gate)
    if config["protocol"]["format_version"] == "F4":
        score["gate_values"]["strict_f4_parse_rate"] = score["gate_values"].pop("strict_f3_parse_rate")
        score["gate_checks"]["strict_f4_parse_rate"] = score["gate_checks"].pop("strict_f3_parse_rate")
    targets = [reference_index[eval_id]["target"] for eval_id in ids]
    fast_cases = [
        {
            "id": case["id"],
            "parse_status": "success" if case["fast"].get("strict_parse_success") else case["fast"].get("parse_status", "incomplete"),
            "parsed": case["fast"].get("parsed"),
        }
        for case in cases
    ]
    fast_gate = {
        "min_strict_f3_parse_rate": 0.0,
        "min_cases_with_nonempty_profile_prediction": 0,
        "min_profile_true_positives": 0,
        "max_profile_false_positives": 999999,
        "min_profile_micro_precision": 0.0,
        "min_emotion_accuracy": 0.0,
        "max_single_emotion_prediction_share": 1.0,
    }
    profile_score = score_fast_cases(targets, fast_cases, assets["enums"], fast_gate)
    response_predictions = [
        case["full"].get("parsed", {}).get("response_text", "")
        if case["full"].get("strict_parse_success") else ""
        for case in cases
    ]
    response_references = [target["response_text"] for target in targets]
    from statistics import fmean
    result = {
        "status": "complete",
        "stage": f"{config['protocol']['format_version']} full-generation Smoke 32",
        "created_at_utc": _now(),
        "cohort_count": len(cases),
        "cohort_ids_sha256": selected_ids_sha256(ids),
        "references_read_by_generator": False,
        "smoke": score,
        "fast_profile_metrics_on_smoke_cohort": profile_score["fixed_denominator_profile_metrics"],
        "response_proxy_metrics": {
            "scope": "local_proxy_not_official_score",
            "bleu4_char": corpus_bleu4_char(response_references, response_predictions, expected_version="2.5.1"),
            "rouge_l_char_macro_f1": fmean(
                rouge_l_f1(reference, prediction)
                for reference, prediction in zip(response_references, response_predictions)
            ) if response_predictions else 0.0,
            "response_characters_mean": fmean(map(len, response_predictions)) if response_predictions else 0.0,
            "empty_response_count": sum(not response.strip() for response in response_predictions),
        },
        "prefix_comparisons": [
            {"id": case["id"], **case["prefix_equivalence"]}
            for case in cases
        ],
    }
    _write_json_atomic(output_dir / "summary.json", result)
    return result


def run_evaluation(config_path: Path, config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"fast", "smoke"}:
        raise ValueError("mode must be fast or smoke")
    import torch
    from participant import run_inference as runner

    assets = _load_assets(config)
    training_config = yaml.safe_load((assets["run_root"] / "config_snapshot.yaml").read_text(encoding="utf-8"))
    _validate_training_config(training_config)
    if training_config.get("pilot_quality_gates") != config.get("pilot_quality_gates"):
        raise ValueError("Pilot evaluation gates differ from the values frozen before training")
    identity = _runtime_identity(config_path, config, assets, mode)
    mode_name = "fast-160" if mode == "fast" else "smoke-32"
    output_dir = _resolve(config["output_root"]) / config["experiment_id"] / mode_name
    ids = [row["id"] for row in assets["input_rows"]] if mode == "fast" else list(assets["smoke_ids"])
    selected_ids = set(ids)
    selected = [row for row in assets["input_rows"] if row["id"] in selected_ids]
    if mode == "smoke":
        selected_by_id = {row["id"]: row for row in selected}
        selected = [selected_by_id[eval_id] for eval_id in ids]
    existing_manifest_path = output_dir / "run_manifest.json"
    if existing_manifest_path.exists():
        existing_manifest = _read_json(existing_manifest_path)
        if existing_manifest.get("run_identity") != identity:
            raise ValueError("existing Pilot evaluation artifacts have a different frozen run identity")
        summary_path = output_dir / "summary.json"
        if existing_manifest.get("status") == "complete" and summary_path.is_file():
            return _read_json(summary_path)
    manifest, completed = _prepare_run_directory(output_dir, identity, ids)
    runtime_config_fact = _model_runtime_config(assets["run_config"], output_dir)
    manifest["runtime_inference_config"] = runtime_config_fact
    _write_json_atomic(output_dir / "run_manifest.json", manifest)
    if len(completed) == len(ids):
        summary = (
            _score_fast(config, assets, completed, output_dir)
            if mode == "fast"
            else _score_smoke(config, assets, completed, output_dir)
        )
        manifest.update(
            status="complete",
            gate_passed=(summary[config["protocol"]["format_version"].lower()]["gate_passed"] if mode == "fast" else summary["smoke"]["gate_passed"]),
            scored_at_utc=_now(),
            references_sha256=_sha256(_resolve(config["sources"]["references_path"])),
            summary_sha256=_sha256(output_dir / "summary.json"),
            cases_sha256=_sha256(output_dir / "cases.jsonl"),
            references_read_by_generator=False,
        )
        _write_json_atomic(output_dir / "run_manifest.json", manifest)
        return summary
    if not torch.cuda.is_available():
        manifest.update(status="blocked_environment", reason="cuda_unavailable", gpu_used=False)
        _write_json_atomic(output_dir / "run_manifest.json", manifest)
        return manifest

    format_version = config["protocol"]["format_version"]
    stopper_class = F3FastPrefixStopper if format_version == "F3" else F4FastPrefixStopper
    parser = parse_f3 if format_version == "F3" else parse_f4
    parse_error_types = (F3ParseError,) if format_version == "F3" else (F4ParseError,)
    prediction_parser = runner.parse_f3_prediction if format_version == "F3" else runner.parse_f4_prediction
    budget_seconds = int(config["generation"][f"{mode}_budget_seconds"])
    deadline = time.monotonic() + budget_seconds
    current_id: str | None = None
    model = tokenizer = None
    try:
        model, tokenizer, model_facts = _load_model(assets["run_config"], torch)
        manifest.update(
            gpu_used=True,
            model_facts=model_facts,
            model_load_seconds=round(budget_seconds - max(0, deadline - time.monotonic()), 6),
            generation_started=True,
            gpu_budget_seconds_including_model_load=budget_seconds,
        )
        _write_json_atomic(output_dir / "run_manifest.json", manifest)
        completed_ids = {row["id"] for row in completed}
        for sample in selected:
            current_id = sample["id"]
            if current_id in completed_ids:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Pilot GPU budget exhausted before the next sample")
            selection, _messages = runner.prepare_sample_messages(
                sample,
                assets["prompt_text"],
                tokenizer,
                int(assets["run_config"]["context"]["max_prompt_tokens"]),
                selection_prompt_text=assets["selection_prompt_text"],
            )
            if selection.get("floor_exceeds_cap"):
                raise ValueError(f"{format_version} prompt floor exceeds the frozen cap: {current_id}")
            prompt_ids = list(selection["prompt_ids"])
            prompt_sha = hashlib.sha256(json.dumps(prompt_ids, separators=(",", ":")).encode()).hexdigest()
            if mode == "fast":
                stopper = stopper_class(
                    lambda values: tokenizer.decode(list(values), skip_special_tokens=True),
                    assets["enums"], deadline=deadline,
                )
                generated_ids, raw_text, elapsed, eos_generated = _run_generate(
                    model, tokenizer, torch, prompt_ids,
                    max_new_tokens=int(config["generation"]["fast_max_new_tokens"]),
                    stopper=stopper,
                )
                result = stopper.result or stopper.finish(
                    "eos" if eos_generated else "token_limit" if len(generated_ids) >= int(config["generation"]["fast_max_new_tokens"]) else "generation_ended",
                    generated_ids,
                )
                row = {
                    "id": current_id,
                    "parse_status": "success" if result.valid else result.status,
                    "reason": result.reason,
                    "raw_prefix": result.raw_prefix,
                    "parsed": result.parsed,
                    "generated_token_ids": list(result.generated_token_ids),
                    "generated_token_count": len(result.generated_token_ids),
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": prompt_sha,
                    "generation_seconds": elapsed,
                    "eos_generated": eos_generated,
                    "stop_reason": stopper.stop_reason or "generation_ended",
                }
            else:
                generated_ids, raw_text, elapsed, eos_generated = _run_generate(
                    model, tokenizer, torch, prompt_ids,
                    max_new_tokens=int(config["generation"]["full_max_new_tokens"]),
                )
                try:
                    full_parsed = parser(raw_text, assets["enums"])
                    full_parse_success = True
                    full_parse_status = "success"
                except parse_error_types as exc:
                    full_parsed = None
                    full_parse_success = False
                    full_parse_status = exc.code
                prediction, runner_status, strict_parse, failure_code = prediction_parser(
                    current_id, raw_text, assets["enums"],
                )
                try:
                    runner.validate_submission_prediction(prediction, assets["schema"])
                    schema_valid = True
                    schema_error = None
                except Exception as exc:
                    schema_valid = False
                    schema_error = f"{type(exc).__name__}: {exc}"
                full = {
                    "raw_text": raw_text,
                    "parsed": full_parsed,
                    "parse_status": full_parse_status,
                    "strict_parse_success": full_parse_success,
                    "runner_parse_status": runner_status,
                    "runner_strict_parse_success": strict_parse,
                    "runner_failure_code": failure_code,
                    "official_schema_valid": schema_valid,
                    "schema_error": schema_error,
                    "nonempty_response": bool(full_parsed and full_parsed["response_text"].strip()),
                    "eos_generated": eos_generated,
                    "stop_reason": "eos" if eos_generated else "max_new_tokens" if len(generated_ids) >= int(config["generation"]["full_max_new_tokens"]) else "generation_ended",
                    "generated_token_ids": generated_ids,
                    "generated_token_count": len(generated_ids),
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": prompt_sha,
                    "generation_seconds": elapsed,
                }
                full_prefix_result: dict[str, Any] | None = None
                try:
                    from src.training.e03_a6_contract import (
                        f3_prefix_through_r_marker,
                        f4_prefix_through_response_marker,
                    )

                    prefix_helper = f3_prefix_through_r_marker if format_version == "F3" else f4_prefix_through_response_marker
                    full_prefix_ids, full_prefix_text = prefix_helper(
                        generated_ids,
                        lambda values: tokenizer.decode(list(values), skip_special_tokens=True),
                    )
                    parsed_prefix = parser(full_prefix_text, assets["enums"])
                    full_prefix_result = {
                        "token_ids": full_prefix_ids,
                        "parsed": parsed_prefix,
                        "text": full_prefix_text,
                    }
                except (ValueError, *parse_error_types) as exc:
                    full_prefix_result = {"token_ids": None, "parsed": None, "text": None, "error": f"{type(exc).__name__}: {exc}"}

                stopper = stopper_class(
                    lambda values: tokenizer.decode(list(values), skip_special_tokens=True),
                    assets["enums"], deadline=deadline,
                )
                fast_ids, _fast_text, fast_elapsed, fast_eos = _run_generate(
                    model, tokenizer, torch, prompt_ids,
                    max_new_tokens=int(config["generation"]["fast_max_new_tokens"]),
                    stopper=stopper,
                )
                fast_result = stopper.result or stopper.finish(
                    "eos" if fast_eos else "token_limit" if len(fast_ids) >= int(config["generation"]["fast_max_new_tokens"]) else "generation_ended",
                    fast_ids,
                )
                semantic_match = bool(
                    full_parsed is not None
                    and fast_result.parsed is not None
                    and full_parsed["emotion_label"] == fast_result.parsed["emotion_label"]
                    and all(
                        full_parsed["user_profile"][name] == fast_result.parsed["user_profile"][name]
                        for name in ("personality_traits", "interests", "style")
                    )
                )
                token_prefix_match = bool(
                    full_prefix_result and full_prefix_result.get("token_ids") is not None
                    and fast_result.generated_token_ids == tuple(full_prefix_result["token_ids"])
                )
                row = {
                    "id": current_id,
                    "full": full,
                    "fast": {
                        "raw_prefix": fast_result.raw_prefix,
                        "parsed": fast_result.parsed,
                        "parse_status": "success" if fast_result.valid else fast_result.status,
                        "strict_parse_success": fast_result.valid,
                        "reason": fast_result.reason,
                        "generated_token_ids": list(fast_result.generated_token_ids),
                        "generated_token_count": len(fast_result.generated_token_ids),
                        "generation_seconds": fast_elapsed,
                        "eos_generated": fast_eos,
                        "stop_reason": stopper.stop_reason or "generation_ended",
                    },
                    "full_prefix": {
                        "token_ids": full_prefix_result.get("token_ids") if full_prefix_result else None,
                        "text": full_prefix_result.get("text") if full_prefix_result else None,
                        "error": full_prefix_result.get("error") if full_prefix_result else "missing_prefix",
                    },
                    "prefix_equivalence": {
                        "equivalent": semantic_match and token_prefix_match,
                        "semantic_match": semantic_match,
                        "token_prefix_match": token_prefix_match,
                        "full_prefix_token_count": len(full_prefix_result["token_ids"]) if full_prefix_result and full_prefix_result.get("token_ids") is not None else None,
                        "fast_token_count": len(fast_result.generated_token_ids),
                    },
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": prompt_sha,
                }
            _append_jsonl_fsynced(output_dir / "cases.jsonl", row)
            completed.append(row)
            completed_ids.add(current_id)
            manifest.update(completed_case_count=len(completed), current_case_id=None)
            _write_json_atomic(output_dir / "run_manifest.json", manifest)
        manifest.update(status="scoring", completed_case_count=len(completed), ended_generation_at_utc=_now())
        _write_json_atomic(output_dir / "run_manifest.json", manifest)
        if mode == "fast":
            summary = _score_fast(config, assets, completed, output_dir)
        else:
            summary = _score_smoke(config, assets, completed, output_dir)
        manifest.update(
            status="complete",
            gate_passed=(summary[config["protocol"]["format_version"].lower()]["gate_passed"] if mode == "fast" else summary["smoke"]["gate_passed"]),
            scored_at_utc=_now(),
            references_sha256=_sha256(_resolve(config["sources"]["references_path"])),
            summary_sha256=_sha256(output_dir / "summary.json"),
            cases_sha256=_sha256(output_dir / "cases.jsonl"),
            references_read_by_generator=False,
        )
        _write_json_atomic(output_dir / "run_manifest.json", manifest)
        return summary
    except BaseException as exc:
        _record_failure(output_dir, manifest, exc, current_id)
        if isinstance(exc, KeyboardInterrupt):
            return manifest
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_preflight(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    assets = _load_assets(config)
    training_config = yaml.safe_load((assets["run_root"] / "config_snapshot.yaml").read_text(encoding="utf-8"))
    _validate_training_config(training_config)
    if training_config.get("pilot_quality_gates") != config.get("pilot_quality_gates"):
        raise ValueError("evaluation quality gates differ from the values frozen before Pilot training")
    source_report = {
        "status": "preflight_passed",
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "input_count": len(assets["input_rows"]),
        "input_ids_sha256": config["sources"]["expected_ids_sha256"],
        "historical_f2_raw_rows": len(assets["historical_raw"]),
        "smoke_count": len(assets["smoke_ids"]),
        "smoke_ids_sha256": config["selection"]["expected_smoke_ids_sha256"],
        "training_status": assets["training_manifest"]["status"],
        "reload_check_status": assets["reload_check"]["status"],
        "adapter_weight_sha256": assets["adapter_identity"]["weight_sha256"],
        "adapter_config_sha256": assets["adapter_identity"]["config_sha256"],
        "prompt_facts": assets["prompt_facts"],
        "selection_facts": assets["selection_facts"],
        "gpu_used": False,
        "references_read_by_generator": False,
    }
    output = _resolve(config["output_root"]) / config["experiment_id"] / "preflight.json"
    _write_json_atomic(output, source_report)
    return source_report


def score_existing_generation(config_path: Path, config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Score a fully generated journal without loading a model or repeating inference."""
    import yaml

    if mode not in {"fast", "smoke"}:
        raise ValueError("score mode must be fast or smoke")
    assets = _load_assets(config)
    training_config = yaml.safe_load((assets["run_root"] / "config_snapshot.yaml").read_text(encoding="utf-8"))
    _validate_training_config(training_config)
    if training_config.get("pilot_quality_gates") != config.get("pilot_quality_gates"):
        raise ValueError("Pilot evaluation quality gates changed after training")
    mode_name = "fast-160" if mode == "fast" else "smoke-32"
    output_dir = _resolve(config["output_root"]) / config["experiment_id"] / mode_name
    manifest_path = output_dir / "run_manifest.json"
    cases_path = output_dir / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise FileNotFoundError("existing Pilot generation journal or manifest is missing")
    manifest = _read_json(manifest_path)
    recorded_identity = manifest.get("run_identity")
    expected_identity = _runtime_identity(config_path, config, assets, mode)
    if not isinstance(recorded_identity, dict):
        raise ValueError("existing Pilot generation has no frozen run identity")
    # The generation code hash may differ because this recovery command uses a
    # corrected scorer. All source/model/config identities must still match.
    for key in expected_identity:
        if key in {"script_sha256", f"{config['protocol']['format_version'].lower()}_eval_helpers_sha256"}:
            continue
        if recorded_identity.get(key) != expected_identity[key]:
            raise ValueError(f"existing Pilot generation identity changed: {key}")
    if not isinstance(recorded_identity.get("script_sha256"), str) or len(recorded_identity["script_sha256"]) != 64:
        raise ValueError("existing Pilot generation code identity is missing")
    ids = (
        [row["id"] for row in assets["input_rows"]]
        if mode == "fast" else list(assets["smoke_ids"])
    )
    cases = _read_jsonl(cases_path)
    if len(cases) != len(ids) or [case.get("id") for case in cases] != ids:
        raise ValueError("existing Pilot generation is not a complete frozen-cohort journal")
    old_status = manifest.get("status")
    old_error = manifest.get("error")
    summary = _score_fast(config, assets, cases, output_dir) if mode == "fast" else _score_smoke(config, assets, cases, output_dir)
    manifest.update(
        status="complete",
        gate_passed=(summary[config["protocol"]["format_version"].lower()]["gate_passed"] if mode == "fast" else summary["smoke"]["gate_passed"]),
        references_sha256=_sha256(_resolve(config["sources"]["references_path"])),
        summary_sha256=_sha256(output_dir / "summary.json"),
        cases_sha256=_sha256(cases_path),
        references_read_by_generator=False,
        rescored_at_utc=_now(),
        scoring_identity={
            "script_sha256": _sha256(Path(__file__).resolve()),
            f"{config['protocol']['format_version'].lower()}_eval_helpers_sha256": _sha256(ROOT / f"src/evaluation/{config['protocol']['format_version'].lower()}_pilot.py")
            if config["protocol"]["format_version"] == "F4"
            else _sha256(ROOT / "src/evaluation/f3_pilot.py"),
            "scorer": "fixed-cohort-subset-reference-coverage-v1",
        },
        scoring_recovery=(
            manifest.get("scoring_recovery")
            if old_status == "complete" and manifest.get("scoring_recovery") is not None
            else
            {
                "previous_status": old_status,
                "previous_error": old_error,
                "reason": "the original score check required all 160 Reference IDs for the 32-row Smoke subset",
                "generation_repeated": False,
            }
            if old_status != "complete" else None
        ),
    )
    _write_json_atomic(manifest_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/e03_a6_f3_pilot.yaml")
    parser.add_argument("--mode", choices=("preflight", "fast", "smoke", "score-fast", "score-smoke"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.mode == "preflight":
        result = run_preflight(config_path, config)
    elif args.mode.startswith("score-"):
        result = score_existing_generation(config_path, config, args.mode.removeprefix("score-"))
    else:
        result = run_evaluation(config_path, config, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
