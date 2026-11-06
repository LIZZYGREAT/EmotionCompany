#!/usr/bin/env python3
"""Evaluate a saved F4 Full checkpoint on fixed target-free 160/224 cohorts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from src.evaluation.e03_a6_f4_staged_metrics import (
    canonical_sha256,
    compare_same_ids,
    historical_baselines,
    load_gold_references,
    load_target_free_cohorts,
    response_body_token_count,
    score_cohort,
    sha256_file,
)
from src.evaluation.f4_pilot import F4FastPrefixStopper
from src.training.e03_a6_f4_staged_contract import validate_eval_config
from src.training.e03_a6_f4_staged_full import (
    DEFAULT_CONFIG,
    _load_config,
    _read_json,
    _resolve,
    _write_json,
    config_sha256,
)
from src.shared.f4_protocol import parse_f4


EVAL_CONFIG = ROOT / "configs/evaluation/e03_a6_f4_staged_fast.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _append_jsonl_fsynced(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_eval_config(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = _resolve(path)
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("F4 staged evaluation config must be a YAML object")
    validate_eval_config(config)
    return config, resolved


def _prediction_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("F4 generated predictions must have unique IDs")
    return {"count": len(ids), "ids_sha256": canonical_sha256(ids)}


def _reported_stop_reason(result: Any, stopper: F4FastPrefixStopper) -> str:
    """Use the parser result's `reason` field and the stopper's optional override."""
    return stopper.stop_reason or result.reason


def _render_node_report(summary: dict[str, Any]) -> str:
    rows = []
    for name in ("pilot160", "dev224"):
        metric = summary["cohorts"][name]
        profile = metric["profile_33_label"]
        emotion = metric["emotion"]
        rows.append(
            f"| {name} | {metric['fixed_denominator_count']} | {metric['valid_count']} / {metric['fixed_denominator_count']} "
            f"| {emotion['accuracy']:.4f} | {profile['overall_33_label_tp']} / {profile['overall_33_label_fp']} / "
            f"{profile['overall_33_label_fn']} | {profile['overall_33_label_micro_precision']:.4f} / "
            f"{profile['overall_33_label_micro_recall']:.4f} / {profile['overall_33_label_micro_f1']:.4f} "
            f"| {metric['nonempty_profile_prediction_count']} |"
        )
    return "\n".join([
        f"# E03-A6 F4 staged Fast-only — Step {summary['checkpoint_step']}",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Checkpoint manifest SHA-256: `{summary['checkpoint_manifest_sha256']}`",
        f"- Adapter weight SHA-256: `{summary['adapter_identity']['weight_sha256']}`",
        f"- Train rows / samples seen / supervised tokens: `{summary['training']['train_rows']}` / `{summary['training']['samples_seen']}` / `{summary['training']['supervised_tokens_seen']}`",
        f"- Training loss start / node: `{summary['training']['loss_start']:.6f}` / `{summary['training']['loss_node']:.6f}`; cumulative training time: `{summary['training']['cumulative_training_seconds']:.2f}s`",
        f"- Evaluation: `{summary['quality_status']}`; engineering gate: `{summary['engineering_gate']}`",
        f"- Authorization: `{summary['progression_authorization_status']}`; node metrics individually reviewed: `{summary['node_metrics_individually_reviewed']}`",
        "",
        "| Cohort | 固定分母 | F4 前缀有效 | 情绪 Accuracy | 画像 TP / FP / FN | Micro P / R / F1 | 非空画像预测 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "| Cohort | Macro-F1 | 三维 Exact | Personality TP/FP/FN | Interests TP/FP/FN | Style TP/FP/FN | P/I/S 非空预测 | Gold 非空 Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {name} | {summary['cohorts'][name]['profile_33_label']['overall_33_label_macro_f1']:.4f} "
            f"| {summary['cohorts'][name]['profile_33_label']['all_three_dimensions_exact_match']:.4f} "
            f"| {summary['cohorts'][name]['profile_dimensions']['personality_traits']['tp']}/"
            f"{summary['cohorts'][name]['profile_dimensions']['personality_traits']['fp']}/"
            f"{summary['cohorts'][name]['profile_dimensions']['personality_traits']['fn']} "
            f"| {summary['cohorts'][name]['profile_dimensions']['interests']['tp']}/"
            f"{summary['cohorts'][name]['profile_dimensions']['interests']['fp']}/"
            f"{summary['cohorts'][name]['profile_dimensions']['interests']['fn']} "
            f"| {summary['cohorts'][name]['profile_dimensions']['style']['tp']}/"
            f"{summary['cohorts'][name]['profile_dimensions']['style']['fp']}/"
            f"{summary['cohorts'][name]['profile_dimensions']['style']['fn']} "
            f"| {summary['cohorts'][name]['predicted_nonempty_by_dimension']['personality_traits']}/"
            f"{summary['cohorts'][name]['predicted_nonempty_by_dimension']['interests']}/"
            f"{summary['cohorts'][name]['predicted_nonempty_by_dimension']['style']} "
            f"| {summary['cohorts'][name]['recall_when_gold_profile_nonempty']:.4f} |"
            for name in ("pilot160", "dev224")
        ],
        "",
        f"- Fast-only tokens: `{summary['resources']['generated_tokens_total']}`; model load: `{summary['resources']['model_load_seconds']:.2f}s`; generation: `{summary['resources']['generation_seconds']:.2f}s`; peak allocated VRAM: `{summary['resources']['peak_memory_bytes']}` bytes.",
        f"- Strict Response marker closure count: `{summary['resources']['response_boundary_closed_count']}/{summary['resources']['unique_generation_count']}`; Response body tokens: `{summary['resources']['response_body_tokens_generated']}`.",
        "- References were read after prediction file hash freeze and after releasing the model. Metrics are local development proxies.",
        "",
    ])


def _prepare_eval_directory(
    output_dir: Path,
    identity: dict[str, Any],
    union_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = output_dir / "run_manifest.json"
    predictions_path = output_dir / "predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        rows = _read_jsonl(predictions_path)
        if manifest.get("evaluation_identity") != identity:
            if manifest.get("status") != "failed_resumable" or rows:
                raise ValueError("existing Fast-only output has a different Run/checkpoint/cohort identity")
            previous_attempts = list(manifest.get("previous_attempts", []))
            previous_identity = manifest.get("evaluation_identity", {})
            attempt_path = output_dir / "attempts" / (
                f"attempt-{len(previous_attempts) + 1:02d}-"
                f"{str(previous_identity.get('evaluation_code_sha256', 'unknown'))[:12]}.json"
            )
            _write_json(attempt_path, manifest, exclusive=True)
            previous_attempts.append({
                "manifest_path": str(attempt_path.relative_to(output_dir)),
                "identity_sha256": canonical_sha256(previous_identity),
                "error": manifest.get("error"),
            })
            manifest = {
                "status": "running",
                "created_at_utc": _now(),
                "evaluation_identity": identity,
                "expected_generation_count": len(union_ids),
                "references_read_by_generator": False,
                "output_contract": "one append-and-fsync record per unique target-free ID",
                "previous_attempts": previous_attempts,
            }
            _write_json(manifest_path, manifest)
    else:
        if predictions_path.exists():
            raise ValueError("prediction journal exists without its identity Manifest")
        manifest = {
            "status": "running",
            "created_at_utc": _now(),
            "evaluation_identity": identity,
            "expected_generation_count": len(union_ids),
            "references_read_by_generator": False,
            "output_contract": "one append-and-fsync record per unique target-free ID",
                "previous_attempts": [],
        }
        _write_json(manifest_path, manifest, exclusive=True)
        rows = []
    if len(rows) > len(union_ids):
        raise ValueError("prediction journal has more rows than the frozen ID union")
    for index, row in enumerate(rows):
        if row.get("id") != union_ids[index]:
            raise ValueError("prediction journal is not a prefix of the frozen union order")
    if manifest.get("status") == "complete" and len(rows) == len(union_ids):
        return manifest, rows
    if not predictions_path.exists():
        predictions_path.touch(exist_ok=False)
    manifest.update(status="running", completed_generation_count=len(rows), resumed_generation_count=len(rows))
    _write_json(manifest_path, manifest)
    return manifest, rows


def _run_generate(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt_ids: list[int],
    enums: dict[str, list[str]],
    *,
    max_new_tokens: int,
    deadline: float,
) -> dict[str, Any]:
    from transformers import StoppingCriteria, StoppingCriteriaList

    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    stopper = F4FastPrefixStopper(
        lambda values: tokenizer.decode(list(values), skip_special_tokens=True),
        enums,
        deadline=deadline,
    )

    class PrefixCriteria(StoppingCriteria):
        def __call__(self, sequences: Any, scores: Any, **kwargs: Any) -> bool:
            generated = sequences[0, len(prompt_ids):].detach().cpu().tolist()
            return stopper.inspect(generated)

    inference.synchronize(torch)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([PrefixCriteria()]),
        )
    inference.synchronize(torch)
    elapsed = time.monotonic() - started
    generated_ids = [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
    eos = tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in generated_ids
    result = stopper.result or stopper.finish(
        "eos" if eos else "token_limit" if len(generated_ids) >= max_new_tokens else "generation_ended",
        generated_ids,
    )
    stop_reason = _reported_stop_reason(result, stopper)
    if stop_reason == "response_marker_closed" and not result.raw_prefix.endswith("\nResponse="):
        raise RuntimeError("Fast stopper did not terminate exactly at the complete F4 Response marker")
    body_token_count = response_body_token_count(
        generated_ids,
        lambda values: tokenizer.decode(list(values), skip_special_tokens=True),
    )
    return {
        "status": result.status,
        "reason": result.reason,
        "stop_reason": stop_reason,
        "raw_prefix": result.raw_prefix,
        "parsed": result.parsed,
        "generated_token_ids": list(result.generated_token_ids),
        "generated_token_count": len(result.generated_token_ids),
        "generation_seconds": elapsed,
        "eos_generated": bool(eos),
        "response_body_tokens_generated": body_token_count,
    }


def _generate_and_score(
    training_config: dict[str, Any],
    training_path: Path,
    eval_config: dict[str, Any],
    eval_path: Path,
    run_id: str,
    checkpoint_step: int,
) -> dict[str, Any]:
    training_root = _resolve(training_config["data"]["output_root"]) / run_id
    node_dir = training_root / f"step-{checkpoint_step:04d}"
    checkpoint = training_root / "trainer" / f"checkpoint-{checkpoint_step}"
    training_manifest = _read_json(training_root / "manifest.json")
    checkpoint_record = _read_json(checkpoint / "checkpoint_manifest.json")
    if training_manifest.get("optimizer_steps") != checkpoint_step:
        raise ValueError("Fast-only checkpoint step differs from the current staged-training node")
    if training_manifest.get("checkpoint_manifest_sha256") != sha256_file(checkpoint / "checkpoint_manifest.json"):
        raise ValueError("Fast-only checkpoint SHA differs from the staged-training Manifest")
    if checkpoint_record.get("run_id") != run_id or checkpoint_record.get("optimizer_step") != checkpoint_step:
        raise ValueError("Fast-only checkpoint identity or Trainer step is incorrect")
    from src.training.training_sft import _load_train_model, _verify_checkpoint

    checkpoint_identity = training_manifest.get("checkpoint_identity")
    _verify_checkpoint(checkpoint, expected=checkpoint_identity, run_id=run_id)
    adapter_identity, adapter_config = inference.adapter_weight_identity(checkpoint)
    if adapter_config.get("r") != 8 or adapter_config.get("lora_alpha") != 16:
        raise ValueError("stage checkpoint LoRA identity differs from the F4 Full contract")
    base_path = _resolve(training_config["base_model"]["local_path"])
    inference.validate_adapter_base_reference(adapter_config, training_config["base_model"]["model_id"], base_path)
    cohorts = load_target_free_cohorts(eval_config, ROOT)
    eval_config_hash = sha256_file(eval_path)
    identity = {
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
        "adapter_weight_sha256": adapter_identity["weight_sha256"],
        "adapter_config_sha256": adapter_identity["config_sha256"],
        "base_weight_sha256": training_config["base_model"]["expected_weight_sha256"],
        "eval_config_sha256": eval_config_hash,
        "training_config_sha256": training_manifest["config_sha256"],
        "training_code_sha256": training_manifest.get("checkpoint_identity", {}).get("training_code_sha256"),
        "train_sha256": training_manifest.get("checkpoint_identity", {}).get("train_sha256"),
        "f4_data_manifest_sha256": training_manifest.get("checkpoint_identity", {}).get("f4_data_manifest_sha256"),
        "tokenizer_signature_sha256": training_manifest.get("checkpoint_identity", {}).get("tokenizer_signature_sha256"),
        "chat_template_sha256": training_manifest.get("checkpoint_identity", {}).get("chat_template_sha256"),
        "evaluation_code_sha256": canonical_sha256({
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/e03_a6_f4_staged_eval.py",
                "src/evaluation/e03_a6_f4_staged_metrics.py",
                "src/evaluation/f4_pilot.py",
                "src/shared/f4_protocol.py",
                "participant/run_inference.py",
            )
        }),
        "cohorts": {
            name: {key: cohort[key] for key in ("ids_sha256", "input_sha256", "id_map_sha256", "conversation_count")}
            for name, cohort in cohorts["cohorts"].items()
        },
        "union_count": cohorts["union_count"],
        "union_ids_sha256": cohorts["union_ids_sha256"],
        "prompt_sha256": eval_config["protocol"]["prompt_sha256"],
        "p0_sha256": eval_config["protocol"]["p0_sha256"],
        "truncation_version": "T1",
        "fast_max_new_tokens": eval_config["generation"]["fast_max_new_tokens"],
    }
    output_root = _resolve(eval_config["output_root"]) / run_id / f"step-{checkpoint_step:04d}"
    manifest, predicted_rows = _prepare_eval_directory(output_root, identity, cohorts["union_ids"])
    predictions_path = output_root / "predictions.jsonl"
    if manifest.get("status") == "complete" and len(predicted_rows) == cohorts["union_count"]:
        return _read_json(output_root / "evaluation.json")

    torch = __import__("torch")
    torch.manual_seed(int(eval_config["generation"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(eval_config["generation"]["seed"]))
    model = tokenizer = None
    model_load_seconds = 0.0
    generation_seconds = 0.0
    inference_cuda_memory: dict[str, Any] = {}
    try:
        runner_config = json.loads(_resolve(training_config["protocol"]["runner_config"]).read_text(encoding="utf-8"))
        prompt_text, schema, enums, prompt_facts = inference.load_runtime_protocol(runner_config)
        selection_prompt_text, selection_facts = inference.load_f3_selection_prompt(runner_config)
        if prompt_facts.get("format_version") != "F4" or selection_facts.get("version") != "P0":
            raise ValueError("stage Fast-only did not load the frozen F4/P0 protocol")
        if prompt_facts.get("sha256") != eval_config["protocol"]["prompt_sha256"]:
            raise ValueError("runtime F4 prompt hash differs from the frozen evaluation config")
        if selection_facts.get("sha256") != eval_config["protocol"]["p0_sha256"]:
            raise ValueError("runtime P0 history-selection hash differs from the frozen evaluation config")
        tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runner_config, base_path)
        started_load = time.monotonic()
        model, torch, model_facts = _load_train_model(
            training_config,
            base_path,
            adapter_path=checkpoint,
            autocast_adapter_dtype=True,
        )
        model_load_seconds = time.monotonic() - started_load
        model.eval()
        lora_dtypes = sorted({str(p.dtype) for name, p in model.named_parameters() if ".lora_A." in name or ".lora_B." in name})
        if lora_dtypes != ["torch.float32"]:
            raise RuntimeError(f"independent checkpoint reload did not preserve FP32 LoRA weights: {lora_dtypes}")
        if not model_facts.get("real_nf4_verified") or model_facts.get("offload_devices"):
            raise RuntimeError("independent Fast-only model reload failed NF4/BF16/CUDA placement verification")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        base_before = inference._weight_identity(base_path, {"model_id": training_config["base_model"]["model_id"], "model_revision": training_config["base_model"]["revision"]})
        if base_before.get("observed_weight_hash") != training_config["base_model"]["expected_weight_sha256"]:
            raise ValueError("F4 Fast-only Base SHA differs from the frozen training identity")
        deadline = time.monotonic() + int(eval_config["generation"]["max_run_seconds"])
        indexed_predictions = {row["id"]: row for row in predicted_rows}
        for sample in cohorts["union_inputs"]:
            eval_id = str(sample["id"])
            if eval_id in indexed_predictions:
                continue
            mapping = None
            for cohort in cohorts["cohorts"].values():
                if eval_id in cohort["mapping"]:
                    mapping = cohort["mapping"][eval_id]
                    break
            if mapping is None:
                raise ValueError(f"no frozen ID Map entry for target-free input {eval_id}")
            if time.monotonic() >= deadline:
                generated = {
                    "status": "budget_exhausted", "reason": "generation_budget_exhausted_before_sample",
                    "stop_reason": "budget_exhausted", "raw_prefix": "", "parsed": None,
                    "generated_token_ids": [], "generated_token_count": 0,
                    "generation_seconds": 0.0, "eos_generated": False,
                    "response_body_tokens_generated": 0,
                }
                prompt_count = 0
                prompt_hash = None
            else:
                selection, _messages = inference.prepare_sample_messages(
                    sample, prompt_text, tokenizer,
                    int(eval_config["protocol"]["max_prompt_tokens"]),
                    selection_prompt_text=selection_prompt_text,
                )
                if selection.get("floor_exceeds_cap"):
                    raise ValueError(f"F4 P0/T1 prompt floor exceeds fixed budget: {eval_id}")
                prompt_ids = list(selection["prompt_ids"])
                generated = _run_generate(
                    model, tokenizer, torch, prompt_ids, enums,
                    max_new_tokens=int(eval_config["generation"]["fast_max_new_tokens"]),
                    deadline=deadline,
                )
                generation_seconds += float(generated["generation_seconds"])
                prompt_count = len(prompt_ids)
                prompt_hash = canonical_sha256(prompt_ids)
                if generated["status"] == "valid":
                    strict = parse_f4(generated["raw_prefix"], enums)
                    if strict != generated["parsed"]:
                        raise RuntimeError("F4 strict parser output changed after Fast-prefix stopping")
            row = {
                "id": eval_id,
                "sample_id": mapping.get("sample_id"),
                "status": generated["status"],
                "reason": generated["reason"],
                "stop_reason": generated["stop_reason"],
                "raw_prefix": generated["raw_prefix"],
                "parsed": generated["parsed"],
                "generated_token_ids": generated["generated_token_ids"],
                "generated_token_count": generated["generated_token_count"],
                "prompt_token_count": prompt_count,
                "prompt_token_ids_sha256": prompt_hash,
                "generation_seconds": generated["generation_seconds"],
                "eos_generated": generated["eos_generated"],
                "response_body_tokens_generated": generated["response_body_tokens_generated"],
            }
            _append_jsonl_fsynced(predictions_path, row)
            predicted_rows.append(row)
            indexed_predictions[eval_id] = row
        expected_identity = _prediction_identity(predicted_rows)
        if expected_identity != {"count": cohorts["union_count"], "ids_sha256": canonical_sha256(cohorts["union_ids"])}:
            raise ValueError("F4 Fast-only prediction coverage differs from the deduplicated fixed union")
        base_after = inference._weight_identity(base_path, {"model_id": training_config["base_model"]["model_id"], "model_revision": training_config["base_model"]["revision"]})
        adapter_after, _ = inference.adapter_weight_identity(checkpoint)
        if base_after.get("observed_weight_hash") != base_before.get("observed_weight_hash"):
            raise RuntimeError("Base weight SHA changed during independent Fast-only evaluation")
        if adapter_after["weight_sha256"] != adapter_identity["weight_sha256"]:
            raise RuntimeError("checkpoint Adapter SHA changed during read-only Fast-only evaluation")
        inference_cuda_memory = inference.cuda_memory_facts(torch)
        prediction_hash = sha256_file(predictions_path)
        manifest.update(
            status="predictions_frozen",
            generation_complete=True,
            generated_count=len(predicted_rows),
            generated_ids_sha256=expected_identity["ids_sha256"],
            predictions_sha256=prediction_hash,
            references_read_by_generator=False,
            response_body_tokens_generated=sum(int(row["response_body_tokens_generated"]) for row in predicted_rows),
            model_load_seconds=model_load_seconds,
            generation_seconds=generation_seconds,
            independent_checkpoint_reload="passed",
            adapter_weight_sha256_before=adapter_identity["weight_sha256"],
            adapter_weight_sha256_after=adapter_after["weight_sha256"],
            base_weight_sha256_before=base_before["observed_weight_hash"],
            base_weight_sha256_after=base_after["observed_weight_hash"],
            tokenizer_facts=tokenizer_facts,
            model_facts=model_facts,
            cuda_memory=inference_cuda_memory,
            lora_dtypes=lora_dtypes,
            generation_frozen_at_utc=_now(),
        )
        _write_json(output_root / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(
            status="blocked" if isinstance(exc, RuntimeError) else "failed_resumable",
            error=f"{type(exc).__name__}: {exc}",
            failed_at_utc=_now(),
            references_read_by_generator=False,
        )
        _write_json(output_root / "run_manifest.json", manifest)
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Gold data is first opened after the prediction file has been frozen and
    # the independently loaded model has been released from GPU memory.
    manifest = _read_json(output_root / "run_manifest.json")
    if manifest.get("status") != "predictions_frozen" or manifest.get("predictions_sha256") != sha256_file(predictions_path):
        raise ValueError("prediction journal is not frozen before reading References")
    reference_index = load_gold_references(eval_config, ROOT, cohorts["cohorts"])
    prediction_index = {row["id"]: row for row in predicted_rows}
    enums = inference.submission_enums(inference.load_submission_schema())
    metrics = {
        name: score_cohort(cohort, reference_index[name], prediction_index, enums)
        for name, cohort in cohorts["cohorts"].items()
    }
    previous_summary = None
    previous_predictions: dict[str, dict[str, Any]] = {}
    previous_stage = max((step for step in (512, 1024, 2048, 4048) if step < checkpoint_step), default=None)
    if previous_stage is not None:
        previous_dir = _resolve(eval_config["output_root"]) / run_id / f"step-{previous_stage:04d}"
        previous_summary_path = previous_dir / "evaluation.json"
        previous_predictions_path = previous_dir / "predictions.jsonl"
        if previous_summary_path.is_file() and previous_predictions_path.is_file():
            previous_summary = _read_json(previous_summary_path)
            previous_predictions = {row["id"]: row for row in _read_jsonl(previous_predictions_path)}
    transitions = {}
    for name, cohort in cohorts["cohorts"].items():
        transitions[name] = (
            compare_same_ids(previous_predictions, prediction_index, reference_index[name], cohort["ids"])
            if previous_predictions else None
        )
    baselines = historical_baselines(ROOT)
    engineering_reasons = []
    if not manifest.get("generation_complete") or len(predicted_rows) != cohorts["union_count"]:
        engineering_reasons.append("incomplete_fixed_union_coverage")
    if any(metric["fast_prefix_success_rate"] < float(eval_config["comparison"]["fast_prefix_parse_minimum"]) for metric in metrics.values()):
        engineering_reasons.append("strict_f4_prefix_parse_below_98_percent")
    if sum(int(row.get("response_body_tokens_generated", 0)) for row in predicted_rows) != 0:
        engineering_reasons.append("response_body_tokens_generated")
    if manifest.get("independent_checkpoint_reload") != "passed":
        engineering_reasons.append("independent_checkpoint_reload_failed")
    if manifest.get("base_weight_sha256_before") != manifest.get("base_weight_sha256_after"):
        engineering_reasons.append("base_weight_changed_during_evaluation")
    if manifest.get("adapter_weight_sha256_before") != manifest.get("adapter_weight_sha256_after"):
        engineering_reasons.append("adapter_checkpoint_changed_during_evaluation")
    memory = manifest.get("cuda_memory", {})
    if memory and memory.get("allocator_peak_allocated_bytes", 0) > memory.get("total_memory_bytes", 0) * float(training_config["quantization"]["max_cuda_memory_fraction"]):
        engineering_reasons.append("cuda_allocated_memory_budget_exceeded")
    engineering_gate = "passed" if not engineering_reasons else "failed"
    if engineering_gate != "passed":
        quality_status = "failed"
        recommendation = "blocked_by_hard_engineering_gate"
    elif checkpoint_step == 512:
        quality_status = "inconclusive"
        recommendation = "review_sparse-profile signal and continue only under recorded authorization"
    else:
        current_profile = metrics["dev224"]["profile_33_label"]
        if current_profile["overall_33_label_tp"] == 0 and metrics["dev224"]["nonempty_profile_prediction_count"] == 0:
            quality_status = "failed" if checkpoint_step >= 2048 else "inconclusive"
            recommendation = "near-empty profile output; stop recommendation at the 2,048-step loss-control gate" if checkpoint_step >= 2048 else "profile learning remains inconclusive"
        else:
            quality_status = "passed" if transitions.get("dev224", {}).get("profile_tp_delta", 0) > 0 and transitions.get("dev224", {}).get("profile_fp_delta", 0) <= 0 else "inconclusive"
            recommendation = "positive paired profile signal" if quality_status == "passed" else "continue interpreting TP together with FP and precision"
    resources = {
        "model_load_seconds": float(manifest.get("model_load_seconds", 0.0)),
        "generation_seconds": float(manifest.get("generation_seconds", 0.0)),
        "elapsed_model_and_generation_seconds": float(manifest.get("model_load_seconds", 0.0)) + float(manifest.get("generation_seconds", 0.0)),
        "generated_tokens_total": sum(int(row.get("generated_token_count", 0)) for row in predicted_rows),
        "peak_memory_bytes": manifest.get("cuda_memory", {}).get("allocator_peak_allocated_bytes"),
        "peak_reserved_memory_bytes": manifest.get("cuda_memory", {}).get("allocator_peak_reserved_bytes"),
        "response_boundary_closed_count": sum(row.get("stop_reason") == "response_marker_closed" for row in predicted_rows),
        "response_body_tokens_generated": sum(int(row.get("response_body_tokens_generated", 0)) for row in predicted_rows),
        "unique_generation_count": len(predicted_rows),
        "cohort_overlap_count": cohorts["overlap_count"],
        "deduplicated_union_count": cohorts["union_count"],
    }
    training_node = _read_json(training_root / f"step-{checkpoint_step:04d}" / "training.json")
    summary = {
        "status": quality_status,
        "quality_status": quality_status,
        "engineering_gate": engineering_gate,
        "engineering_gate_reasons": engineering_reasons,
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "max_optimizer_steps": 4048,
        "checkpoint_manifest_sha256": identity["checkpoint_manifest_sha256"],
        "adapter_identity": adapter_identity,
        "base_weight_sha256": training_config["base_model"]["expected_weight_sha256"],
        "config_sha256": training_manifest["config_sha256"],
        "evaluation_identity": identity,
        "training": {
            "train_rows": int(training_manifest["preflight_data"]["train_rows"]),
            "optimizer_steps": checkpoint_step,
            "samples_seen": int(training_node["training_samples_seen"]),
            "micro_batches_seen": int(training_node["micro_batches_seen"]),
            "supervised_tokens_seen": int(training_node["supervised_tokens_seen"]),
            "input_tokens_seen": int(training_node["input_tokens_seen"]),
            "loss_start": float(training_node["training_loss_start"]),
            "loss_node": float(training_node["training_loss_node"]),
            "cumulative_training_seconds": float(training_node["cumulative_training_seconds"]),
            "sample_order": training_node["sample_order"],
        },
        "tokenizer_facts": manifest.get("tokenizer_facts"),
        "model_facts": manifest.get("model_facts"),
        "cohorts": metrics,
        "same_id_transitions_from_previous_node": transitions,
        "historical_baselines": baselines,
        "previous_node_summary_loaded": previous_summary is not None,
        "resources": resources,
        "generation_protocol": {
            "fast_max_new_tokens": eval_config["generation"]["fast_max_new_tokens"],
            "greedy": True,
            "response_marker_stop": "\\nResponse=",
            "response_body_generated": resources["response_body_tokens_generated"] > 0,
            "references_read_by_generator": False,
            "prediction_sha256": sha256_file(predictions_path),
            "prediction_count": len(predicted_rows),
            "references_read_after_prediction_freeze": True,
        },
        "quality_recommendation": recommendation,
        "progression_authorization_status": "awaiting_manual_review",
        "node_metrics_individually_reviewed": False,
        "created_at_utc": _now(),
    }
    _write_json(output_root / "evaluation.json", summary)
    (output_root / "report.md").write_text(_render_node_report(summary), encoding="utf-8")
    manifest.update(
        status="complete",
        scored_at_utc=_now(),
        references_read_by_generator=False,
        references_read_after_prediction_freeze=True,
        references_sha256={name: eval_config["cohorts"][name]["expected_references_sha256"] for name in metrics},
        predictions_sha256=sha256_file(predictions_path),
        summary_sha256=sha256_file(output_root / "evaluation.json"),
        quality_status=quality_status,
        engineering_gate=engineering_gate,
        response_body_tokens_generated=resources["response_body_tokens_generated"],
    )
    _write_json(output_root / "run_manifest.json", manifest)
    training_manifest["nodes"][str(checkpoint_step)].update(
        evaluation_status=quality_status,
        evaluation_engineering_gate=engineering_gate,
        evaluation_summary_path=str((output_root / "evaluation.json").relative_to(ROOT)),
        evaluation_summary_sha256=sha256_file(output_root / "evaluation.json"),
        evaluation_predictions_sha256=sha256_file(predictions_path),
        quality_recommendation=recommendation,
        review_status="awaiting_manual_review",
        progression_authorization_status="not_recorded",
        node_metrics_individually_reviewed=False,
    )
    training_manifest["status"] = "awaiting_manual_review" if checkpoint_step < 4048 else "completed"
    training_manifest["stage"] = "awaiting_manual_review" if checkpoint_step < 4048 else "full_training_completed_awaiting_final_review"
    training_manifest["final_evaluation_status"] = quality_status if checkpoint_step == 4048 else training_manifest.get("final_evaluation_status")
    _write_json(training_root / "manifest.json", training_manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed F4 Fast-only inference on the deduplicated 160+224 development cohorts. "
            "The generator reads only target-free input/ID maps, stops at the complete Response= marker, "
            "and opens Gold References only after prediction hashes are frozen."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="frozen staged-full training YAML")
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG, help="frozen 160/224 Fast-only YAML")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True, choices=(512, 1024, 2048, 4048))
    parser.add_argument("--mode", choices=("fast-only",), required=True)
    args = parser.parse_args(argv)
    train_config, train_path = _load_config(args.config)
    eval_config, eval_path = _load_eval_config(args.eval_config)
    summary = _generate_and_score(
        train_config, train_path, eval_config, eval_path,
        args.run_id, args.checkpoint_step,
    )
    print(json.dumps({
        "status": summary["quality_status"],
        "engineering_gate": summary["engineering_gate"],
        "run_id": args.run_id,
        "checkpoint_step": args.checkpoint_step,
        "summary_path": str((_resolve(eval_config["output_root"]) / args.run_id / f"step-{args.checkpoint_step:04d}" / "evaluation.json")),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["engineering_gate"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
