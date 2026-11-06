#!/usr/bin/env python3
"""Generate and score fixed, target-free A8 classification cohorts per checkpoint."""

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
from src.data.e03_a8_sft import canonical_sha256, select_common_history
from src.evaluation.e03_a6_f4_staged_metrics import (
    compare_same_ids,
    historical_baselines,
    _conversation_id,
    load_gold_references,
    load_target_free_cohorts,
    sha256_file,
)
from src.evaluation.e03_a8_staged_metrics import bootstrap_conversation_micro_f1, score_a8_cohort
from src.shared.e03_a8_protocol import A8ParseError, parse_specialist_output, validate_a8_prompt_enum_order
from src.shared.prompting import select_recent_complete_suffix
from src.training.e03_a8_contract import (
    AUTHORIZED_MAX_STEP,
    EVALUATION_CODE_PATHS,
    STAGE_STEPS,
    TRAINING_CODE_PATHS,
    canonical_sha256 as contract_canonical_sha256,
)
from src.training.e03_a8_staged import (
    DEFAULT_CONFIGS,
    _base_path,
    _code_identity,
    _load_config as _load_training_config,
    _read_json,
    _read_jsonl,
    _resolve,
    _write_json,
)

EVAL_CONFIG_PATH = ROOT / "configs/evaluation/e03_a8_staged_fast.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl_fsynced(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_eval_config(path: Path = EVAL_CONFIG_PATH) -> tuple[dict[str, Any], Path]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E03-A8-staged-fast-eval-v1" or config.get("experiment_id") != "E03-A8":
        raise ValueError("unsupported A8 staged evaluation config")
    if config.get("comparison") != {
        "minimum_strict_parse_rate": 0.98,
        "fixed_denominator": True,
        "deduplicate_union_generation": True,
        "score_failures_as_empty": True,
        "references_read_only_after_prediction_freeze": True,
        "bootstrap_replicates": 1000,
    }:
        raise ValueError("A8 evaluation fixed-denominator/quality contract changed")
    return config, resolved


def _eval_root(eval_config: dict[str, Any], arm: str, run_id: str, step: int) -> Path:
    root = _resolve(eval_config["output_root"]) / arm / run_id / f"step-{step:04d}"
    if not root.is_relative_to(_resolve(eval_config["output_root"])):
        raise ValueError("A8 evaluation output path escapes its namespace")
    return root


def _conversation_for_id(cohorts: dict[str, Any], eval_id: str) -> dict[str, Any]:
    for cohort in cohorts["cohorts"].values():
        if eval_id in cohort["mapping"]:
            return cohort["mapping"][eval_id]
    raise ValueError(f"A8 target-free eval ID has no frozen ID Map row: {eval_id}")


def _generate_rows(
    *,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    run_id: str,
    step: int,
    cohorts: dict[str, Any],
    tokenizer: Any,
    model: Any,
    p_prompt: str,
    ep_prompt: str,
    p0_prompt: str,
    schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    arm = config["arm"]
    started_all = time.monotonic()
    deadline = started_all + int(eval_config["generation"]["max_run_seconds"])
    generated = []
    generation_seconds = 0.0
    generation_latencies = []
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    eos_id = int(tokenizer.eos_token_id)
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    for index, sample in enumerate(cohorts["union_inputs"], 1):
        eval_id = str(sample["id"])
        mapping = _conversation_for_id(cohorts, eval_id)
        history = sample.get("history")
        p0_selection = select_recent_complete_suffix(
            history,
            p0_prompt,
            int(eval_config["protocol"]["max_prompt_tokens"]),
            tokenizer,
        )
        if p0_selection.get("status") != "ok":
            raise ValueError(f"{eval_id}: P0/T1 current-user floor exceeds the frozen prompt cap")
        common = select_common_history(
            history,
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            tokenizer=tokenizer,
            initial_start_index=int(p0_selection["selected_start_index"]),
            prompt_cap=int(eval_config["protocol"]["max_prompt_tokens"]),
        )
        prompt_ids = list(common["prompt_ids"][arm])
        if len(prompt_ids) > 1280:
            raise ValueError(f"{eval_id}/{arm}: selected prompt exceeds P=1280")
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=model.get_input_embeddings().weight.device)
        attention = torch.ones_like(prompt_tensor)
        status = "invalid"
        reason = None
        parsed = None
        raw_text = ""
        eos_generated = False
        generated_ids: list[int] = []
        if time.monotonic() >= deadline:
            raise TimeoutError(f"A8 evaluation exceeded {eval_config['generation']['max_run_seconds']} seconds")
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                input_ids=prompt_tensor,
                attention_mask=attention,
                max_new_tokens=int(eval_config["generation"]["max_new_tokens"]),
                do_sample=False,
                num_beams=1,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=eos_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        generation_seconds += elapsed
        generation_latencies.append(elapsed)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"A8 evaluation exceeded {eval_config['generation']['max_run_seconds']} seconds")
        generated_ids = [int(value) for value in output[0, prompt_tensor.shape[1]:].detach().cpu().tolist()]
        eos_positions = [position for position, token_id in enumerate(generated_ids) if token_id == eos_id]
        eos_generated = bool(eos_positions)
        if eos_positions:
            eos_position = eos_positions[0]
            if eos_position != len(generated_ids) - 1:
                reason = "tokens_after_eos"
            visible_ids = generated_ids[:eos_position]
        else:
            visible_ids = generated_ids
            reason = "missing_eos_or_truncated"
        forbidden_specials = [token for token in visible_ids if token in special_ids]
        raw_text = tokenizer.decode(
            visible_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if reason is None and forbidden_specials:
            reason = "unexpected_special_token"
        if reason is None:
            try:
                parsed = parse_specialist_output(raw_text, schema, arm)
                status = "valid"
            except A8ParseError as error:
                reason = error.code
        generated.append({
            "id": eval_id,
            "sample_id": mapping.get("sample_id"),
            "conversation_id": _conversation_id(mapping),
            "status": status,
            "reason": reason,
            "parsed": parsed,
            "raw_text": raw_text,
            "generated_token_count": len(generated_ids),
            "eos_generated": eos_generated,
            "stop_reason": "eos" if eos_generated else "token_limit_or_generation_end",
            "selected_start_index": common["selected_start_index"],
            "p0_selected_start_index": p0_selection["selected_start_index"],
            "extra_removed_messages": common["extra_removed_messages"],
            "selected_history_sha256": common["selected_history_sha256"],
            "p_prompt_tokens": common["prompt_lengths"]["P"],
            "ep_prompt_tokens": common["prompt_lengths"]["EP"],
            "generation_seconds": elapsed,
        })
        if index % 32 == 0 or index == cohorts["union_count"]:
            print(f"[E03-A8 eval {arm} step={step}] generated {index}/{cohorts['union_count']} valid={sum(row['status']=='valid' for row in generated)}", flush=True)
    if len(generated) != int(cohorts["union_count"]):
        raise RuntimeError("A8 evaluation did not generate the complete deduplicated cohort union")
    facts = {
        "generation_count": len(generated),
        "generation_seconds": generation_seconds,
        "elapsed_generation_loop_seconds": time.monotonic() - started_all,
        "generation_latency_seconds": {
            "p50": sorted(generation_latencies)[int(0.50 * (len(generation_latencies) - 1))],
            "p90": sorted(generation_latencies)[int(0.90 * (len(generation_latencies) - 1))],
            "p99": sorted(generation_latencies)[int(0.99 * (len(generation_latencies) - 1))],
            "max": max(generation_latencies, default=0.0),
        },
        "generated_token_count": sum(int(row["generated_token_count"]) for row in generated),
        "eos_completion_count": sum(bool(row["eos_generated"]) for row in generated),
    }
    return generated, facts


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# E03-A8 {summary['arm']} — Step {summary['checkpoint_step']}",
        "",
        f"- Run ID：`{summary['run_id']}`",
        f"- Checkpoint Manifest SHA-256：`{summary['checkpoint_manifest_sha256']}`",
        f"- 画像质量门：`{summary['quality_decision']['decision']}`（{summary['quality_decision']['reason']}）",
        f"- 授权状态：`{summary['progression_authorization_status']}`",
        f"- E01-B Fast-only 基线画像 Micro-F1：`{summary['historical_baselines'].get('e01_b_full_224', {}).get('profile_micro_f1', 'not_available')}`",
        "",
        "| Cohort | 有效格式 | TP / FP / FN | Precision / Recall / Micro-F1 | 非空画像 | EOS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("pilot160", "dev224"):
        item = summary["cohorts"][name]
        profile = item["profile_33_label"]
        lines.append(
            f"| {name} | {item['valid_count']} / {item['fixed_denominator_count']} | "
            f"{profile['overall_33_label_tp']} / {profile['overall_33_label_fp']} / {profile['overall_33_label_fn']} | "
            f"{profile['overall_33_label_micro_precision']:.4f} / {profile['overall_33_label_micro_recall']:.4f} / {profile['overall_33_label_micro_f1']:.4f} | "
            f"{item['nonempty_profile_prediction_count']} | {item['eos_completion_count']} / {item['fixed_denominator_count']} |"
        )
    lines += [
        "",
        f"- Train 唯一样本数：`{summary['training']['samples_seen']}`；Completion Token 数（累计到该节点）：`{summary['training']['completion_tokens_seen']}`",
        f"- P50/P90/P99 单样本生成秒数：`{summary['resources']['generation_latency_seconds']['p50']:.3f}` / `{summary['resources']['generation_latency_seconds']['p90']:.3f}` / `{summary['resources']['generation_latency_seconds']['p99']:.3f}`",
        f"- 预测 JSONL SHA-256：`{summary['prediction_sha256']}`",
        f"- Conversation bootstrap Dev224 Micro-F1 95% 区间：`{summary['bootstrap']['dev224']['micro_f1_2_5_percentile']:.4f}` – `{summary['bootstrap']['dev224']['micro_f1_97_5_percentile']:.4f}`（探索性）",
        "",
        "逐 Cohort、逐维度与逐标签统计见 `evaluation.json`；本地画像指标不等同于官方赛事评分。",
    ]
    return "\n".join(lines) + "\n"


def evaluate_checkpoint(training_config_path: Path, arm: str, run_id: str, step: int, eval_config_path: Path = EVAL_CONFIG_PATH) -> dict[str, Any]:
    if step not in STAGE_STEPS:
        raise ValueError(f"A8 evaluation step must be one of {STAGE_STEPS}")
    training_config, _training_path = _load_training_config(training_config_path, arm)
    eval_config, eval_path = _load_eval_config(eval_config_path)
    training_root = _resolve(training_config["data"]["output_root"]) / run_id
    training_manifest_path = training_root / "manifest.json"
    training_manifest = _read_json(training_manifest_path)
    checkpoint = training_root / "trainer" / f"checkpoint-{step}"
    checkpoint_path = checkpoint / "checkpoint_manifest.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"A8 checkpoint is missing: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint_path)
    node = training_manifest.get("nodes", {}).get(str(step), {})
    if (
        training_manifest.get("run_id") != run_id
        or training_manifest.get("arm") != arm
        or training_manifest.get("optimizer_steps") != step
        or training_manifest.get("config_sha256") != contract_canonical_sha256(training_config)
        or training_manifest.get("checkpoint_manifest_sha256") != checkpoint_sha
        or node.get("training_status") != "passed"
        or node.get("checkpoint_manifest_sha256") != checkpoint_sha
    ):
        raise ValueError("A8 evaluation Run/Arm/step/checkpoint identity differs")
    checkpoint_record = _read_json(checkpoint_path)
    if checkpoint_record.get("run_id") != run_id or checkpoint_record.get("arm") != arm or checkpoint_record.get("optimizer_step") != step:
        raise ValueError("A8 checkpoint directory and identity differ")
    if checkpoint_record.get("config_sha256") != training_manifest.get("config_sha256") or checkpoint_record.get("dataset_manifest_sha256") != training_manifest.get("dataset_manifest_sha256"):
        raise ValueError("A8 checkpoint input identity differs from the training Run")
    training_code = _code_identity(TRAINING_CODE_PATHS)
    evaluation_code = _code_identity(EVALUATION_CODE_PATHS)
    if training_code != training_manifest.get("training_code_sha256") or evaluation_code != training_manifest.get("evaluation_code_sha256"):
        raise ValueError("A8 training or evaluation code changed after the Run was frozen")
    if training_manifest.get("preflight_sha256") != sha256_file(training_root / "preflight.json"):
        raise ValueError("A8 CPU preflight file changed after training")
    from src.training.e03_a8_staged import _verify_checkpoint

    _verify_checkpoint(
        checkpoint,
        {
            "config_sha256": training_manifest["config_sha256"],
            "dataset_manifest_sha256": training_manifest["dataset_manifest_sha256"],
            "sample_order_sha256": training_manifest["sample_order_sha256"],
            "training_code_sha256": training_manifest["training_code_sha256"],
        },
        run_id,
        arm,
    )
    cohorts = load_target_free_cohorts(eval_config, ROOT)
    output_root = _eval_root(eval_config, arm, run_id, step)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite A8 evaluation artifacts: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    prompt_path = _resolve(training_config["protocol"]["prompt_path"])
    ep_prompt = (ROOT / "configs/prompts/p_a8_emotion_profile.txt").read_text(encoding="utf-8")
    p_prompt = (ROOT / "configs/prompts/p_a8_profile.txt").read_text(encoding="utf-8")
    p0_path = _resolve(eval_config["protocol"]["p0_path"])
    p0_prompt = p0_path.read_text(encoding="utf-8")
    schema_path = ROOT / "data/train/train_public.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enums = validate_a8_prompt_enum_order(p_prompt, schema, "P")
    validate_a8_prompt_enum_order(ep_prompt, schema, "EP")
    if sha256_file(prompt_path) != training_config["protocol"]["prompt_sha256"] or sha256_file(p0_path) != eval_config["protocol"]["p0_sha256"]:
        raise ValueError("A8 prompt/P0 identity changed before evaluation")
    if sha256_file(schema_path) != "a9467e520b06a96859acaa2ff022e395d08ff6ad5e8ae3532a22cc24c630cada":
        raise ValueError("A8 official Train Schema differs from the frozen repository baseline")
    base_path = _base_path(training_config)
    base_before = inference._weight_identity(base_path, {
        "model_id": training_config["base_model"]["model_id"],
        "model_revision": training_config["base_model"]["revision"],
    })
    if base_before.get("observed_weight_hash") != training_config["base_model"]["expected_weight_sha256"]:
        raise ValueError("A8 eval Base SHA differs from the frozen training Base")
    runner_config = _read_json(_resolve(training_config["protocol"]["runner_config"]))
    tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runner_config)
    started_load = time.monotonic()
    from src.training.training_sft import _load_train_model

    model, torch, model_facts = _load_train_model(
        training_config,
        base_path,
        adapter_path=checkpoint,
        autocast_adapter_dtype=bool(training_config["evaluation"]["autocast_adapter_dtype"]),
    )
    model_load_seconds = time.monotonic() - started_load
    model.eval()
    dtypes = sorted({str(parameter.dtype) for name, parameter in model.named_parameters() if ".lora_A." in name or ".lora_B." in name})
    if dtypes != ["torch.float32"]:
        raise RuntimeError(f"A8 independent evaluation reload changed LoRA dtype: {dtypes}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adapter_identity, _adapter_config = inference.adapter_weight_identity(checkpoint)
    predictions: list[dict[str, Any]] = []
    try:
        torch.manual_seed(int(eval_config["generation"]["seed"]))
        torch.cuda.manual_seed_all(int(eval_config["generation"]["seed"]))
        predictions, generation_facts = _generate_rows(
            config=training_config,
            eval_config=eval_config,
            run_id=run_id,
            step=step,
            cohorts=cohorts,
            tokenizer=tokenizer,
            model=model,
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            p0_prompt=p0_prompt,
            schema=schema,
        )
        generation_facts["model_load_seconds"] = model_load_seconds
        prediction_path = output_root / "predictions.jsonl"
        _append_jsonl_fsynced(prediction_path, predictions)
        prediction_sha = sha256_file(prediction_path)
        prediction_index = {row["id"]: row for row in predictions}
        if len(prediction_index) != len(predictions) or set(prediction_index) != set(cohorts["union_ids"]):
            raise RuntimeError("A8 predictions do not cover the frozen unique evaluation ID union")
        generation_manifest = {
            "status": "predictions_frozen",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "prediction_count": len(predictions),
            "prediction_sha256": prediction_sha,
            "references_read": False,
            "created_at_utc": _now(),
        }
        _write_json(output_root / "generation_manifest.json", generation_manifest, exclusive=True)
        # References are deliberately opened only after predictions have a frozen hash.
        reference_index = load_gold_references(eval_config, ROOT, cohorts["cohorts"])
        metrics = {
            name: score_a8_cohort(cohort, reference_index[name], prediction_index, enums, arm=arm)
            for name, cohort in cohorts["cohorts"].items()
        }
        previous_step = max((candidate for candidate in STAGE_STEPS if candidate < step), default=None)
        previous_summary = None
        previous_predictions = {}
        if previous_step is not None:
            previous_dir = _eval_root(eval_config, arm, run_id, previous_step)
            previous_summary_path = previous_dir / "evaluation.json"
            previous_predictions_path = previous_dir / "predictions.jsonl"
            if previous_summary_path.is_file() and previous_predictions_path.is_file():
                previous_summary = _read_json(previous_summary_path)
                previous_predictions = {row["id"]: row for row in _read_jsonl(previous_predictions_path)}
        transitions = {}
        for name, cohort in cohorts["cohorts"].items():
            transitions[name] = compare_same_ids(previous_predictions, prediction_index, reference_index[name], cohort["ids"]) if previous_predictions else None
        bootstrap = {
            name: bootstrap_conversation_micro_f1(
                cohort,
                reference_index[name],
                prediction_index,
                enums,
                replicates=int(eval_config["comparison"]["bootstrap_replicates"]),
                seed=int(eval_config["generation"]["seed"]),
            )
            for name, cohort in cohorts["cohorts"].items()
        }
        base_after = inference._weight_identity(base_path, {
            "model_id": training_config["base_model"]["model_id"],
            "model_revision": training_config["base_model"]["revision"],
        })
        engineering_reasons = []
        if len(predictions) != cohorts["union_count"]:
            engineering_reasons.append("fixed_union_coverage_incomplete")
        if any(metrics[name]["strict_parse_rate"] < float(eval_config["comparison"]["minimum_strict_parse_rate"]) for name in metrics):
            engineering_reasons.append("strict_parse_rate_below_98_percent")
        if any(metrics[name]["eos_completion_rate"] < float(eval_config["comparison"]["minimum_strict_parse_rate"]) for name in metrics):
            engineering_reasons.append("eos_completion_rate_below_98_percent")
        if model_facts.get("offload_devices") or not model_facts.get("real_nf4_verified"):
            engineering_reasons.append("nf4_cuda_placement_or_offload_gate_failed")
        if base_before.get("observed_weight_hash") != base_after.get("observed_weight_hash"):
            engineering_reasons.append("base_weight_changed_during_evaluation")
        reload_check = _read_json(training_root / f"step-{step:04d}" / "training.json").get("independent_reload_check", {})
        if reload_check.get("status") != "passed":
            engineering_reasons.append("independent_checkpoint_reload_failed")
        if sum(int(row.get("generated_token_count", 0)) for row in predictions) != generation_facts["generated_token_count"]:
            engineering_reasons.append("generation_token_ledger_mismatch")
        engineering_gate = "passed" if not engineering_reasons else "failed"
        # Build the current summary shell before applying the pre-authorized gate.
        training_node = _read_json(training_root / f"step-{step:04d}" / "training.json")
        baseline = historical_baselines(ROOT)
        summary = {
            "status": "complete" if engineering_gate == "passed" else "engineering_failed",
            "experiment_id": "E03-A8",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "samples_seen": step * 4,
            "authorized_max_step": AUTHORIZED_MAX_STEP,
            "engineering_gate": engineering_gate,
            "engineering_gate_reasons": engineering_reasons,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "adapter_identity": adapter_identity,
            "base_weight_sha256": base_after.get("observed_weight_hash"),
            "config_sha256": training_manifest["config_sha256"],
            "dataset_manifest_sha256": training_manifest["dataset_manifest_sha256"],
            "sample_order_sha256": training_manifest["sample_order_sha256"],
            "evaluation_config_sha256": sha256_file(eval_path),
            "evaluation_code_sha256": evaluation_code,
            "cohorts": metrics,
            "same_id_transitions_from_previous_node": transitions,
            "historical_baselines": baseline,
            "bootstrap": bootstrap,
            "resources": {
                **generation_facts,
                "peak_memory_bytes": inference.cuda_memory_facts(torch).get("allocator_peak_allocated_bytes"),
                "peak_reserved_memory_bytes": inference.cuda_memory_facts(torch).get("allocator_peak_reserved_bytes"),
                "deduplicated_union_count": cohorts["union_count"],
                "cohort_overlap_count": cohorts["overlap_count"],
            },
            "prediction_sha256": prediction_sha,
            "prediction_count": len(predictions),
            "references_read_by_generator": False,
            "references_read_after_prediction_freeze": True,
            "references_sha256": {name: eval_config["cohorts"][name]["expected_references_sha256"] for name in metrics},
            "training": {
                "train_rows": 16192,
                "samples_seen": step * 4,
                "completion_tokens_seen": sum(int(row.get("completion_tokens", 0)) for row in _read_jsonl(training_root / "step_ledger.jsonl")),
                "loss_start": _read_jsonl(training_root / "step_ledger.jsonl")[0]["mean_completion_loss"],
                "loss_node": training_node["last_loss"],
                "cumulative_training_seconds": training_node["cumulative_training_seconds"],
                "initial_lora_tensor_sha256": training_manifest["initial_lora_tensor_sha256"],
            },
            "tokenizer_facts": tokenizer_facts,
            "model_facts": model_facts,
            "generation_protocol": {
                "max_new_tokens": int(eval_config["generation"]["max_new_tokens"]),
                "greedy": True,
                "eos_required": True,
                "references_read_by_generator": False,
                "common_history_selection": True,
                "union_count": cohorts["union_count"],
                "union_ids_sha256": cohorts["union_ids_sha256"],
                "prediction_sha256": prediction_sha,
            },
            "created_at_utc": _now(),
        }
        from src.training.e03_a8_contract import conditional_quality_decision

        quality = conditional_quality_decision(summary, previous_summary)
        if step == AUTHORIZED_MAX_STEP and quality.get("decision") == "conditionally_authorized":
            quality = {"decision": "awaiting_manual_review", "reason": "user_authorized_step_cap_reached", "authorized_max_step": AUTHORIZED_MAX_STEP}
        summary["quality_decision"] = quality
        summary["progression_authorization_status"] = "awaiting_conditional_receipt" if quality.get("decision") == "conditionally_authorized" else "awaiting_manual_review"
        summary["node_metrics_individually_reviewed"] = False
        summary["quality_recommendation"] = quality.get("reason")
        summary["status"] = "complete" if engineering_gate == "passed" else "engineering_failed"
        summary_path = output_root / "evaluation.json"
        _write_json(summary_path, summary, exclusive=True)
        (output_root / "report.md").write_text(_render_report(summary), encoding="utf-8")
        stage_manifest = {
            "status": "complete",
            "run_id": run_id,
            "arm": arm,
            "step": step,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "evaluation_summary_sha256": sha256_file(summary_path),
            "predictions_sha256": prediction_sha,
            "evaluation_config_sha256": sha256_file(eval_path),
            "evaluation_code_sha256": evaluation_code,
            "engineering_gate": engineering_gate,
            "quality_decision": quality,
            "created_at_utc": _now(),
        }
        _write_json(output_root / "stage_manifest.json", stage_manifest, exclusive=True)
        generation_manifest.update(
            status="complete",
            references_read_after_prediction_freeze=True,
            summary_sha256=sha256_file(summary_path),
            quality_decision=quality,
            engineering_gate=engineering_gate,
        )
        _write_json(output_root / "generation_manifest.json", generation_manifest)
        training_manifest = _read_json(training_manifest_path)
        training_manifest["nodes"][str(step)].update(
            evaluation_status=summary["status"],
            evaluation_engineering_gate=engineering_gate,
            evaluation_summary_path=str(summary_path.relative_to(ROOT)),
            evaluation_summary_sha256=sha256_file(summary_path),
            evaluation_predictions_sha256=prediction_sha,
            quality_decision=quality,
            authorization_status=summary["progression_authorization_status"],
            evaluation_code_sha256=evaluation_code,
        )
        training_manifest["last_evaluation_summary_path"] = str(summary_path.relative_to(ROOT))
        training_manifest["last_evaluation_summary_sha256"] = sha256_file(summary_path)
        if quality.get("decision") == "conditionally_authorized":
            training_manifest.update(status="awaiting_conditional_authorization", stage="quality_gate_passed_awaiting_receipt")
        else:
            training_manifest.update(status="awaiting_manual_review", stage="evaluation_stop_for_review")
        _write_json(training_manifest_path, training_manifest)
        return summary
    except BaseException as exc:
        _write_json(output_root / "evaluation_failure.json", {
            "status": "failed",
            "run_id": run_id,
            "arm": arm,
            "checkpoint_step": step,
            "error": f"{type(exc).__name__}: {exc}",
            "created_at_utc": _now(),
        })
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_paired_nodes(p_run_id: str, ep_run_id: str, step: int, eval_config_path: Path = EVAL_CONFIG_PATH) -> dict[str, Any]:
    eval_config, _ = _load_eval_config(eval_config_path)
    p_dir = _eval_root(eval_config, "P", p_run_id, step)
    ep_dir = _eval_root(eval_config, "EP", ep_run_id, step)
    p_summary, ep_summary = _read_json(p_dir / "evaluation.json"), _read_json(ep_dir / "evaluation.json")
    p_predictions = {row["id"]: row for row in _read_jsonl(p_dir / "predictions.jsonl")}
    ep_predictions = {row["id"]: row for row in _read_jsonl(ep_dir / "predictions.jsonl")}
    if p_summary.get("checkpoint_step") != step or ep_summary.get("checkpoint_step") != step:
        raise ValueError("A8 paired comparison requires the same checkpoint step")
    if set(p_predictions) != set(ep_predictions):
        raise ValueError("A8 paired evaluation predictions do not cover the same union IDs")
    for eval_id in set(p_predictions) & set(ep_predictions):
        if p_predictions[eval_id].get("selected_history_sha256") != ep_predictions[eval_id].get("selected_history_sha256"):
            raise ValueError(f"A8 paired evaluation selected different common histories for {eval_id}")
    metric_deltas = {}
    for cohort in ("pilot160", "dev224"):
        p = p_summary["cohorts"][cohort]["profile_33_label"]
        ep = ep_summary["cohorts"][cohort]["profile_33_label"]
        metric_deltas[cohort] = {
            "ep_minus_p": {
                key: ep[key] - p[key]
                for key in (
                    "overall_33_label_tp", "overall_33_label_fp", "overall_33_label_fn",
                    "overall_33_label_micro_precision", "overall_33_label_micro_recall", "overall_33_label_micro_f1",
                )
            },
            "p": p,
            "ep": ep,
            "p_emotion": "N/A",
            "ep_emotion": ep_summary["cohorts"][cohort]["emotion"],
        }
    result = {
        "status": "complete",
        "experiment_id": "E03-A8",
        "step": step,
        "p_run_id": p_run_id,
        "ep_run_id": ep_run_id,
        "p_summary_sha256": sha256_file(p_dir / "evaluation.json"),
        "ep_summary_sha256": sha256_file(ep_dir / "evaluation.json"),
        "same_union_ids": set(p_predictions) == set(ep_predictions),
        "common_history_verified": True,
        "metrics": metric_deltas,
        "created_at_utc": _now(),
    }
    out = _resolve(eval_config["output_root"]) / "comparison" / f"step-{step:04d}"
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "paired_comparison.json", result, exclusive=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate or pair-compare E03-A8 staged classification checkpoints.")
    parser.add_argument("--mode", choices=("evaluate", "compare"), required=True)
    parser.add_argument("--arm", choices=("P", "EP"))
    parser.add_argument("--run-id")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--p-run-id")
    parser.add_argument("--ep-run-id")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.mode == "evaluate":
        if args.arm is None or args.run_id is None:
            parser.error("evaluate requires --arm and --run-id")
        result = evaluate_checkpoint(args.training_config or DEFAULT_CONFIGS[args.arm], args.arm, args.run_id, args.step, args.eval_config)
    else:
        if args.p_run_id is None or args.ep_run_id is None:
            parser.error("compare requires --p-run-id and --ep-run-id")
        result = compare_paired_nodes(args.p_run_id, args.ep_run_id, args.step, args.eval_config)
    print(json.dumps({"status": result.get("status"), "arm": result.get("arm"), "step": result.get("checkpoint_step", result.get("step")), "quality_decision": result.get("quality_decision")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
