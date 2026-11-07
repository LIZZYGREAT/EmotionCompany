#!/usr/bin/env python3
"""Generate and score the fixed A8-V2 Pilot160 and Dev224 cohorts."""

from __future__ import annotations

import gc
import json
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
from scripts.e03_a8_staged_eval import _append_jsonl_fsynced, _generate_rows, _render_report
from src.data.e03_a8_sft import canonical_sha256
from src.evaluation.e03_a6_f4_staged_metrics import (
    historical_baselines,
    load_gold_references,
    load_target_free_cohorts,
    selected_ids_sha256,
    sha256_file,
)
from src.evaluation.e03_a8_v2_staged_metrics import bootstrap_conversation_micro_f1, score_a8_cohort
from src.shared.e03_a8_protocol import validate_a8_prompt_enum_order
from src.training.e03_a8_staged import _read_json, _read_jsonl, _write_json
from src.training.e03_a8_v2_staged import (
    DEFAULT_CONFIGS,
    V2_EVALUATION_CODE_PATHS,
    V2_TRAINING_CODE_PATHS,
    _base_identity,
    _base_path,
    _code_identity,
    _config_sha256,
    _load_config as _load_training_config,
    _resolve,
    _runtime_tokenizer,
    _validate_prepared_data,
    _verify_v2_checkpoint,
)

EVAL_CONFIG_PATH = ROOT / "configs/evaluation/e03_a8_v2_staged_fast.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_eval_config(path: Path = EVAL_CONFIG_PATH) -> tuple[dict[str, Any], Path]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E03-A8-V2-staged-fast-eval-v1" or config.get("experiment_id") != "E03-A8-V2":
        raise ValueError("unsupported A8-V2 evaluation config")
    if config.get("comparison") != {
        "minimum_strict_parse_rate": 0.98,
        "fixed_denominator": True,
        "deduplicate_union_generation": True,
        "score_failures_as_empty": True,
        "references_read_only_after_prediction_freeze": True,
        "bootstrap_replicates": 1000,
    }:
        raise ValueError("A8-V2 fixed denominator or scoring contract changed")
    return config, resolved


def _eval_root(eval_config: dict[str, Any], arm: str, run_id: str, step: int) -> Path:
    root = _resolve(eval_config["output_root"]) / arm / run_id / f"step-{step:04d}"
    if not root.is_relative_to(_resolve(eval_config["output_root"])):
        raise ValueError("A8-V2 evaluation output path escapes its namespace")
    return root


def _validate_run(training_config: dict[str, Any], run_id: str, step: int) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    if step not in {512, 1024} or (step == 1024 and training_config["arm"] != "P"):
        raise ValueError("A8-V2 supports Step 512, plus the explicitly authorized P Step 1024 continuation")
    training_root = _resolve(training_config["data"]["output_root"]) / run_id
    training_manifest_path = training_root / "manifest.json"
    training_manifest = _read_json(training_manifest_path)
    checkpoint = training_root / "trainer" / f"checkpoint-{step}"
    checkpoint_path = checkpoint / "checkpoint_manifest.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"A8-V2 checkpoint is missing: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint_path)
    node = training_manifest.get("nodes", {}).get(str(step), {})
    if (
        training_manifest.get("run_id") != run_id
        or training_manifest.get("arm") != training_config["arm"]
        or training_manifest.get("optimizer_steps") != step
        or training_manifest.get("config_sha256") != _config_sha256(training_config)
        or training_manifest.get("checkpoint_manifest_sha256") != checkpoint_sha
        or node.get("training_status") != "passed"
        or node.get("checkpoint_manifest_sha256") != checkpoint_sha
    ):
        raise ValueError("A8-V2 Run/Arm/Step/checkpoint identity differs")
    record = _read_json(checkpoint_path)
    if record.get("run_id") != run_id or record.get("arm") != training_config["arm"] or record.get("optimizer_step") != step:
        raise ValueError("A8-V2 checkpoint directory and identity differ")
    if record.get("config_sha256") != training_manifest.get("config_sha256") or record.get("dataset_manifest_sha256") != training_manifest.get("dataset_manifest_sha256"):
        raise ValueError("A8-V2 checkpoint input identity differs from its Run")
    current_training_code = _code_identity(V2_TRAINING_CODE_PATHS)
    current_evaluation_code = _code_identity(V2_EVALUATION_CODE_PATHS)
    if current_training_code != training_manifest.get("training_code_sha256") or current_evaluation_code != training_manifest.get("evaluation_code_sha256"):
        raise ValueError("A8-V2 training or evaluation code changed after Run freeze")
    preflight_path = training_root / "preflight.json"
    if training_manifest.get("preflight_sha256") != sha256_file(preflight_path):
        raise ValueError("A8-V2 preflight changed after training")
    data_facts = _validate_prepared_data(training_config)
    if data_facts["dataset_manifest_sha256"] != training_manifest.get("dataset_manifest_sha256"):
        raise ValueError("A8-V2 dataset identity differs from the Run")
    _verify_v2_checkpoint(checkpoint, {
        "config_sha256": training_manifest["config_sha256"],
        "dataset_manifest_sha256": training_manifest["dataset_manifest_sha256"],
        "sample_order_sha256": training_manifest["sample_order_sha256"],
        "training_code_sha256": training_manifest["training_code_sha256"],
    })
    return training_manifest, data_facts, training_root, checkpoint


def evaluate_checkpoint(
    training_config_path: Path,
    arm: str,
    run_id: str,
    step: int,
    eval_config_path: Path = EVAL_CONFIG_PATH,
) -> dict[str, Any]:
    training_config, _training_path = _load_training_config(training_config_path, arm)
    eval_config, eval_path = _load_eval_config(eval_config_path)
    training_manifest, data_facts, training_root, checkpoint = _validate_run(training_config, run_id, step)
    checkpoint_sha = sha256_file(checkpoint / "checkpoint_manifest.json")
    cohorts = load_target_free_cohorts(eval_config, ROOT)
    output_root = _eval_root(eval_config, arm, run_id, step)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite A8-V2 evaluation: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    p_prompt_path = ROOT / "configs/prompts/p_a8_v2_profile.txt"
    ep_prompt_path = ROOT / "configs/prompts/p_a8_v2_emotion_profile.txt"
    p_prompt, ep_prompt = p_prompt_path.read_text(encoding="utf-8"), ep_prompt_path.read_text(encoding="utf-8")
    prompt_path = _resolve(training_config["protocol"]["prompt_path"])
    p0_path = _resolve(eval_config["protocol"]["p0_path"])
    schema_path = ROOT / "data/train/train_public.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enums = validate_a8_prompt_enum_order(p_prompt, schema, "P")
    validate_a8_prompt_enum_order(ep_prompt, schema, "EP")
    if sha256_file(prompt_path) != training_config["protocol"]["prompt_sha256"] or sha256_file(p0_path) != eval_config["protocol"]["p0_sha256"]:
        raise ValueError("A8-V2 or protected P0 Prompt identity changed before generation")
    if sha256_file(schema_path) != "a9467e520b06a96859acaa2ff022e395d08ff6ad5e8ae3532a22cc24c630cada":
        raise ValueError("official Train Schema differs from the protected repository baseline")
    base_path = _base_path(training_config)
    base_before = _base_identity(training_config)
    if base_before.get("observed_weight_hash") != training_config["base_model"]["expected_weight_sha256"]:
        raise ValueError("A8-V2 evaluation Base SHA differs from the training Base")
    tokenizer, tokenizer_facts = _runtime_tokenizer(training_config)
    model = None
    try:
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
        adapter_identity, _adapter_config = inference.adapter_weight_identity(checkpoint)
        torch.manual_seed(int(eval_config["generation"]["seed"]))
        torch.cuda.manual_seed_all(int(eval_config["generation"]["seed"]))
        generated, generation_facts = _generate_rows(
            config=training_config,
            eval_config=eval_config,
            run_id=run_id,
            step=step,
            cohorts=cohorts,
            tokenizer=tokenizer,
            model=model,
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            p0_prompt=p0_path.read_text(encoding="utf-8"),
            schema=schema,
        )
        generation_facts["model_load_seconds"] = model_load_seconds
        prediction_path = output_root / "predictions.jsonl"
        _append_jsonl_fsynced(prediction_path, generated)
        prediction_sha = sha256_file(prediction_path)
        prediction_index = {row["id"]: row for row in generated}
        if len(prediction_index) != len(generated) or set(prediction_index) != set(cohorts["union_ids"]):
            raise RuntimeError("A8-V2 predictions do not cover the frozen deduplicated ID union")
        generation_manifest = {
            "status": "predictions_frozen",
            "experiment_id": "E03-A8-V2",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "prediction_count": len(generated),
            "prediction_sha256": prediction_sha,
            "references_read": False,
            "created_at_utc": _now(),
        }
        _write_json(output_root / "generation_manifest.json", generation_manifest, exclusive=True)
        references = load_gold_references(eval_config, ROOT, cohorts["cohorts"])
        metrics = {
            name: score_a8_cohort(cohort, references[name], prediction_index, enums, arm=arm)
            for name, cohort in cohorts["cohorts"].items()
        }
        bootstraps = {
            name: bootstrap_conversation_micro_f1(
                cohort,
                references[name],
                prediction_index,
                enums,
                replicates=int(eval_config["comparison"]["bootstrap_replicates"]),
                seed=int(eval_config["generation"]["seed"]),
            )
            for name, cohort in cohorts["cohorts"].items()
        }
        base_after = _base_identity(training_config)
        engineering_reasons = []
        if len(generated) != cohorts["union_count"]:
            engineering_reasons.append("fixed_union_coverage_incomplete")
        if any(metrics[name]["strict_parse_rate"] < float(eval_config["comparison"]["minimum_strict_parse_rate"]) for name in metrics):
            engineering_reasons.append("strict_parse_rate_below_98_percent")
        if any(metrics[name]["eos_completion_rate"] < float(eval_config["comparison"]["minimum_strict_parse_rate"]) for name in metrics):
            engineering_reasons.append("eos_completion_rate_below_98_percent")
        if model_facts.get("offload_devices") or not model_facts.get("real_nf4_verified"):
            engineering_reasons.append("nf4_cuda_placement_or_offload_gate_failed")
        if base_before.get("observed_weight_hash") != base_after.get("observed_weight_hash"):
            engineering_reasons.append("base_weight_changed_during_evaluation")
        train_summary = _read_json(training_root / f"step-{step:04d}" / "training.json")
        if train_summary.get("independent_reload_check", {}).get("status") != "passed":
            engineering_reasons.append("independent_checkpoint_reload_failed")
        if sum(int(row.get("generated_token_count", 0)) for row in generated) != generation_facts["generated_token_count"]:
            engineering_reasons.append("generation_token_ledger_mismatch")
        engineering_gate = "passed" if not engineering_reasons else "failed"
        summary = {
            "status": "complete" if engineering_gate == "passed" else "engineering_failed",
            "experiment_id": "E03-A8-V2",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "samples_seen": step * 4,
            "authorized_max_step": int(training_manifest["authorized_max_step"]),
            "engineering_gate": engineering_gate,
            "engineering_gate_reasons": engineering_reasons,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "adapter_identity": adapter_identity,
            "base_weight_sha256": base_after.get("observed_weight_hash"),
            "config_sha256": training_manifest["config_sha256"],
            "dataset_manifest_sha256": training_manifest["dataset_manifest_sha256"],
            "sample_order_sha256": training_manifest["sample_order_sha256"],
            "evaluation_config_sha256": sha256_file(eval_path),
            "evaluation_code_sha256": _code_identity(V2_EVALUATION_CODE_PATHS),
            "cohorts": metrics,
            "same_id_transitions_from_previous_node": {name: None for name in metrics},
            "historical_baselines": historical_baselines(ROOT),
            "bootstrap": bootstraps,
            "resources": {
                **generation_facts,
                "peak_memory_bytes": inference.cuda_memory_facts(torch).get("allocator_peak_allocated_bytes"),
                "peak_reserved_memory_bytes": inference.cuda_memory_facts(torch).get("allocator_peak_reserved_bytes"),
                "deduplicated_union_count": cohorts["union_count"],
                "cohort_overlap_count": cohorts["overlap_count"],
                "training_step_gpu_seconds": train_summary["step_gpu_seconds"],
                "training_step_host_elapsed_seconds": train_summary["step_host_elapsed_seconds"],
            },
            "prediction_sha256": prediction_sha,
            "prediction_count": len(generated),
            "references_read_by_generator": False,
            "references_read_after_prediction_freeze": True,
            "references_sha256": {name: eval_config["cohorts"][name]["expected_references_sha256"] for name in metrics},
            "training": {
                "train_rows": 16192,
                "samples_seen": step * 4,
                "completion_tokens_seen": int(train_summary["completion_tokens_seen"]),
                "loss_start": _read_jsonl(
                    _resolve(training_config["data"]["output_root"])
                    / training_manifest["continuation"]["parent_run_id"]
                    / "step_ledger.jsonl"
                    if training_manifest.get("continuation")
                    else training_root / "step_ledger.jsonl"
                )[0]["mean_completion_loss"],
                "loss_node": train_summary["last_loss"],
                "loss_mean": train_summary["loss_mean"],
                "segment_loss_mean": train_summary.get("segment_loss_mean", train_summary["loss_mean"]),
                "cumulative_training_seconds": train_summary["cumulative_training_seconds"],
                "initial_lora_tensor_sha256": training_manifest["initial_lora_tensor_sha256"],
                "sample_order_sha256": data_facts["schedule_sha256"],
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
            "quality_decision": {
                "decision": "awaiting_manual_review",
                "reason": f"plan_stop_after_step_{step}_for_manual_review",
            },
            "progression_authorization_status": "awaiting_manual_review",
            "node_metrics_individually_reviewed": False,
            "created_at_utc": _now(),
        }
        summary_path = output_root / "evaluation.json"
        _write_json(summary_path, summary, exclusive=True)
        (output_root / "report.md").write_text(_render_report(summary), encoding="utf-8")
        stage_manifest = {
            "status": "complete" if engineering_gate == "passed" else "engineering_failed",
            "experiment_id": "E03-A8-V2",
            "run_id": run_id,
            "arm": arm,
            "step": step,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "evaluation_summary_sha256": sha256_file(summary_path),
            "predictions_sha256": prediction_sha,
            "evaluation_config_sha256": sha256_file(eval_path),
            "evaluation_code_sha256": summary["evaluation_code_sha256"],
            "engineering_gate": engineering_gate,
            "quality_decision": summary["quality_decision"],
            "created_at_utc": _now(),
        }
        _write_json(output_root / "stage_manifest.json", stage_manifest, exclusive=True)
        generation_manifest.update(
            status="complete" if engineering_gate == "passed" else "engineering_failed",
            references_read_after_prediction_freeze=True,
            summary_sha256=sha256_file(summary_path),
            quality_decision=summary["quality_decision"],
            engineering_gate=engineering_gate,
        )
        _write_json(output_root / "generation_manifest.json", generation_manifest)
        training_manifest["nodes"][str(step)].update(
            evaluation_status=summary["status"],
            evaluation_engineering_gate=engineering_gate,
            evaluation_summary_path=str(summary_path.relative_to(ROOT)),
            evaluation_summary_sha256=sha256_file(summary_path),
            evaluation_predictions_sha256=prediction_sha,
            quality_decision=summary["quality_decision"],
            authorization_status="awaiting_manual_review",
            evaluation_code_sha256=summary["evaluation_code_sha256"],
        )
        training_manifest["last_evaluation_summary_path"] = str(summary_path.relative_to(ROOT))
        training_manifest["last_evaluation_summary_sha256"] = sha256_file(summary_path)
        training_manifest.update(status="awaiting_manual_review", stage=f"step_{step}_manual_review_boundary")
        _write_json(training_root / "manifest.json", training_manifest)
        return summary
    except BaseException as exc:
        _write_json(output_root / "evaluation_failure.json", {
            "status": "failed", "experiment_id": "E03-A8-V2", "run_id": run_id,
            "arm": arm, "checkpoint_step": step, "error": f"{type(exc).__name__}: {exc}",
            "created_at_utc": _now(),
        })
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_paired_nodes(p_run_id: str, ep_run_id: str, step: int, eval_config_path: Path = EVAL_CONFIG_PATH) -> dict[str, Any]:
    if step != 512:
        raise ValueError("A8-V2 comparison is frozen to Step 512")
    eval_config, _ = _load_eval_config(eval_config_path)
    p_dir = _eval_root(eval_config, "P", p_run_id, step)
    ep_dir = _eval_root(eval_config, "EP", ep_run_id, step)
    p_summary, ep_summary = _read_json(p_dir / "evaluation.json"), _read_json(ep_dir / "evaluation.json")
    p_predictions = {row["id"]: row for row in _read_jsonl(p_dir / "predictions.jsonl")}
    ep_predictions = {row["id"]: row for row in _read_jsonl(ep_dir / "predictions.jsonl")}
    if (
        p_summary.get("status") != "complete" or ep_summary.get("status") != "complete"
        or p_summary.get("engineering_gate") != "passed" or ep_summary.get("engineering_gate") != "passed"
        or p_summary.get("arm") != "P" or ep_summary.get("arm") != "EP"
        or p_summary.get("checkpoint_step") != step or ep_summary.get("checkpoint_step") != step
        or p_summary.get("run_id") != p_run_id or ep_summary.get("run_id") != ep_run_id
    ):
        raise ValueError("A8-V2 paired comparison requires the same checkpoint step")
    if (
        sha256_file(p_dir / "predictions.jsonl") != p_summary.get("prediction_sha256")
        or sha256_file(ep_dir / "predictions.jsonl") != ep_summary.get("prediction_sha256")
    ):
        raise ValueError("A8-V2 paired prediction files differ from their frozen summary hashes")
    if set(p_predictions) != set(ep_predictions):
        raise ValueError("A8-V2 paired predictions do not cover the same union IDs")
    for eval_id in p_predictions:
        if p_predictions[eval_id].get("selected_history_sha256") != ep_predictions[eval_id].get("selected_history_sha256"):
            raise ValueError(f"A8-V2 P/EP selected different common histories for {eval_id}")
    metrics = {}
    for cohort in ("pilot160", "dev224"):
        p_profile = p_summary["cohorts"][cohort]["profile_33_label"]
        ep_profile = ep_summary["cohorts"][cohort]["profile_33_label"]
        metric_keys = (
            "overall_33_label_tp", "overall_33_label_fp", "overall_33_label_fn",
            "overall_33_label_micro_precision", "overall_33_label_micro_recall", "overall_33_label_micro_f1",
        )
        metrics[cohort] = {
            "p": p_profile,
            "ep": ep_profile,
            "ep_minus_p": {key: ep_profile[key] - p_profile[key] for key in metric_keys},
            "p_emotion": "N/A",
            "ep_emotion": ep_summary["cohorts"][cohort]["emotion"],
        }
    result = {
        "status": "complete",
        "experiment_id": "E03-A8-V2",
        "step": step,
        "p_run_id": p_run_id,
        "ep_run_id": ep_run_id,
        "p_summary_sha256": sha256_file(p_dir / "evaluation.json"),
        "ep_summary_sha256": sha256_file(ep_dir / "evaluation.json"),
        "same_union_ids": True,
        "common_history_verified": True,
        "metrics": metrics,
        "created_at_utc": _now(),
    }
    output = _resolve(eval_config["output_root"]) / "comparison" / f"step-{step:04d}"
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "paired_comparison.json", result, exclusive=True)
    return result


def evaluate_train_positive_diagnostic(
    training_config_path: Path,
    arm: str,
    run_id: str,
    step: int,
    eval_config_path: Path = EVAL_CONFIG_PATH,
    *,
    count: int = 64,
) -> dict[str, Any]:
    """Generate on consumed nonempty Train examples without changing the Run."""
    if step != 512 or count != 64:
        raise ValueError("A8-V2 Train diagnostic is fixed to 64 positives at Step 512")
    training_config, _ = _load_training_config(training_config_path, arm)
    eval_config, _ = _load_eval_config(eval_config_path)
    training_manifest, _data_facts, training_root, checkpoint = _validate_run(training_config, run_id, step)
    dataset_manifest = _read_json(_resolve(training_config["data"]["dataset_manifest"]))
    normalized_path = _resolve(dataset_manifest["inputs"]["normalized_train"]["path"])
    first_ids = [sample_id for row in _read_jsonl(_resolve(training_config["data"]["sample_order_path"]))[:512] for sample_id in row["sample_ids"]]
    first_id_set = set(first_ids)
    selected_rows = []
    with normalized_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sample_id") not in first_id_set:
                continue
            profile = row["target"]["user_profile"]
            if any(profile[dimension] for dimension in ("personality_traits", "interests", "style")):
                selected_rows.append(row)
                if len(selected_rows) == count:
                    break
    if len(selected_rows) != count:
        raise ValueError("A8-V2 consumed first 2,048 rows contain fewer than 64 nonempty profiles")

    diagnostic_ids = [f"train-positive:{row['sample_id']}" for row in selected_rows]
    mapping = {
        diagnostic_id: {"sample_id": row["sample_id"], "conversation_id": row["conversation_id"]}
        for diagnostic_id, row in zip(diagnostic_ids, selected_rows)
    }
    cohorts = {
        "cohorts": {"train_positive64": {"mapping": mapping}},
        "union_count": count,
        "union_ids": diagnostic_ids,
        "union_ids_sha256": selected_ids_sha256(diagnostic_ids),
        "union_inputs": [
            {"id": diagnostic_id, "history": row["history"]}
            for diagnostic_id, row in zip(diagnostic_ids, selected_rows)
        ],
    }
    output_root = _resolve(eval_config["output_root"]) / "train_positive" / arm / run_id / f"step-{step:04d}"
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Train-positive diagnostic: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    p_prompt = (ROOT / "configs/prompts/p_a8_v2_profile.txt").read_text(encoding="utf-8")
    ep_prompt = (ROOT / "configs/prompts/p_a8_v2_emotion_profile.txt").read_text(encoding="utf-8")
    p0_path = _resolve(eval_config["protocol"]["p0_path"])
    schema_path = ROOT / "data/train/train_public.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enums = validate_a8_prompt_enum_order(p_prompt, schema, "P")
    validate_a8_prompt_enum_order(ep_prompt, schema, "EP")
    tokenizer, tokenizer_facts = _runtime_tokenizer(training_config)
    model = None
    try:
        from src.training.training_sft import _load_train_model

        model, torch, model_facts = _load_train_model(
            training_config,
            _base_path(training_config),
            adapter_path=checkpoint,
            autocast_adapter_dtype=True,
        )
        model.eval()
        generated, generation_facts = _generate_rows(
            config=training_config,
            eval_config=eval_config,
            run_id=run_id,
            step=step,
            cohorts=cohorts,
            tokenizer=tokenizer,
            model=model,
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            p0_prompt=p0_path.read_text(encoding="utf-8"),
            schema=schema,
        )
        prediction_path = output_root / "predictions.jsonl"
        _append_jsonl_fsynced(prediction_path, generated)
        prediction_sha = sha256_file(prediction_path)
        prediction_index = {row["id"]: row for row in generated}
        if set(prediction_index) != set(diagnostic_ids):
            raise RuntimeError("A8-V2 Train-positive diagnostic prediction coverage differs")
        generation_manifest = {
            "status": "predictions_frozen",
            "experiment_id": "E03-A8-V2",
            "diagnostic": "consumed_train_nonempty_profiles",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "sample_count": count,
            "sample_ids_sha256": canonical_sha256([row["sample_id"] for row in selected_rows]),
            "prediction_sha256": prediction_sha,
            "references_read": False,
            "created_at_utc": _now(),
        }
        _write_json(output_root / "generation_manifest.json", generation_manifest, exclusive=True)
        references = {
            diagnostic_id: {"target": row["target"]}
            for diagnostic_id, row in zip(diagnostic_ids, selected_rows)
        }
        diagnostic_cohort = {
            "name": "train_positive64",
            "ids": diagnostic_ids,
            "ids_sha256": selected_ids_sha256(diagnostic_ids),
            "conversation_count": len({row["conversation_id"] for row in selected_rows}),
        }
        metrics = score_a8_cohort(diagnostic_cohort, references, prediction_index, enums, arm=arm)
        result = {
            "status": "complete",
            "experiment_id": "E03-A8-V2",
            "diagnostic": "consumed_train_nonempty_profiles",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "sample_count": count,
            "sample_ids_sha256": generation_manifest["sample_ids_sha256"],
            "prediction_sha256": prediction_sha,
            "references_read_after_prediction_freeze": True,
            "metrics": metrics,
            "generation": generation_facts,
            "tokenizer_facts": tokenizer_facts,
            "model_facts": model_facts,
            "created_at_utc": _now(),
        }
        _write_json(output_root / "train_positive_evaluation.json", result, exclusive=True)
        generation_manifest.update(status="complete", references_read_after_prediction_freeze=True, metrics_sha256=sha256_file(output_root / "train_positive_evaluation.json"))
        _write_json(output_root / "generation_manifest.json", generation_manifest)
        return result
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate A8-V2 checkpoint quality on frozen Pilot160 and Dev224.")
    parser.add_argument("--mode", choices=("evaluate", "train-positive-diagnostic", "compare"), required=True)
    parser.add_argument("--arm", choices=("P", "EP"))
    parser.add_argument("--run-id")
    parser.add_argument("--step", type=int, default=512)
    parser.add_argument("--p-run-id")
    parser.add_argument("--ep-run-id")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.mode == "evaluate":
        if args.arm is None or args.run_id is None:
            parser.error("evaluate requires --arm and --run-id")
        result = evaluate_checkpoint(args.training_config or DEFAULT_CONFIGS[args.arm], args.arm, args.run_id, args.step, args.eval_config)
    elif args.mode == "train-positive-diagnostic":
        if args.arm is None or args.run_id is None:
            parser.error("train-positive-diagnostic requires --arm and --run-id")
        result = evaluate_train_positive_diagnostic(
            args.training_config or DEFAULT_CONFIGS[args.arm],
            args.arm,
            args.run_id,
            args.step,
            args.eval_config,
        )
    else:
        if args.p_run_id is None or args.ep_run_id is None:
            parser.error("compare requires --p-run-id and --ep-run-id")
        result = compare_paired_nodes(args.p_run_id, args.ep_run_id, args.step, args.eval_config)
    print(json.dumps({
        "status": result.get("status"),
        "arm": result.get("arm"),
        "step": result.get("checkpoint_step", result.get("step")),
        "quality_decision": result.get("quality_decision"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
