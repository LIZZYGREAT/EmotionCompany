#!/usr/bin/env python3
"""Run read-only Train-Calibration and fixed 224-point evaluation for E03-A7."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e03_a5_evaluate, e03_a5_probe, e03_a7_train
from src.data.e03_a5_labels import canonical_sha256, iter_jsonl, profile_to_target, schema_label_order, sha256_file
from src.data.e03_a7_classification import SFTPromptIndex, build_supervision, emotion_label_order, read_splits
from src.evaluation.e03_a7_classification_metrics import (
    DIMENSIONS,
    average_precision_by_label,
    choose_calibration_threshold,
    compare_arm_profile_predictions,
    emotion_summary,
    profile_from_probabilities,
    summarize_profile_pair,
)
from src.evaluation.metrics import profile_metrics
from src.evaluation.profile_fast_eval import selected_id_sha256
from src.evaluation.profile_reconcile import index_profile_prediction_rows
from src.evaluation.source_integrity import unique_index, validate_target_free_inputs
from src.training.e03_a7_staged import verify_checkpoint_manifest, write_manifest


INFERENCE_BATCH_SIZE = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    return e03_a7_train._resolve(value)


def _write_once(path: Path, value: Any) -> None:
    e03_a7_train._write_once(path, value)


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _load_evaluation_sources(config: Mapping[str, Any], a5_config: Mapping[str, Any]) -> dict[str, Any]:
    assets = e03_a5_evaluate._load_eval_assets(a5_config)
    expected = config["identity"]
    if (
        assets["source_hashes"]["frozen_input"] != expected["frozen_dev_input_sha256"]
        or assets["source_hashes"]["references"] != expected["frozen_dev_references_sha256"]
        or assets["source_hashes"]["id_map"] != expected["frozen_dev_id_map_sha256"]
        or len(assets["frozen_ids"]) != int(config["data"]["expected_dev_rows"])
        or selected_id_sha256(assets["frozen_ids"]) != expected["frozen_dev_ids_sha256"]
    ):
        raise ValueError("A7 fixed Dev cohort or source hashes differ from the frozen contract")
    input_rows = [row for _line, row, _sha in iter_jsonl(assets["source_paths"]["frozen_input"])]
    input_index = validate_target_free_inputs(input_rows)
    if set(input_index) != set(assets["frozen_ids"]):
        raise ValueError("A7 frozen target-free input IDs differ from the fixed Dev cohort")
    id_map_rows = [row for _line, row, _sha in iter_jsonl(assets["source_paths"]["id_map"])]
    id_map = unique_index(id_map_rows, "eval_id", "A7 frozen Dev ID map")
    if set(id_map) != set(assets["frozen_ids"]):
        raise ValueError("A7 frozen Dev ID map IDs differ from the target-free input")
    return {**assets, "input_index": input_index, "id_map": id_map}


def _load_run_assets(config: Mapping[str, Any], config_sha: str, training_code_sha: str, arm: str, step: int) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file

    run_id = f"{config['training']['pair_id']}-{arm}"
    run_root = _resolve(config["outputs"]["training_root"]) / run_id
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = _json(run_manifest_path)
    node = run_manifest.get("nodes", {}).get(str(step))
    if (
        run_manifest.get("run_id") != run_id
        or run_manifest.get("arm") != arm
        or run_manifest.get("config_sha256") != config_sha
        or run_manifest.get("code_sha256") != training_code_sha
        or run_manifest.get("optimizer_step") != step
        or not isinstance(node, Mapping)
        or node.get("training_status") != "passed"
    ):
        raise ValueError("A7 evaluation Run is incomplete or its config/code identity changed")
    checkpoint_dir = run_root / "trainer" / f"checkpoint-{step}"
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = verify_checkpoint_manifest(
        checkpoint_manifest_path,
        expected_run_id=run_id,
        expected_arm=arm,
        expected_step=step,
        expected_identity=run_manifest["checkpoint_identity"],
        expected_schedule_sha256=run_manifest["schedule_sha256"],
    )
    if sha256_file(checkpoint_manifest_path) != node.get("checkpoint_manifest_sha256"):
        raise ValueError("A7 evaluation checkpoint differs from the training Run Manifest")
    base = e03_a7_train._load_base(config, torch, trainable=False)
    model = PeftModel.from_pretrained(
        base,
        str(checkpoint_dir / "inference_export/adapter"),
        is_trainable=False,
        local_files_only=True,
        # Classifier training stores LoRA weights in FP32; retain that dtype
        # so Dev predictions use the same Adapter values as the reload gate.
        autocast_adapter_dtype=True,
    )
    model.eval()
    heads = e03_a7_train.build_heads(
        int(model.config.hidden_size),
        arm,
        profile_seed=int(config["model"]["heads"]["profile_seed"]),
        emotion_seed=int(config["model"]["heads"]["emotion_seed"]),
    ).to(torch.device("cuda", torch.cuda.current_device()))
    export_root = checkpoint_dir / "inference_export"
    profile_state = load_file(str(export_root / "profile_head.safetensors"), device="cuda:0")
    heads.profile_head.load_state_dict(profile_state, strict=True)
    if arm == "EP":
        emotion_state = load_file(str(export_root / "emotion_head.safetensors"), device="cuda:0")
        heads.emotion_head.load_state_dict(emotion_state, strict=True)
    heads.eval()
    return {
        "run_id": run_id,
        "run_root": run_root,
        "run_manifest_path": run_manifest_path,
        "run_manifest": run_manifest,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_manifest_path": checkpoint_manifest_path,
        "checkpoint_manifest": checkpoint_manifest,
        "model": model,
        "heads": heads,
    }


def _prompts_for_dev(config: Mapping[str, Any], a5_config: Mapping[str, Any], sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    from participant import run_inference as inference

    runtime_config = _json(_resolve(a5_config["data"]["runtime_inference_config"]))
    prompt_text, _schema, _enums, prompt_facts = inference.load_runtime_protocol(runtime_config)
    tokenizer, facts = inference.load_runtime_tokenizer(
        runtime_config, local_source=_resolve(config["identity"]["base_model_path"])
    )
    if (
        prompt_facts.get("sha256") != config["identity"]["p0_sha256"]
        or facts.get("signature_sha256") != config["identity"]["tokenizer_signature_sha256"]
        or facts.get("template_sha256") != config["identity"]["chat_template_sha256"]
    ):
        raise ValueError("A7 Dev Prompt Tokenizer or P0 facts differ from the frozen identity")
    prompt_rows = []
    for sample_id in sources["frozen_ids"]:
        selection, _messages = inference.prepare_sample_messages(
            dict(sources["input_index"][sample_id]),
            prompt_text,
            tokenizer,
            int(config["protocol"]["max_prompt_tokens"]),
        )
        prompt_ids = selection.get("prompt_ids")
        if selection.get("floor_exceeds_cap") or not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"{sample_id}: A7 frozen Dev Prompt violates the P=1280 boundary")
        prompt_rows.append({"sample_id": sample_id, "input_ids": [int(value) for value in prompt_ids], "prompt_length": len(prompt_ids)})
    if len(prompt_rows) != 224 or [row["sample_id"] for row in prompt_rows] != list(sources["frozen_ids"]):
        raise ValueError("A7 Dev Prompt Token rows do not preserve all 224 frozen IDs in order")
    return prompt_rows


def _infer_probabilities(
    model: Any,
    heads: Any,
    arm: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
) -> list[dict[str, Any]]:
    import torch
    from src.training.e03_a7_classifier import forward_prompt_end

    device = torch.device("cuda", torch.cuda.current_device())
    predictions: list[dict[str, Any]] = []
    model.eval()
    heads.eval()
    started_at = time.monotonic()
    next_progress_report = 256
    with torch.inference_mode():
        for start in range(0, len(rows), INFERENCE_BATCH_SIZE):
            batch_rows = rows[start : start + INFERENCE_BATCH_SIZE]
            lengths = [len(row["input_ids"]) for row in batch_rows]
            max_length = max(lengths)
            input_ids = [
                list(row["input_ids"]) + [int(pad_token_id)] * (max_length - length)
                for row, length in zip(batch_rows, lengths)
            ]
            attention_masks = [
                [1] * length + [0] * (max_length - length)
                for length in lengths
            ]
            ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
            hidden = forward_prompt_end(model, ids, mask, torch.tensor(lengths, device=device))
            output = heads(hidden)
            profile_probabilities = torch.sigmoid(output["profile_logits"].float()).cpu().tolist()
            emotion_logits = output["emotion_logits"].float().cpu() if arm == "EP" else None
            for index, row in enumerate(batch_rows):
                result: dict[str, Any] = {
                    "id": row["sample_id"],
                    "profile_probabilities": profile_probabilities[index],
                }
                if arm == "EP":
                    logits = emotion_logits[index]
                    result["emotion_logits"] = logits.tolist()
                    result["emotion_probabilities"] = torch.softmax(logits, dim=-1).tolist()
                predictions.append(result)
            processed = start + len(batch_rows)
            if processed >= next_progress_report or processed == len(rows):
                elapsed = time.monotonic() - started_at
                print(f"A7 {arm} evaluation inference: {processed}/{len(rows)} samples ({elapsed:.1f}s)", flush=True)
                next_progress_report = processed + 256
    return predictions


def _profile_metrics_subset(gold: Sequence[Mapping[str, Sequence[str]]], predictions: Sequence[Mapping[str, Sequence[str]]], enums: Mapping[str, Sequence[str]], indices: Sequence[int]) -> dict[str, Any]:
    if not indices:
        return {"status": "empty_group", "sample_count": 0}
    return profile_metrics(
        [dict(gold[index]) for index in indices],
        [dict(predictions[index]) for index in indices],
        {key: list(enums[key]) for key in DIMENSIONS},
    )


def _baseline_reconciliation_enums(
    profile_enums: Mapping[str, Sequence[str]], emotion_labels: Sequence[str],
) -> dict[str, list[str]]:
    result = {key: list(values) for key, values in profile_enums.items()}
    result["emotion_label"] = list(emotion_labels)
    return result


def _mark_scoring_interrupted(eval_manifest_path: Path, error: BaseException) -> None:
    manifest = _json(eval_manifest_path)
    if manifest.get("status") not in {"scoring_started_after_prediction_lock", "scoring_interrupted_after_gold_read"}:
        return
    manifest.update(
        status="scoring_interrupted_after_gold_read",
        gold_read=True,
        gold_read_after_predictions_locked=True,
        scoring_failure={
            "exception_type": type(error).__name__,
            "failed_at_utc": _now(),
        },
    )
    write_manifest(eval_manifest_path, manifest, overwrite=True)


def _score_after_prediction_freeze(
    *,
    config: Mapping[str, Any],
    arm: str,
    step: int,
    eval_dir: Path,
    eval_manifest_path: Path,
    prediction_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    sources: Mapping[str, Any],
    a5_config: Mapping[str, Any],
    supervision: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, Sequence[str]],
    evaluation_assets: Mapping[str, Any],
    start_time: float,
    training_code_sha256: str,
) -> dict[str, Any]:
    locked_predictions_sha = sha256_file(eval_dir / "predictions.jsonl")
    eval_manifest = _json(eval_manifest_path)
    if eval_manifest.get("predictions_sha256") != locked_predictions_sha:
        raise ValueError("A7 prediction lock Manifest changed before Gold scoring")
    if eval_manifest.get("status") == "dev_predictions_frozen_gold_not_read":
        if eval_manifest.get("gold_read") is not False:
            raise ValueError("A7 fresh score must start from a target-free prediction lock")
        eval_manifest.update(
            status="scoring_started_after_prediction_lock",
            gold_read=True,
            gold_read_after_predictions_locked=True,
            scoring_attempts=[{"started_at_utc": _now(), "prediction_sha256": locked_predictions_sha}],
        )
        write_manifest(eval_manifest_path, eval_manifest, overwrite=True)
    elif eval_manifest.get("status") in {"scoring_started_after_prediction_lock", "scoring_interrupted_after_gold_read"}:
        if eval_manifest.get("gold_read") is not True or eval_manifest.get("gold_read_after_predictions_locked") is not True:
            raise ValueError("A7 scoring recovery lacks a target-read-after-lock record")
        eval_manifest.setdefault("scoring_attempts", []).append(
            {"started_at_utc": _now(), "prediction_sha256": locked_predictions_sha, "recovery": True}
        )
        eval_manifest["status"] = "scoring_started_after_prediction_lock"
        write_manifest(eval_manifest_path, eval_manifest, overwrite=True)
    else:
        raise ValueError("A7 score requires a fresh prediction lock or an interrupted post-lock scoring record")

    # This is deliberately the first read of frozen Dev Gold in the A7 evaluation path.
    references_path = evaluation_assets["source_paths"]["references"]
    references = unique_index(
        [row for _line, row, _sha in iter_jsonl(references_path)],
        "eval_id",
        "A7 frozen Dev References",
    )
    ids = list(sources["frozen_ids"])
    if set(references) != set(ids):
        raise ValueError("A7 frozen Gold Reference IDs differ from locked predictions")
    output_schema = _json(_resolve(config["data"]["output_schema"]))
    train_schema = _json(_resolve(config["data"]["train_schema"]))
    profile_labels, enums = schema_label_order(train_schema, output_schema)
    emotion_labels = emotion_label_order(train_schema, output_schema)
    gold_profiles = []
    gold_bits = []
    gold_emotions = []
    for sample_id in ids:
        target = references[sample_id].get("target")
        if not isinstance(target, Mapping):
            raise ValueError(f"A7 Reference {sample_id} has no Gold target")
        profile = target.get("user_profile")
        y, mask, normalized = profile_to_target(profile, enums, sample_id=sample_id)
        if mask != [1] * 33:
            raise ValueError(f"A7 Dev Gold {sample_id} violates the closed-world 33-label contract")
        emotion = target.get("emotion_label")
        if emotion not in emotion_labels:
            raise ValueError(f"A7 Dev Gold {sample_id} has an invalid emotion label")
        gold_profiles.append(normalized)
        gold_bits.append(y)
        gold_emotions.append(emotion)

    probabilities = [row["profile_probabilities"] for row in prediction_rows]
    if [row["id"] for row in prediction_rows] != ids:
        raise ValueError("A7 locked predictions no longer align with frozen Dev ID order")
    profile_05 = profile_from_probabilities(probabilities, threshold=0.5, enums=enums)
    profile_05_metrics = profile_metrics(gold_profiles, profile_05, enums)
    calibrated_threshold = calibration.get("threshold")
    if calibrated_threshold is None:
        calibrated_profile = None
        calibrated_metrics = {"status": "threshold_constraint_unsatisfied", "threshold": None}
    else:
        calibrated_profile = profile_from_probabilities(probabilities, threshold=float(calibrated_threshold), enums=enums)
        calibrated_metrics = profile_metrics(gold_profiles, calibrated_profile, enums)

    prediction_row_targets = [row for row in prediction_rows]
    ap = average_precision_by_label(probabilities, gold_bits, profile_labels)
    histories = [sources["input_index"][sample_id]["history"] for sample_id in ids]
    short_indices = [index for index, history in enumerate(histories) if len(history) <= 4]
    long_indices = [index for index, history in enumerate(histories) if len(history) > 4]
    history_groups = {
        "short_le_4_messages": {
            "sample_count": len(short_indices),
            "threshold_0_5": _profile_metrics_subset(gold_profiles, profile_05, enums, short_indices),
            "calibrated": _profile_metrics_subset(gold_profiles, calibrated_profile, enums, short_indices) if calibrated_profile is not None else calibrated_metrics,
        },
        "long_gt_4_messages": {
            "sample_count": len(long_indices),
            "threshold_0_5": _profile_metrics_subset(gold_profiles, profile_05, enums, long_indices),
            "calibrated": _profile_metrics_subset(gold_profiles, calibrated_profile, enums, long_indices) if calibrated_profile is not None else calibrated_metrics,
        },
    }
    baseline_rows = [row for _line, row, _sha in iter_jsonl(evaluation_assets["baseline_paths"]["predictions"])]
    baseline_profile_index, baseline_statuses = index_profile_prediction_rows(
        baseline_rows, ids, _baseline_reconciliation_enums(enums, emotion_labels), require_all_valid=True,
    )
    if baseline_statuses.get("valid") != 224:
        raise ValueError("A7 E01-B Fast-only baseline is not valid for all 224 frozen IDs")
    baseline_profiles = [baseline_profile_index[sample_id] for sample_id in ids]
    conversation_ids = [sources["id_map"][sample_id]["conversation_id"] for sample_id in ids]
    baseline_comparisons = {
        "threshold_0_5": summarize_profile_pair(
            ids=ids, conversation_ids=conversation_ids, gold=gold_profiles,
            baseline=baseline_profiles, candidate=profile_05, enums=enums,
            seed=int(config["evaluation"]["bootstrap_seed"]),
            bootstrap_repetitions=int(config["evaluation"]["bootstrap_repetitions"]),
        ),
    }
    if calibrated_profile is not None:
        baseline_comparisons["calibrated"] = summarize_profile_pair(
            ids=ids, conversation_ids=conversation_ids, gold=gold_profiles,
            baseline=baseline_profiles, candidate=calibrated_profile, enums=enums,
            seed=int(config["evaluation"]["bootstrap_seed"]),
            bootstrap_repetitions=int(config["evaluation"]["bootstrap_repetitions"]),
        )

    emotion = {"status": "not_applicable"}
    if arm == "EP":
        emotion = emotion_summary(
            gold_emotions,
            [row["emotion_logits"] for row in prediction_rows],
            emotion_labels,
        )
    report = {
        "version": "E03-A7-evaluation-report-v1",
        "status": "scored",
        "experiment_id": config["experiment_id"],
        "pair_id": config["training"]["pair_id"],
        "run_id": f"{config['training']['pair_id']}-{arm}",
        "arm": arm,
        "checkpoint_step": step,
        "training_code_sha256": training_code_sha256,
        "evaluation_code_sha256": e03_a7_train._code_sha256(),
        "engineering_gate": "passed",
        "model_quality_status": "inconclusive",
        "progression_authorization_status": "awaiting_manual_review",
        "sample_count": len(ids),
        "conversation_count": len(set(conversation_ids)),
        "frozen_ids_sha256": config["identity"]["frozen_dev_ids_sha256"],
        "locked_predictions_sha256": sha256_file(eval_dir / "predictions.jsonl"),
        "gold_read_after_predictions_locked": True,
        "official_val_read": False,
        "response_text_generated": False,
        "profile_label_order_sha256": canonical_sha256(profile_labels),
        "emotion_label_order_sha256": canonical_sha256(emotion_labels),
        "calibration": dict(calibration),
        "profile_threshold_0_5": profile_05_metrics,
        "profile_calibrated": calibrated_metrics,
        "profile_average_precision": ap,
        "profile_history_groups": history_groups,
        "comparison_to_e01_b_fast_only": baseline_comparisons,
        "emotion": emotion,
        "response_metrics": "not_run",
        "official_end_to_end_metrics": "not_run",
        "source_facts": {
            "references_sha256": sha256_file(references_path),
            "predictions_sha256": sha256_file(eval_dir / "predictions.jsonl"),
            "calibration_predictions_sha256": sha256_file(eval_dir / "calibration_probabilities.jsonl"),
            "baseline_predictions_sha256": evaluation_assets["baseline_hashes"]["predictions"],
            "a5_audit_manifest_sha256": config["identity"]["a5_audit_manifest_sha256"],
        },
        "created_at_utc": _now(),
        "elapsed_seconds": time.monotonic() - start_time,
    }
    report_path = eval_dir / "evaluation_report.json"
    if report_path.exists():
        existing_report = _json(report_path)
        if (
            existing_report.get("status") != "scored"
            or existing_report.get("locked_predictions_sha256") != report["locked_predictions_sha256"]
            or existing_report.get("arm") != arm
            or existing_report.get("checkpoint_step") != step
        ):
            raise ValueError("A7 existing score report does not match its immutable prediction lock")
        report = existing_report
    else:
        _write_once(report_path, report)
    eval_manifest = _json(eval_manifest_path)
    if eval_manifest.get("gold_read") is not True or eval_manifest.get("predictions_sha256") != report["locked_predictions_sha256"]:
        raise ValueError("A7 prediction lock Manifest changed before Gold scoring")
    eval_manifest.update(
        status="scored",
        gold_read=True,
        gold_read_after_predictions_locked=True,
        references_sha256=report["source_facts"]["references_sha256"],
        evaluation_report_sha256=sha256_file(report_path),
        scored_at_utc=_now(),
    )
    write_manifest(eval_manifest_path, eval_manifest, overwrite=True)
    _render_report(report, eval_dir / "report.md")

    run_root = _resolve(config["outputs"]["training_root"]) / report["run_id"]
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = _json(run_manifest_path)
    node = run_manifest["nodes"][str(step)]
    node.update(
        evaluation_status="passed",
        evaluation_manifest_path=str(eval_manifest_path),
        evaluation_manifest_sha256=sha256_file(eval_manifest_path),
        evaluation_report_path=str(report_path),
        evaluation_report_sha256=sha256_file(report_path),
        progression_authorization_status="awaiting_manual_review",
    )
    run_manifest.update(
        status="awaiting_manual_review",
        manual_review_state=f"awaiting_manual_review_step_{step:04d}",
        last_updated_at_utc=_now(),
    )
    write_manifest(run_manifest_path, run_manifest, overwrite=True)
    print(json.dumps({"status": report["status"], "arm": arm, "step": step, "profile_micro_f1_0_5": profile_05_metrics["overall_33_label_micro_f1"], "profile_tp_fp_fn_0_5": [profile_05_metrics["overall_33_label_tp"], profile_05_metrics["overall_33_label_fp"], profile_05_metrics["overall_33_label_fn"]], "calibrated_threshold": calibrated_threshold, "report": str(report_path)}, ensure_ascii=False))
    return report


def _render_report(report: Mapping[str, Any], path: Path) -> None:
    profile = report["profile_threshold_0_5"]
    calibrated = report["profile_calibrated"]
    lines = [
        f"# E03-A7 {report['arm']} Step {report['checkpoint_step']} 节点评测",
        "",
        f"- 工程门禁：`{report['engineering_gate']}`；画像质量：`{report['model_quality_status']}`；继续授权：`{report['progression_authorization_status']}`。",
        f"- 固定开发代理集：{report['sample_count']} 点 / {report['conversation_count']} 段会话；Dev Gold 在锁定预测 SHA 后读取。",
        f"- 0.5 阈值：Micro/Macro-F1 `{profile['overall_33_label_micro_f1']:.6f}` / `{profile['overall_33_label_macro_f1']:.6f}`；TP/FP/FN `{profile['overall_33_label_tp']}/{profile['overall_33_label_fp']}/{profile['overall_33_label_fn']}`；Precision/Recall `{profile['overall_33_label_micro_precision']:.6f}` / `{profile['overall_33_label_micro_recall']:.6f}`。",
        f"- Calibration 阈值：`{calibrated.get('threshold')}`；状态 `{calibrated.get('status', 'scored')}`。",
        f"- 与 E01-B Fast-only 同 ID 比较及会话 Bootstrap：见 `evaluation_report.json`。",
        f"- EP Emotion：`{report['emotion'].get('metrics', {}).get('accuracy', 'not_applicable')}`；P 情绪：`not_applicable`。",
        "- 回复生成、完整 JSON 与官方端到端评测：`not_run`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_arm(config: Mapping[str, Any], config_sha: str, training_code_sha: str, a5_config: Mapping[str, Any], arm: str, step: int, *, assets: Mapping[str, Any], supervision: Mapping[str, Mapping[str, Any]], splits: Mapping[str, Sequence[str]], prompt_index: SFTPromptIndex) -> dict[str, Any]:
    import torch

    run_assets = _load_run_assets(config, config_sha, training_code_sha, arm, step)
    eval_root = _resolve(config["outputs"]["evaluation_root"]) / run_assets["run_id"] / f"step-{step:04d}"
    if eval_root.exists():
        raise FileExistsError(f"A7 evaluation output already exists and is immutable: {eval_root}")
    eval_root.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    calibration_ids = list(splits["calibration"])
    calibration_rows = list(prompt_index.iter_prompt_rows(calibration_ids))
    dev_rows = _prompts_for_dev(config, a5_config, assets)
    calibration_predictions = _infer_probabilities(
        run_assets["model"], run_assets["heads"], arm, calibration_rows,
        pad_token_id=int(config["identity"]["pad_token_id"]),
    )
    calibration_path = eval_root / "calibration_probabilities.jsonl"
    calibration_sha = _write_jsonl_once(calibration_path, calibration_predictions)
    calibration_targets = [supervision[sample_id]["profile_target"] for sample_id in calibration_ids]
    calibration_result = choose_calibration_threshold(
        [row["profile_probabilities"] for row in calibration_predictions],
        calibration_targets,
        thresholds=[float(value) for value in config["evaluation"]["calibration_thresholds"]],
        minimum_precision=float(config["evaluation"]["calibration_min_micro_precision"]),
    )
    calibration_manifest = {
        "version": "E03-A7-calibration-manifest-v1",
        "status": calibration_result["status"],
        "run_id": run_assets["run_id"],
        "arm": arm,
        "checkpoint_step": step,
        "sample_count": len(calibration_ids),
        "sample_ids_sha256": canonical_sha256(calibration_ids),
        "split": "train_calibration",
        "predictions_sha256": calibration_sha,
        "threshold_policy": calibration_result,
        "dev_rows_read": 0,
        "gold_dev_read": False,
        "created_at_utc": _now(),
    }
    calibration_report_path = eval_root / "calibration.json"
    _write_once(calibration_report_path, calibration_manifest)

    dev_predictions = _infer_probabilities(
        run_assets["model"], run_assets["heads"], arm, dev_rows,
        pad_token_id=int(config["identity"]["pad_token_id"]),
    )
    predictions_path = eval_root / "predictions.jsonl"
    predictions_sha = _write_jsonl_once(predictions_path, dev_predictions)
    eval_manifest_path = eval_root / "evaluation_manifest.json"
    eval_manifest = {
        "version": "E03-A7-evaluation-manifest-v1",
        "status": "dev_predictions_frozen_gold_not_read",
        "run_id": run_assets["run_id"],
        "arm": arm,
        "checkpoint_step": step,
        "config_sha256": config_sha,
        "code_sha256": e03_a7_train._code_sha256(),
        "training_code_sha256": run_assets["run_manifest"]["code_sha256"],
        "evaluation_code_sha256": e03_a7_train._code_sha256(),
        "run_manifest_sha256_before_scoring": sha256_file(run_assets["run_manifest_path"]),
        "checkpoint_manifest_sha256": sha256_file(run_assets["checkpoint_manifest_path"]),
        "profile_head_sha256": run_assets["checkpoint_manifest"]["profile_head_sha256"],
        "emotion_head_sha256": run_assets["checkpoint_manifest"]["emotion_head_sha256"],
        "base_weight_sha256": config["identity"]["base_weight_sha256"],
        "adapter_sha256": run_assets["checkpoint_manifest"]["adapter_sha256"],
        "profile_label_order_sha256": canonical_sha256(schema_label_order(_json(_resolve(config["data"]["train_schema"])), _json(_resolve(config["data"]["output_schema"])))[0]),
        "input_ids_sha256": config["identity"]["frozen_dev_ids_sha256"],
        "input_source_sha256": assets["source_hashes"]["frozen_input"],
        "calibration_manifest_sha256": sha256_file(calibration_report_path),
        "calibration_predictions_sha256": calibration_sha,
        "predictions_path": str(predictions_path),
        "predictions_sha256": predictions_sha,
        "prediction_count": len(dev_predictions),
        "prediction_ids_sha256": selected_id_sha256([row["id"] for row in dev_predictions]),
        "gold_read": False,
        "references_sha256_pinned_not_read": assets["source_hashes"]["references"],
        "gold_targets_passed_to_model": False,
        "model_generate_called": False,
        "response_text_generated": False,
        "created_at_utc": _now(),
    }
    _write_once(eval_manifest_path, eval_manifest)
    # Release the classifier model before any later training stage can load a model.
    del run_assets["model"], run_assets["heads"]
    gc.collect()
    torch.cuda.empty_cache()
    try:
        report = _score_after_prediction_freeze(
            config=config,
            arm=arm,
            step=step,
            eval_dir=eval_root,
            eval_manifest_path=eval_manifest_path,
            prediction_rows=dev_predictions,
            calibration=calibration_result,
            sources=assets,
            a5_config=a5_config,
            supervision=supervision,
            splits=splits,
            evaluation_assets=assets,
            start_time=start_time,
            training_code_sha256=run_assets["run_manifest"]["code_sha256"],
        )
    except BaseException as error:
        _mark_scoring_interrupted(eval_manifest_path, error)
        raise
    return report


def _write_partial_pair_report(config: Mapping[str, Any], step: int, arm: str, eval_root: Path) -> dict[str, Any]:
    pair_id = config["training"]["pair_id"]
    report_path = eval_root / "evaluation_report.json"
    manifest_path = eval_root / "evaluation_manifest.json"
    predictions_path = eval_root / "predictions.jsonl"
    report = _json(report_path)
    manifest = _json(manifest_path)
    predictions_sha = sha256_file(predictions_path)
    if (
        report.get("status") != "scored"
        or report.get("arm") != arm
        or report.get("checkpoint_step") != step
        or report.get("run_id") != f"{pair_id}-{arm}"
        or report.get("sample_count") != int(config["data"]["expected_dev_rows"])
        or report.get("gold_read_after_predictions_locked") is not True
        or report.get("locked_predictions_sha256") != predictions_sha
        or manifest.get("status") != "scored"
        or manifest.get("prediction_count") != int(config["data"]["expected_dev_rows"])
        or manifest.get("predictions_sha256") != predictions_sha
    ):
        raise ValueError(f"A7 {arm} Step {step} report is not complete for review")
    pair_dir = _resolve(config["outputs"]["evaluation_root"]) / "pairs" / pair_id / f"step-{step:04d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    paired_report_path = pair_dir / "paired_report.json"
    if paired_report_path.is_file():
        previous = _json(paired_report_path)
        if previous.get("status") != "awaiting_manual_review_missing_matched_arm":
            raise FileExistsError("A7 paired node report already exists and is immutable")
        if previous.get("available_arm") != arm or previous.get("evaluation_report_sha256") != sha256_file(report_path):
            raise ValueError("A7 existing partial pair report no longer matches its scored Arm")
    result = {
        "version": "E03-A7-paired-node-report-v1",
        "status": "awaiting_manual_review_missing_matched_arm",
        "pair_id": pair_id,
        "checkpoint_step": step,
        "same_cohort": True,
        "cohort_count": report.get("sample_count"),
        "cohort_ids_sha256": report.get("frozen_ids_sha256"),
        "available_arm": arm,
        "missing_arm": "EP" if arm == "P" else "P",
        "run_id": report.get("run_id"),
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "evaluation_report_sha256": sha256_file(report_path),
        "p_vs_ep": "not_applicable_missing_matched_arm",
        "recommendation": "Review only the available Arm; no paired P/EP conclusion is available at this node.",
        "next_step": {512: 1024, 1024: 2048, 2048: 4048}.get(step),
        "created_at_utc": _now(),
    }
    write_manifest(paired_report_path, result, overwrite=paired_report_path.exists())
    md = [
        f"# E03-A7 Step {step} 单 Arm 节点评审",
        "",
        f"- 状态：`{result['status']}`；已评测 Arm：`{arm}`；缺少匹配 Arm：`{result['missing_arm']}`。",
        f"- 当前 Run：`{result['run_id']}`；固定开发集：{result['cohort_count']} 点。",
        "- P/EP 配对差异：`not_applicable_missing_matched_arm`。",
        "- 下一节点需要该 Arm 的明确用户授权；缺少匹配 Arm 时不形成双路结论。",
        "",
    ]
    (pair_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "pair_id": pair_id, "step": step, "available_arm": arm, "missing_arm": result["missing_arm"], "report": str(paired_report_path)}, ensure_ascii=False))
    return result


def _pair_report(config: Mapping[str, Any], step: int) -> dict[str, Any]:
    pair_id = config["training"]["pair_id"]
    run_ids = {arm: f"{pair_id}-{arm}" for arm in ("P", "EP")}
    eval_roots = {
        arm: _resolve(config["outputs"]["evaluation_root"]) / run_id / f"step-{step:04d}"
        for arm, run_id in run_ids.items()
    }
    completed: list[str] = []
    for arm, root in eval_roots.items():
        if not root.exists():
            continue
        report_path = root / "evaluation_report.json"
        manifest_path = root / "evaluation_manifest.json"
        if not report_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"A7 {arm} Step {step} evaluation output is incomplete")
        report = _json(report_path)
        manifest = _json(manifest_path)
        if (
            report.get("status") != "scored"
            or report.get("arm") != arm
            or report.get("checkpoint_step") != step
            or report.get("run_id") != run_ids[arm]
            or report.get("sample_count") != int(config["data"]["expected_dev_rows"])
            or report.get("frozen_ids_sha256") != config["identity"]["frozen_dev_ids_sha256"]
            or report.get("gold_read_after_predictions_locked") is not True
            or manifest.get("status") != "scored"
            or manifest.get("prediction_count") != int(config["data"]["expected_dev_rows"])
            or manifest.get("predictions_sha256") != report.get("locked_predictions_sha256")
            or report.get("locked_predictions_sha256") != sha256_file(root / "predictions.jsonl")
        ):
            raise ValueError(f"A7 {arm} Step {step} report is not complete for review")
        completed.append(arm)
    if not completed:
        raise FileNotFoundError("A7 has no scored Arm report to review")
    if len(completed) == 1:
        return _write_partial_pair_report(config, step, completed[0], eval_roots[completed[0]])

    reports = {arm: _json(eval_roots[arm] / "evaluation_report.json") for arm in ("P", "EP")}
    ids = [row["id"] for _line, row, _sha in iter_jsonl(eval_roots["P"] / "predictions.jsonl")]
    if ids != [row["id"] for _line, row, _sha in iter_jsonl(eval_roots["EP"] / "predictions.jsonl")]:
        raise ValueError("A7 P and EP predictions are not aligned to the same frozen ID order")
    p_predictions = [row for _line, row, _sha in iter_jsonl(eval_roots["P"] / "predictions.jsonl")]
    ep_predictions = [row for _line, row, _sha in iter_jsonl(eval_roots["EP"] / "predictions.jsonl")]
    profile_schema = _json(_resolve(config["data"]["train_schema"]))
    output_schema = _json(_resolve(config["data"]["output_schema"]))
    _labels, enums = schema_label_order(profile_schema, output_schema)
    gold_profiles = []
    references = unique_index(
        [row for _line, row, _sha in iter_jsonl(_resolve(config["data"]["frozen_dev_references"]))],
        "eval_id", "A7 pair report references",
    )
    for sample_id in ids:
        profile = references[sample_id]["target"]["user_profile"]
        _bits, _mask, normalized = profile_to_target(profile, enums, sample_id=sample_id)
        gold_profiles.append(normalized)
    p_calibration = _json(eval_roots["P"] / "calibration.json")["threshold_policy"]
    ep_calibration = _json(eval_roots["EP"] / "calibration.json")["threshold_policy"]
    p_threshold = p_calibration.get("threshold")
    ep_threshold = ep_calibration.get("threshold")
    threshold = float(p_threshold) if p_threshold is not None and ep_threshold == p_threshold else 0.5
    p_profiles = profile_from_probabilities([row["profile_probabilities"] for row in p_predictions], threshold=threshold, enums=enums)
    ep_profiles = profile_from_probabilities([row["profile_probabilities"] for row in ep_predictions], threshold=threshold, enums=enums)
    arm_comparison = compare_arm_profile_predictions(
        ids=ids, gold=gold_profiles, p_predictions=p_profiles, ep_predictions=ep_profiles, enums=enums,
    )
    pair_dir = _resolve(config["outputs"]["evaluation_root"]) / "pairs" / pair_id / f"step-{step:04d}"
    pair_report_path = pair_dir / "paired_report.json"
    if pair_report_path.is_file():
        previous = _json(pair_report_path)
        if previous.get("status") != "awaiting_manual_review_missing_matched_arm":
            raise FileExistsError("A7 paired node report already exists and is immutable")
        available = previous.get("available_arm")
        if not available or previous.get("evaluation_report_sha256") != sha256_file(eval_roots[available] / "evaluation_report.json"):
            raise ValueError("A7 partial pair report no longer matches its previously scored Arm")
    pair_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "E03-A7-paired-node-report-v1",
        "status": "awaiting_manual_review",
        "pair_id": pair_id,
        "checkpoint_step": step,
        "same_cohort": True,
        "cohort_count": len(ids),
        "cohort_ids_sha256": config["identity"]["frozen_dev_ids_sha256"],
        "shared_calibration_threshold": threshold,
        "calibration_thresholds_match": p_threshold == ep_threshold,
        "p_evaluation_manifest_sha256": sha256_file(eval_roots["P"] / "evaluation_manifest.json"),
        "ep_evaluation_manifest_sha256": sha256_file(eval_roots["EP"] / "evaluation_manifest.json"),
        "profile_only_emotion_score": "not_applicable",
        "p_vs_ep": arm_comparison,
        "recommendation": "continue both to the next review node only if the user approves; Step 512 is an early trend check, not a retention gate.",
        "next_step": {512: 1024, 1024: 2048, 2048: 4048}.get(step),
        "created_at_utc": _now(),
    }
    write_manifest(pair_report_path, result, overwrite=pair_report_path.exists())
    md = [
        f"# E03-A7 Step {step} P/EP 配对审阅",
        "",
        f"- 状态：`{result['status']}`；同一固定开发代理集：{len(ids)} 点。",
        f"- P/EP 共用阈值：`{threshold}`（两条 Cal 未同时满足同一阈值时回退到 0.5 口径）。",
        f"- EP 相对 P：画像 Micro-F1 Δ `{arm_comparison['ep_minus_p']['micro_f1']:.6f}`；TP Δ `{arm_comparison['ep_minus_p']['tp']}`；FP Δ `{arm_comparison['ep_minus_p']['fp']}`；Precision Δ `{arm_comparison['ep_minus_p']['micro_precision']:.6f}`。",
        "- P 的 Emotion 指标为 `not_applicable`。",
        "- Codex 建议只供审阅，不构成继续训练授权；下一节点须取得用户明确指令。",
        "",
    ]
    (pair_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "pair_id": result["pair_id"], "step": step, "next_step": result["next_step"], "report": str(pair_report_path)}, ensure_ascii=False))
    return result


def _load_training_data(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]], SFTPromptIndex]:
    supervision, _summary = build_supervision(
        _resolve(config["data"]["normalized_train"]),
        _resolve(config["data"]["a5_supervision"]),
        _json(_resolve(config["data"]["train_schema"])),
        _json(_resolve(config["data"]["output_schema"])),
        expected_rows=int(config["data"]["expected_d03_train_rows"]),
    )
    splits = read_splits(_resolve(config["data"]["a5_splits"]))
    prompt_index = SFTPromptIndex(_resolve(config["data"]["sft_train"]), list(supervision))
    return supervision, splits, prompt_index


def _resume_locked_scoring(
    config: Mapping[str, Any],
    config_sha: str,
    training_code_sha: str,
    pair_id: str,
    arm: str,
    step: int,
    eval_root: Path,
    a5_config: Mapping[str, Any],
    assets: Mapping[str, Any],
    supervision: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Resume scoring only from immutable Dev predictions and Calibration artifacts."""
    manifest_path = eval_root / "evaluation_manifest.json"
    manifest = _json(manifest_path)
    predictions_path = eval_root / "predictions.jsonl"
    calibration_predictions_path = eval_root / "calibration_probabilities.jsonl"
    calibration_manifest_path = eval_root / "calibration.json"
    expected_run_id = f"{pair_id}-{arm}"
    prediction_sha = sha256_file(predictions_path)
    calibration_sha = sha256_file(calibration_predictions_path)
    if (
        manifest.get("status") not in {"scoring_started_after_prediction_lock", "scoring_interrupted_after_gold_read"}
        or manifest.get("gold_read") is not True
        or manifest.get("gold_read_after_predictions_locked") is not True
        or manifest.get("run_id") != expected_run_id
        or manifest.get("arm") != arm
        or manifest.get("checkpoint_step") != step
        or manifest.get("config_sha256") != config_sha
        or manifest.get("training_code_sha256") != training_code_sha
        or manifest.get("evaluation_code_sha256") != e03_a7_train._code_sha256()
        or manifest.get("predictions_sha256") != prediction_sha
        or manifest.get("calibration_predictions_sha256") != calibration_sha
        or manifest.get("prediction_count") != int(config["data"]["expected_dev_rows"])
        or manifest.get("prediction_ids_sha256") != config["identity"]["frozen_dev_ids_sha256"]
    ):
        raise ValueError("A7 locked-scoring recovery is not bound to the current Arm, node, code, and prediction hashes")

    run_root = _resolve(config["outputs"]["training_root"]) / expected_run_id
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = _json(run_manifest_path)
    node = run_manifest.get("nodes", {}).get(str(step), {})
    checkpoint_manifest_path = run_root / "trainer" / f"checkpoint-{step}" / "checkpoint_manifest.json"
    if (
        run_manifest.get("code_sha256") != training_code_sha
        or run_manifest.get("optimizer_step") != step
        or node.get("training_status") != "passed"
        or sha256_file(checkpoint_manifest_path) != manifest.get("checkpoint_manifest_sha256")
    ):
        raise ValueError("A7 locked-scoring recovery checkpoint differs from its completed training node")

    calibration_manifest = _json(calibration_manifest_path)
    calibration_ids = list(splits["calibration"])
    if (
        calibration_manifest.get("run_id") != expected_run_id
        or calibration_manifest.get("arm") != arm
        or calibration_manifest.get("checkpoint_step") != step
        or calibration_manifest.get("sample_count") != len(calibration_ids)
        or calibration_manifest.get("sample_ids_sha256") != canonical_sha256(calibration_ids)
        or calibration_manifest.get("predictions_sha256") != calibration_sha
    ):
        raise ValueError("A7 locked-scoring recovery Calibration artifacts are incomplete or changed")

    prediction_rows = [row for _line, row, _sha in iter_jsonl(predictions_path)]
    if (
        [row.get("id") for row in prediction_rows] != list(assets["frozen_ids"])
        or len(prediction_rows) != int(config["data"]["expected_dev_rows"])
        or selected_id_sha256([row["id"] for row in prediction_rows]) != config["identity"]["frozen_dev_ids_sha256"]
    ):
        raise ValueError("A7 locked-scoring recovery prediction IDs differ from the frozen Dev cohort")

    try:
        return _score_after_prediction_freeze(
            config=config,
            arm=arm,
            step=step,
            eval_dir=eval_root,
            eval_manifest_path=manifest_path,
            prediction_rows=prediction_rows,
            calibration=calibration_manifest["threshold_policy"],
            sources=assets,
            a5_config=a5_config,
            supervision=supervision,
            splits=splits,
            evaluation_assets=assets,
            start_time=time.monotonic(),
            training_code_sha256=training_code_sha,
        )
    except BaseException as error:
        _mark_scoring_interrupted(manifest_path, error)
        raise


def _evaluate_pair(args: argparse.Namespace) -> int:
    import torch

    config, config_sha = e03_a7_train._load_config(_resolve(args.config))
    pair_id = args.pair_id or config["training"]["pair_id"]
    if pair_id != config["training"]["pair_id"]:
        raise ValueError("A7 evaluation pair ID must match the frozen config")
    step = int(args.checkpoint_step)
    if step not in (512, 1024, 2048, 4048):
        raise ValueError("A7 evaluation node must be 512, 1024, 2048, or 4048")
    preflight, _preflight_root = e03_a7_train._load_preflight_for_training(config, config_sha)
    a5_config, _a5_sha, _audit = e03_a7_train._a5_assets(config)
    evaluation_assets = _load_evaluation_sources(config, a5_config)
    supervision, splits, prompt_index = _load_training_data(config)
    arms = ("P", "EP") if args.mode == "evaluate-pair" else (args.arm,)
    for arm in arms:
        eval_root = _resolve(config["outputs"]["evaluation_root"]) / f"{pair_id}-{arm}" / f"step-{step:04d}"
        if eval_root.exists():
            report_path = eval_root / "evaluation_report.json"
            manifest_path = eval_root / "evaluation_manifest.json"
            if report_path.is_file() and manifest_path.is_file():
                report = _json(report_path)
                manifest = _json(manifest_path)
                if (
                    report.get("status") != "scored"
                    or report.get("arm") != arm
                    or report.get("checkpoint_step") != step
                    or manifest.get("status") != "scored"
                    or manifest.get("predictions_sha256") != report.get("locked_predictions_sha256")
                ):
                    raise ValueError(f"A7 {arm} existing evaluation output does not match this node")
            elif manifest_path.is_file() and (eval_root / "predictions.jsonl").is_file() and (eval_root / "calibration.json").is_file():
                _resume_locked_scoring(
                    config,
                    config_sha,
                    preflight["code_sha256"],
                    pair_id,
                    arm,
                    step,
                    eval_root,
                    a5_config,
                    evaluation_assets,
                    supervision,
                    splits,
                )
            else:
                raise FileExistsError(f"A7 {arm} evaluation output is incomplete and cannot resume from a prediction lock: {eval_root}")
        else:
            _evaluate_arm(
                config, config_sha, preflight["code_sha256"], a5_config, arm, step,
                assets=evaluation_assets,
                supervision=supervision,
                splits=splits,
                prompt_index=prompt_index,
            )
            torch.cuda.empty_cache()
    _pair_report(config, step)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=e03_a7_train.DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("evaluate-arm", "evaluate-pair"), required=True)
    parser.add_argument("--arm", choices=("P", "EP"))
    parser.add_argument("--pair-id")
    parser.add_argument("--checkpoint-step", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "evaluate-arm" and args.arm is None:
        raise SystemExit("--mode evaluate-arm requires --arm P|EP")
    if args.mode == "evaluate-pair" and args.arm is not None:
        raise SystemExit("--mode evaluate-pair does not accept --arm")
    return _evaluate_pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
