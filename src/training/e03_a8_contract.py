"""Frozen E03-A8 config, conditional stage authorization, and quality gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

EXPERIMENT_ID = "E03-A8"
TRAINING_SCOPE = "generative_profile_specialist"
PLANNED_STEPS = 4048
AUTHORIZED_MAX_STEP = 2048
STAGE_STEPS = (512, 1024, 2048)
ACCUMULATION_STEPS = 4
USER_AUTHORIZATION_SOURCE = "explicit_user_request_2026-11-07"
USER_AUTHORIZATION_TEXT = (
    "A8的方案继续跑，如果它的质量过关就可以一直跑下去，如果不过关就需要留给我review，最多跑到2048步，"
    "前面的步数，我可以后来在补review，但再往后就必须要有我的批准了。"
    "A8的质量关不需要卡的太死，反正2048步之前的可以宽松，但是完全没希望的糟糕结果的那种就不需要继续了。"
)
DATA_CODE_PATHS = (
    "configs/data/e03_a8.yaml",
    "configs/prompts/p_a8_profile.txt",
    "configs/prompts/p_a8_emotion_profile.txt",
    "scripts/e03_a8_prepare.py",
    "src/data/e03_a8_sft.py",
    "src/shared/e03_a8_protocol.py",
    "src/shared/prompting.py",
)
TRAINING_CODE_PATHS = (
    "configs/training/e03_a8_p_staged.yaml",
    "configs/training/e03_a8_ep_staged.yaml",
    "src/training/e03_a8_contract.py",
    "src/training/e03_a8_staged.py",
    "scripts/e03_a8_staged_train.py",
    *DATA_CODE_PATHS,
)
EVALUATION_CODE_PATHS = (
    "configs/evaluation/e03_a8_staged_fast.yaml",
    "scripts/e03_a8_staged_eval.py",
    "src/evaluation/e03_a8_staged_metrics.py",
    "src/evaluation/e03_a6_f4_staged_metrics.py",
    "src/shared/e03_a8_protocol.py",
    "src/shared/prompting.py",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_a8_training_config(config: Mapping[str, Any], arm: str) -> None:
    if arm not in {"P", "EP"}:
        raise ValueError(f"unsupported E03-A8 arm: {arm!r}")
    if (
        config.get("version") != "E03-A8-staged-training-v1"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("training_scope") != TRAINING_SCOPE
        or config.get("arm") != arm
        or config.get("status") != "ready_for_preflight"
    ):
        raise ValueError("A8 config version, experiment, scope, arm, or preflight status differs")
    base = config.get("base_model", {})
    if base != {
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "local_path": "participant/model",
        "expected_weight_sha256": "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5",
    }:
        raise ValueError("A8 Base model identity differs from the frozen E01-B Base")
    data = config.get("data", {})
    expected_train = "outputs/data/e03_a8/p_train.jsonl" if arm == "P" else "outputs/data/e03_a8/ep_train.jsonl"
    expected_root = "outputs/training/E03-A8/P" if arm == "P" else "outputs/training/E03-A8/EP"
    if (
        data.get("split") != "train"
        or data.get("expected_train_rows") != 16192
        or data.get("train_path") != expected_train
        or data.get("dataset_manifest") != "outputs/data/e03_a8/manifest.json"
        or data.get("sample_order_path") != "outputs/data/e03_a8/sample_order.jsonl"
        or data.get("output_root") != expected_root
        or data.get("init_adapter") is not None
        or data.get("resume_from_checkpoint") is not None
    ):
        raise ValueError("A8 must use the frozen paired Train data and an independent new Run namespace")
    protocol = config.get("protocol", {})
    expected_prompt_path = "configs/prompts/p_a8_profile.txt" if arm == "P" else "configs/prompts/p_a8_emotion_profile.txt"
    expected_prompt_sha = "a765196b4d182cd792c7c322e2f052d62e4d1c18f619c321cee3893b50ca3ec7" if arm == "P" else "cdde50dd10f2ea0bdf8f316b85d9ca5c593c3aa305cf4152812c7c647333556b"
    if (
        protocol.get("arm") != arm
        or protocol.get("completion_version") != "A8-GENCLASS-v1"
        or protocol.get("prompt_path") != expected_prompt_path
        or protocol.get("prompt_sha256") != expected_prompt_sha
        or protocol.get("selection_prompt_path") != "configs/prompts/p0.txt"
        or protocol.get("p0_sha256") != "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2"
        or protocol.get("history_policy") != "P0_T1_then_common_A8_P_EP_history"
        or protocol.get("max_prompt_tokens") != 1280
        or protocol.get("max_train_seq_length") != 2048
        or protocol.get("packing") is not False
        or protocol.get("padding_free") is not False
    ):
        raise ValueError("A8 prompt, P0/T1, sequence budget, or completion protocol changed")
    quant = config.get("quantization", {})
    if quant != {
        "method": "bitsandbytes_4bit",
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quant": True,
        "device_map": {"": 0},
        "max_cuda_memory_fraction": 0.92,
    }:
        raise ValueError("A8 requires the frozen NF4/double-quant/BF16 CUDA profile")
    lora = config.get("lora", {})
    if (
        lora.get("r") != 8
        or lora.get("alpha") != 16
        or lora.get("dropout") != 0.0
        or lora.get("bias") != "none"
        or lora.get("checkpoint_dtype") != "float32"
        or set(lora.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ):
        raise ValueError("A8 LoRA identity differs from the frozen seven-target profile")
    if config.get("optimizer") != {
        "name": "AdamW", "learning_rate": 0.0001, "lr_scheduler": "constant",
        "warmup_steps": 0, "weight_decay": 0.0,
    }:
        raise ValueError("A8 optimizer must remain AdamW 1e-4 with a constant schedule and no decay")
    training = config.get("training", {})
    exact = {
        "seed": 42,
        "per_device_micro_batch_size": 1,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "max_optimizer_steps": PLANNED_STEPS,
        "save_interval_steps": 512,
        "gradient_checkpointing": True,
    }
    if any(training.get(key) != value for key, value in exact.items()):
        raise ValueError("A8 training schedule or batch configuration changed")
    authorization = config.get("authorization", {})
    if authorization != {
        "authorized_max_step": AUTHORIZED_MAX_STEP,
        "conditional_until_step": AUTHORIZED_MAX_STEP,
        "no_tp_consecutive_nodes": 2,
        "minimum_strict_parse_rate": 0.98,
        "source": USER_AUTHORIZATION_SOURCE,
    }:
        raise ValueError("A8 bounded conditional authorization or loose stop gate changed")
    evaluation = config.get("evaluation", {})
    if evaluation.get("do_sample") is not False or evaluation.get("max_new_tokens") != 256 or evaluation.get("autocast_adapter_dtype") is not True:
        raise ValueError("A8 evaluation decode or adapter reload dtype contract changed")


def validate_stage_transition(from_step: int | None, to_step: int) -> None:
    expected = {None: 512, 512: 1024, 1024: 2048}
    if expected.get(from_step) != to_step:
        raise ValueError(f"A8 transition {from_step!r}->{to_step} exceeds the frozen staged path")
    if to_step > AUTHORIZED_MAX_STEP:
        raise ValueError("A8 training above Step 2048 requires new explicit user approval")


def _is_zero_tp_summary(summary: Mapping[str, Any]) -> bool:
    cohorts = summary.get("cohorts", {})
    return set(cohorts) == {"pilot160", "dev224"} and all(
        int(cohorts[name].get("profile_33_label", {}).get("overall_33_label_tp", -1)) == 0
        for name in ("pilot160", "dev224")
    )


def conditional_quality_decision(
    current: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> dict[str, Any]:
    if current.get("engineering_gate") != "passed":
        return {"decision": "awaiting_manual_review", "reason": "engineering_gate_failed"}
    cohorts = current.get("cohorts", {})
    if set(cohorts) != {"pilot160", "dev224"}:
        return {"decision": "awaiting_manual_review", "reason": "fixed_cohort_coverage_incomplete"}
    parse_floor = 0.98
    parse_rates = {name: float(cohorts[name].get("strict_parse_rate", 0.0)) for name in cohorts}
    if any(rate < parse_floor for rate in parse_rates.values()):
        return {
            "decision": "awaiting_manual_review",
            "reason": "strict_parse_rate_below_98_percent",
            "parse_rates": parse_rates,
        }
    consecutive_zero_tp = _is_zero_tp_summary(current) and previous is not None and _is_zero_tp_summary(previous)
    if consecutive_zero_tp:
        return {
            "decision": "awaiting_manual_review",
            "reason": "two_consecutive_nodes_with_zero_tp_on_both_cohorts",
            "current_zero_tp_both_cohorts": True,
            "previous_zero_tp_both_cohorts": True,
            "parse_rates": parse_rates,
        }
    return {
        "decision": "conditionally_authorized",
        "reason": "loose_quality_gate_passed_within_user_authorized_range",
        "current_zero_tp_both_cohorts": _is_zero_tp_summary(current),
        "previous_zero_tp_both_cohorts": _is_zero_tp_summary(previous) if previous else None,
        "parse_rates": parse_rates,
        "authorized_next_step_limit": AUTHORIZED_MAX_STEP,
    }


def write_conditional_authorization(
    path: Path,
    *,
    run_id: str,
    arm: str,
    from_step: int,
    to_step: int,
    checkpoint_manifest_sha256: str,
    evaluation_summary_sha256: str,
    previous_evaluation_summary_sha256: str | None,
    config_sha256: str,
    data_manifest_sha256: str,
    code_sha256: str,
    quality_decision: Mapping[str, Any],
) -> str:
    validate_stage_transition(from_step, to_step)
    if arm not in {"P", "EP"} or quality_decision.get("decision") != "conditionally_authorized":
        raise ValueError("A8 conditional receipt requires a passing per-Arm quality gate")
    record = {
        "version": "E03-A8-conditional-authorization-v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "arm": arm,
        "from_step": from_step,
        "to_step": to_step,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "evaluation_summary_sha256": evaluation_summary_sha256,
        "previous_evaluation_summary_sha256": previous_evaluation_summary_sha256,
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "code_sha256": code_sha256,
        "quality_decision": dict(quality_decision),
        "authorization_source": USER_AUTHORIZATION_SOURCE,
        "authorization_text": USER_AUTHORIZATION_TEXT,
    }
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite A8 conditional authorization: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def validate_conditional_authorization(
    path: Path,
    *,
    run_id: str,
    arm: str,
    from_step: int,
    to_step: int,
    checkpoint_manifest_sha256: str,
    evaluation_summary_sha256: str,
    previous_evaluation_summary_sha256: str | None,
    config_sha256: str,
    data_manifest_sha256: str,
    code_sha256: str,
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "version": "E03-A8-conditional-authorization-v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "arm": arm,
        "from_step": from_step,
        "to_step": to_step,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "evaluation_summary_sha256": evaluation_summary_sha256,
        "previous_evaluation_summary_sha256": previous_evaluation_summary_sha256,
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "code_sha256": code_sha256,
        "authorization_source": USER_AUTHORIZATION_SOURCE,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"A8 conditional authorization {key} differs from the current Run")
    if record.get("quality_decision", {}).get("decision") != "conditionally_authorized":
        raise ValueError("A8 conditional authorization does not contain a passing quality decision")
    validate_stage_transition(from_step, to_step)
    return record
