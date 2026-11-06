"""Frozen config, checkpoint progression, and sampler evidence for E03-A6 F4 Full."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT_ID = "E03-A6-F4-STAGED-FULL"
TRAINING_SCOPE = "staged_full_train"
MAX_STEPS = 4048
STAGE_STEPS = (512, 1024, 2048, 4048)
ACCUMULATION_STEPS = 4
USER_AUTHORIZATION_SOURCE = "explicit_user_request_2026-11-06"
USER_AUTHORIZATION_TEXT = (
    "先查看当前的文档，了解当前项目进度，然后进行执行E0.3-A6_F4分阶段Full训练与FastOnly评测.md 该任务。"
    "你可以在需要涉及到到GPU指令进行训练的时候进行在沙箱外执行。"
    "然后我要求你必须要把任务执行完整全部反馈全部做完才准结束，不准擅自停止执行。"
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


def validate_staged_full_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "E03-A6-F4-STAGED-FULL-training-v1":
        raise ValueError("unsupported E03-A6 F4 staged-full training config")
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("training_scope") != TRAINING_SCOPE:
        raise ValueError("F4 Full requires its isolated experiment ID and staged_full_train scope")
    if config.get("status") != "ready_for_staged_full_preflight":
        raise ValueError("F4 Full config must retain its preflight-only status")
    base = config.get("base_model", {})
    if (
        base.get("model_id") != "Qwen/Qwen3-1.7B"
        or base.get("revision") != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        or base.get("expected_weight_sha256") != "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
    ):
        raise ValueError("F4 Full Base model identity differs from the frozen E01-B Base")
    data = config.get("data", {})
    if (
        data.get("split") != "train"
        or data.get("expected_train_rows") != 16192
        or data.get("expected_source_train_rows") != 16312
        or data.get("expected_historical_excluded_rows") != 120
        or data.get("init_adapter") is not None
        or data.get("resume_from_checkpoint") is not None
    ):
        raise ValueError("F4 Full must use all 16,192 frozen F2 Train IDs, no adapter init, and Train only")
    protocol = config.get("protocol", {})
    if (
        protocol.get("format_version") != "F4"
        or protocol.get("prompt_version") != "P_A6_F4"
        or protocol.get("prompt_sha256") != "84a29d425d1ed2bc61cdf83b47713538738c80b0b625daf486ea9d744b2832cf"
        or protocol.get("selection_prompt_version") != "P0"
        or protocol.get("p0_sha256") != "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2"
        or protocol.get("truncation_version") != "T1"
        or protocol.get("max_prompt_tokens") != 1280
        or protocol.get("max_train_seq_length") != 2048
        or protocol.get("packing") is not False
        or protocol.get("padding_free") is not False
    ):
        raise ValueError("F4 Full must preserve P_A6_F4, P0/T1, and fixed P/S budgets")
    quant = config.get("quantization", {})
    if (
        quant.get("method") != "bitsandbytes_4bit"
        or quant.get("quant_type") != "nf4"
        or quant.get("compute_dtype") != "bfloat16"
        or quant.get("use_double_quant") is not True
        or quant.get("device_map") != {"": 0}
        or float(quant.get("max_cuda_memory_fraction", 0)) != 0.92
    ):
        raise ValueError("F4 Full quantization must remain NF4/BF16 with double quantization and the frozen 92% CUDA budget")
    lora = config.get("lora", {})
    if (
        lora.get("r") != 8
        or lora.get("alpha") != 16
        or lora.get("dropout") != 0.0
        or lora.get("bias") != "none"
        or lora.get("checkpoint_dtype") != "float32"
        or set(lora.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ):
        raise ValueError("F4 Full LoRA identity differs from the frozen seven-target FP32 profile")
    optimizer = config.get("optimizer", {})
    if optimizer != {
        "name": "AdamW", "learning_rate": 0.0001, "lr_scheduler": "constant",
        "warmup_steps": 0, "weight_decay": 0.0,
    }:
        raise ValueError("F4 Full optimizer must remain AdamW 1e-4, constant, no warmup/decay")
    training = config.get("training", {})
    exact = {
        "seed": 42,
        "per_device_micro_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "max_optimizer_steps": MAX_STEPS,
        "epochs": 1,
        "save_interval_steps": 512,
        "gradient_checkpointing": True,
        "evaluation_interval_steps": 0,
        "checkpointing": True,
        "train_sampling_strategy": "random",
    }
    changed = {key: (training.get(key), value) for key, value in exact.items() if training.get(key) != value}
    if changed:
        raise ValueError(f"F4 Full fixed training plan changed: {changed}")
    if int(training.get("save_total_limit", 0)) < len(STAGE_STEPS) + 4:
        raise ValueError("F4 Full must retain every stage checkpoint plus intermediate saves")
    if int(training.get("max_training_seconds", 0)) < 1 or int(training.get("min_free_disk_gb", 0)) < 1:
        raise ValueError("F4 Full requires explicit positive training-time and free-disk budgets")
    evaluation = config.get("evaluation", {})
    if evaluation.get("do_sample") is not False or evaluation.get("autocast_adapter_dtype") is not True:
        raise ValueError("F4 Full requires Greedy generation and FP32 Adapter restore")
    if int(evaluation.get("sft_dev_loss_rows", -1)) != 0:
        raise ValueError("F4 staged Full does not read or score teacher-forced Dev targets")


def validate_eval_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "E03-A6-F4-STAGED-FAST-eval-v1":
        raise ValueError("unsupported F4 staged-fast evaluation config")
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("training_scope") != TRAINING_SCOPE:
        raise ValueError("F4 Fast-only evaluation must bind to the staged-full experiment")
    protocol = config.get("protocol", {})
    if (
        protocol.get("format_version") != "F4"
        or protocol.get("prompt_version") != "P_A6_F4"
        or protocol.get("prompt_sha256") != "84a29d425d1ed2bc61cdf83b47713538738c80b0b625daf486ea9d744b2832cf"
        or protocol.get("selection_prompt_version") != "P0"
        or protocol.get("p0_sha256") != "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2"
        or protocol.get("truncation_version") != "T1"
        or protocol.get("max_prompt_tokens") != 1280
    ):
        raise ValueError("F4 evaluation prompt or P0/T1 history identity changed")
    generation = config.get("generation", {})
    if generation.get("do_sample") is not False or generation.get("seed") != 42 or generation.get("fast_max_new_tokens") != 256:
        raise ValueError("F4 Fast-only decode must remain Greedy with seed 42 and 256-token frozen cap")
    if config.get("comparison") != {
        "fixed_denominator": True,
        "deduplicate_union_generation": True,
        "score_failures_as_empty": True,
        "fast_prefix_parse_minimum": 0.98,
        "references_read_only_after_prediction_freeze": True,
    }:
        raise ValueError("F4 evaluation fixed-cohort and scoring policy changed")
    cohorts = config.get("cohorts", {})
    if set(cohorts) != {"pilot160", "dev224"}:
        raise ValueError("F4 Fast-only must retain the frozen 160 and 224 cohorts")
    expected = {"pilot160": (160, "98f24f1c1b41e1676618f4e557daa402b3cbb80686916ca818e1bdc1b89ae49c"),
                "dev224": (224, "d3c84beb6227e87c0fd09e425619635cd9fc6bac5b087e3f1010fb089f4ff8d2")}
    for name, (count, ids_hash) in expected.items():
        cohort = cohorts[name]
        if cohort.get("expected_count") != count or cohort.get("expected_ids_sha256") != ids_hash:
            raise ValueError(f"F4 evaluation {name} cohort identity changed")


def validate_stage_transition(from_step: int | None, to_step: int) -> None:
    expected = {None: 512, 512: 1024, 1024: 2048, 2048: 4048}
    if expected.get(from_step) != to_step:
        raise ValueError(f"invalid F4 staged transition: {from_step!r} -> {to_step}; allowed={expected}")


def _read_order_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"sample-order log has a blank row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"sample-order row {line_number} is not an object")
            rows.append(row)
    return rows


class SampleOrderRecorder:
    """Persist exactly four sample IDs per optimizer step and verify one-pass coverage."""

    def __init__(self, path: Path, ordered_train_ids: Sequence[str], *, accumulation_steps: int = ACCUMULATION_STEPS):
        if len(ordered_train_ids) != len(set(ordered_train_ids)) or any(not value for value in ordered_train_ids):
            raise ValueError("ordered F4 Train IDs must be non-empty and unique")
        self.path = path
        self.expected_ids = set(ordered_train_ids)
        self.accumulation_steps = accumulation_steps
        self.rows = _read_order_rows(path)
        self.pending: list[str] = []
        self._validate_rows()

    def _flatten(self) -> list[str]:
        return [sample_id for row in self.rows for sample_id in row["sample_ids"]]

    def _validate_rows(self) -> None:
        observed: list[str] = []
        for index, row in enumerate(self.rows, 1):
            ids = row.get("sample_ids")
            if row.get("optimizer_step") != index or not isinstance(ids, list) or len(ids) != self.accumulation_steps:
                raise ValueError("sample-order log does not have one four-ID row per optimizer step")
            if any(sample_id not in self.expected_ids for sample_id in ids):
                raise ValueError("sample-order log contains an ID outside the frozen F4 Train cohort")
            observed.extend(ids)
        if len(observed) != len(set(observed)):
            raise ValueError("sample-order log repeats a Train ID before the one-epoch boundary")

    @property
    def optimizer_steps(self) -> int:
        return len(self.rows)

    def record_micro_batch(self, sample_ids: Sequence[str]) -> None:
        values = [str(value) for value in sample_ids]
        if not values or any(value not in self.expected_ids for value in values):
            raise ValueError("actual Trainer micro-batch contains an unknown F4 Train ID")
        if len(values) != 1:
            raise ValueError("F4 Full micro_batch must remain exactly one sample")
        self.pending.extend(values)

    def commit_optimizer_step(self, step: int) -> dict[str, Any]:
        if step != len(self.rows) + 1 or len(self.pending) != self.accumulation_steps:
            raise ValueError("actual Trainer optimizer step and recorded sample consumption disagree")
        known = set(self._flatten())
        if len(set(self.pending)) != len(self.pending) or any(value in known for value in self.pending):
            raise ValueError("actual Trainer repeated a sample in the one-epoch F4 cohort")
        row = {"optimizer_step": step, "sample_ids": list(self.pending)}
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        self.rows.append(row)
        self.pending.clear()
        return self.checkpoint_facts(step)

    def checkpoint_facts(self, step: int) -> dict[str, Any]:
        if step != len(self.rows) or self.pending:
            raise ValueError("sample-order checkpoint requested before a complete optimizer-step boundary")
        ids = self._flatten()
        if len(ids) != step * self.accumulation_steps or len(ids) != len(set(ids)):
            raise ValueError("sample-order checkpoint has duplicated/missing micro-batch consumption")
        return {
            "sample_order_steps": step,
            "sample_order_micro_batches": len(ids),
            "sample_order_unique_ids": len(set(ids)),
            "sample_order_ordered_ids_sha256": canonical_sha256(ids),
            "sample_order_consumed_ids_set_sha256": canonical_sha256(sorted(ids)),
            "sample_order_log_sha256": sha256_file(self.path),
            "sample_order_no_duplicates": True,
        }

    def verify_full_epoch(self) -> dict[str, Any]:
        ids = self._flatten()
        if self.optimizer_steps != MAX_STEPS or set(ids) != self.expected_ids:
            raise ValueError("F4 Full sample order did not consume every Train ID exactly once")
        facts = self.checkpoint_facts(MAX_STEPS)
        facts["sample_order_exact_one_epoch"] = True
        return facts


def rewind_sample_order_log(path: Path, checkpoint_step: int, run_root: Path) -> None:
    """Archive any post-checkpoint sampler rows and preserve the verified checkpoint prefix."""
    rows = _read_order_rows(path)
    if len(rows) < checkpoint_step:
        raise ValueError("sample-order log ends before the requested checkpoint")
    prefix = rows[:checkpoint_step]
    if [row.get("optimizer_step") for row in prefix] != list(range(1, checkpoint_step + 1)):
        raise ValueError("sample-order log does not cover the exact checkpoint prefix")
    if len(rows) > checkpoint_step:
        from datetime import datetime, timezone

        archive = run_root / "interrupted_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "sample_order.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in prefix),
            encoding="utf-8",
        )


def write_stage_authorization(
    path: Path,
    *,
    run_id: str,
    from_step: int,
    to_step: int,
    checkpoint_manifest_sha256: str,
) -> dict[str, Any]:
    validate_stage_transition(from_step, to_step)
    if len(checkpoint_manifest_sha256) != 64:
        raise ValueError("stage authorization requires the previous checkpoint manifest SHA-256")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite stage authorization: {path}")
    value = {
        "status": "approved_by_explicit_user_instruction",
        "run_id": run_id,
        "from_step": from_step,
        "to_step": to_step,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "authorization_source": USER_AUTHORIZATION_SOURCE,
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "node_metrics_individually_reviewed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return value


def validate_stage_authorization(
    path: Path,
    *,
    run_id: str,
    from_step: int,
    to_step: int,
    checkpoint_manifest_sha256: str,
) -> dict[str, Any]:
    validate_stage_transition(from_step, to_step)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "approved_by_explicit_user_instruction",
        "run_id": run_id,
        "from_step": from_step,
        "to_step": to_step,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "authorization_source": USER_AUTHORIZATION_SOURCE,
        "authorization_text": USER_AUTHORIZATION_TEXT,
        "node_metrics_individually_reviewed": False,
    }
    if value != expected:
        raise ValueError("stage approval does not match the current Run, node, checkpoint, or user directive")
    return value
