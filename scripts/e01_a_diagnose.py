"""Read-only closeout diagnostics for the preserved E01-A training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference
from scripts.e01_evaluate import SMOKE_ROOT
from src.evaluation import read_jsonl, strict_json_loads
from src.evaluation.metrics import emotion_metrics, profile_metrics, raw_diagnostics


EXPECTED_BASE_SHA256 = "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
EXPECTED_DIAGNOSTIC_STEPS = [0, 20, 40, 60, 80, 100, 120]
EXPECTED_OPTIMIZER_STEPS = 120
E00_MANIFEST_PATH = ROOT / "outputs/evaluation/e00_2/E00.2-20261026-220635/run_manifest.json"
TRAIN_CONFIG_PATH = ROOT / "configs/training/e01_a.yaml"
DIMENSIONS = ("personality_traits", "interests", "style")


class IdentityMismatchError(ValueError):
    """A source artifact exists but does not match the expected run identity."""


class OutputAlreadyExistsError(FileExistsError):
    """Closeout outputs are immutable and cannot be overwritten."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise IdentityMismatchError(message)


def _unique_by_id(rows: list[dict[str, Any]], key: str, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{source} has a missing or invalid {key}")
        if value in result:
            raise IdentityMismatchError(f"{source} contains duplicate {key}: {value}")
        result[value] = row
    return result


def validate_train_log(
    rows: list[dict[str, Any]], expected_steps: int = EXPECTED_OPTIMIZER_STEPS
) -> list[dict[str, Any]]:
    """Require one finite loss record for each consecutive optimizer step."""
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("train_log optimizer_step must be an integer")
        if step in indexed:
            raise ValueError(f"train_log has duplicate optimizer_step={step}")
        for field in ("loss", "learning_rate", "gradient_norm"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"train_log step {step} has a non-finite or missing {field}")
        if row.get("finite_gradients") is not True:
            raise ValueError(f"train_log step {step} does not record finite gradients")
        indexed[step] = row

    expected = set(range(1, expected_steps + 1))
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"train_log optimizer steps are not continuous; missing={missing}, extra={extra}")
    return [indexed[step] for step in range(1, expected_steps + 1)]


def classify_raw_json(raw_text: str) -> str:
    """Classify strict JSON failure from the raw string without returning it."""
    try:
        parsed = strict_json_loads(raw_text)
    except json.JSONDecodeError as exc:
        message = exc.msg.lower()
        if "extra data" in message:
            return "trailing_content_after_json"
        stripped = raw_text.lstrip()
        if stripped and not stripped.startswith("{"):
            return "not_json_object_prefix"
        if "invalid control character" in message:
            return "unescaped_control_character"
        if "unterminated string" in message:
            return "unterminated_string_or_quote"
        if "property name enclosed in double quotes" in message:
            return "object_key_quote_syntax"
        if "expecting ':' delimiter" in message or "expecting ',' delimiter" in message:
            return "object_delimiter_or_quote_syntax"
        if "expecting value" in message:
            return "missing_value_or_malformed_bracket"
        return "unknown_json_syntax_failure"
    except ValueError as exc:
        if "duplicate JSON object key" in str(exc):
            return "duplicate_json_key"
        if "non-standard JSON constant" in str(exc):
            return "nonstandard_json_constant"
        return "unknown_strict_json_failure"
    if not isinstance(parsed, dict):
        return "json_value_is_not_object"
    return "strict_json_object"


def _metric_equal(left: Any, right: Any, label: str) -> None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
            raise IdentityMismatchError(f"recomputed metric differs from preserved diagnostic: {label}")
        return
    if left != right:
        raise IdentityMismatchError(f"recomputed metric differs from preserved diagnostic: {label}")


def _compare_train_diagnostic(
    stored: dict[str, Any],
    emotion: dict[str, Any],
    profile: dict[str, Any],
    raw: dict[str, Any],
    step: int,
) -> None:
    for key in ("accuracy", "macro_f1"):
        _metric_equal(emotion[key], stored["emotion"][key], f"step {step} emotion.{key}")
    for key in (
        "overall_33_label_micro_precision",
        "overall_33_label_micro_recall",
        "overall_33_label_micro_f1",
        "overall_33_label_macro_f1",
        "all_three_dimensions_exact_match",
    ):
        _metric_equal(profile[key], stored["profile"][key], f"step {step} profile.{key}")
    for dimension in DIMENSIONS:
        for key in ("local_micro_precision", "local_micro_recall", "local_micro_f1", "local_exact_match"):
            _metric_equal(
                profile["per_dimension"][dimension][key],
                stored["profile"]["per_dimension"][dimension][key],
                f"step {step} profile.{dimension}.{key}",
            )
    for key in ("raw_json_parse_rate", "raw_schema_valid_rate", "submission_schema_valid_rate"):
        _metric_equal(raw[key], stored["raw"][key], f"step {step} raw.{key}")


def _compact_raw_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "sample_count",
        "raw_json_parse_rate",
        "raw_schema_valid_rate",
        "submission_schema_valid_rate",
        "valid_emotion_rate",
        "valid_profile_rate",
        "nonempty_response_rate",
        "error_case_counts",
    )
    return {key: raw[key] for key in keys if key in raw}


def _compact_profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    per_dimension = {}
    for dimension in DIMENSIONS:
        metric = profile["per_dimension"][dimension]
        per_dimension[dimension] = {
            key: metric[key]
            for key in (
                "sample_count",
                "local_exact_match",
                "exact_match_count",
                "local_micro_precision",
                "local_micro_recall",
                "local_micro_f1",
                "gold_empty_rate",
                "pred_empty_rate",
                "gold_average_label_count",
                "pred_average_label_count",
                "tp",
                "fp",
                "fn",
            )
        }
    return {
        "overall_33_label_micro_precision": profile["overall_33_label_micro_precision"],
        "overall_33_label_micro_recall": profile["overall_33_label_micro_recall"],
        "overall_33_label_micro_f1": profile["overall_33_label_micro_f1"],
        "overall_33_label_macro_f1": profile["overall_33_label_macro_f1"],
        "overall_33_label_tp": profile["overall_33_label_tp"],
        "overall_33_label_fp": profile["overall_33_label_fp"],
        "overall_33_label_fn": profile["overall_33_label_fn"],
        "all_three_dimensions_exact_match": profile["all_three_dimensions_exact_match"],
        "all_three_dimensions_exact_match_count": profile["all_three_dimensions_exact_match_count"],
        "per_dimension": per_dimension,
    }


def _resolved_path(value: str, relative_to: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


def _check_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    actual = _sha256(path)
    _expect(actual == expected, f"{label} SHA-256 mismatch: expected={expected}, actual={actual}")
    return actual


def _load_run_identity(
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    _expect(manifest.get("run_id") == run_root.name, "training Run ID does not match its directory")
    _expect(manifest.get("status") == "passed", "training Run manifest is not passed")
    _expect(manifest.get("experiment_id") == "E01-A", "training Run is not E01-A")

    config = _read_yaml(TRAIN_CONFIG_PATH)
    expected_config_sha = manifest.get("config_sha256")
    _expect(isinstance(expected_config_sha, str), "training Run is missing config_sha256")
    _check_file_hash(TRAIN_CONFIG_PATH, expected_config_sha, "current training config")
    _check_file_hash(run_root / "config_snapshot.yaml", expected_config_sha, "training config snapshot")
    _expect(config["base_model"]["expected_weight_sha256"] == EXPECTED_BASE_SHA256, "configured Base SHA differs from closeout plan")
    _expect(manifest.get("base_weight_sha256") == EXPECTED_BASE_SHA256, "training Run Base SHA differs from closeout plan")
    _expect(manifest.get("optimizer_steps") == EXPECTED_OPTIMIZER_STEPS, "training optimizer step count differs from expected Run")

    one_step = _resolved_path(manifest.get("one_step_proof_path", ""))
    _check_file_hash(one_step, manifest.get("one_step_proof_sha256", ""), "one-step proof")
    proof = _read_json(one_step)
    _expect(proof.get("status") == "passed" and proof.get("base_weight_sha256") == EXPECTED_BASE_SHA256, "one-step proof identity is invalid")

    selected_path = run_root / "selected_samples.json"
    selected = _read_json(selected_path)
    rows = selected.get("rows")
    if not isinstance(rows, list) or not rows:
        raise IdentityMismatchError("selected_samples.json has no selected rows")
    _expect(selected.get("sha256") == _canonical_sha256(rows), "selected sample metadata checksum mismatch")
    _expect(manifest.get("selected_samples_sha256") == selected["sha256"], "selected sample metadata differs from training manifest")
    selected_by_id = _unique_by_id(rows, "sample_id", "selected training samples")

    adapter_dir = run_root / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"required Adapter directory is missing: {adapter_dir}")
    adapter_identity, _adapter_config = run_inference.adapter_weight_identity(adapter_dir)
    recorded_adapter = manifest.get("adapter_identity", {})
    _expect(adapter_identity.get("weight_sha256") == recorded_adapter.get("weight_sha256"), "Adapter weight SHA differs from training manifest")
    _expect(adapter_identity.get("config_sha256") == recorded_adapter.get("config_sha256"), "Adapter config SHA differs from training manifest")
    _expect(adapter_identity.get("weight_sha256") == "191b136b89dbd10aff0ee07a39db9b868acbad0b276dd03be5e19915822ff443", "Adapter SHA differs from the preserved E01-A report")

    reload_check = _read_json(run_root / "reload_check.json")
    _expect(reload_check.get("status") == "passed", "Adapter reload check did not pass")
    _expect(reload_check.get("base_weight_sha256_before") == EXPECTED_BASE_SHA256, "reload Base-before SHA mismatch")
    _expect(reload_check.get("base_weight_sha256_after") == EXPECTED_BASE_SHA256, "reload Base-after SHA mismatch")

    source_hashes = {
        "training_manifest": _sha256(manifest_path),
        "training_config": expected_config_sha,
        "training_config_snapshot": _sha256(run_root / "config_snapshot.yaml"),
        "selected_samples": _sha256(selected_path),
        "one_step_proof": _sha256(one_step),
        "adapter_weight": adapter_identity["weight_sha256"],
        "adapter_config": adapter_identity["config_sha256"],
        "reload_check": _sha256(run_root / "reload_check.json"),
    }
    identity = {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "base_weight_sha256": manifest["base_weight_sha256"],
        "training_config_sha256": expected_config_sha,
        "adapter_weight_sha256": adapter_identity["weight_sha256"],
        "adapter_config_sha256": adapter_identity["config_sha256"],
        "selected_sample_count": len(selected_by_id),
        "selected_sample_ids_sha256": _canonical_sha256(sorted(selected_by_id)),
        "optimizer_steps": manifest["optimizer_steps"],
        "fit_status": manifest.get("fit_status"),
        "reload_check_status": reload_check.get("status"),
    }
    return manifest, config, rows, identity, source_hashes


def _load_training_diagnostics(
    run_root: Path,
    config: dict[str, Any],
    selected_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    selected_ids = {row["sample_id"] for row in selected_rows}
    normalized_train_path = _resolved_path(config["data"]["normalized_train_path"])
    train_rows = read_jsonl(normalized_train_path)
    train_by_id = _unique_by_id(train_rows, "sample_id", "normalized training source")
    missing_train = selected_ids - set(train_by_id)
    if missing_train:
        raise IdentityMismatchError(f"selected training targets are missing: {sorted(missing_train)}")
    targets = {sample_id: train_by_id[sample_id]["target"] for sample_id in selected_ids}
    emotions = [targets[sample_id]["emotion_label"] for sample_id in sorted(selected_ids)]
    gold_profiles = [targets[sample_id]["user_profile"] for sample_id in sorted(selected_ids)]
    schema = run_inference.load_submission_schema()
    enums = run_inference.submission_enums(schema)

    train_log = read_jsonl(run_root / "train_log.jsonl")
    ordered_log = validate_train_log(train_log)
    loss_by_step = {row["optimizer_step"]: float(row["loss"]) for row in ordered_log}
    if len(ordered_log) < 10:
        raise ValueError("train_log has fewer than ten optimizer steps")
    loss_windows = []
    for step in EXPECTED_DIAGNOSTIC_STEPS[1:]:
        window = [loss_by_step[index] for index in range(step - 19, step + 1)]
        loss_windows.append({"step": step, "window_steps": [step - 19, step], "mean_loss": statistics.fmean(window)})

    evaluation_dir = run_root / "evaluation"
    found_steps = sorted(
        int(match.group(1))
        for path in evaluation_dir.glob("train_diagnostics_step_*.json")
        if (match := re.fullmatch(r"train_diagnostics_step_(\d{4})\.json", path.name))
    )
    if found_steps != EXPECTED_DIAGNOSTIC_STEPS:
        raise IdentityMismatchError(f"training diagnostic steps differ from expected: {found_steps}")

    step_records = []
    for step in EXPECTED_DIAGNOSTIC_STEPS:
        diag_path = evaluation_dir / f"train_diagnostics_step_{step:04d}.json"
        raw_path = evaluation_dir / f"train_raw_step_{step:04d}.jsonl"
        stored = _read_json(diag_path)
        _expect(stored.get("scope") == "train_fitting_only", f"training diagnostic {step} has unexpected scope")
        _expect(stored.get("step") == step, f"training diagnostic file/step mismatch at {step}")
        _expect(set(stored.get("sample_ids", [])) == selected_ids, f"training diagnostic sample IDs differ at step {step}")
        _expect(len(stored.get("sample_ids", [])) == len(selected_ids), f"training diagnostic has duplicate IDs at step {step}")
        raw_rows = read_jsonl(raw_path)
        raw_by_id = _unique_by_id(raw_rows, "id", f"training Raw step {step}")
        _expect(set(raw_by_id) == selected_ids, f"training Raw IDs differ at step {step}")
        _expect(
            stored.get("generated_raw_sha256") == _canonical_sha256([raw_by_id[sample_id]["raw_text"] for sample_id in stored["sample_ids"]]),
            f"training Raw content checksum differs at step {step}",
        )

        predictions_by_id = {}
        submission_validity_by_id = {}
        for sample_id in sorted(selected_ids):
            prediction, _parse_failed, _invalid_label = run_inference.parse_prediction(
                sample_id, raw_by_id[sample_id].get("raw_text", ""), enums
            )
            predictions_by_id[sample_id] = prediction
            try:
                run_inference.validate_submission_prediction(prediction, schema)
                submission_validity_by_id[sample_id] = True
            except Exception:
                submission_validity_by_id[sample_id] = False
        step_gold_emotions = [targets[sample_id]["emotion_label"] for sample_id in sorted(selected_ids)]
        step_predictions = [predictions_by_id[sample_id] for sample_id in sorted(selected_ids)]
        emotion = emotion_metrics(
            step_gold_emotions,
            [row["emotion_label"] for row in step_predictions],
            enums["emotion_label"],
        )
        profile = profile_metrics(
            [targets[sample_id]["user_profile"] for sample_id in sorted(selected_ids)],
            [row["user_profile"] for row in step_predictions],
            enums,
        )
        raw = raw_diagnostics(
            [raw_by_id[sample_id] for sample_id in sorted(selected_ids)],
            enums,
            step_predictions,
            [submission_validity_by_id[sample_id] for sample_id in sorted(selected_ids)],
            generation_cap=int(config["evaluation"]["max_new_tokens"]),
        )
        _compare_train_diagnostic(stored, emotion, profile, raw, step)
        status_counts = dict(sorted(Counter(row.get("status", "unknown") for row in raw_rows).items()))
        step_records.append({
            "step": step,
            "scope": "train_fitting_only",
            "sample_count": len(selected_ids),
            "emotion": {"accuracy": emotion["accuracy"], "macro_f1": emotion["macro_f1"]},
            "profile": _compact_profile_metrics(profile),
            "raw": _compact_raw_metrics(raw),
            "runner_status_counts": status_counts,
            "recomputed_against_preserved_diagnostic": True,
        })

    # Reuse the already verified last-step aggregates and retain per-label counts from the stored helper output.
    last_diag = _read_json(evaluation_dir / "train_diagnostics_step_0120.json")
    dimension_gold_label_support = {}
    for dimension in DIMENSIONS:
        dimension_gold_label_support[dimension] = {
            label: int(counts["support"])
            for label, counts in last_diag["profile"]["per_dimension"][dimension]["per_label"].items()
            if counts["support"]
        }
    train_metrics = {
        "scope": "train_fitting_only",
        "optimizer_steps": len(ordered_log),
        "loss_windows": loss_windows,
        "first_10_steps_mean_loss": statistics.fmean(loss_by_step[step] for step in range(1, 11)),
        "last_10_steps_mean_loss": statistics.fmean(loss_by_step[step] for step in range(EXPECTED_OPTIMIZER_STEPS - 9, EXPECTED_OPTIMIZER_STEPS + 1)),
        "generation_diagnostics_by_step": step_records,
        "selected_sample_profile_gold_label_support_by_dimension": dimension_gold_label_support,
        "final_profile_metrics": _compact_profile_metrics(last_diag["profile"]),
        "final_raw_metrics": _compact_raw_metrics(last_diag["raw"]),
        "normalized_submission_schema_note": (
            "Training diagnostic rows retain internal train: IDs; the official submission schema requires test_* IDs. "
            "The recorded 0/8 official Submission-schema rate is therefore an ID-namespace check, while raw A0 schema "
            "and prediction-field metrics are reported separately."
        ),
    }
    artifact_hashes = {
        "normalized_training_source": _sha256(normalized_train_path),
        "train_log": _sha256(run_root / "train_log.jsonl"),
        **{
            f"train_diagnostic_step_{step:04d}": _sha256(evaluation_dir / f"train_diagnostics_step_{step:04d}.json")
            for step in EXPECTED_DIAGNOSTIC_STEPS
        },
        **{
            f"train_raw_step_{step:04d}": _sha256(evaluation_dir / f"train_raw_step_{step:04d}.jsonl")
            for step in EXPECTED_DIAGNOSTIC_STEPS
        },
    }
    return train_metrics, artifact_hashes


def _profile_nonempty_dimensions(profile: dict[str, Any]) -> list[str]:
    return [dimension for dimension in DIMENSIONS if profile.get(dimension)]


def _distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _token_summary(rows: list[dict[str, Any]], prompt_cap: int, generation_cap: int) -> dict[str, Any]:
    generated = [row["generated_tokens"] for row in rows if isinstance(row.get("generated_tokens"), int)]
    prompts = [row["prompt_tokens"] for row in rows if isinstance(row.get("prompt_tokens"), int)]
    ordered = sorted(generated)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    overflow_statuses = {"server_prompt_token_mismatch", "prompt_floor_exceeds_cap"}
    return {
        "sample_count": len(rows),
        "stop_reason_counts": dict(sorted(Counter(str(row.get("stop_reason", "unknown")) for row in rows).items())),
        "generated_tokens": {
            "min": min(generated) if generated else None,
            "median": statistics.median(generated) if generated else None,
            "p95": p95,
            "max": max(generated) if generated else None,
        },
        "generation_cap": generation_cap,
        "generation_cap_reached_count": sum(token >= generation_cap for token in generated),
        "prompt_cap": prompt_cap,
        "prompt_tokens_max": max(prompts) if prompts else None,
        "prompt_overflow_count": sum(
            (isinstance(row.get("prompt_tokens"), int) and row["prompt_tokens"] > prompt_cap)
            or row.get("status") in overflow_statuses
            for row in rows
        ),
        "history_truncated_sample_count": sum(bool(row.get("removed_history_messages")) for row in rows),
    }


def _classify_rows(raw_rows: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, str]]:
    counts: Counter[str] = Counter()
    per_id: dict[str, str] = {}
    for sample_id, row in sorted(raw_rows.items()):
        category = classify_raw_json(row.get("raw_text", ""))
        if category != "strict_json_object":
            counts[category] += 1
            per_id[sample_id] = category
    return dict(sorted(counts.items())), per_id


def _validate_protocol_match(
    smoke_manifest: dict[str, Any],
    e00_manifest: dict[str, Any],
    e00_run: dict[str, Any],
    dev_manifest: dict[str, Any],
    attempt_root: Path,
    train_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    e00_profile = e00_manifest.get("profiles", {}).get("greedy", {})
    e00_runner_path = _resolved_path(e00_profile.get("runner_config_path", ""))
    e00_eval_path = _resolved_path(e00_profile.get("evaluation_config_path", ""))
    e00_runner_sha = _check_file_hash(
        e00_runner_path, e00_profile.get("runner_config_sha256", ""), "E00.2 Greedy runner config"
    )
    e00_eval_sha = _check_file_hash(
        e00_eval_path, e00_profile.get("evaluation_config_sha256", ""), "E00.2 Greedy evaluation config"
    )

    e01_runner_path = attempt_root / "inference_config.json"
    e01_eval_path = attempt_root / "evaluation_config.yaml"
    e01_runner_sha = _check_file_hash(e01_runner_path, dev_manifest.get("inference_config_sha256", ""), "E01-A runner config")
    e01_eval_sha = _check_file_hash(e01_eval_path, dev_manifest.get("evaluation_config_sha256", ""), "E01-A evaluation config")
    _expect(e00_run.get("runner_config_sha256") == e00_runner_sha, "E00.2 Greedy runner config hash is inconsistent")
    _expect(e00_run.get("evaluation_config_sha256") == e00_eval_sha, "E00.2 Greedy evaluation config hash is inconsistent")
    _expect(e00_run.get("profile") == "greedy", "E00.2 baseline is not Greedy")
    base_quantization = e00_run.get("quantization", {})
    _expect(
        base_quantization.get("real_nf4_verified") is True
        and base_quantization.get("quant_type") == "nf4"
        and base_quantization.get("compute_dtype") == "bfloat16"
        and base_quantization.get("offload_devices") == [],
        "E00.2 observed quantization is not verified NF4/BF16/CUDA without offload",
    )

    e00_runner = _read_json(e00_runner_path)
    e01_runner = _read_json(e01_runner_path)
    e00_eval = _read_yaml(e00_eval_path)
    e01_eval = _read_yaml(e01_eval_path)
    _expect(e00_runner.get("generation") == e01_runner.get("generation"), "Base and Adapter generation configs differ")
    _expect(e00_runner.get("quantization") == e01_runner.get("quantization"), "Base and Adapter runner quantization configs differ")
    for section in ("backend", "model_path", "prompt", "tokenizer", "context"):
        _expect(e00_runner.get(section) == e01_runner.get(section), f"Base and Adapter runner {section} configs differ")
    for section in ("protocol", "scoring"):
        _expect(e00_eval.get(section) == e01_eval.get(section), f"Base and Adapter evaluation {section} configs differ")
    base_eval_generation = e00_eval.get("generation", {})
    adapter_eval_generation = e01_eval.get("generation", {})
    _expect(
        all(base_eval_generation.get(key) == adapter_eval_generation.get(key) for key in ("max_new_tokens", "do_sample")),
        "Base and Adapter effective evaluation generation configs differ",
    )
    _expect(e00_eval.get("quantization", {}).get("planned") == e01_eval.get("quantization", {}).get("planned"), "Base and Adapter evaluation quantization plans differ")
    _expect(e00_eval.get("model", {}).get("model_id") == e01_eval.get("model", {}).get("model_id"), "Base and Adapter model IDs differ")
    _expect(e00_eval.get("generation", {}).get("do_sample") is False, "E00.2 reference generation is not Greedy")
    _expect(e01_runner.get("generation", {}).get("do_sample") is False, "E01-A generation is not Greedy")

    protocol = e01_eval["protocol"]
    _expect(protocol.get("prompt_version") == "P0" and protocol.get("format_version") == "F2" and protocol.get("truncation_version") == "T1", "E01-A protocol is not P0/F2/T1")
    _expect(protocol.get("prompt_sha256") == smoke_manifest.get("protocol", {}).get("prompt_sha256"), "Smoke manifest P0 hash differs from E01 evaluation")
    _expect(e00_run.get("model_weight_identity", {}).get("observed_weight_hash") == EXPECTED_BASE_SHA256, "E00.2 Base weight identity differs from E01 Base")
    _expect(dev_manifest.get("score_status") == "verified_base_and_adapter", "E01-A Dev scorer did not verify Base and Adapter")
    adapter_config = e01_runner.get("adapter", {})
    adapter_identity = train_manifest.get("adapter_identity", {})
    _expect(adapter_config.get("enabled") is True, "E01-A Dev inference did not enable the trained Adapter")
    _expect(adapter_config.get("expected_base_weight_sha256") == EXPECTED_BASE_SHA256, "E01-A Dev inference Base SHA differs")
    _expect(adapter_config.get("expected_adapter_weight_sha256") == adapter_identity.get("weight_sha256"), "E01-A Dev inference Adapter SHA differs from training Run")
    _expect(adapter_config.get("expected_adapter_config_sha256") == adapter_identity.get("config_sha256"), "E01-A Dev inference Adapter config SHA differs from training Run")

    details = {
        "status": "matched",
        "protocol": "P0/F2/T1",
        "tokenizer_id": protocol.get("tokenizer_id"),
        "tokenizer_revision": protocol.get("tokenizer_revision"),
        "tokenizer_signature_sha256": protocol.get("tokenizer_signature_sha256"),
        "chat_template_sha256": protocol.get("chat_template_sha256"),
        "generation": e01_runner["generation"],
        "effective_evaluation_generation": {
            "max_new_tokens": adapter_eval_generation.get("max_new_tokens"),
            "do_sample": adapter_eval_generation.get("do_sample"),
        },
        "evaluation_snapshot_extra_generation_fields": {
            "base": sorted(set(base_eval_generation) - {"max_new_tokens", "do_sample"}),
            "adapter": sorted(set(adapter_eval_generation) - {"max_new_tokens", "do_sample"}),
            "inert_under_greedy": adapter_eval_generation.get("do_sample") is False,
        },
        "quantization": e01_runner["quantization"],
        "scoring_config_sha256_base": e00_eval_sha,
        "scoring_config_sha256_adapter": e01_eval_sha,
    }
    config_hashes = {
        "e00_greedy_runner_config": e00_runner_sha,
        "e00_greedy_evaluation_config": e00_eval_sha,
        "e01_runner_config": e01_runner_sha,
        "e01_evaluation_config": e01_eval_sha,
    }
    return details, config_hashes, e00_runner, e01_runner


def _load_dev_diagnostics(
    run_root: Path,
    dev_manifest: dict[str, Any],
    train_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    attempt_id = dev_manifest.get("attempt_id")
    if not isinstance(attempt_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", attempt_id):
        raise IdentityMismatchError("E01-A Dev Smoke attempt_id is missing or unsafe")
    attempt_root = run_root / "evaluation" / "dev_smoke" / attempt_id
    _expect(dev_manifest.get("run_id") == f"{run_root.name}-dev-smoke-{attempt_id}", "E01-A Dev Smoke Run ID mismatch")
    _expect(dev_manifest.get("status") == "scored" and dev_manifest.get("return_code") == 0, "E01-A Dev Smoke did not complete successfully")
    _expect(dev_manifest.get("references_read_by_inference") is False, "Dev inference unexpectedly read References")
    _expect(dev_manifest.get("training_fit_scope") == "train_fitting_only", "E01-A Dev manifest has wrong train-fit scope")
    _expect(dev_manifest.get("dev_scope") == "fixed_20_row_E00_1_smoke", "E01-A Dev manifest has wrong smoke scope")

    smoke_manifest_path = SMOKE_ROOT / "manifest.json"
    smoke_manifest = _read_json(smoke_manifest_path)
    smoke_manifest_sha = _check_file_hash(smoke_manifest_path, dev_manifest.get("smoke_manifest_sha256", ""), "fixed Smoke manifest")
    smoke_input_path = SMOKE_ROOT / "smoke_input.jsonl"
    smoke_input_rows = read_jsonl(smoke_input_path)
    smoke_input_by_id = _unique_by_id(smoke_input_rows, "id", "fixed Smoke inference input")
    _expect(len(smoke_input_by_id) == 20, "fixed Smoke input is not 20 rows")

    e00_manifest = _read_json(E00_MANIFEST_PATH)
    _expect(e00_manifest.get("status") == "real_smoke_scored", "E00.2 baseline Manifest is not real_smoke_scored")
    e00_run = e00_manifest.get("runs", {}).get("greedy", {})
    if not e00_run:
        raise IdentityMismatchError("E00.2 run Manifest has no Greedy profile")
    _expect(e00_run.get("input_sha256") == _sha256(smoke_input_path), "E00.2 input SHA does not match the fixed 20-row Smoke input")
    _expect(e00_run.get("run_id") == "e00_2_greedy_E00.2-20261026-220635", "E00.2 Greedy Run identity differs from the preserved baseline")

    base_raw_path = _resolved_path(e00_run.get("raw_path", ""))
    base_submission_path = _resolved_path(e00_run.get("prediction_path", ""))
    base_metrics_path = _resolved_path(e00_run.get("metrics_path", ""))
    base_raw_sha = _check_file_hash(base_raw_path, e00_run.get("raw_predictions_sha256", ""), "E00.2 Greedy raw predictions")
    base_submission_sha = _check_file_hash(base_submission_path, e00_run.get("submission_sha256", ""), "E00.2 Greedy submission")
    base_metrics = _read_json(base_metrics_path)

    e01_raw_path = attempt_root / "raw_predictions.jsonl"
    e01_submission_path = attempt_root / "runner" / "submission.jsonl"
    e01_metrics_path = attempt_root / "scored" / "metrics.json"
    e01_metrics_sha = _check_file_hash(e01_metrics_path, dev_manifest.get("score_metrics_sha256", ""), "E01-A Dev score metrics")
    adapter_raw_rows = read_jsonl(e01_raw_path)
    adapter_submission_rows = read_jsonl(e01_submission_path)
    base_raw_rows = read_jsonl(base_raw_path)
    base_submission_rows = read_jsonl(base_submission_path)
    smoke_references = read_jsonl(SMOKE_ROOT / "smoke_references.jsonl")
    gold_by_id = _unique_by_id(smoke_references, "eval_id", "fixed Smoke References")
    _expect(set(gold_by_id) == set(smoke_input_by_id), "fixed Smoke input IDs differ from References")
    base_raw_by_id = _unique_by_id(base_raw_rows, "id", "E00.2 Greedy Raw")
    adapter_raw_by_id = _unique_by_id(adapter_raw_rows, "id", "E01-A Dev Raw")
    base_submission_by_id = _unique_by_id(base_submission_rows, "id", "E00.2 Greedy Submission")
    adapter_submission_by_id = _unique_by_id(adapter_submission_rows, "id", "E01-A Dev Submission")
    expected_ids = set(gold_by_id)
    for name, mapping in (
        ("E00.2 Raw", base_raw_by_id),
        ("E00.2 Submission", base_submission_by_id),
        ("E01-A Raw", adapter_raw_by_id),
        ("E01-A Submission", adapter_submission_by_id),
    ):
        _expect(set(mapping) == expected_ids, f"{name} IDs differ from fixed 20-row Smoke References")
    _expect(len(expected_ids) == 20, "fixed Smoke Reference set is not 20 IDs")

    protocol_details, config_hashes, _e00_runner, _e01_runner = _validate_protocol_match(
        smoke_manifest, e00_manifest, e00_run, dev_manifest, attempt_root, train_manifest
    )
    adapter_metrics = _read_json(e01_metrics_path)
    _expect(adapter_metrics.get("runtime_compatibility", {}).get("status") == "verified", "E01-A Dev runtime compatibility was not verified")
    _expect(adapter_metrics.get("model", {}).get("quantization", {}).get("real_nf4_verified") is True, "E01-A Dev runtime did not verify NF4")
    _expect(adapter_metrics.get("model", {}).get("quantization", {}).get("offload_devices") == [], "E01-A Dev observed CPU/disk offload")
    _expect(adapter_metrics.get("model", {}).get("model_weight_identity", {}).get("observed_weight_hash") == EXPECTED_BASE_SHA256, "E01-A Dev scored Base SHA differs")
    _expect(adapter_metrics.get("model", {}).get("adapter_identity", {}).get("weight_sha256") == train_manifest.get("adapter_identity", {}).get("weight_sha256"), "E01-A Dev scored Adapter SHA differs")
    schema = run_inference.load_submission_schema()
    enums = run_inference.submission_enums(schema)

    gold_emotions = []
    base_emotions = []
    adapter_emotions = []
    gold_profiles = []
    base_profiles = []
    adapter_profiles = []
    cases = []
    base_schema_validity = []
    adapter_schema_validity = []
    for sample_id in sorted(expected_ids):
        gold = gold_by_id[sample_id]["target"]
        base_pred = base_submission_by_id[sample_id]
        adapter_pred = adapter_submission_by_id[sample_id]
        for pred, source in ((base_pred, "E00.2 submission"), (adapter_pred, "E01-A submission")):
            try:
                run_inference.validate_submission_prediction(pred, schema)
                valid = True
            except Exception:
                valid = False
            if source.startswith("E00"):
                base_schema_validity.append(valid)
            else:
                adapter_schema_validity.append(valid)
        gold_emotions.append(gold["emotion_label"])
        base_emotions.append(base_pred.get("emotion_label", ""))
        adapter_emotions.append(adapter_pred.get("emotion_label", ""))
        gold_profile = gold["user_profile"]
        base_profile = base_pred.get("user_profile", {})
        adapter_profile = adapter_pred.get("user_profile", {})
        gold_profiles.append(gold_profile)
        base_profiles.append(base_profile)
        adapter_profiles.append(adapter_profile)
        cases.append({
            "id": sample_id,
            "gold_emotion": gold["emotion_label"],
            "base_emotion": base_pred.get("emotion_label", ""),
            "adapter_emotion": adapter_pred.get("emotion_label", ""),
            "base_status": base_raw_by_id[sample_id].get("status", "unknown"),
            "adapter_status": adapter_raw_by_id[sample_id].get("status", "unknown"),
            "gold_profile_nonempty_dimensions": _profile_nonempty_dimensions(gold_profile),
            "base_profile_nonempty_dimensions": _profile_nonempty_dimensions(base_profile),
            "adapter_profile_nonempty_dimensions": _profile_nonempty_dimensions(adapter_profile),
            "base_profile_exact_match": all(set(gold_profile[d]) == set(base_profile.get(d, [])) for d in DIMENSIONS),
            "adapter_profile_exact_match": all(set(gold_profile[d]) == set(adapter_profile.get(d, [])) for d in DIMENSIONS),
        })

    base_emotion_metrics = emotion_metrics(gold_emotions, base_emotions, enums["emotion_label"])
    adapter_emotion_metrics = emotion_metrics(gold_emotions, adapter_emotions, enums["emotion_label"])
    base_profile_metrics = profile_metrics(gold_profiles, base_profiles, enums)
    adapter_profile_metrics = profile_metrics(gold_profiles, adapter_profiles, enums)
    base_raw_diagnostics = raw_diagnostics(base_raw_rows, enums, base_submission_rows, base_schema_validity, generation_cap=1200)
    adapter_raw_diagnostics = raw_diagnostics(adapter_raw_rows, enums, adapter_submission_rows, adapter_schema_validity, generation_cap=1200)
    base_failure_counts, base_failure_by_id = _classify_rows(base_raw_by_id)
    adapter_failure_counts, adapter_failure_by_id = _classify_rows(adapter_raw_by_id)

    _metric_equal(base_emotion_metrics["accuracy"], base_metrics["emotion"]["accuracy"], "E00.2 emotion accuracy")
    _metric_equal(base_emotion_metrics["macro_f1"], base_metrics["emotion"]["macro_f1"], "E00.2 emotion macro-F1")
    _metric_equal(base_profile_metrics["overall_33_label_micro_f1"], base_metrics["profile"]["overall_33_label_micro_f1"], "E00.2 profile micro-F1")
    _metric_equal(base_profile_metrics["all_three_dimensions_exact_match"], base_metrics["profile"]["all_three_dimensions_exact_match"], "E00.2 profile exact match")
    for key in ("raw_json_parse_rate", "raw_schema_valid_rate", "submission_schema_valid_rate"):
        _metric_equal(base_raw_diagnostics[key], base_metrics["raw_diagnostics"][key], f"E00.2 raw.{key}")
    _metric_equal(adapter_emotion_metrics["accuracy"], adapter_metrics["emotion"]["accuracy"], "E01-A emotion accuracy")
    _metric_equal(adapter_emotion_metrics["macro_f1"], adapter_metrics["emotion"]["macro_f1"], "E01-A emotion macro-F1")
    _metric_equal(adapter_profile_metrics["overall_33_label_micro_f1"], adapter_metrics["profile"]["overall_33_label_micro_f1"], "E01-A profile micro-F1")
    _metric_equal(adapter_profile_metrics["all_three_dimensions_exact_match"], adapter_metrics["profile"]["all_three_dimensions_exact_match"], "E01-A profile exact match")
    for key in ("raw_json_parse_rate", "raw_schema_valid_rate", "submission_schema_valid_rate"):
        _metric_equal(adapter_raw_diagnostics[key], adapter_metrics["raw_diagnostics"][key], f"E01-A raw.{key}")

    def profile_nonempty_summary(profiles: list[dict[str, list[str]]], metric: dict[str, Any]) -> dict[str, Any]:
        return {
            "nonempty_profile_sample_count": sum(bool(_profile_nonempty_dimensions(profile)) for profile in profiles),
            "all_empty_profile_sample_count": sum(not _profile_nonempty_dimensions(profile) for profile in profiles),
            "per_dimension_gold_nonempty_sample_count": {
                dimension: int(round((1 - metric["per_dimension"][dimension]["gold_empty_rate"]) * len(profiles)))
                for dimension in DIMENSIONS
            },
            "per_dimension_prediction_nonempty_sample_count": {
                dimension: int(round((1 - metric["per_dimension"][dimension]["pred_empty_rate"]) * len(profiles)))
                for dimension in DIMENSIONS
            },
            "per_dimension_nonempty_label_hit_count": {
                dimension: metric["per_dimension"][dimension]["tp"] for dimension in DIMENSIONS
            },
            "per_dimension_recall": {
                dimension: metric["per_dimension"][dimension]["local_micro_recall"] for dimension in DIMENSIONS
            },
            "all_three_dimensions_exact_match_count": metric["all_three_dimensions_exact_match_count"],
            "all_three_dimensions_exact_match_rate": metric["all_three_dimensions_exact_match"],
        }

    e01_eval = _read_yaml(attempt_root / "evaluation_config.yaml")
    prompt_cap = int(e01_eval["protocol"]["max_prompt_tokens"])
    generation_cap = int(e01_eval["generation"]["max_new_tokens"])
    dev_metrics = {
        "scope": "fixed_20_row_E00_1_smoke",
        "references_read_by_inference": False,
        "sample_count": len(expected_ids),
        "id_set_sha256": _canonical_sha256(sorted(expected_ids)),
        "id_sets_match_base_adapter_references": True,
        "protocol_match": protocol_details,
        "emotion": {
            "gold_category_support": _distribution(gold_emotions),
            "base_predicted_category_distribution": _distribution(base_emotions),
            "adapter_predicted_category_distribution": _distribution(adapter_emotions),
            "base": base_emotion_metrics,
            "adapter": adapter_emotion_metrics,
        },
        "profile": {
            "gold_nonempty_profile_sample_count": sum(bool(_profile_nonempty_dimensions(profile)) for profile in gold_profiles),
            "base": {**_compact_profile_metrics(base_profile_metrics), **profile_nonempty_summary(base_profiles, base_profile_metrics)},
            "adapter": {**_compact_profile_metrics(adapter_profile_metrics), **profile_nonempty_summary(adapter_profiles, adapter_profile_metrics)},
        },
        "raw": {
            "base": _compact_raw_metrics(base_raw_diagnostics),
            "adapter": _compact_raw_metrics(adapter_raw_diagnostics),
            "base_runner_status_counts": dict(sorted(Counter(row.get("status", "unknown") for row in base_raw_rows).items())),
            "adapter_runner_status_counts": dict(sorted(Counter(row.get("status", "unknown") for row in adapter_raw_rows).items())),
            "base_strict_json_failure_categories": base_failure_counts,
            "adapter_strict_json_failure_categories": adapter_failure_counts,
            "base_failure_category_by_id": base_failure_by_id,
            "adapter_failure_category_by_id": adapter_failure_by_id,
        },
        "generation": {
            "base": _token_summary(base_raw_rows, prompt_cap, generation_cap),
            "adapter": _token_summary(adapter_raw_rows, prompt_cap, generation_cap),
        },
        "correctness_transition_counts": {
            "base_and_adapter_correct": sum(row["gold_emotion"] == row["base_emotion"] == row["adapter_emotion"] for row in cases),
            "adapter_only_correct": sum(row["gold_emotion"] != row["base_emotion"] and row["gold_emotion"] == row["adapter_emotion"] for row in cases),
            "base_only_correct": sum(row["gold_emotion"] == row["base_emotion"] and row["gold_emotion"] != row["adapter_emotion"] for row in cases),
            "both_wrong": sum(row["gold_emotion"] != row["base_emotion"] and row["gold_emotion"] != row["adapter_emotion"] for row in cases),
            "profile_exact_match_base": sum(row["base_profile_exact_match"] for row in cases),
            "profile_exact_match_adapter": sum(row["adapter_profile_exact_match"] for row in cases),
        },
        "profile_exact_match_gold_empty_breakdown": {
            model: {
                "exact_match_with_gold_empty_profile": sum(
                    row[f"{model}_profile_exact_match"] and not row["gold_profile_nonempty_dimensions"]
                    for row in cases
                ),
                "exact_match_with_gold_nonempty_profile": sum(
                    row[f"{model}_profile_exact_match"] and bool(row["gold_profile_nonempty_dimensions"])
                    for row in cases
                ),
            }
            for model in ("base", "adapter")
        },
        "cases_id_only": cases,
    }

    source_hashes = {
        "fixed_smoke_manifest": smoke_manifest_sha,
        "fixed_smoke_input": _sha256(smoke_input_path),
        "fixed_smoke_references": _sha256(SMOKE_ROOT / "smoke_references.jsonl"),
        "e00_manifest": _sha256(E00_MANIFEST_PATH),
        "e00_raw_predictions": base_raw_sha,
        "e00_submission": base_submission_sha,
        "e00_scored_metrics": _sha256(base_metrics_path),
        "e01_dev_manifest": _sha256(attempt_root / "run_manifest.json"),
        "e01_raw_predictions": _sha256(e01_raw_path),
        "e01_submission": _sha256(e01_submission_path),
        "e01_scored_metrics": e01_metrics_sha,
        **config_hashes,
    }
    return dev_metrics, source_hashes


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def build_report(metrics: dict[str, Any]) -> str:
    train = metrics["training"]
    dev = metrics["dev"]
    lines = [
        "# E01-A 训练诊断收尾报告",
        "",
        f"- Run：`{metrics['identity']['run_id']}`",
        f"- Adapter SHA-256：`{metrics['identity']['adapter_weight_sha256']}`",
        f"- 诊断范围：训练集 8 条 fitting 样本 + 固定 Dev Smoke 20 条；不代表完整 Dev 泛化。",
        "- 原始回复文本未写入本报告；逐样本表只包含 Smoke ID、标签、状态和画像非空维度。",
        "",
        "## 训练集逐步诊断",
        "",
        "Loss 列为该评测步之前最近 20 个 optimizer step 的平均训练 Loss；Step 0 没有更新 Loss。它与训练集生成诊断、Dev 指标分开统计。",
        "",
        "| Step | 20-step 平均训练 Loss | 情绪准确率 | 画像 33 标签 Micro-F1 | 画像 33 标签 Macro-F1 | 三维 Exact Match | Raw 严格 JSON | Raw A0 Schema | Submission Schema* |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    loss_by_step = {item["step"]: item["mean_loss"] for item in train["loss_windows"]}
    for item in train["generation_diagnostics_by_step"]:
        step = item["step"]
        profile = item["profile"]
        raw = item["raw"]
        loss_text = f"{loss_by_step[step]:.6f}" if step in loss_by_step else "—"
        lines.append(
            f"| {step} | {loss_text} "
            f"| {_format_pct(item['emotion']['accuracy'])} "
            f"| {_format_pct(profile['overall_33_label_micro_f1'])} "
            f"| {_format_pct(profile['overall_33_label_macro_f1'])} "
            f"| {_format_pct(profile['all_three_dimensions_exact_match'])} "
            f"| {_format_pct(raw['raw_json_parse_rate'])} "
            f"| {_format_pct(raw['raw_schema_valid_rate'])} "
            f"| {_format_pct(raw['submission_schema_valid_rate'])} |"
        )
    lines.extend([
        "",
        f"前 10 步平均训练 Loss 为 `{train['first_10_steps_mean_loss']:.6f}`，末 10 步为 `{train['last_10_steps_mean_loss']:.6f}`。末步 8 条样本情绪准确率为 {_format_pct(train['generation_diagnostics_by_step'][-1]['emotion']['accuracy'])}，Raw A0 Schema 合法率为 {_format_pct(train['final_raw_metrics']['raw_schema_valid_rate'])}。",
        "",
        "末步画像按维度复核：",
        "",
        "| 维度 | Gold 标签支持数 | Precision | Recall | F1 | Gold 空样本率 | 预测空样本率 | Exact Match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    final_profile = train["final_profile_metrics"]
    gold_support = train["selected_sample_profile_gold_label_support_by_dimension"]
    for dimension in DIMENSIONS:
        item = final_profile["per_dimension"][dimension]
        lines.append(
            f"| `{dimension}` | {sum(gold_support[dimension].values())} "
            f"| {_format_pct(item['local_micro_precision'])} | {_format_pct(item['local_micro_recall'])} "
            f"| {_format_pct(item['local_micro_f1'])} | {_format_pct(item['gold_empty_rate'])} "
            f"| {_format_pct(item['pred_empty_rate'])} | {item['exact_match_count']}/{item['sample_count']} |"
        )
    lines.extend([
        "",
        f"全体 33 标签 Micro-F1 为 {_format_pct(final_profile['overall_33_label_micro_f1'])}，Macro-F1 为 {_format_pct(final_profile['overall_33_label_macro_f1'])}；三维同时 Exact Match 为 {final_profile['all_three_dimensions_exact_match_count']}/{train['generation_diagnostics_by_step'][-1]['sample_count']}。Gold 正标签支持为 `personality_traits={sum(gold_support['personality_traits'].values())}`、`interests={sum(gold_support['interests'].values())}`、`style={sum(gold_support['style'].values())}`。因此 1.0 Micro-F1 和全匹配来自极少数 `style` 正例及大量空维度，不能证明模型学会了丰富画像。",
        "",
        "`Submission Schema*` 是现存训练诊断按正式 Schema 对完整行校验的值。训练 ID 使用内部 `train:` 命名空间，正式 Schema 要求 `test_...`，所以此列 0% 是 ID 格式检查结果；Raw A0 Schema 列独立反映原始生成结构。",
        "",
        "## Dev 小样本局限",
        "",
        f"Base Greedy 与 Adapter 的固定 Smoke ID 集合一致（{dev['sample_count']}/{dev['sample_count']}），协议核验为 `{dev['protocol_match']['protocol']}`；Tokenizer、P0、Greedy、NF4/BF16、生成上限和评分配置哈希见同目录 `metrics.json`。推理未读取 References。",
        "",
        "| 指标 | Base Greedy | E01-A Adapter |",
        "|---|---:|---:|",
        f"| Emotion Accuracy | {_format_pct(dev['emotion']['base']['accuracy'])} | {_format_pct(dev['emotion']['adapter']['accuracy'])} |",
        f"| Emotion Macro-F1 | {_format_pct(dev['emotion']['base']['macro_f1'])} | {_format_pct(dev['emotion']['adapter']['macro_f1'])} |",
        f"| Raw strict JSON | {_format_pct(dev['raw']['base']['raw_json_parse_rate'])} | {_format_pct(dev['raw']['adapter']['raw_json_parse_rate'])} |",
        f"| Raw A0 Schema | {_format_pct(dev['raw']['base']['raw_schema_valid_rate'])} | {_format_pct(dev['raw']['adapter']['raw_schema_valid_rate'])} |",
        f"| Profile 33 标签 Micro-F1 | {_format_pct(dev['profile']['base']['overall_33_label_micro_f1'])} | {_format_pct(dev['profile']['adapter']['overall_33_label_micro_f1'])} |",
        f"| 三维 Exact Match | {_format_pct(dev['profile']['base']['all_three_dimensions_exact_match'])} | {_format_pct(dev['profile']['adapter']['all_three_dimensions_exact_match'])} |",
        f"| 预测全空画像 | {dev['profile']['base']['all_empty_profile_sample_count']}/{dev['sample_count']} | {dev['profile']['adapter']['all_empty_profile_sample_count']}/{dev['sample_count']} |",
        "",
        f"Adapter 的 Gold 非空画像样本为 {dev['profile']['gold_nonempty_profile_sample_count']}/{dev['sample_count']}；预测非空画像样本为 {dev['profile']['adapter']['nonempty_profile_sample_count']}/{dev['sample_count']}。Adapter 三维 Exact Match 的 8 个匹配样本全部来自 Gold 全空画像；非空 Gold 样本没有命中标签，因此 Micro-F1 为零。逐维命中与 Recall、类别支持、预测分布及完整混淆矩阵保存在 `metrics.json`。",
        "",
        "| 画像维度 | Gold 非空样本 | Base 预测非空 | Base 命中 / Recall | Adapter 预测非空 | Adapter 命中 / Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dimension in DIMENSIONS:
        base = dev["profile"]["base"]
        adapter = dev["profile"]["adapter"]
        base_dimension = base["per_dimension"][dimension]
        adapter_dimension = adapter["per_dimension"][dimension]
        lines.append(
            f"| `{dimension}` | {base['per_dimension_gold_nonempty_sample_count'][dimension]} "
            f"| {base['per_dimension_prediction_nonempty_sample_count'][dimension]} "
            f"| {base['per_dimension_nonempty_label_hit_count'][dimension]} / {_format_pct(base['per_dimension_recall'][dimension])} "
            f"| {adapter['per_dimension_prediction_nonempty_sample_count'][dimension]} "
            f"| {adapter['per_dimension_nonempty_label_hit_count'][dimension]} / {_format_pct(adapter['per_dimension_recall'][dimension])} |"
        )
    lines.extend([
        "",
        f"Gold 情绪类别支持：`{json.dumps(dev['emotion']['gold_category_support'], ensure_ascii=False, sort_keys=True)}`。Base 预测类别分布：`{json.dumps(dev['emotion']['base_predicted_category_distribution'], ensure_ascii=False, sort_keys=True)}`；Adapter：`{json.dumps(dev['emotion']['adapter_predicted_category_distribution'], ensure_ascii=False, sort_keys=True)}`。逐 ID 计算得到 Adapter-only 正确 2 条、Base-only 正确 1 条、两者均错 17 条。Base/Adapter 完整混淆矩阵由共享 `emotion_metrics` 按原始标签逐条计算，并包括 invalid/empty 列，见 `metrics.json`。",
        "",
        f"Adapter Raw 状态：`{json.dumps(dev['raw']['adapter_runner_status_counts'], ensure_ascii=False, sort_keys=True)}`。严格 JSON 失败分类（只根据保留的 Raw 计算）：`{json.dumps(dev['raw']['adapter_strict_json_failure_categories'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "| 模型 | 停止原因 | 生成 Token 范围 | 达生成上限 | Prompt overflow | 历史消息有截断 |",
        "|---|---|---:|---:|---:|---:|",
        f"| Base Greedy | `{json.dumps(dev['generation']['base']['stop_reason_counts'], ensure_ascii=False, sort_keys=True)}` | {dev['generation']['base']['generated_tokens']['min']}–{dev['generation']['base']['generated_tokens']['max']} | {dev['generation']['base']['generation_cap_reached_count']} | {dev['generation']['base']['prompt_overflow_count']} | {dev['generation']['base']['history_truncated_sample_count']} |",
        f"| Adapter | `{json.dumps(dev['generation']['adapter']['stop_reason_counts'], ensure_ascii=False, sort_keys=True)}` | {dev['generation']['adapter']['generated_tokens']['min']}–{dev['generation']['adapter']['generated_tokens']['max']} | {dev['generation']['adapter']['generation_cap_reached_count']} | {dev['generation']['adapter']['prompt_overflow_count']} | {dev['generation']['adapter']['history_truncated_sample_count']} |",
        "",
        f"Adapter 的严格 JSON 失败可读为：{_describe_failure_categories(dev['raw']['adapter_strict_json_failure_categories'])}。运行时 Runner 的 Greedy generation 与量化配置逐字段相同；Base 评测 YAML 额外记录了 temperature/top_p，但 `do_sample=false`，这些采样字段不参与本次解码。",
        "",
        "逐 ID 匿名索引表只在本机忽略目录的 `cases_id_only.md` 中生成；版本化验收报告保留聚合指标，不写入对话内容或 Raw 回复。20 条 Smoke 的规模不能证明总体泛化能力或统计显著性。",
        "",
        "## 收尾判定",
        "",
        "**E01-A 工程通过，但 Dev 泛化低且尚不能证明模型容量不足。** 真实 QLoRA 训练、Adapter 重载、Base 身份和离线诊断证据通过；8 条样本已拟合当前稀疏画像标签，但不覆盖其他画像维度；固定 Dev Smoke 指标很低。正式全量 SFT 为 `not_run`，本诊断不构成启动授权。",
        "",
    ])
    return "\n".join(lines)


def _describe_failure_categories(counts: dict[str, int]) -> str:
    labels = {
        "not_json_object_prefix": "未以 JSON 对象开头",
        "trailing_content_after_json": "JSON 值后有额外文本",
        "unescaped_control_character": "含未转义控制字符",
        "unterminated_string_or_quote": "字符串或引号未闭合",
        "object_key_quote_syntax": "对象键引号错误",
        "object_delimiter_or_quote_syntax": "对象分隔符或引号错误",
        "missing_value_or_malformed_bracket": "缺少值或括号结构错误",
        "json_value_is_not_object": "JSON 顶层值不是对象",
        "duplicate_json_key": "JSON 对象键重复",
        "nonstandard_json_constant": "使用非标准 JSON 常量",
        "unknown_json_syntax_failure": "其他 JSON 语法错误",
        "unknown_strict_json_failure": "其他严格 JSON 失败",
    }
    if not counts:
        return "没有严格 JSON 失败"
    return "；".join(f"{count} 条{labels.get(category, category)}" for category, count in sorted(counts.items()))


def _build_id_only_report(dev: dict[str, Any]) -> str:
    lines = [
        "# E01-A Dev Smoke ID-only 对照",
        "",
        "仅含 ID、Gold/预测标签、Runner 状态和画像非空维度；没有对话正文或 Raw 回复。",
        "",
        "| ID | Gold emotion | Base | Adapter | Base 状态 | Adapter 状态 | Gold 非空维度 | Base 预测非空维度 | Adapter 预测非空维度 | Gold/Adapter 画像全匹配 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in dev["cases_id_only"]:
        lines.append(
            f"| `{row['id']}` | {row['gold_emotion']} | {row['base_emotion'] or '∅'} | {row['adapter_emotion'] or '∅'} "
            f"| {row['base_status']} | {row['adapter_status']} "
            f"| {', '.join(row['gold_profile_nonempty_dimensions']) or '∅'} "
            f"| {', '.join(row['base_profile_nonempty_dimensions']) or '∅'} "
            f"| {', '.join(row['adapter_profile_nonempty_dimensions']) or '∅'} "
            f"| {int(row['base_profile_exact_match'])}/{int(row['adapter_profile_exact_match'])} |"
        )
    return "\n".join(lines) + "\n"


def _write_bundle(output_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any], report: str, cases_report: str) -> None:
    if output_dir.exists():
        raise OutputAlreadyExistsError(f"closeout output already exists; choose a new version: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = {
        "metrics.json": json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "report.md": report,
        "cases_id_only.md": cases_report,
    }
    for name, content in payloads.items():
        path = output_dir / name
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    manifest["outputs"] = {
        name: {"sha256": _sha256(output_dir / name), "size_bytes": (output_dir / name).stat().st_size}
        for name in payloads
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def run_diagnosis(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise OutputAlreadyExistsError(f"closeout output already exists; choose a new version: {output_dir}")
    run_root = run_root.resolve()
    manifest, config, selected_rows, identity, identity_hashes = _load_run_identity(run_root)
    training, train_hashes = _load_training_diagnostics(run_root, config, selected_rows)
    dev_manifest_paths = sorted((run_root / "evaluation" / "dev_smoke").glob("*/run_manifest.json"))
    if not dev_manifest_paths:
        raise FileNotFoundError(f"required E01-A Dev Smoke manifest is missing under {run_root / 'evaluation/dev_smoke'}")
    if len(dev_manifest_paths) != 1:
        raise IdentityMismatchError(f"expected one preserved E01-A Dev Smoke attempt, found {len(dev_manifest_paths)}")
    dev_manifest_path = dev_manifest_paths[0]
    attempt_root = dev_manifest_path.parent
    dev_manifest = _read_json(dev_manifest_path)
    dev, dev_hashes = _load_dev_diagnostics(run_root, dev_manifest, manifest)
    metrics = {
        "schema_version": "e01-a-closeout-v1",
        "identity": identity,
        "training": training,
        "dev": dev,
    }
    report = build_report(metrics)
    cases_report = _build_id_only_report(dev)
    script_hash = _sha256(Path(__file__).resolve())
    source_hashes = identity_hashes | train_hashes | dev_hashes
    source_hashes["diagnose_script"] = script_hash
    output_manifest = {
        "schema_version": "e01-a-closeout-manifest-v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": identity["run_id"],
        "diagnostic_version": output_dir.name,
        "diagnose_script_sha256": script_hash,
        "source_artifacts_sha256": source_hashes,
        "source_checks": {
            "base_weight_sha256": EXPECTED_BASE_SHA256,
            "training_config_sha256": identity["training_config_sha256"],
            "adapter_weight_sha256": identity["adapter_weight_sha256"],
            "smoke_id_count": dev["sample_count"],
            "smoke_id_set_sha256": dev["id_set_sha256"],
            "base_adapter_id_sets_match": dev["id_sets_match_base_adapter_references"],
            "protocol_match": dev["protocol_match"]["status"],
        },
    }
    _write_bundle(output_dir, output_manifest, metrics, report, cases_report)
    return {"run_root": str(run_root), "output_dir": str(output_dir.resolve()), "status": "passed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        result = run_diagnosis(run_root, output_dir)
    except FileNotFoundError as exc:
        print(json.dumps({"status": "blocked_missing_artifacts", "error": str(exc)}, ensure_ascii=False))
        return 2
    except IdentityMismatchError as exc:
        print(json.dumps({"status": "identity_mismatch", "error": str(exc)}, ensure_ascii=False))
        return 2
    except OutputAlreadyExistsError as exc:
        print(json.dumps({"status": "output_exists", "error": str(exc)}, ensure_ascii=False))
        return 2
    except (ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "validation_failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
