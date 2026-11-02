"""Deterministic local proxy metrics; none are mapped to official competition points."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from src.evaluation import strict_json_loads


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _ratio(2 * precision * recall, precision + recall)


def _quantile(values: list[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def emotion_metrics(
    gold: list[str], predicted: list[str], labels: list[str]
) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError(f"情绪指标输入长度不一致: gold={len(gold)}, predicted={len(predicted)}")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("情绪 labels 必须是非空且无重复的固定枚举")
    if any(label not in labels for label in gold):
        raise ValueError("情绪 gold 包含固定枚举以外的标签")
    label_index = {label: index for index, label in enumerate(labels)}
    confusion = [[0 for _ in labels] for _ in labels]
    diagnostic = [[0 for _ in range(len(labels) + 1)] for _ in labels]
    invalid_count = 0
    for expected, actual in zip(gold, predicted):
        row = label_index[expected]
        if actual in label_index:
            column = label_index[actual]
            confusion[row][column] += 1
            diagnostic[row][column] += 1
        else:
            invalid_count += 1
            diagnostic[row][-1] += 1
    per_label: dict[str, Any] = {}
    f1_values = []
    for label in labels:
        index = label_index[label]
        tp = confusion[index][index]
        support = sum(confusion[index]) + diagnostic[index][-1]
        predicted_count = sum(confusion[row][index] for row in range(len(labels)))
        fp = predicted_count - tp
        fn = support - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        label_f1 = _f1(precision, recall)
        f1_values.append(label_f1)
        per_label[label] = {
            "support": support,
            "predicted_count": predicted_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": label_f1,
        }
    count = len(gold)
    return {
        "sample_count": count,
        "accuracy": _ratio(sum(expected == actual for expected, actual in zip(gold, predicted)), count),
        "macro_f1": fmean(f1_values) if f1_values else 0.0,
        "zero_division": 0,
        "invalid_empty_prediction_count": invalid_count,
        "labels": labels,
        "per_label": per_label,
        "confusion_matrix": {"labels": labels, "rows_gold_columns_predicted": confusion},
        "diagnostic_confusion_matrix": {
            "labels": labels + ["__invalid_or_empty_prediction__"],
            "rows_gold_columns_predicted": diagnostic,
        },
    }


def _dimension_metrics(
    gold_profiles: list[dict[str, list[str]]],
    predicted_profiles: list[dict[str, list[str]]],
    labels: list[str],
    dimension: str,
) -> dict[str, Any]:
    exact_count = 0
    gold_empty_count = 0
    pred_empty_count = 0
    gold_label_total = 0
    pred_label_total = 0
    label_counts: dict[str, dict[str, int]] = {
        label: {"support": 0, "predicted_count": 0, "tp": 0, "fp": 0, "fn": 0}
        for label in labels
    }
    for gold_profile, predicted_profile in zip(gold_profiles, predicted_profiles):
        expected = set(gold_profile[dimension])
        actual = set(predicted_profile[dimension])
        exact_count += expected == actual
        gold_empty_count += not expected
        pred_empty_count += not actual
        gold_label_total += len(expected)
        pred_label_total += len(actual)
        for label in labels:
            expected_has = label in expected
            actual_has = label in actual
            item = label_counts[label]
            item["support"] += expected_has
            item["predicted_count"] += actual_has
            item["tp"] += expected_has and actual_has
            item["fp"] += (not expected_has) and actual_has
            item["fn"] += expected_has and (not actual_has)

    tp = sum(item["tp"] for item in label_counts.values())
    fp = sum(item["fp"] for item in label_counts.values())
    fn = sum(item["fn"] for item in label_counts.values())
    per_label: dict[str, Any] = {}
    label_f1s = []
    for label, counts in label_counts.items():
        precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
        score = _f1(precision, recall)
        label_f1s.append(score)
        per_label[label] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": score,
        }
    sample_count = len(gold_profiles)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "sample_count": sample_count,
        "local_exact_match": _ratio(exact_count, sample_count),
        "exact_match_count": exact_count,
        "local_micro_precision": precision,
        "local_micro_recall": recall,
        "local_micro_f1": _f1(precision, recall),
        "local_macro_f1": fmean(label_f1s) if label_f1s else 0.0,
        "gold_empty_rate": _ratio(gold_empty_count, sample_count),
        "pred_empty_rate": _ratio(pred_empty_count, sample_count),
        "gold_average_label_count": _ratio(gold_label_total, sample_count),
        "pred_average_label_count": _ratio(pred_label_total, sample_count),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "labels": labels,
        "per_label": per_label,
    }


def profile_metrics(
    gold_profiles: list[dict[str, list[str]]],
    predicted_profiles: list[dict[str, list[str]]],
    enums: dict[str, list[str]],
) -> dict[str, Any]:
    if len(gold_profiles) != len(predicted_profiles):
        raise ValueError("画像指标输入长度不一致")
    dimensions = ("personality_traits", "interests", "style")
    per_dimension = {
        dimension: _dimension_metrics(gold_profiles, predicted_profiles, enums[dimension], dimension)
        for dimension in dimensions
    }
    gold_exact = [
        all(set(gold[dimension]) == set(predicted[dimension]) for dimension in dimensions)
        for gold, predicted in zip(gold_profiles, predicted_profiles)
    ]
    combined_labels = [f"{dimension}:{label}" for dimension in dimensions for label in enums[dimension]]
    # Reuse the same exact TP/FP/FN definitions over the full 33-label vector.
    total_tp = total_fp = total_fn = 0
    total_label_f1: list[float] = []
    for dimension in dimensions:
        metric = per_dimension[dimension]
        total_tp += metric["tp"]
        total_fp += metric["fp"]
        total_fn += metric["fn"]
        total_label_f1.extend(metric["per_label"][label]["f1"] for label in enums[dimension])
    micro_precision = _ratio(total_tp, total_tp + total_fp)
    micro_recall = _ratio(total_tp, total_tp + total_fn)
    return {
        "metrics_scope": "local_proxy",
        "per_dimension": per_dimension,
        "all_three_dimensions_exact_match": _ratio(sum(gold_exact), len(gold_exact)),
        "all_three_dimensions_exact_match_count": sum(gold_exact),
        "overall_33_label_micro_precision": micro_precision,
        "overall_33_label_micro_recall": micro_recall,
        "overall_33_label_micro_f1": _f1(micro_precision, micro_recall),
        "overall_33_label_macro_f1": fmean(total_label_f1) if total_label_f1 else 0.0,
        "overall_33_label_tp": total_tp,
        "overall_33_label_fp": total_fp,
        "overall_33_label_fn": total_fn,
        "overall_label_names": combined_labels,
    }


def _unicode_nonwhitespace(text: str) -> list[str]:
    return [character for character in text if not character.isspace()]


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for column, right_char in enumerate(right, 1):
            if left_char == right_char:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[column - 1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    """Character-codepoint ROUGE-L F1 after removing all Unicode whitespace."""
    reference_chars = _unicode_nonwhitespace(reference)
    prediction_chars = _unicode_nonwhitespace(prediction)
    if not prediction_chars or not reference_chars:
        return 0.0
    lcs = _lcs_length(reference_chars, prediction_chars)
    precision = _ratio(lcs, len(prediction_chars))
    recall = _ratio(lcs, len(reference_chars))
    return _f1(precision, recall)


def _sacrebleu_version() -> str | None:
    try:
        import sacrebleu

        return str(sacrebleu.__version__)
    except ImportError:
        return None


def corpus_bleu4_char(
    references: list[str],
    predictions: list[str],
    *,
    expected_version: str = "2.5.1",
) -> dict[str, Any]:
    if len(references) != len(predictions):
        raise ValueError("BLEU references/predictions 数量不一致")
    try:
        import sacrebleu
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise RuntimeError("缺少 corpus BLEU 依赖，请安装 requirements-eval.txt") from exc
    actual_version = str(sacrebleu.__version__)
    if actual_version != expected_version:
        raise RuntimeError(f"SacreBLEU 版本不符: expected={expected_version}, actual={actual_version}")
    scorer = BLEU(
        tokenize="char",
        max_ngram_order=4,
        smooth_method="exp",
        effective_order=True,
    )
    result = scorer.corpus_score(predictions, [references])
    return {
        "score_percent": float(result.score),
        "implementation": "sacrebleu",
        "version": actual_version,
        "tokenize": "char",
        "max_ngram_order": 4,
        "smooth_method": "exp",
        "effective_order": True,
        "signature": str(scorer.get_signature()),
        "sample_count": len(references),
        "empty_prediction_count": sum(not prediction.strip() for prediction in predictions),
    }


def _to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def bert_score_optional(
    references: list[str],
    predictions: list[str],
    config: dict[str, Any],
    *,
    scorer: Callable[[list[str], list[str]], tuple[Any, Any, Any]] | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    root = workspace_root or Path.cwd()
    model_path_value = config.get("local_model_path")
    model_path = Path(model_path_value).expanduser() if model_path_value else None
    if model_path is not None and not model_path.is_absolute():
        model_path = root / model_path
    revision = config.get("revision")
    base = {
        "implementation": config.get("implementation", "bert-score"),
        "implementation_version": config.get("implementation_version"),
        "model_id": config.get("model_id"),
        "model_revision": revision,
        "local_model_path": str(model_path) if model_path is not None else None,
        "language": config.get("language", "zh"),
    }
    if not config.get("enabled", False):
        return {**base, "status": "not_run", "reason": "disabled_by_config", "precision": None, "recall": None, "f1": None}
    if model_path is None or not model_path.is_dir():
        return {**base, "status": "not_run", "reason": "local_model_path_missing", "precision": None, "recall": None, "f1": None}
    if not isinstance(revision, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision, re.I):
        return {**base, "status": "not_run", "reason": "model_revision_not_pinned_to_commit_or_sha256", "precision": None, "recall": None, "f1": None}
    if scorer is None:
        try:
            from bert_score import score as bert_score
        except ImportError:
            return {**base, "status": "not_run", "reason": "bert_score_package_not_installed", "precision": None, "recall": None, "f1": None}

        def scorer(predictions_arg: list[str], references_arg: list[str]):
            prior_hub = os.environ.get("HF_HUB_OFFLINE")
            prior_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                return bert_score(
                    predictions_arg,
                    references_arg,
                    model_type=str(model_path),
                    lang=config.get("language", "zh"),
                    verbose=False,
                    rescale_with_baseline=False,
                )
            finally:
                if prior_hub is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prior_hub
                if prior_transformers is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = prior_transformers

    precision, recall, f1 = scorer(predictions, references)
    precision_values = _to_float_list(precision)
    recall_values = _to_float_list(recall)
    f1_values = _to_float_list(f1)
    if not (len(precision_values) == len(recall_values) == len(f1_values) == len(references)):
        raise ValueError("BERTScore 返回数量与输入样本数不一致")
    return {
        **base,
        "status": "completed",
        "reason": None,
        "precision": fmean(precision_values) if precision_values else None,
        "recall": fmean(recall_values) if recall_values else None,
        "f1": fmean(f1_values) if f1_values else None,
        "sample_count": len(references),
    }


def response_metrics(
    references: list[str],
    predictions: list[str],
    scoring_config: dict[str, Any],
    *,
    bert_scorer: Callable[[list[str], list[str]], tuple[Any, Any, Any]] | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    if len(references) != len(predictions):
        raise ValueError("回复 references/predictions 数量不一致")
    bleu_config = scoring_config["bleu"]
    required_bleu = {
        "implementation": "sacrebleu",
        "tokenize": "char",
        "max_ngram_order": 4,
        "smooth_method": "exp",
        "effective_order": True,
    }
    for key, expected in required_bleu.items():
        if bleu_config.get(key) != expected:
            raise ValueError(f"Corpus BLEU 配置漂移: {key} must be {expected!r}")
    rouge_config = scoring_config["rouge_l"]
    if rouge_config.get("implementation") != "local_unicode_character_lcs" or rouge_config.get("version") != 1:
        raise ValueError("ROUGE-L 实现必须锁定为 local_unicode_character_lcs v1")
    if rouge_config.get("whitespace") != "remove_all_unicode_whitespace":
        raise ValueError("ROUGE-L whitespace 配置必须为 remove_all_unicode_whitespace")
    bleu = corpus_bleu4_char(references, predictions, expected_version=str(bleu_config["version"]))
    rouge_values = [rouge_l_f1(reference, prediction) for reference, prediction in zip(references, predictions)]
    predicted_lengths = [len(text) for text in predictions]
    reference_lengths = [len(text) for text in references]
    bert = bert_score_optional(
        references,
        predictions,
        scoring_config.get("bert_score", {}),
        scorer=bert_scorer,
        workspace_root=workspace_root,
    )
    return {
        "metrics_scope": "local_proxy",
        "local_corpus_bleu4": bleu,
        "local_rouge_l_macro_f1": {
            "value": fmean(rouge_values) if rouge_values else 0.0,
            "implementation": "local_unicode_character_lcs",
            "version": 1,
            "whitespace": "remove_all_unicode_whitespace",
            "sample_count": len(references),
            "empty_prediction_count": sum(not text.strip() for text in predictions),
            "empty_prediction_score": 0.0,
        },
        "bert_score": bert,
        "empty_prediction_rate": _ratio(sum(not text.strip() for text in predictions), len(predictions)),
        "prediction_character_length": {
            "mean": fmean(predicted_lengths) if predicted_lengths else 0.0,
            "p50": _quantile(predicted_lengths, 0.50),
            "p90": _quantile(predicted_lengths, 0.90),
            "max": max(predicted_lengths) if predicted_lengths else 0,
        },
        "reference_character_length": {
            "mean": fmean(reference_lengths) if reference_lengths else 0.0,
            "p50": _quantile(reference_lengths, 0.50),
            "p90": _quantile(reference_lengths, 0.90),
            "max": max(reference_lengths) if reference_lengths else 0,
        },
    }


def raw_diagnostics(
    raw_rows: list[dict[str, Any]],
    enums: dict[str, list[str]],
    submission_rows: list[dict[str, Any]],
    submission_schema_valid: list[bool],
    *,
    generation_cap: int | None,
) -> dict[str, Any]:
    if len(raw_rows) != len(submission_rows) or len(submission_schema_valid) != len(submission_rows):
        raise ValueError("Raw/submission/schema 诊断长度不一致")
    parsed_objects: list[dict[str, Any] | None] = []
    for row in raw_rows:
        raw_text = row.get("raw_text")
        if not isinstance(raw_text, str):
            parsed_objects.append(None)
            continue
        try:
            parsed = strict_json_loads(raw_text)
        except (json.JSONDecodeError, ValueError, TypeError):
            parsed_objects.append(None)
            continue
        parsed_objects.append(parsed if isinstance(parsed, dict) else None)
    parse_count = sum(value is not None for value in parsed_objects)
    raw_schema_count = 0
    valid_emotion_count = 0
    valid_profile_count = 0
    nonempty_response_count = 0
    errors: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    for row, raw_object in zip(raw_rows, parsed_objects):
        stops[str(row.get("stop_reason", "unknown"))] += 1
        if raw_object is None:
            errors["strict_json_parse_failure"] += 1
        else:
            expected_fields = {"emotion_label", "user_profile", "response_text", "memory_refs"}
            if set(raw_object) != expected_fields:
                errors["raw_fields_missing_or_extra"] += 1
            emotion_valid = raw_object.get("emotion_label") in enums["emotion_label"]
            valid_emotion_count += emotion_valid
            if not emotion_valid:
                errors["invalid_emotion"] += 1
            profile = raw_object.get("user_profile")
            profile_valid = isinstance(profile, dict) and set(profile) == {
                "personality_traits", "interests", "style"
            }
            if profile_valid:
                for dimension in ("personality_traits", "interests", "style"):
                    values = profile[dimension]
                    if (
                        not isinstance(values, list)
                        or any(not isinstance(value, str) for value in values)
                        or len(values) != len(set(values))
                        or not set(values) <= set(enums[dimension])
                    ):
                        profile_valid = False
                        break
            valid_profile_count += profile_valid
            if not profile_valid:
                errors["invalid_profile"] += 1
            response = raw_object.get("response_text")
            response_valid = isinstance(response, str) and bool(response.strip())
            nonempty_response_count += response_valid
            if not response_valid:
                errors["empty_or_invalid_response"] += 1
            memory_refs_valid = raw_object.get("memory_refs") == []
            raw_schema_valid = (
                set(raw_object) == expected_fields
                and emotion_valid
                and profile_valid
                and response_valid
                and memory_refs_valid
            )
            raw_schema_count += raw_schema_valid
            if not memory_refs_valid:
                errors["memory_refs_not_empty_array"] += 1
            if raw_schema_valid is False and set(raw_object) == expected_fields:
                errors["raw_a0_completion_schema_invalid"] += 1
        status = str(row.get("status", "unknown"))
        if status in {"parse_failure", "invalid_label", "generation_failure", "template_equivalence_failure", "server_prompt_token_mismatch", "prompt_floor_exceeds_cap"}:
            errors[f"runner_status:{status}"] += 1
        generated_tokens = row.get("generated_tokens")
        if generation_cap is not None and isinstance(generated_tokens, int) and generated_tokens >= generation_cap:
            errors["reached_generation_token_cap"] += 1
        if row.get("stop_reason") in {"max_new_tokens", "server_finish_reason:length", "server_finish_reason:max_tokens"}:
            errors["generation_stopped_at_limit"] += 1

    count = len(raw_rows)
    return {
        "sample_count": count,
        "raw_json_parse_rate": _ratio(parse_count, count),
        "raw_schema_valid_rate": _ratio(raw_schema_count, count),
        "valid_emotion_rate": _ratio(valid_emotion_count, count),
        "valid_profile_rate": _ratio(valid_profile_count, count),
        "nonempty_response_rate": _ratio(nonempty_response_count, count),
        "submission_schema_valid_rate": _ratio(sum(submission_schema_valid), len(submission_schema_valid)),
        "generation_stop_counts": dict(sorted(stops.items())),
        "error_case_counts": dict(sorted(errors.items())),
        "definitions": {
            "raw_json_parse_rate": "json.loads(raw_text) must consume the entire string and produce one JSON object",
            "raw_schema_valid_rate": "exact A0 fields; 16-class emotion; legal unique profile tags; nonempty response; memory_refs=[]",
            "submission_schema_valid_rate": "official Draft 2020-12 submission schema; empty emotion sentinel allowed",
            "invalid_or_empty_emotion": "counted as incorrect, never added as a 17th class",
        },
    }
