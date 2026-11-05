"""CPU-only helpers for the E03-A6 F3 Pilot Dev evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.evaluation.metrics import profile_metrics
from src.shared.f3_protocol import F3ParseError, parse_f3


PROFILE_DIMENSIONS = ("personality_traits", "interests", "style")


def empty_profile() -> dict[str, list[str]]:
    return {name: [] for name in PROFILE_DIMENSIONS}


def selected_ids_sha256(ids: Sequence[str]) -> str:
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("selected IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("selected IDs contain duplicates")
    payload = "[" + ",".join(json.dumps(value, ensure_ascii=False) for value in sorted(ids)) + "]"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_reference_coverage(reference_ids: Sequence[str], selected_ids: Sequence[str]) -> None:
    """Require every selected generation ID to have a scorer reference.

    A Smoke cohort is a strict subset of the 160-row Pilot set, so unrelated
    extra Reference rows are allowed while missing or duplicate IDs are not.
    """
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("reference IDs contain duplicates")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected IDs contain duplicates")
    missing = set(selected_ids) - set(reference_ids)
    if missing:
        raise ValueError("References do not cover every selected generation ID")


def conversation_id_from_sample_id(sample_id: str) -> str:
    """Recover D0.3 conversation identity from the frozen sample ID only."""
    parts = sample_id.split(":")
    if len(parts) < 3 or not all(parts[:2]):
        raise ValueError(f"sample ID does not encode a conversation: {sample_id!r}")
    return ":".join(parts[:2])


def derive_smoke_selection(
    input_ids: Sequence[str],
    sample_id_by_eval_id: Mapping[str, str],
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Select one target-free ID per conversation using deterministic SHA ranks."""
    if isinstance(count, bool) or count < 1:
        raise ValueError("smoke selection count must be a positive integer")
    if len(input_ids) != len(set(input_ids)) or set(input_ids) != set(sample_id_by_eval_id):
        raise ValueError("input and ID map IDs must be unique and identical")
    by_conversation: dict[str, list[str]] = {}
    for eval_id in input_ids:
        conversation = conversation_id_from_sample_id(sample_id_by_eval_id[eval_id])
        by_conversation.setdefault(conversation, []).append(eval_id)
    if len(by_conversation) < count:
        raise ValueError("not enough unique conversations for the requested smoke cohort")
    ranked: list[tuple[str, str, str]] = []
    for conversation, eval_ids in by_conversation.items():
        representative = min(
            eval_ids,
            key=lambda eval_id: hashlib.sha256(
                f"{seed}:point:{eval_id}".encode("utf-8")
            ).hexdigest(),
        )
        rank = hashlib.sha256(f"{seed}:conversation:{conversation}".encode("utf-8")).hexdigest()
        ranked.append((rank, conversation, representative))
    ranked.sort()
    return [eval_id for _rank, _conversation, eval_id in ranked[:count]]


@dataclass(frozen=True)
class F3FastResult:
    status: str
    reason: str
    raw_prefix: str
    parsed: Mapping[str, Any] | None
    generated_token_ids: tuple[int, ...]

    @property
    def valid(self) -> bool:
        return self.status == "valid"


class F3FastPrefixStopper:
    """Stop only when generated token IDs decode through the complete ``\nR=`` marker."""

    def __init__(
        self,
        decode: Callable[[Sequence[int]], str],
        enums: Mapping[str, list[str]],
        *,
        deadline: float | None = None,
    ) -> None:
        self.decode = decode
        self.enums = enums
        self.deadline = deadline
        self.stop_reason: str | None = None
        self.result: F3FastResult | None = None

    def inspect(self, generated_token_ids: Sequence[int]) -> bool:
        ids = tuple(int(value) for value in generated_token_ids)
        if self.deadline is not None:
            import time

            if time.monotonic() >= self.deadline:
                self.stop_reason = "budget_exhausted"
                self.result = F3FastResult("budget_exhausted", self.stop_reason, "", None, ids)
                return True
        decoded = self.decode(ids)
        marker = decoded.find("\nR=")
        if marker < 0:
            return False
        boundary = marker + len("\nR=")
        prefix = decoded[:boundary]
        try:
            parsed = parse_f3(prefix, dict(self.enums))
        except F3ParseError as exc:
            self.stop_reason = "invalid_at_r_marker"
            self.result = F3FastResult("invalid", exc.code, prefix, None, ids)
            return True
        self.stop_reason = "r_marker_closed"
        self.result = F3FastResult("valid", "r_marker_closed", prefix, parsed, ids)
        return True

    def finish(self, stop_reason: str, generated_token_ids: Sequence[int]) -> F3FastResult:
        if self.result is not None:
            return self.result
        ids = tuple(int(value) for value in generated_token_ids)
        decoded = self.decode(ids)
        result = F3FastResult("incomplete", stop_reason, decoded, None, ids)
        self.stop_reason = stop_reason
        self.result = result
        return result


def score_fast_cases(
    gold_targets: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    enums: dict[str, list[str]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    if len(gold_targets) != len(cases):
        raise ValueError("F3 Fast cases and references differ in row count")
    gold_profiles: list[dict[str, list[str]]] = []
    predicted_profiles: list[dict[str, list[str]]] = []
    gold_emotions: list[str] = []
    predicted_emotions: list[str | None] = []
    status_counts: dict[str, int] = {}
    nonempty_cases = 0
    for target, case in zip(gold_targets, cases):
        if not isinstance(target.get("user_profile"), Mapping):
            raise ValueError("reference target has no profile object")
        gold_profiles.append({name: list(target["user_profile"][name]) for name in PROFILE_DIMENSIONS})
        parse_success = case.get("parse_status") == "success"
        status = str(case.get("parse_status", case.get("status", "missing")))
        status_counts[status] = status_counts.get(status, 0) + 1
        parsed = case.get("parsed")
        if parse_success and isinstance(parsed, Mapping):
            profile = parsed.get("user_profile")
            emotion = parsed.get("emotion_label")
            if not isinstance(profile, Mapping) or not isinstance(emotion, str):
                raise ValueError("successful F3 Fast case has no parsed labels")
            normalized = {name: list(profile[name]) for name in PROFILE_DIMENSIONS}
            nonempty_cases += any(normalized[name] for name in PROFILE_DIMENSIONS)
            predicted_profiles.append(normalized)
            predicted_emotions.append(emotion)
        else:
            predicted_profiles.append(empty_profile())
            predicted_emotions.append(None)
        gold_emotions.append(str(target["emotion_label"]))

    profile_summary = profile_metrics(gold_profiles, predicted_profiles, enums)
    emotion_counts: dict[str, int] = {}
    emotion_correct = 0
    for gold, predicted in zip(gold_emotions, predicted_emotions):
        if predicted is not None:
            emotion_counts[predicted] = emotion_counts.get(predicted, 0) + 1
            emotion_correct += int(gold == predicted)
    predicted_emotion_count = sum(emotion_counts.values())
    single_emotion_share = (
        max(emotion_counts.values(), default=0) / predicted_emotion_count
        if predicted_emotion_count else 0.0
    )
    fast_gates = {
        "strict_f3_parse_rate": sum(
            case.get("parse_status") == "success" for case in cases
        ) / len(cases) if cases else 0.0,
        "cases_with_nonempty_profile_prediction": nonempty_cases,
        "profile_true_positives": profile_summary["overall_33_label_tp"],
        "profile_false_positives": profile_summary["overall_33_label_fp"],
        "profile_micro_precision": profile_summary["overall_33_label_micro_precision"],
        "emotion_accuracy": emotion_correct / len(cases) if cases else 0.0,
        "max_single_emotion_prediction_share": single_emotion_share,
    }
    checks = {
        "strict_f3_parse_rate": fast_gates["strict_f3_parse_rate"] >= float(gates["min_strict_f3_parse_rate"]),
        "cases_with_nonempty_profile_prediction": fast_gates["cases_with_nonempty_profile_prediction"] >= int(gates["min_cases_with_nonempty_profile_prediction"]),
        "profile_true_positives": fast_gates["profile_true_positives"] >= int(gates["min_profile_true_positives"]),
        "profile_false_positives": fast_gates["profile_false_positives"] <= int(gates["max_profile_false_positives"]),
        "profile_micro_precision": fast_gates["profile_micro_precision"] >= float(gates["min_profile_micro_precision"]),
        "emotion_accuracy": fast_gates["emotion_accuracy"] >= float(gates["min_emotion_accuracy"]),
        "max_single_emotion_prediction_share": fast_gates["max_single_emotion_prediction_share"] <= float(gates["max_single_emotion_prediction_share"]),
    }
    return {
        "fixed_denominator_count": len(cases),
        "status_counts": dict(sorted(status_counts.items())),
        "fixed_denominator_profile_metrics": profile_summary,
        "emotion_accuracy_fixed_denominator": fast_gates["emotion_accuracy"],
        "emotion_prediction_counts": dict(sorted(emotion_counts.items())),
        "nonempty_profile_prediction_count": nonempty_cases,
        "gate_values": fast_gates,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def score_smoke_cases(cases: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    if not cases:
        raise ValueError("F3 full-generation Smoke must contain at least one case")
    count = len(cases)
    strict_parse = sum(bool(case.get("full", {}).get("strict_parse_success")) for case in cases)
    schema_valid = sum(bool(case.get("full", {}).get("official_schema_valid")) for case in cases)
    prefix_equal = sum(bool(case.get("prefix_equivalence", {}).get("equivalent")) for case in cases)
    nonempty_response = sum(bool(case.get("full", {}).get("nonempty_response")) for case in cases)
    eos = sum(bool(case.get("full", {}).get("eos_generated")) for case in cases)
    values = {
        "strict_f3_parse_rate": strict_parse / count,
        "official_schema_valid_rate": schema_valid / count,
        "prefix_full_equivalence_rate": prefix_equal / count,
        "nonempty_response_rate": nonempty_response / count,
        "eos_rate": eos / count,
    }
    checks = {
        "strict_f3_parse_rate": values["strict_f3_parse_rate"] >= float(gates["min_strict_f3_parse_rate"]),
        "official_schema_valid_rate": values["official_schema_valid_rate"] >= float(gates["min_official_schema_valid_rate"]),
        "prefix_full_equivalence_rate": values["prefix_full_equivalence_rate"] >= float(gates["min_prefix_full_equivalence_rate"]),
        "nonempty_response_rate": values["nonempty_response_rate"] >= float(gates["min_nonempty_response_rate"]),
        "eos_rate": values["eos_rate"] >= float(gates["min_eos_rate"]),
    }
    return {
        "fixed_denominator_count": count,
        "status_counts": _case_status_counts(cases),
        "gate_values": values,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "full_generated_tokens_total": sum(int(case.get("full", {}).get("generated_token_count", 0)) for case in cases),
        "fast_generated_tokens_total": sum(int(case.get("fast", {}).get("generated_token_count", 0)) for case in cases),
    }


def _case_status_counts(cases: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        status = str(case.get("full", {}).get("parse_status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
