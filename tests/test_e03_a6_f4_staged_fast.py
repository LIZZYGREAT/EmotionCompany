"""CPU-only F4 staged Fast-only boundary and fixed-denominator tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.e03_a6_f4_staged_metrics import (
    load_target_free_cohorts,
    response_body_token_count,
    score_cohort,
)
from src.evaluation.f3_pilot import selected_ids_sha256
from src.evaluation.f4_pilot import F4FastPrefixStopper, F4FastResult
from src.shared.f3_protocol import submission_enums
from scripts.e03_a6_f4_staged_eval import _prepare_eval_directory, _reported_stop_reason


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "participant/templates/submission_schema.json").read_text())
ENUMS = submission_enums(SCHEMA)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_f4_fast_stopper_handles_a_response_marker_split_across_tokens():
    first = "Emotion=anxiety\nPersonality=-\nInterests=-\nStyle=-\nResp"
    pieces = {1: first, 2: "onse="}
    stopper = F4FastPrefixStopper(lambda ids: "".join(pieces[value] for value in ids), ENUMS)
    assert not stopper.inspect([1])
    assert stopper.inspect([1, 2])
    assert stopper.result is not None and stopper.result.valid
    assert stopper.result.raw_prefix.endswith("\nResponse=")


def test_fast_evaluator_reads_stop_reason_from_result_reason_field():
    stopper = F4FastPrefixStopper(lambda _ids: "", ENUMS)
    parsed = F4FastResult("incomplete", "token_limit", "", None, ())
    assert _reported_stop_reason(parsed, stopper) == "token_limit"

    stopper.stop_reason = "response_marker_closed"
    complete = F4FastResult("valid", "response_marker_closed", "", {}, ())
    assert _reported_stop_reason(complete, stopper) == "response_marker_closed"


def test_fast_evaluator_archives_empty_failed_attempt_before_identity_change(tmp_path: Path):
    output = tmp_path / "step-0512"
    output.mkdir()
    (output / "predictions.jsonl").write_text("", encoding="utf-8")
    old_manifest = {
        "status": "failed_resumable",
        "evaluation_identity": {"evaluation_code_sha256": "old-code-hash"},
        "error": "AttributeError: old evaluator field contract",
        "references_read_by_generator": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")

    identity = {"evaluation_code_sha256": "new-code-hash", "checkpoint_step": 512}
    manifest, rows = _prepare_eval_directory(output, identity, ["eval-1"])

    assert manifest["status"] == "running"
    assert manifest["evaluation_identity"] == identity
    assert manifest["previous_attempts"][0]["error"] == old_manifest["error"]
    archived = output / manifest["previous_attempts"][0]["manifest_path"]
    assert json.loads(archived.read_text(encoding="utf-8")) == old_manifest
    assert rows == []


def test_fast_evaluator_rejects_identity_change_after_partial_predictions(tmp_path: Path):
    output = tmp_path / "step-0512"
    output.mkdir()
    _write_jsonl(output / "predictions.jsonl", [{"id": "eval-1"}])
    (output / "run_manifest.json").write_text(json.dumps({
        "status": "failed_resumable",
        "evaluation_identity": {"evaluation_code_sha256": "old-code-hash"},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="different Run/checkpoint/cohort identity"):
        _prepare_eval_directory(output, {"evaluation_code_sha256": "new-code-hash"}, ["eval-1", "eval-2"])


def test_response_body_token_count_includes_text_in_the_marker_token():
    separate = {1: "Emotion=anxiety\nPersonality=-\nInterests=-\nStyle=-\n", 2: "Response=", 3: "Hello"}
    decode_separate = lambda ids: "".join(separate[value] for value in ids)
    assert response_body_token_count([1, 2], decode_separate) == 0
    assert response_body_token_count([1, 2, 3], decode_separate) == 1

    same_token = {1: "Emotion=anxiety\nPersonality=-\nInterests=-\nStyle=-\nResponse=Hi"}
    assert response_body_token_count([1], lambda ids: "".join(same_token[value] for value in ids)) == 1


def test_target_free_cohort_loader_does_not_open_gold_references(tmp_path: Path, monkeypatch):
    import src.evaluation.e03_a6_f4_staged_metrics as metrics

    original_read = metrics._read_jsonl
    reads: list[Path] = []

    def tracked_read(path: Path):
        reads.append(path)
        if path.name.endswith("references.jsonl"):
            raise AssertionError("target-free cohort loading opened Gold References")
        return original_read(path)

    monkeypatch.setattr(metrics, "_read_jsonl", tracked_read)
    cohorts = {}
    for name in ("pilot160", "dev224"):
        eval_id = f"{name}-id"
        input_path = tmp_path / f"{name}-input.jsonl"
        map_path = tmp_path / f"{name}-id-map.jsonl"
        input_hash = _write_jsonl(input_path, [{"id": eval_id, "history": [{"role": "user", "content": "synthetic"}]}])
        map_hash = _write_jsonl(map_path, [{"eval_id": eval_id, "sample_id": f"train:{name}:001", "conversation_id": name}])
        cohorts[name] = {
            "input_path": str(input_path.relative_to(tmp_path)),
            "id_map_path": str(map_path.relative_to(tmp_path)),
            "references_path": f"{name}-references.jsonl",
            "expected_input_sha256": input_hash,
            "expected_id_map_sha256": map_hash,
            "expected_references_sha256": "0" * 64,
            "expected_count": 1,
            "expected_ids_sha256": selected_ids_sha256([eval_id]),
        }
    result = load_target_free_cohorts({"cohorts": cohorts}, tmp_path)
    assert result["references_read"] is False
    assert result["union_count"] == 2
    assert all("references.jsonl" not in str(path) for path in reads)


def test_score_cohort_keeps_invalid_generation_in_the_fixed_denominator():
    cohort = {
        "name": "synthetic",
        "ids": ["id-1", "id-2"],
        "ids_sha256": "fixed-ids",
        "conversation_count": 2,
    }
    references = {
        eval_id: {
            "target": {
                "emotion_label": emotion,
                "user_profile": {
                    "personality_traits": ["sensitive"],
                    "interests": [],
                    "style": ["brief"],
                },
            }
        }
        for eval_id, emotion in (("id-1", "anxiety"), ("id-2", "sadness"))
    }
    predictions = {
        "id-1": {
            "status": "valid",
            "parsed": {"emotion_label": "anxiety", "user_profile": {"personality_traits": ["sensitive"], "interests": [], "style": []}},
            "generated_token_count": 12,
            "stop_reason": "response_marker_closed",
        },
        "id-2": {
            "status": "incomplete",
            "parsed": None,
            "generated_token_count": 256,
            "stop_reason": "token_limit",
            "reason": "truncated",
        },
    }
    metric = score_cohort(cohort, references, predictions, ENUMS)
    assert metric["fixed_denominator_count"] == 2
    assert metric["valid_count"] == 1
    assert metric["emotion"]["accuracy"] == 0.5
    assert metric["profile_33_label"]["overall_33_label_tp"] == 1
    assert metric["nonempty_profile_prediction_count"] == 1
    assert metric["predicted_nonempty_by_dimension"] == {
        "personality_traits": 1, "interests": 0, "style": 0,
    }
    assert metric["error_reason_counts"] == {"truncated": 1}
