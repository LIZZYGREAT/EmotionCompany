from __future__ import annotations

import json

import pytest

from scripts import e02_benchmark_selection as selection


def _candidate(index: int, prompt_tokens: int) -> dict:
    return {"id": f"sample-{index:03d}", "history": [], "prompt_tokens": prompt_tokens}


def test_selection_is_deterministic_and_covers_prompt_length_tertiles():
    candidates = [_candidate(index, 200 + index * 3) for index in range(180)]

    first, facts = selection.select_stratified_candidates(candidates)
    second, _ = selection.select_stratified_candidates(candidates)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 100
    assert {name: facts[name]["selected_count"] for name in facts} == {
        "short": 34,
        "medium": 33,
        "long": 33,
    }
    assert facts["short"]["prompt_token_max"] < facts["long"]["prompt_token_max"]
    assert len({row["id"] for row in first}) == 100


def test_projection_ignores_source_target_fields():
    history = [{"role": "user", "content": "synthetic"}]
    first = selection.project_target_free_sample(
        {"sample_id": "sample-1", "history": history, "target": {"emotion_label": "joy"}},
        line_number=1,
    )
    second = selection.project_target_free_sample(
        {"sample_id": "sample-1", "history": history, "target": {"emotion_label": "sadness"}},
        line_number=1,
    )

    assert first == second == {"id": "sample-1", "history": history}


def test_target_free_projection_hash_does_not_depend_on_target_values(tmp_path):
    history = [{"role": "user", "content": "synthetic"}]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"sample_id": "one", "history": history, "target": {"x": 1}}) + "\n", encoding="utf-8")
    _first_rows, first_hash = selection.read_target_free_projection(source)
    source.write_text(json.dumps({"sample_id": "one", "history": history, "target": {"x": 2}}) + "\n", encoding="utf-8")
    _second_rows, second_hash = selection.read_target_free_projection(source)

    assert first_hash == second_hash


def test_projection_rejects_duplicate_ids_and_nonstandard_history(tmp_path):
    source = tmp_path / "source.jsonl"
    row = {"id": "duplicate", "history": [{"role": "user", "content": "synthetic"}]}
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        selection.read_target_free_projection(source)

    row["history"] = [{"role": "system", "content": "synthetic"}]
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非标准 History"):
        selection.read_target_free_projection(source)


def test_stratified_selection_rejects_insufficient_stratum_capacity():
    candidates = [_candidate(index, index + 1) for index in range(20)]

    with pytest.raises(ValueError, match="候选不足"):
        selection.select_stratified_candidates(
            candidates,
            quotas={"short": 10, "medium": 10, "long": 10},
        )


def test_candidate_writer_emits_only_id_and_history(tmp_path):
    rows = [{"id": "one", "history": [{"role": "user", "content": "synthetic"}], "prompt_tokens": 1, "stratum": "short"}]

    encoded = selection._write_jsonl_bytes(rows)
    output = json.loads(encoded.decode("utf-8"))

    assert set(output) == {"id", "history"}
    assert output["id"] == "one"
