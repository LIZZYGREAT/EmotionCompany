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


def test_smoke_eval_id_maps_to_distinct_d03_sample_id(tmp_path):
    smoke = tmp_path / "smoke.jsonl"
    id_map = tmp_path / "id_map.jsonl"
    smoke.write_text(json.dumps({
        "id": "test_001058_t0",
        "history": [{"role": "user", "content": "synthetic"}],
    }) + "\n", encoding="utf-8")
    id_map.write_text(json.dumps({
        "eval_id": "test_001058_t0",
        "sample_id": "train:train_conv_005336:pp2",
        "source": {"split": "train"},
    }) + "\n", encoding="utf-8")

    mapped, facts = selection.map_smoke_ids_to_sample_ids(smoke, id_map)

    assert {"test_001058_t0"} != mapped
    assert mapped == {"train:train_conv_005336:pp2"}
    assert facts["smoke_eval_id_count"] == facts["mapped_sample_id_count"] == 1


def test_exclusion_removes_actual_mapped_sample_id_overlap_only():
    source_rows = [
        {"id": "train:train_conv_005336:pp2", "history": [{"role": "user", "content": "overlap"}]},
        {"id": "train:train_conv_009999:pp0", "history": [{"role": "user", "content": "keep"}]},
    ]
    # These are local eval IDs, in a different namespace from D0.3 sample_id.
    smoke_eval_ids = {"test_001058_t0"}
    id_map_sample_ids = {"train:train_conv_005336:pp2"}
    assert {row["id"] for row in source_rows} & smoke_eval_ids == set()

    remaining, overlap = selection.exclude_sample_id_overlap(source_rows, id_map_sample_ids)

    assert overlap == {"train:train_conv_005336:pp2"}
    assert [row["id"] for row in remaining] == ["train:train_conv_009999:pp0"]


def test_smoke_mapping_requires_every_eval_id_and_unique_source_ids(tmp_path):
    smoke = tmp_path / "smoke.jsonl"
    id_map = tmp_path / "id_map.jsonl"
    smoke.write_text("".join(json.dumps({
        "id": sample_id,
        "history": [{"role": "user", "content": "synthetic"}],
    }) + "\n" for sample_id in ("test_000001_t0", "test_000002_t0")), encoding="utf-8")
    id_map.write_text(json.dumps({"eval_id": "test_000001_t0", "sample_id": "source-1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未出现在 id_map.jsonl"):
        selection.map_smoke_ids_to_sample_ids(smoke, id_map)

    id_map.write_text("".join(json.dumps({"eval_id": eval_id, "sample_id": "same-source"}) + "\n" for eval_id in ("test_000001_t0", "test_000002_t0")), encoding="utf-8")
    with pytest.raises(ValueError, match="重复 sample_id"):
        selection.map_smoke_ids_to_sample_ids(smoke, id_map)


def test_selected_benchmark_ids_fit_submission_schema_and_preserve_source_identity():
    selected = [
        {"id": f"train:sample-{index}", "history": [], "prompt_tokens": 400 + index, "stratum": "short"}
        for index in range(selection.DEFAULT_SAMPLE_COUNT)
    ]

    rows = selection.assign_benchmark_input_ids(selected)

    assert rows[0]["id"] == "test_900001_t0"
    assert rows[-1]["id"] == "test_900100_t0"
    assert rows[0]["source_sample_id"] == "train:sample-0"
    assert len({row["id"] for row in rows}) == 100
