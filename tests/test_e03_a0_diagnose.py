from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.diagnostics.e03_a0 import (
    AlignmentError,
    PROFILE_DIMS,
    TOP_REGIONS,
    TokenAccumulator,
    _aggregate_ce,
    _ce_for_logits,
    assign_token_regions,
    create_run_dir,
    deterministic_select,
    parse_completion_regions,
    validate_sample_cap,
)


def _completion(response: str = 'calm "quoted" \\ path\n雪 </think>') -> str:
    return json.dumps(
        {
            "emotion_label": "calm",
            "user_profile": {
                "personality_traits": ["calm"],
                "interests": [],
                "style": [],
            },
            "response_text": response,
            "memory_refs": [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _synthetic_assignment(*, unknown_offset: bool = False, mixed_boundary: bool = False):
    completion = _completion()
    prefix = "PROMPT"
    suffix = "<|im_end|>\n"
    rendered = prefix + completion + suffix
    body_end = len(prefix) + len(completion)
    offsets = [(index, index + 1) for index in range(body_end)]
    if unknown_offset:
        offsets[len(prefix) + 2] = (0, 0)
    if mixed_boundary:
        primary_spans, _dims = parse_completion_regions(completion)
        boundary = len(prefix) + primary_spans["emotion_label"][1]
        offsets[boundary - 1] = (boundary - 1, boundary + 1)
    # Chat-template special EOS has no character Offset; trailing newline is ordinary text.
    offsets.append((0, 0))
    offsets.append((len(rendered) - 1, len(rendered)))
    token_ids = [7] * body_end + [99, 7]
    labels = [-100] * len(prefix) + [7] * (len(completion)) + [99, 7]
    offsets.append((0, 0))
    token_ids.append(0)
    labels.append(-100)
    return assign_token_regions(
        rendered_text=rendered,
        completion=completion,
        completion_start=len(prefix),
        offsets=offsets,
        token_ids=token_ids,
        prompt_length=len(prefix),
        labels=labels,
        eos_token_id=99,
        special_token_ids={99},
    )


def test_completion_regions_parse_repeated_values_and_escaped_unicode() -> None:
    completion = _completion()
    primary, dimensions = parse_completion_regions(completion)
    assert completion[primary["emotion_label"][0]:primary["emotion_label"][1]] == '"calm"'
    assert completion[primary["response_text"][0]:primary["response_text"][1]].startswith('"calm')
    assert completion[dimensions["personality_traits"][0]:dimensions["personality_traits"][1]].startswith('"personality_traits"')
    assert "\\n雪 </think>" in completion
    assert primary["emotion_label"] != primary["response_text"]


def test_token_assignment_conserves_all_supervised_regions_and_eos() -> None:
    regions, profile_regions, counts = _synthetic_assignment()
    assert sum(counts.values()) == len(regions) - len("PROMPT") - 1
    assert counts["eos"] == 1
    assert counts["public_structure"] >= 1
    assert counts["user_profile"] > 0
    assert all(region is None for region in profile_regions[:len("PROMPT")])
    assert regions[-1] == "ignored_prompt"  # right padding is excluded from supervision


def test_token_crossing_field_boundary_is_reported_as_mixed() -> None:
    _regions, _profile_regions, counts = _synthetic_assignment(mixed_boundary=True)
    assert counts["boundary_mixed"] == 1


def test_unrecognized_empty_offset_blocks_quantitative_mapping() -> None:
    with pytest.raises(AlignmentError, match="empty Offset"):
        _synthetic_assignment(unknown_offset=True)


def test_ce_shift_includes_first_completion_token_prediction() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]])
    labels = torch.tensor([[-100, -100, 1, 0]])
    regions = ["ignored_prompt", "ignored_prompt", "emotion_label", "eos"]
    profile_regions = [None, None, None, None]
    values, profile_values, total, count = _ce_for_logits(
        torch,
        logits,
        labels,
        regions,
        profile_regions,
        chunk_tokens=1,
    )
    expected = torch.nn.functional.cross_entropy(
        logits[:, 1:3, :].float().reshape(-1, 2),
        labels[:, 2:4].reshape(-1),
        reduction="sum",
    ).item()
    assert count == 2
    assert total == pytest.approx(expected)
    assert values["emotion_label"] > 0
    assert values["eos"] > 0
    assert sum(profile_values.values()) == 0


def test_aggregate_ce_reports_not_applicable_and_conserves_mixed_tokens() -> None:
    counts = {key: 0 for key in TOP_REGIONS}
    counts.update({"emotion_label": 1, "boundary_mixed": 1})
    profile_counts = {key: 0 for key in ("personality_traits", "interests", "style", "profile_structure", "boundary_mixed")}
    record = {
        "groups": (),
        "token_counts": counts,
        "profile_token_counts": profile_counts,
        "ce_sums": {key: float(counts[key]) for key in TOP_REGIONS},
        "profile_ce_sums": profile_counts | {key: 0.0 for key in profile_counts},
        "total_ce_sum": 2.0,
        "valid_token_count": 2,
        "native_loss": 1.0,
    }
    summary = _aggregate_ce([record])
    assert summary["conservation"]["passed"] is True
    assert summary["regions"]["boundary_mixed"]["ce_contribution"]["ratio"] == 0.5
    assert summary["regions"]["response_text"]["pooled_mean_token_ce"] is None
    assert summary["regions"]["response_text"]["macro_sample_count"] == 0


def test_token_accumulator_tracks_empty_and_nonempty_profile_groups() -> None:
    accumulator = TokenAccumulator()
    token_counts = {key: 0 for key in TOP_REGIONS}
    token_counts.update({"user_profile": 2, "public_structure": 1, "eos": 1})
    profile_counts = {"personality_traits": 1, "interests": 0, "style": 0, "profile_structure": 1, "boundary_mixed": 0}
    accumulator.add(
        {"profile_all_empty": False, "dim_nonempty": {dim: dim == "personality_traits" for dim in PROFILE_DIMS}, "fit_outcome": "full_fit", "short_completion": True},
        token_counts,
        profile_counts,
        20,
    )
    assert accumulator.groups["profile_any_nonempty"].samples == 1
    assert "profile_all_empty" not in accumulator.groups
    assert accumulator.groups["personality_traits_gold_nonempty"].samples == 1


def test_stratified_selection_is_stable_and_records_denominators() -> None:
    candidates = []
    for index in range(16):
        dim_nonempty = {dim: bool(index & (1 << bit)) for bit, dim in enumerate(PROFILE_DIMS)}
        candidates.append({
            "sample_id": f"train:synthetic:{index}",
            "profile_all_empty": not any(dim_nonempty.values()),
            "dim_nonempty": dim_nonempty,
            "fit_outcome": "full_fit" if index % 2 else "trimmed_fit",
            "completion_length": 50 + index * 10,
        })
    first, summary1 = deterministic_select(candidates, cap=8, seed=42, stratified=True)
    second, summary2 = deterministic_select(candidates, cap=8, seed=42, stratified=True)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert summary1["selected_sample_ids_sha256"] == summary2["selected_sample_ids_sha256"]
    assert summary1["selection_is_population_representative"] is False
    assert summary1["selected_rows"] == 8
    assert all(item["available"] >= item["selected"] for item in summary1["strata"].values())


def test_ce_mode_requires_a_finite_positive_cap() -> None:
    with pytest.raises(ValueError, match="refuses sample-cap=0"):
        validate_sample_cap("ce", 0, 128)
    with pytest.raises(ValueError, match="safety limit"):
        validate_sample_cap("ce", 129, 128)
    validate_sample_cap("ce", 128, 128)
    validate_sample_cap("token", 0, 128)


def test_run_directory_never_overwrites_existing_output(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    created = create_run_dir(output_root, "E03-A0-test")
    assert created.is_dir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_run_dir(output_root, "E03-A0-test")
    with pytest.raises(ValueError, match="unsupported characters"):
        create_run_dir(output_root, "../outside")
