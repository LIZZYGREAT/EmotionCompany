from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.diagnostics.e03_a0 import PROFILE_DIMS, assign_token_regions
from src.diagnostics.e03_a0_1_profile import (
    FINE_REGIONS,
    FineAccumulator,
    _profile_schema,
    assign_profile_fine_regions,
    build_population_summary,
    select_review_samples,
)


def _fixture():
    completion = json.dumps(
        {
            "emotion_label": "calm",
            "user_profile": {
                "personality_traits": ["conservative"],
                "interests": [],
                "style": ["indirect"],
            },
            "response_text": "conservative [] 雪 </think>",
            "memory_refs": [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prefix = "PROMPT"
    suffix = "<|im_end|>\n"
    rendered = prefix + completion + suffix
    offsets = [(index, index + 1) for index in range(len(rendered))]
    offsets.append((0, 0))  # locked terminal EOS
    offsets.append((0, 0))  # ignored right padding
    token_ids = [7] * len(rendered) + [99, 0]
    labels = [-100] * len(prefix) + [7] * (len(rendered) - len(prefix)) + [99, -100]
    top, profile, _counts = assign_token_regions(
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
    enums = {
        "personality_traits": ["conservative"],
        "interests": ["games"],
        "style": ["indirect"],
    }
    return completion, prefix, rendered, offsets, labels, top, profile, enums


def _fine_fixture(offsets=None):
    completion, prefix, rendered, base_offsets, labels, top, profile, enums = _fixture()
    return assign_profile_fine_regions(
        completion=completion,
        completion_start=len(prefix),
        offsets=base_offsets if offsets is None else offsets,
        labels=labels,
        top_regions=top,
        profile_regions=profile,
        label_enums=enums,
    )


def test_official_schema_has_33_unique_profile_labels() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "data/train/train_public.schema.json"
    enums = _profile_schema(json.loads(schema_path.read_text(encoding="utf-8")))
    assert tuple(enums) == PROFILE_DIMS
    assert sum(map(len, enums.values())) == 33
    assert sum(len(set(values)) for values in enums.values()) == 33


def test_fine_profile_mapping_uses_exact_json_spans_and_conserves_a0_tokens() -> None:
    completion, _prefix, _rendered, offsets, labels, top, profile, enums = _fixture()
    mapped = assign_profile_fine_regions(
        completion=completion,
        completion_start=len("PROMPT"),
        offsets=offsets,
        labels=labels,
        top_regions=top,
        profile_regions=profile,
        label_enums=enums,
    )
    assert set(mapped["region_counts"]) == set(FINE_REGIONS)
    assert sum(mapped["region_counts"].values()) == top.count("user_profile")
    assert mapped["region_counts"]["profile_label_value"] == len("conservative") + len("indirect")
    assert mapped["region_counts"]["profile_empty_decision"] == 2
    assert mapped["label_exact_tokens"] == {"conservative": len("conservative"), "indirect": len("indirect")}
    # The same text in response_text is not treated as a profile label occurrence.
    assert [item["label"] for item in mapped["label_occurrences"]].count("conservative") == 1


def test_empty_decision_counts_are_attributed_to_their_own_dimension() -> None:
    completion, prefix, _rendered, offsets, labels, top, profile, enums = _fixture()
    mapped = assign_profile_fine_regions(
        completion=completion,
        completion_start=len(prefix),
        offsets=offsets,
        labels=labels,
        top_regions=top,
        profile_regions=profile,
        label_enums=enums,
    )
    population = {
        "label_support": {
            dim: {label: {"positive_sample_count": 1 if label in {"conservative", "indirect"} else 0} for label in values}
            for dim, values in enums.items()
        }
    }
    accumulator = FineAccumulator(enums)
    accumulator.add(
        mapped,
        {"personality_traits": ["conservative"], "interests": [], "style": ["indirect"]},
        top.count("user_profile"),
    )
    summary = accumulator.summary(enums, population)
    assert summary["dimensions"]["interests"]["gold_empty"]["fine_region_token_counts"]["profile_empty_decision"] == 2
    assert summary["dimensions"]["personality_traits"]["gold_empty"]["fine_region_token_counts"]["profile_empty_decision"] == 0
    assert summary["dimensions"]["personality_traits"]["gold_nonempty"]["fine_region_token_counts"]["profile_empty_decision"] == 0


def test_token_crossing_label_quote_is_mixed_and_counted_for_that_label() -> None:
    completion, prefix, rendered, offsets, labels, top, profile, enums = _fixture()
    label_at = completion.index('"conservative"')
    changed = list(offsets)
    token_index = len(prefix) + label_at + 1
    changed[token_index] = (len(prefix) + label_at, len(prefix) + label_at + 2)
    mapped = assign_profile_fine_regions(
        completion=completion,
        completion_start=len(prefix),
        offsets=changed,
        labels=labels,
        top_regions=top,
        profile_regions=profile,
        label_enums=enums,
    )
    assert mapped["token_regions"][token_index] == "profile_boundary_mixed"
    assert mapped["label_boundary_tokens"]["conservative"] == 1
    assert sum(mapped["region_counts"].values()) == top.count("user_profile")


def test_empty_array_token_crossing_into_punctuation_is_not_forced_into_empty_bucket() -> None:
    completion, prefix, _rendered, offsets, labels, top, profile, enums = _fixture()
    empty_at = completion.index("[]")
    changed = list(offsets)
    # Token covers the array close bracket and the following profile comma.
    changed[len(prefix) + empty_at + 1] = (
        len(prefix) + empty_at + 1,
        len(prefix) + empty_at + 3,
    )
    mapped = assign_profile_fine_regions(
        completion=completion,
        completion_start=len(prefix),
        offsets=changed,
        labels=labels,
        top_regions=top,
        profile_regions=profile,
        label_enums=enums,
    )
    assert mapped["token_regions"][len(prefix) + empty_at + 1] == "profile_boundary_mixed"
    assert mapped["region_counts"]["profile_empty_decision"] == 1
    assert sum(mapped["region_counts"].values()) == top.count("user_profile")


def test_population_summary_reports_support_cooccurrence_and_condition_denominators() -> None:
    labels = {
        "personality_traits": ["conservative", "sensitive"],
        "interests": ["games"],
        "style": ["indirect"],
    }
    rows = [
        {
            "labels_by_dim": {"personality_traits": ["conservative", "sensitive"], "interests": ["games"], "style": []},
            "fit_outcome": "full_fit",
            "history_bin": "short_1_4_turns",
            "profile_all_empty": False,
            "history_turn_count": 3,
        },
        {
            "labels_by_dim": {"personality_traits": [], "interests": [], "style": ["indirect"]},
            "fit_outcome": "trimmed_fit",
            "history_bin": "long_9_plus_turns",
            "profile_all_empty": False,
            "history_turn_count": 10,
        },
    ]
    summary = build_population_summary(rows, labels)
    assert summary["sample_count"] == 2
    assert summary["label_support"]["personality_traits"]["conservative"]["positive_sample_count"] == 1
    pairs = summary["label_pair_cooccurrence"]["personality_traits"]
    assert {"label_a": "conservative", "label_b": "sensitive", "sample_count": 1} in pairs
    assert summary["groups"]["full_fit"]["sample_count"] == 1
    assert summary["groups"]["trimmed_fit"]["sample_count"] == 1
    assert summary["groups"]["short_1_4_turns"]["sample_count"] == 1
    assert summary["groups"]["profile_any_nonempty"]["sample_count"] == 2


def test_audit_sample_selection_is_seeded_deduplicated_and_reports_overlapping_groups() -> None:
    rows = []
    for index in range(30):
        rows.append({
            "sample_id": f"train:synthetic:{index}",
            "labels_by_dim": {
                "personality_traits": ["conservative"] if index % 2 == 0 else [],
                "interests": ["games"] if index % 3 == 0 else [],
                "style": [],
            },
            "history_turn_count": index + 1,
            "history_bin": "short_1_4_turns" if index < 4 else "long_9_plus_turns",
        })
    enums = {
        "personality_traits": ["conservative"],
        "interests": ["games"],
        "style": ["indirect"],
    }
    first = select_review_samples(
        rows, label_enums=enums, focus_labels=["conservative", "games"], seed=31,
        positives_per_label=8, positives_per_history_stratum=4, negatives_per_label=4,
    )
    second = select_review_samples(
        rows, label_enums=enums, focus_labels=["conservative", "games"], seed=31,
        positives_per_label=8, positives_per_history_stratum=4, negatives_per_label=4,
    )
    assert first == second
    selected, memberships, metadata = first
    assert len(selected) == len(set(selected))
    assert metadata["unique_selected_sample_count"] == len(selected)
    assert metadata["selection_is_population_representative"] is False
    assert any(len(groups) > 1 for groups in memberships.values())
