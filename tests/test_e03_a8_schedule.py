from __future__ import annotations

from collections import Counter

from src.data.e03_a8_schedule import build_a8_v2_schedule
from src.shared.e03_a8_protocol import PROFILE_DIMS


def _candidate(index: int, *, empty: bool = False):
    profile = {dimension: [] for dimension in PROFILE_DIMS}
    if not empty:
        dimension = PROFILE_DIMS[index % len(PROFILE_DIMS)]
        profile[dimension] = [
            ("sensitive", "open", "introverted")[index % 3]
            if dimension == "personality_traits"
            else ("games", "pets", "music")[index % 3]
            if dimension == "interests"
            else ("direct", "rational", "brief")[index % 3]
        ]
    return {
        "sample_id": f"sample-{index:03d}",
        "conversation_id": f"conversation-{index // 2:03d}",
        "profile": profile,
        "history_message_count": 1 + (index % 8),
    }


def test_schedule_is_reproducible_complete_and_balanced_per_optimizer_step():
    candidates = [_candidate(index) for index in range(24)] + [_candidate(24 + index, empty=True) for index in range(8)]
    schedule, audit = build_a8_v2_schedule(
        candidates,
        seed=42,
        accumulation_steps=4,
        first_window_rows=8,
        first_window_nonempty=6,
    )
    again, audit_again = build_a8_v2_schedule(
        candidates,
        seed=42,
        accumulation_steps=4,
        first_window_rows=8,
        first_window_nonempty=6,
    )

    assert schedule == again
    assert audit == audit_again
    flattened = [sample_id for step in schedule for sample_id in step["sample_ids"]]
    assert len(flattened) == len(candidates)
    assert set(flattened) == {row["sample_id"] for row in candidates}
    assert len(set(flattened)) == len(flattened)
    assert audit["first_window_nonempty_rows"] == 6
    assert audit["first_window_all_empty_rows"] == 2
    assert audit["first_window_ids_complement_remainder"] is True
    assert len(audit["first_window_profile_label_counts"]) > 0

    row_by_id = {row["sample_id"]: row for row in candidates}
    for step in schedule[:2]:
        nonempty = sum(any(row_by_id[sid]["profile"][dim] for dim in PROFILE_DIMS) for sid in step["sample_ids"])
        assert nonempty == 3


def test_schedule_selects_distinct_ids_and_reports_conversation_repetition():
    candidates = [_candidate(index) for index in range(24)] + [_candidate(24 + index, empty=True) for index in range(8)]
    _schedule, audit = build_a8_v2_schedule(
        candidates,
        seed=7,
        accumulation_steps=4,
        first_window_rows=8,
        first_window_nonempty=6,
    )
    assert audit["first_window_unique_conversations"] > 0
    assert audit["first_window_repeated_conversation_samples"] >= 0
    assert audit["first_window_max_samples_per_conversation"] >= 1
