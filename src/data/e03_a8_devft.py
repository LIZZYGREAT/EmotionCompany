"""Versioned A8 P data preparation for a Dev fine-tuning experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.data.e03_a8_sft import build_a8_training_item


def build_devft_training_item(
    sample: Mapping[str, Any],
    sft_item: Mapping[str, Any],
    budget_audit: Mapping[str, Any],
    **kwargs: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reuse the frozen A8 target serializer while requiring an explicit Dev row."""
    sample_id = sample.get("sample_id")
    if sample.get("source", {}).get("split") != "train":
        raise ValueError(f"{sample_id}: D0.3 Dev source identity is invalid")
    if sft_item.get("source_split") != "dev":
        raise ValueError(f"{sample_id}: expected a D0.4.3 Dev row")
    if sft_item.get("sample_id") != sample_id or sft_item.get("conversation_id") != sample.get("conversation_id"):
        raise ValueError(f"{sample_id}: normalized and D0.4.3 Dev identities differ")
    if budget_audit.get("split") != "dev" or budget_audit.get("sample_id") != sample_id:
        raise ValueError(f"{sample_id}: D0.4.2 Dev audit identity differs")
    if budget_audit.get("outcome") not in {"full_fit", "trimmed_fit"}:
        raise ValueError(f"{sample_id}: D0.4.2 did not accept this Dev sample")

    # The frozen builder enforces the original P0/T1 tokens, common P/EP
    # history, exact target serialization, and the prompt/completion masks.
    # A private copy changes only its legacy split guard; output rows are then
    # explicitly stamped with this new DevFT protocol identity.
    legacy_sft_item = dict(sft_item)
    legacy_sft_item["source_split"] = "train"
    rows, audit = build_a8_training_item(sample, legacy_sft_item, budget_audit, **kwargs)
    for row in rows.values():
        row["source_split"] = "dev_ft"
    audit["source_split"] = "dev_ft"
    return rows, audit
