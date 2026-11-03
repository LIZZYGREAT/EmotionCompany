"""Read-only profile label evidence and fine-grained supervision diagnostics.

Private audit rows contain source histories and Gold labels and are written only
under the ignored ``outputs/`` tree. Committed summaries contain aggregates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.diagnostics import e03_a0 as a0
from src.shared import prompting


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "E03-A0.1-v1"
FINE_REGIONS = (
    "profile_label_value",
    "profile_empty_decision",
    "profile_structure",
    "profile_boundary_mixed",
)
FIT_OUTCOMES = ("full_fit", "trimmed_fit")
HISTORY_BINS = ("short_1_4_turns", "mid_5_8_turns", "long_9_plus_turns")
REVIEW_CATEGORIES = ("explicit", "inferred", "insufficient", "conflicting", "uncertain")
T1_RETENTION = ("retained_by_T1", "removed_by_T1", "not_identified")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"configured path escapes project root: {value}")
    return resolved


def _load_task_config(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    config_path = _resolve(root, path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise ValueError(f"diagnostic config version must be {VERSION}")
    base_config_path = _resolve(root, config.get("base_a0_config", ""))
    base_config, paths = a0.load_config(root, base_config_path)
    output_root = _resolve(root, config.get("output_root", ""))
    if not output_root.is_relative_to((root / "outputs").resolve()):
        raise ValueError("E03-A0.1 output_root must remain under outputs/")
    paths = dict(paths)
    paths["e03_a0_1_output_root"] = output_root

    audit = config.get("audit", {})
    labels = audit.get("focus_labels")
    if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
        raise ValueError("audit.focus_labels must be a non-empty unique list")
    for key in ("seed", "positives_per_label", "positives_per_history_stratum", "negatives_per_label"):
        if not isinstance(audit.get(key), int) or isinstance(audit[key], bool) or audit[key] < 0:
            raise ValueError(f"audit.{key} must be a non-negative integer")
    ce = config.get("ce", {})
    for key in ("seed", "sample_cap", "logit_chunk_tokens", "minimum_label_sample_support"):
        if not isinstance(ce.get(key), int) or isinstance(ce[key], bool) or ce[key] <= 0:
            raise ValueError(f"ce.{key} must be a positive integer")
    return config, base_config, paths


def _profile_schema(schema: dict[str, Any]) -> dict[str, list[str]]:
    enums = prompting._schema_enums(schema)
    result = {dim: enums[dim] for dim in a0.PROFILE_DIMS}
    if sum(len(values) for values in result.values()) != 33:
        raise a0.AlignmentError("official profile schema no longer defines the expected 33 labels")
    return result


def _history_bin(turn_count: int) -> str:
    if turn_count <= 4:
        return HISTORY_BINS[0]
    if turn_count <= 8:
        return HISTORY_BINS[1]
    return HISTORY_BINS[2]


def _median(values: list[int]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _read_at(handle: Any, offset: int) -> dict[str, Any]:
    handle.seek(offset)
    row = json.loads(handle.readline())
    if not isinstance(row, dict):
        raise ValueError("indexed JSONL row is not an object")
    return row


def _collect_population(
    records: list[dict[str, Any]],
    *,
    source_handle: Any,
    label_enums: dict[str, list[str]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        source = _read_at(source_handle, record["normalized_offset"])
        history = source.get("history")
        target = source.get("target")
        if not isinstance(history, list) or not history or history[-1].get("role") != "user":
            raise a0.AlignmentError("Train source history is malformed or lacks its current user turn")
        if not isinstance(target, dict) or not isinstance(target.get("user_profile"), dict):
            raise a0.AlignmentError("Train source target lacks the required user_profile object")
        profile = target["user_profile"]
        labels_by_dim: dict[str, list[str]] = {}
        for dim in a0.PROFILE_DIMS:
            values = profile.get(dim)
            if not isinstance(values, list) or len(values) != len(set(values)):
                raise a0.AlignmentError(f"Train Gold {dim} must be a unique label list")
            unknown = set(values) - set(label_enums[dim])
            if unknown:
                raise a0.AlignmentError(f"Train Gold {dim} contains labels outside the official enum")
            labels_by_dim[dim] = list(values)
        enriched.append({
            **record,
            "labels_by_dim": labels_by_dim,
            "history_turn_count": len(history),
            "history_bin": _history_bin(len(history)),
        })
    return enriched


def _group_summary(rows: list[dict[str, Any]], label_enums: dict[str, list[str]]) -> dict[str, Any]:
    dim_counts = {dim: Counter() for dim in a0.PROFILE_DIMS}
    label_counts = {dim: Counter() for dim in a0.PROFILE_DIMS}
    dim_count_distribution = {dim: Counter() for dim in a0.PROFILE_DIMS}
    all_count_distribution: Counter[int] = Counter()
    for row in rows:
        total = 0
        for dim in a0.PROFILE_DIMS:
            labels = row["labels_by_dim"][dim]
            dim_counts[dim]["gold_nonempty" if labels else "gold_empty"] += 1
            dim_count_distribution[dim][len(labels)] += 1
            label_counts[dim].update(labels)
            total += len(labels)
        all_count_distribution[total] += 1
    return {
        "sample_count": len(rows),
        "dimensions": {
            dim: {
                "gold_empty_samples": dim_counts[dim]["gold_empty"],
                "gold_nonempty_samples": dim_counts[dim]["gold_nonempty"],
                "label_count_distribution": {
                    str(count): dim_count_distribution[dim][count]
                    for count in range(len(label_enums[dim]) + 1)
                },
                "label_support": {label: label_counts[dim][label] for label in label_enums[dim]},
            }
            for dim in a0.PROFILE_DIMS
        },
        "all_dimensions_label_count_distribution": {
            str(count): all_count_distribution[count]
            for count in range(sum(map(len, label_enums.values())) + 1)
        },
    }


def build_population_summary(
    rows: list[dict[str, Any]], label_enums: dict[str, list[str]]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["fit_outcome"]].append(row)
        groups[row["history_bin"]].append(row)
        groups["profile_all_empty" if row["profile_all_empty"] else "profile_any_nonempty"].append(row)

    pairwise: dict[str, list[dict[str, Any]]] = {}
    label_support: dict[str, dict[str, Any]] = {}
    for dim, labels in label_enums.items():
        counts: Counter[tuple[str, str]] = Counter()
        for row in rows:
            present = set(row["labels_by_dim"][dim])
            ordered = [label for label in labels if label in present]
            for index, label_a in enumerate(ordered):
                for label_b in ordered[index:]:
                    counts[(label_a, label_b)] += 1
        pairwise[dim] = [
            {"label_a": left, "label_b": right, "sample_count": counts[(left, right)]}
            for index, left in enumerate(labels)
            for right in labels[index:]
        ]
        label_support[dim] = {
            label: {
                "positive_sample_count": sum(label in row["labels_by_dim"][dim] for row in rows),
                "positive_rate": sum(label in row["labels_by_dim"][dim] for row in rows) / len(rows),
                "population_count": len(rows),
            }
            for label in labels
        }

    return {
        **_group_summary(rows, label_enums),
        "history_turn_count": {
            "minimum": min(row["history_turn_count"] for row in rows),
            "maximum": max(row["history_turn_count"] for row in rows),
            "median": _median([row["history_turn_count"] for row in rows]),
            "bins": {
                "short_1_4_turns": "1-4 messages in source history",
                "mid_5_8_turns": "5-8 messages in source history",
                "long_9_plus_turns": "9 or more messages in source history",
            },
        },
        "groups": {name: _group_summary(group_rows, label_enums) for name, group_rows in sorted(groups.items())},
        "label_support": label_support,
        "label_pair_cooccurrence": pairwise,
    }


def _stable_order(rows: list[dict[str, Any]], seed: int, group: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{group}:{row['sample_id']}".encode("utf-8")
        ).hexdigest(),
    )


def select_review_samples(
    rows: list[dict[str, Any]],
    *,
    label_enums: dict[str, list[str]],
    focus_labels: list[str],
    seed: int,
    positives_per_label: int,
    positives_per_history_stratum: int,
    negatives_per_label: int,
) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    turns_median = _median([row["history_turn_count"] for row in rows])
    memberships: dict[str, list[str]] = defaultdict(list)
    denominators: dict[str, dict[str, int]] = {}
    row_by_id = {row["sample_id"]: row for row in rows}
    valid_labels = {label for values in label_enums.values() for label in values}
    if set(focus_labels) - valid_labels:
        raise ValueError("audit.focus_labels contains a label outside the official profile enum")

    for label in focus_labels:
        dim = next(dim for dim, labels in label_enums.items() if label in labels)
        positives = [row for row in rows if label in row["labels_by_dim"][dim]]
        negatives = [row for row in rows if label not in row["labels_by_dim"][dim]]
        short = [row for row in positives if row["history_turn_count"] <= turns_median]
        long = [row for row in positives if row["history_turn_count"] > turns_median]
        group_stats: dict[str, int] = {}
        chosen_positive: set[str] = set()
        for group_name, candidates in (("history_le_median", short), ("history_gt_median", long)):
            key = f"positive:{label}:{group_name}"
            ordered = _stable_order(candidates, seed, key)
            take = min(positives_per_history_stratum, positives_per_label, len(ordered))
            group_stats[f"{group_name}_available"] = len(candidates)
            group_stats[f"{group_name}_selected"] = take
            for row in ordered[:take]:
                chosen_positive.add(row["sample_id"])
                memberships[row["sample_id"]].append(key)

        if len(chosen_positive) < positives_per_label:
            fill_key = f"positive:{label}:history_stratum_fill"
            remaining = [row for row in positives if row["sample_id"] not in chosen_positive]
            fill = _stable_order(remaining, seed, fill_key)[:positives_per_label - len(chosen_positive)]
            group_stats["history_stratum_fill_selected"] = len(fill)
            for row in fill:
                chosen_positive.add(row["sample_id"])
                memberships[row["sample_id"]].append(fill_key)

        neg_key = f"negative:{label}"
        neg_selected = _stable_order(negatives, seed, neg_key)[:negatives_per_label]
        group_stats.update({
            "positive_available": len(positives),
            "positive_selected_unique": len(chosen_positive),
            "negative_available": len(negatives),
            "negative_selected": len(neg_selected),
        })
        for row in neg_selected:
            memberships[row["sample_id"]].append(neg_key)
        denominators[label] = group_stats

    selected_ids = sorted(
        memberships,
        key=lambda sample_id: hashlib.sha256(
            f"{seed}:selected:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    for sample_id in selected_ids:
        row = row_by_id[sample_id]
        memberships[sample_id].append(f"history_bin:{row['history_bin']}")
    return selected_ids, dict(memberships), {
        "seed": seed,
        "selection_rule": "sha256_seeded_per_label_positive_history_median_strata_and_negative_controls_v1",
        "candidate_rows": len(rows),
        "history_turn_median": turns_median,
        "focus_label_groups": denominators,
        "unique_selected_sample_count": len(selected_ids),
        "selected_sample_ids_sha256": _json_sha256(selected_ids),
        "selection_is_population_representative": False,
        "note": "group denominators overlap; memberships must not be summed to infer unique sample count",
    }


def _parse_profile_spans(
    completion: str, label_enums: dict[str, list[str]]
) -> tuple[tuple[int, int], list[str | None], dict[str, tuple[int, int]], list[dict[str, Any]]]:
    root, end = a0._parse_json_node(completion, 0)
    if a0._skip_json_ws(completion, end) != len(completion) or root.get("kind") != "object":
        raise a0.AlignmentError("completion is not one complete JSON object")
    top_members = root.get("members", {})
    profile_member = top_members.get("user_profile")
    if not profile_member or profile_member["value"].get("kind") != "object":
        raise a0.AlignmentError("user_profile is not a JSON object")
    profile_node = profile_member["value"]
    profile_span = (profile_node["start"], profile_node["end"])
    members = profile_node.get("members", {})
    if set(members) != set(a0.PROFILE_DIMS):
        raise a0.AlignmentError("user_profile dimensions differ from the fixed P0 contract")

    categories: list[str | None] = [None] * len(completion)
    for pos in range(profile_span[0], profile_span[1]):
        categories[pos] = "profile_structure"
    dimension_spans: dict[str, tuple[int, int]] = {}
    label_occurrences: list[dict[str, Any]] = []
    for dim in a0.PROFILE_DIMS:
        member = members[dim]
        array = member["value"]
        if array.get("kind") != "array":
            raise a0.AlignmentError(f"user_profile.{dim} is not an array")
        dimension_spans[dim] = (member["key_start"], member["entry_end"])
        items = array.get("items", [])
        if not items:
            if completion[array["start"]:array["end"]] != "[]":
                raise a0.AlignmentError("empty-array span is not the canonical compact JSON value")
            for pos in range(array["start"], array["end"]):
                categories[pos] = "profile_empty_decision"
            continue
        seen: set[str] = set()
        for item in items:
            if item.get("kind") != "string" or item.get("value") not in label_enums[dim]:
                raise a0.AlignmentError(f"user_profile.{dim} contains a non-enum label")
            label = item["value"]
            if label in seen:
                raise a0.AlignmentError(f"user_profile.{dim} contains a duplicate Gold label")
            seen.add(label)
            start, finish = item["start"] + 1, item["end"] - 1
            if completion[start:finish] != label:
                raise a0.AlignmentError("Gold label JSON escaping prevents exact raw-value span mapping")
            for pos in range(start, finish):
                categories[pos] = "profile_label_value"
            label_occurrences.append({
                "dimension": dim,
                "label": label,
                "span": (start, finish),
            })
    return profile_span, categories, dimension_spans, label_occurrences


def assign_profile_fine_regions(
    *,
    completion: str,
    completion_start: int,
    offsets: list[tuple[int, int]],
    labels: list[int],
    top_regions: list[str],
    profile_regions: list[str | None],
    label_enums: dict[str, list[str]],
) -> dict[str, Any]:
    if not (len(offsets) == len(labels) == len(top_regions) == len(profile_regions)):
        raise a0.AlignmentError("fine profile mapping input lengths differ")
    profile_span, char_regions, dimension_spans, occurrences = _parse_profile_spans(completion, label_enums)
    category_set = set(FINE_REGIONS)
    by_dim = set(a0.PROFILE_DIMS) | {"profile_structure", "boundary_mixed"}
    token_regions: list[str | None] = [None] * len(offsets)
    token_labels: list[tuple[str, ...]] = [()] * len(offsets)
    token_dimensions: list[str | None] = [None] * len(offsets)
    region_counts = Counter({key: 0 for key in FINE_REGIONS})
    label_exact = Counter()
    label_boundary = Counter()

    for index, ((start, end), label_id, primary) in enumerate(zip(offsets, labels, top_regions)):
        if label_id == -100 or primary != "user_profile":
            continue
        if start == end:
            raise a0.AlignmentError("active user_profile Token has an empty Offset")
        relative_start = start - completion_start
        relative_end = end - completion_start
        if relative_start < profile_span[0] or relative_end > profile_span[1] or relative_end <= relative_start:
            raise a0.AlignmentError("user_profile Token Offset crosses its parsed JSON object boundary")
        span_regions = char_regions[relative_start:relative_end]
        distinct = set(span_regions)
        if None in distinct or not distinct:
            raise a0.AlignmentError("user_profile Token did not map to a fine-grained character region")
        fine = next(iter(distinct)) if len(distinct) == 1 else "profile_boundary_mixed"
        if fine not in category_set:
            raise a0.AlignmentError("fine-grained profile Token has an unknown category")
        touched = tuple(sorted({
            occurrence["label"]
            for occurrence in occurrences
            if max(relative_start, occurrence["span"][0]) < min(relative_end, occurrence["span"][1])
        }))
        if fine == "profile_label_value" and len(touched) != 1:
            raise a0.AlignmentError("label-value Token does not map to exactly one Gold label")
        token_regions[index] = fine
        token_labels[index] = touched
        dim = profile_regions[index]
        if dim not in by_dim:
            raise a0.AlignmentError("user_profile Token lacks a valid existing A0 dimension assignment")
        token_dimensions[index] = dim
        region_counts[fine] += 1
        if fine == "profile_label_value":
            label_exact[touched[0]] += 1
        elif fine == "profile_boundary_mixed":
            label_boundary.update(touched)

    return {
        "token_regions": token_regions,
        "token_label_hits": token_labels,
        "token_dimensions": token_dimensions,
        "region_counts": dict(region_counts),
        "label_exact_tokens": dict(label_exact),
        "label_boundary_tokens": dict(label_boundary),
        "label_occurrences": [
            {"dimension": item["dimension"], "label": item["label"]}
            for item in occurrences
        ],
    }


class FineAccumulator:
    def __init__(self, label_enums: dict[str, list[str]]) -> None:
        self.samples = 0
        self.a0_profile_tokens = 0
        self.category_totals = Counter({key: 0 for key in FINE_REGIONS})
        self.category_per_sample = {key: [] for key in FINE_REGIONS}
        self.category_coverage = Counter()
        self.dimension_totals = {
            dim: Counter({key: 0 for key in FINE_REGIONS})
            for dim in (*a0.PROFILE_DIMS, "profile_structure", "boundary_mixed")
        }
        self.dimension_state = {
            dim: {state: Counter({key: 0 for key in FINE_REGIONS}) for state in ("gold_empty", "gold_nonempty")}
            for dim in a0.PROFILE_DIMS
        }
        self.dimension_state_samples = {
            dim: Counter({"gold_empty": 0, "gold_nonempty": 0}) for dim in a0.PROFILE_DIMS
        }
        self.label_exact_tokens = Counter()
        self.label_exact_samples = Counter()
        self.label_boundary_tokens = Counter()
        self.label_support = {label: 0 for values in label_enums.values() for label in values}

    def add(self, mapped: dict[str, Any], labels_by_dim: dict[str, list[str]], a0_profile_count: int) -> None:
        counts = mapped["region_counts"]
        if sum(counts.values()) != a0_profile_count:
            raise a0.AlignmentError("fine profile buckets do not conserve the existing A0 user_profile Token count")
        self.samples += 1
        self.a0_profile_tokens += a0_profile_count
        for key in FINE_REGIONS:
            count = counts.get(key, 0)
            self.category_totals[key] += count
            self.category_per_sample[key].append(count)
            self.category_coverage[key] += count > 0
        dimension_sample_counts = {
            dim: Counter({key: 0 for key in FINE_REGIONS})
            for dim in (*a0.PROFILE_DIMS, "profile_structure", "boundary_mixed")
        }
        for dim, category in zip(mapped["token_dimensions"], mapped["token_regions"]):
            if dim is not None and category is not None:
                self.dimension_totals[dim][category] += 1
                dimension_sample_counts[dim][category] += 1
        for dim in a0.PROFILE_DIMS:
            state = "gold_nonempty" if labels_by_dim[dim] else "gold_empty"
            self.dimension_state_samples[dim][state] += 1
            self.dimension_state[dim][state].update(dimension_sample_counts[dim])
            for label in labels_by_dim[dim]:
                self.label_support[label] += 1
        for label, count in mapped["label_exact_tokens"].items():
            self.label_exact_tokens[label] += count
            self.label_exact_samples[label] += 1
        self.label_boundary_tokens.update(mapped["label_boundary_tokens"])

    def summary(self, label_enums: dict[str, list[str]], population: dict[str, Any]) -> dict[str, Any]:
        categories: dict[str, Any] = {}
        for key in FINE_REGIONS:
            distribution = a0._distribution(self.category_per_sample[key], self.a0_profile_tokens)
            categories[key] = {
                **distribution,
                "sample_count_with_tokens": self.category_coverage[key],
                "sample_coverage": self.category_coverage[key] / self.samples if self.samples else None,
            }
        label_rows = {}
        for dim, labels in label_enums.items():
            for label in labels:
                label_rows[label] = {
                    "dimension": dim,
                    "positive_sample_count": population["label_support"][dim][label]["positive_sample_count"],
                    "gold_occurrence_count": population["label_support"][dim][label]["positive_sample_count"],
                    "exact_label_value_token_count": self.label_exact_tokens[label],
                    "samples_with_exact_label_value_tokens": self.label_exact_samples[label],
                    "boundary_mixed_tokens_touching_label_value": self.label_boundary_tokens[label],
                }
        dimensions = {}
        for dim in a0.PROFILE_DIMS:
            dimensions[dim] = {
                "gold_empty_sample_count": self.dimension_state_samples[dim]["gold_empty"],
                "gold_nonempty_sample_count": self.dimension_state_samples[dim]["gold_nonempty"],
                "gold_empty": {
                    "fine_region_token_counts": dict(self.dimension_state[dim]["gold_empty"]),
                    "empty_decision_attribution": (
                        "attributed" if self.dimension_state[dim]["gold_empty"]["profile_empty_decision"] else "not_separately_attributable"
                    ),
                },
                "gold_nonempty": {
                    "fine_region_token_counts": dict(self.dimension_state[dim]["gold_nonempty"]),
                },
                "tokens_assigned_to_dimension": dict(self.dimension_totals[dim]),
            }
        return {
            "sample_count": self.samples,
            "a0_user_profile_token_count": self.a0_profile_tokens,
            "fine_category_conservation": {
                "fine_category_token_sum": sum(self.category_totals.values()),
                "a0_user_profile_token_count": self.a0_profile_tokens,
                "passed": sum(self.category_totals.values()) == self.a0_profile_tokens,
                "top_level_cross_field_tokens": "remain in A0 boundary_mixed and are outside the user_profile denominator",
            },
            "categories": categories,
            "dimensions": dimensions,
            "a0_dimension_region_token_counts": {
                dim: dict(self.dimension_totals[dim])
                for dim in (*a0.PROFILE_DIMS, "profile_structure", "boundary_mixed")
            },
            "labels": label_rows,
        }


def _write_review_material(
    *,
    run_dir: Path,
    selected_ids: list[str],
    memberships: dict[str, list[str]],
    records_by_id: dict[str, dict[str, Any]],
    source_handle: Any,
    tokenizer: Any,
    prompt_text: str,
    max_prompt_tokens: int,
    label_enums: dict[str, list[str]],
) -> None:
    rows: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        record = records_by_id[sample_id]
        source = _read_at(source_handle, record["normalized_offset"])
        history = source["history"]
        target_profile = source["target"]["user_profile"]
        t1 = prompting.select_recent_complete_suffix(history, prompt_text, max_prompt_tokens, tokenizer)
        focus_labels = sorted({
            group.split(":", 2)[1]
            for group in memberships[sample_id]
            if group.startswith("positive:") or group.startswith("negative:")
        })
        review = {}
        for label in focus_labels:
            dim = next(dim for dim, values in label_enums.items() if label in values)
            review[label] = {
                "gold_positive": label in target_profile[dim],
                "evidence_classification": "unreviewed",
                "support_message_indices": None,
                "t1_retention": "not_identified",
                "ambiguity_or_conflict": "",
                "reviewer_rationale": "",
                "review_status": "unreviewed",
            }
        rows.append({
            "sample_id": sample_id,
            "fit_outcome": record["fit_outcome"],
            "history_turn_count": len(history),
            "history_bin": record["history_bin"],
            "selection_groups": memberships[sample_id],
            "gold_profile": {dim: list(target_profile[dim]) for dim in a0.PROFILE_DIMS},
            "original_legal_history": history,
            "actual_t1_history": t1["selected_history"],
            "t1": {
                "selected_start_index": t1["selected_start_index"],
                "retained_message_count": len(t1["selected_history"]),
                "removed_message_count": len(history) - len(t1["selected_history"]),
                "prompt_tokens": t1["prompt_tokens"],
                "prompt_floor_exceeds_cap": t1["floor_exceeds_cap"],
            },
            "review": {
                "reviewer_id": "",
                "status": "unreviewed",
                "labels": review,
            },
        })
    _write_jsonl(run_dir / "private_review_template.jsonl", rows)
    _write_json(run_dir / "private_review_instructions.json", {
        "privacy": "contains raw dialogue and Gold; keep local under ignored outputs/",
        "reviewer_instructions": "Read original_legal_history and actual_t1_history independently. Leave evidence_classification=unreviewed until a person reviews. Do not infer evidence status from model outputs.",
        "evidence_classification_allowed": [*REVIEW_CATEGORIES, "unreviewed", "not_fully_reviewed"],
        "t1_retention_allowed": list(T1_RETENTION),
        "support_message_indices": "Use zero-based message indices into original_legal_history; keep null when not reviewed or not identified.",
        "required_decision": "Record a concise rationale and any conflict/ambiguity for every reviewed label. This template does not alter Gold.",
        "record_count": len(rows),
    })


def _fine_ce_for_logits(
    torch: Any,
    logits: Any,
    labels: Any,
    mapping: dict[str, Any],
    chunk_tokens: int,
) -> dict[str, Any]:
    import torch.nn.functional as F

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[0] != labels.shape[0] or labels.shape[0] != 1:
        raise a0.AlignmentError("fine CE expects one unpadded causal sequence")
    targets = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    if targets.numel() and int(targets[0].item()) == 0:
        raise a0.AlignmentError("first sequence token cannot be a shifted causal target")
    sums = Counter({key: 0.0 for key in FINE_REGIONS})
    counts = Counter({key: 0 for key in FINE_REGIONS})
    label_ce = Counter()
    label_counts = Counter()
    for start in range(0, int(targets.numel()), chunk_tokens):
        target_positions = targets[start:start + chunk_tokens]
        shifted_positions = target_positions - 1
        chunk_logits = logits[0].index_select(0, shifted_positions).float()
        chunk_targets = labels[0].index_select(0, target_positions)
        losses = F.cross_entropy(chunk_logits, chunk_targets, reduction="none")
        for offset, target_position in enumerate(target_positions.tolist()):
            fine = mapping["token_regions"][target_position]
            if fine is None:
                continue
            loss_value = float(losses[offset].item())
            sums[fine] += loss_value
            counts[fine] += 1
            if fine == "profile_label_value":
                touched = mapping["token_label_hits"][target_position]
                if len(touched) != 1:
                    raise a0.AlignmentError("fine label-value CE token has ambiguous Gold-label attribution")
                label_ce[touched[0]] += loss_value
                label_counts[touched[0]] += 1
    return {
        "ce_sums": dict(sums),
        "token_counts": dict(counts),
        "label_exact_ce_sums": dict(label_ce),
        "label_exact_token_counts": dict(label_counts),
        "total_profile_ce_sum": sum(sums.values()),
        "total_profile_token_count": sum(counts.values()),
    }


def _run_cpu(
    *,
    root: Path,
    config: dict[str, Any],
    base_config: dict[str, Any],
    paths: dict[str, Path],
    run_id: str,
    run_dir: Path,
    identity: dict[str, Any],
    manifests: dict[str, Any],
    tokenizer: Any,
    prompt_text: str,
    schema: dict[str, Any],
    tokenizer_facts: dict[str, Any],
) -> dict[str, Any]:
    label_enums = _profile_schema(schema)
    population, _budget, _unused = a0._candidate_metadata(paths, identity, base_config)
    with paths["normalized_train"].open("rb") as source_handle:
        enriched = _collect_population(population, source_handle=source_handle, label_enums=label_enums)
    population_summary = build_population_summary(enriched, label_enums)
    selected_ids, memberships, audit_selection = select_review_samples(
        enriched,
        label_enums=label_enums,
        focus_labels=config["audit"]["focus_labels"],
        seed=config["audit"]["seed"],
        positives_per_label=config["audit"]["positives_per_label"],
        positives_per_history_stratum=config["audit"]["positives_per_history_stratum"],
        negatives_per_label=config["audit"]["negatives_per_label"],
    )
    audit_selection["group_memberships"] = {
        sample_id: memberships[sample_id] for sample_id in selected_ids
    }
    _write_json(run_dir / "private_review_selection.json", audit_selection)
    records_by_id = {record["sample_id"]: record for record in enriched}
    with paths["normalized_train"].open("rb") as source_handle:
        _write_review_material(
            run_dir=run_dir,
            selected_ids=selected_ids,
            memberships=memberships,
            records_by_id=records_by_id,
            source_handle=source_handle,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_prompt_tokens=identity["max_prompt_tokens"],
            label_enums=label_enums,
        )

    accumulator = FineAccumulator(label_enums)
    processed = 0
    with paths["normalized_train"].open("rb") as source_handle, paths["sft_train"].open("rb") as sft_handle:
        for record in enriched:
            source = _read_at(source_handle, record["normalized_offset"])
            sft_row = _read_at(sft_handle, record["sample_id_offset"])
            result = a0._build_sample_mapping(
                source=source,
                sft_row=sft_row,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                schema=schema,
                max_prompt_tokens=identity["max_prompt_tokens"],
            )
            completion, _rendered, offsets, regions, profile_regions, counts, _profile_counts, metadata = result
            fine = assign_profile_fine_regions(
                completion=completion,
                completion_start=metadata["completion_start"],
                offsets=offsets,
                labels=list(map(int, sft_row["labels"])),
                top_regions=regions,
                profile_regions=profile_regions,
                label_enums=label_enums,
            )
            accumulator.add(fine, record["labels_by_dim"], counts["user_profile"])
            processed += 1
            if processed % 1000 == 0:
                print(f"E03-A0.1 CPU aligned {processed}/{len(enriched)} Train rows", flush=True)
    if processed != identity["sft_train_rows"]:
        raise a0.AlignmentError("fine-grained mapping did not cover every verified SFT Train row")

    baseline_token_dir = paths["output_root"] / config["baseline"]["token_run_id"]
    baseline_token_summary = json.loads((baseline_token_dir / "token_summary.json").read_text(encoding="utf-8"))
    baseline_profile_count = baseline_token_summary["groups"]["all"]["top_level_regions"]["user_profile"]["sum"]
    if accumulator.a0_profile_tokens != baseline_profile_count:
        raise a0.AlignmentError("new A0.1 user_profile denominator differs from the frozen A0 full-run total")
    baseline_token_manifest = json.loads((baseline_token_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "status": "completed",
        "population": "verified D0.4.3 SFT Train only; no Dev or official Val rows",
        "data_rows": len(enriched),
        "population_statistics": population_summary,
        "audit_sample": {
            "selection": {key: value for key, value in audit_selection.items() if key != "group_memberships"},
            "unique_sample_count": len(selected_ids),
            "review_status": "unreviewed",
            "private_material_paths": [
                "private_review_template.jsonl",
                "private_review_selection.json",
                "private_review_instructions.json",
            ],
            "human_evidence_conclusions_recorded": 0,
        },
        "fine_grained_profile_tokens": accumulator.summary(label_enums, population_summary),
        "a0_comparison": {
            "baseline_token_run_id": config["baseline"]["token_run_id"],
            "baseline_run_manifest_sha256": a0.sha256_file(baseline_token_dir / "manifest.json"),
            "baseline_sample_count": baseline_token_summary["verified_rows"],
            "baseline_a0_user_profile_token_count": baseline_profile_count,
            "new_a0_user_profile_token_count": accumulator.a0_profile_tokens,
            "sample_count_equal": baseline_token_summary["verified_rows"] == processed,
            "profile_token_count_equal": baseline_profile_count == accumulator.a0_profile_tokens,
            "baseline_model_identity_status": baseline_token_manifest.get("model_identity", {}).get("status"),
        },
        "mapping": {
            "method": "A0 whole-sequence verified IDs and fast-tokenizer offsets; parsed exact JSON AST spans for label values and empty arrays",
            "saved_input_ids_verified_by_a0_per_row": True,
            "all_33_labels_from_official_schema": True,
            "secondary_profile_token_conservation_passed": sum(accumulator.category_totals.values()) == accumulator.a0_profile_tokens,
            "top_level_cross_field_boundary_tokens": "retained in existing A0 boundary_mixed; excluded from user_profile denominator",
        },
        "tokenizer": tokenizer_facts["tokenizer"],
    }
    _write_json(run_dir / "profile_summary.json", summary)
    return summary


def _run_ce(
    *,
    root: Path,
    config: dict[str, Any],
    base_config: dict[str, Any],
    paths: dict[str, Path],
    run_dir: Path,
    identity: dict[str, Any],
    tokenizer: Any,
    prompt_text: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    import torch

    label_enums = _profile_schema(schema)
    cpu_run_id = config.get("_cpu_run_id")
    if not cpu_run_id:
        raise ValueError("CE mode requires --cpu-run-id for a completed A0.1 CPU alignment Run")
    cpu_dir = paths["e03_a0_1_output_root"] / cpu_run_id
    cpu_manifest_path = cpu_dir / "manifest.json"
    cpu_summary_path = cpu_dir / "profile_summary.json"
    if not cpu_manifest_path.is_file() or not cpu_summary_path.is_file():
        raise FileNotFoundError(f"completed A0.1 CPU Run not found: {cpu_run_id}")
    cpu_manifest = json.loads(cpu_manifest_path.read_text(encoding="utf-8"))
    cpu_summary = json.loads(cpu_summary_path.read_text(encoding="utf-8"))
    if cpu_manifest.get("status") != "completed" or cpu_summary.get("status") != "completed":
        raise ValueError("CE mode requires a completed A0.1 CPU Run")
    if cpu_manifest.get("source_identity") != identity:
        raise ValueError("CPU Run source identity differs from current validated inputs")

    candidate_rows, _budget, _unused = a0._candidate_metadata(paths, identity, base_config)
    selected, selection = a0.deterministic_select(
        candidate_rows,
        cap=config["ce"]["sample_cap"],
        seed=config["ce"]["seed"],
        stratified=True,
    )
    baseline_ce_path = paths["output_root"] / config["baseline"]["ce_run_id"] / "ce_summary.json"
    baseline_ce = json.loads(baseline_ce_path.read_text(encoding="utf-8"))
    if baseline_ce.get("status") != "completed":
        raise ValueError("configured A0 baseline CE Run is not completed")
    if selection["selected_sample_ids_sha256"] != baseline_ce.get("sample_ids_sha256"):
        raise a0.AlignmentError("A0.1 CE selection differs from the fixed A0 128-sample selection")

    if not torch.cuda.is_available():
        return {
            "status": "not_run",
            "reason": "torch.cuda.is_available() is false in this runtime",
            "gpu_model_loaded": False,
            "sample_count": len(selected),
            "sample_ids_sha256": selection["selected_sample_ids_sha256"],
            "baseline_a0_ce_run_id": config["baseline"]["ce_run_id"],
            "selection_verified_against_a0": True,
            "model_identity": "not_rehashed because no GPU forward could run",
        }

    selected_ids = {row["sample_id"] for row in selected}
    selected_mappings: list[dict[str, Any]] = []
    with paths["normalized_train"].open("rb") as source_handle, paths["sft_train"].open("rb") as sft_handle:
        for row in selected:
            source = _read_at(source_handle, row["normalized_offset"])
            sft_row = _read_at(sft_handle, row["sample_id_offset"])
            built = a0._build_sample_mapping(
                source=source,
                sft_row=sft_row,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                schema=schema,
                max_prompt_tokens=identity["max_prompt_tokens"],
            )
            completion, _rendered, offsets, regions, profile_regions, counts, profile_counts, metadata = built
            fine = assign_profile_fine_regions(
                completion=completion,
                completion_start=metadata["completion_start"],
                offsets=offsets,
                labels=list(map(int, sft_row["labels"])),
                top_regions=regions,
                profile_regions=profile_regions,
                label_enums=label_enums,
            )
            if sum(fine["region_counts"].values()) != counts["user_profile"]:
                raise a0.AlignmentError("selected CE sample failed fine profile token conservation")
            selected_mappings.append({
                "row": row,
                "source": source,
                "sft_row": sft_row,
                "regions": regions,
                "profile_regions": profile_regions,
                "token_counts": counts,
                "profile_token_counts": profile_counts,
                "fine": fine,
            })
    if len(selected_mappings) != len(selected) or {item["row"]["sample_id"] for item in selected_mappings} != selected_ids:
        raise a0.AlignmentError("fine CE mapping did not cover the fixed selected sample set")

    model_identity = a0._model_identity(
        root,
        paths,
        base_config,
        json.loads((paths["training_run_root"] / "manifest.json").read_text(encoding="utf-8")),
    )
    if model_identity.get("status") != "verified":
        return {
            "status": "gpu_blocked",
            "reason": "local Full B Base or Adapter identity differs from the historical Run",
            "gpu_model_loaded": False,
            "sample_count": len(selected),
            "sample_ids_sha256": selection["selected_sample_ids_sha256"],
            "model_identity": model_identity,
        }
    try:
        model, torch, runtime_facts = a0._load_inference_model(
            root,
            paths,
            a0._load_yaml(paths["training_config"]),
            paths["training_run_root"] / "adapter",
        )
    except (RuntimeError, a0.QuantizationPlacementError) as exc:
        result = {
            "status": "gpu_blocked",
            "reason": str(exc),
            "gpu_model_loaded": isinstance(exc, a0.QuantizationPlacementError),
            "sample_count": len(selected),
            "sample_ids_sha256": selection["selected_sample_ids_sha256"],
            "model_identity": model_identity,
        }
        if isinstance(exc, a0.QuantizationPlacementError):
            result["runtime_facts"] = exc.runtime_facts
        return result

    chunk_tokens = config["ce"]["logit_chunk_tokens"]
    native_tolerance = float(base_config["gpu"]["native_loss_abs_tolerance"])
    repeat_tolerance = float(base_config["gpu"]["repeat_abs_tolerance"])
    category_ce = Counter({key: 0.0 for key in FINE_REGIONS})
    category_tokens = Counter({key: 0 for key in FINE_REGIONS})
    per_label_ce = Counter()
    per_label_tokens = Counter()
    per_label_sample_support = Counter()
    total_ce_sum = 0.0
    total_valid_tokens = 0
    profile_ce_sum = 0.0
    profile_tokens = 0
    max_native_error = 0.0
    repeat_error: float | None = None
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for index, item in enumerate(selected_mappings):
                sft_row = item["sft_row"]
                input_ids = torch.tensor([sft_row["input_ids"]], dtype=torch.long, device="cuda:0")
                labels = torch.tensor([sft_row["labels"]], dtype=torch.long, device="cuda:0")
                output = model(input_ids=input_ids, labels=labels, use_cache=False)
                native_loss = float(output.loss.float().item())
                ce_sums, old_profile_ce, sample_total_ce, valid_count = a0._ce_for_logits(
                    torch, output.logits, labels, item["regions"], item["profile_regions"], chunk_tokens
                )
                fine_ce = _fine_ce_for_logits(torch, output.logits, labels, item["fine"], chunk_tokens)
                if fine_ce["total_profile_token_count"] != item["token_counts"]["user_profile"]:
                    raise a0.AlignmentError("fine CE token denominator differs from A0 user_profile")
                if not math.isclose(
                    fine_ce["total_profile_ce_sum"],
                    old_profile_ce["profile_structure"]
                    + old_profile_ce["personality_traits"]
                    + old_profile_ce["interests"]
                    + old_profile_ce["style"]
                    + old_profile_ce["boundary_mixed"],
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                ):
                    raise a0.AlignmentError("fine CE categories do not conserve A0 user_profile CE")
                if valid_count != sum(item["token_counts"].values()):
                    raise a0.AlignmentError("A0 manual CE valid-token denominator differs from mapped tokens")
                manual_mean = sample_total_ce / valid_count
                error = abs(manual_mean - native_loss)
                max_native_error = max(max_native_error, error)
                if index == 0:
                    del output
                    repeat = model(input_ids=input_ids, labels=labels, use_cache=False)
                    repeat_error = abs(float(repeat.loss.float().item()) - native_loss)
                    del repeat
                else:
                    del output
                total_ce_sum += sample_total_ce
                total_valid_tokens += valid_count
                profile_ce_sum += fine_ce["total_profile_ce_sum"]
                profile_tokens += fine_ce["total_profile_token_count"]
                category_ce.update(fine_ce["ce_sums"])
                category_tokens.update(fine_ce["token_counts"])
                for label, value in fine_ce["label_exact_ce_sums"].items():
                    per_label_ce[label] += value
                per_label_tokens.update(fine_ce["label_exact_token_counts"])
                for label in fine_ce["label_exact_token_counts"]:
                    per_label_sample_support[label] += 1
                del input_ids, labels
                if index % 16 == 15:
                    print(f"E03-A0.1 CE forwarded {index + 1}/{len(selected_mappings)} samples", flush=True)
    except torch.cuda.OutOfMemoryError:
        del model
        torch.cuda.empty_cache()
        return {
            "status": "gpu_blocked",
            "reason": "CUDA OOM during bounded per-sample forward",
            "gpu_model_loaded": True,
            "completed_before_oom": index,
            "sample_count": len(selected),
            "sample_ids_sha256": selection["selected_sample_ids_sha256"],
            "model_identity": model_identity,
            "runtime_facts": runtime_facts,
        }
    elapsed = time.monotonic() - started
    del model
    baseline_profile_ce = baseline_ce["groups"]["all"]["regions"]["user_profile"]["ce_sum"]
    baseline_tokens = baseline_ce["groups"]["all"]["regions"]["user_profile"]["token_count"]
    conservation_passed = (
        sum(category_tokens.values()) == profile_tokens
        and math.isclose(sum(category_ce.values()), profile_ce_sum, rel_tol=1e-10, abs_tol=1e-6)
        and profile_tokens == baseline_tokens
        and math.isclose(profile_ce_sum, baseline_profile_ce, rel_tol=1e-6, abs_tol=1e-4)
    )
    labels_summary = {}
    minimum_support = config["ce"]["minimum_label_sample_support"]
    for dim, labels in label_enums.items():
        for label in labels:
            token_count = per_label_tokens[label]
            sample_support = per_label_sample_support[label]
            enough = token_count > 0 and sample_support >= minimum_support
            labels_summary[label] = {
                "dimension": dim,
                "status": "measured" if enough else "insufficient_support",
                "sample_support": sample_support,
                "minimum_sample_support": minimum_support,
                "exact_label_value_token_count": token_count,
                "exact_label_value_ce_sum": per_label_ce[label] if enough else None,
                "pooled_mean_exact_label_value_token_ce": per_label_ce[label] / token_count if enough else None,
            }
    native_passed = max_native_error <= native_tolerance
    repeat_passed = repeat_error is not None and repeat_error <= repeat_tolerance
    return {
        "status": "completed" if native_passed and repeat_passed and conservation_passed else "consistency_check_failed",
        "measurement": "frozen Full B teacher-forced CE; fixed A0 128-sample Train selection",
        "sample_count": len(selected),
        "sample_ids_sha256": selection["selected_sample_ids_sha256"],
        "selection": selection,
        "model_identity": model_identity,
        "runtime_facts": runtime_facts,
        "dtype": "NF4 weights with BF16 compute; causal shift and chunked FP32 CE",
        "inference_mode": True,
        "model_eval": True,
        "native_loss_check": {
            "max_absolute_difference": max_native_error,
            "tolerance": native_tolerance,
            "passed": native_passed,
            "supervised_token_count": total_valid_tokens,
        },
        "repeat_forward_check": {
            "absolute_loss_difference": repeat_error,
            "tolerance": repeat_tolerance,
            "passed": repeat_passed,
        },
        "profile_ce_conservation": {
            "fine_profile_token_count": profile_tokens,
            "a0_user_profile_token_count": baseline_tokens,
            "fine_profile_ce_sum": profile_ce_sum,
            "a0_user_profile_ce_sum": baseline_profile_ce,
            "passed": conservation_passed,
        },
        "categories": {
            key: {
                "token_count": category_tokens[key],
                "token_share_within_user_profile": category_tokens[key] / profile_tokens if profile_tokens else None,
                "ce_sum": category_ce[key],
                "pooled_mean_token_ce": category_ce[key] / category_tokens[key] if category_tokens[key] else None,
                "ce_contribution_within_user_profile": category_ce[key] / profile_ce_sum if profile_ce_sum else None,
            }
            for key in FINE_REGIONS
        },
        "labels": labels_summary,
        "total_supervised_ce_sum": total_ce_sum,
        "total_supervised_token_count": total_valid_tokens,
        "elapsed_seconds": elapsed,
    }


def run_diagnosis(*, config_path: Path, run_id: str, mode: str, cpu_run_id: str | None = None) -> Path:
    root = PROJECT_ROOT
    if mode not in {"cpu", "ce"}:
        raise ValueError("mode must be cpu or ce")
    config, base_config, paths = _load_task_config(root, config_path)
    if mode == "ce" and not cpu_run_id:
        raise ValueError("CE mode requires --cpu-run-id")
    run_dir = a0.create_run_dir(paths["e03_a0_1_output_root"], run_id)
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "version": VERSION,
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": a0._git_facts(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": {
            "path": str(_resolve(root, config_path)),
            "sha256": a0.sha256_file(_resolve(root, config_path)),
        },
        "population": "D0.4.3 SFT Train only; no Dev or official Val rows",
    }
    _write_json(run_dir / "manifest.json", manifest)
    try:
        identity, source_manifests = a0.verify_source_identity(root, base_config, paths)
        manifest["source_identity"] = identity
        manifest["source_hashes_before"] = source_manifests["hashes"]
        tokenizer, prompt_text, schema, tokenizer_facts = a0._load_tokenizer_context(root, paths)
        if tokenizer_facts["tokenizer"].get("signature_sha256") != identity["tokenizer_signature_sha256"]:
            raise a0.AlignmentError("runtime Tokenizer signature differs from accepted D0.4.3 identity")
        if tokenizer_facts["tokenizer"].get("template_sha256") != identity["chat_template_sha256"]:
            raise a0.AlignmentError("runtime Chat Template differs from accepted D0.4.3 identity")
        if not getattr(tokenizer, "is_fast", False):
            raise a0.AlignmentError("Tokenizer lacks verified fast Offset support")
        manifest["runtime_tokenizer"] = {
            "signature_sha256": tokenizer_facts["tokenizer"].get("signature_sha256"),
            "template_sha256": tokenizer_facts["tokenizer"].get("template_sha256"),
            "source": tokenizer_facts["tokenizer"].get("source"),
            "is_fast": True,
        }
        manifest["environment"]["packages"] = a0._package_versions()
        _write_json(run_dir / "manifest.json", manifest)

        if mode == "cpu":
            result = _run_cpu(
                root=root,
                config=config,
                base_config=base_config,
                paths=paths,
                run_id=run_id,
                run_dir=run_dir,
                identity=identity,
                manifests=source_manifests,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                schema=schema,
                tokenizer_facts=tokenizer_facts,
            )
        else:
            config = {**config, "_cpu_run_id": cpu_run_id}
            result = _run_ce(
                root=root,
                config=config,
                base_config=base_config,
                paths=paths,
                run_dir=run_dir,
                identity=identity,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                schema=schema,
            )
            _write_json(run_dir / "ce_summary.json", result)

        source_after = a0._source_hashes_after(paths)
        source_unchanged = all(
            source_after.get(name) == digest
            for name, digest in source_manifests["hashes"].items()
            if name in source_after
        )
        manifest["source_hashes_after"] = source_after
        manifest["source_hashes_unchanged"] = source_unchanged
        if not source_unchanged:
            raise RuntimeError("protected source file hash changed during E03-A0.1 run")
        result_status = result["status"]
        manifest.update(
            status="completed" if result_status in {"completed", "not_run"} else result_status,
            result_status=result_status,
            elapsed_seconds=time.monotonic() - started,
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(run_dir / "manifest.json", manifest)
        return run_dir
    except BaseException as exc:
        manifest.update(
            status="alignment_blocked" if isinstance(exc, a0.AlignmentError) else "blocked",
            error_type=type(exc).__name__,
            error_message=str(exc),
            elapsed_seconds=time.monotonic() - started,
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(run_dir / "manifest.json", manifest)
        raise
