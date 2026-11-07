"""Deterministic candidate mining and exposure scheduling for E03-A8 Replay."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.shared.e03_a8_protocol import PROFILE_DIMS


def canonical_json_sha256(value: Any) -> str:
    import json

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_replay_learning_rate(optimizer: Any, scheduler: Any, learning_rate: float) -> dict[str, Any]:
    """Reset all AdamW and constant-scheduler LR state after recovery."""
    if learning_rate <= 0:
        raise ValueError("Replay learning rate must be positive")
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
        group["initial_lr"] = float(learning_rate)
    optimizer.defaults["lr"] = float(learning_rate)
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [float(learning_rate)] * len(optimizer.param_groups)
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [float(learning_rate)] * len(optimizer.param_groups)
    actual = [float(group["lr"]) for group in optimizer.param_groups]
    if actual != [float(learning_rate)] * len(optimizer.param_groups):
        raise RuntimeError(f"Replay LR reset failed: {actual}")
    return {"learning_rate": float(learning_rate), "optimizer_group_lrs": actual, "scheduler_base_lrs": list(scheduler.base_lrs)}


def assert_safe_replay_checkpoint_path(
    output_path: Path,
    *,
    parent_checkpoint: Path,
    local_baseline: Path,
    archive_root: Path,
) -> None:
    """Reject writes into any immutable parent, local baseline, or archive."""
    output = output_path.resolve()
    for protected in (parent_checkpoint.resolve(), local_baseline.resolve(), archive_root.resolve()):
        if output == protected or output.is_relative_to(protected):
            raise ValueError(f"Replay training output cannot write to a protected model path: {output}")


def tie_hash(seed: int, source: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{source}|{sample_id}".encode("utf-8")).hexdigest()


def profile_label_keys(profile: Mapping[str, Sequence[str]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for dimension in PROFILE_DIMS:
        labels = profile.get(dimension)
        if not isinstance(labels, (list, tuple)) or any(not isinstance(label, str) for label in labels):
            raise ValueError(f"profile dimension {dimension} is malformed")
        if len(labels) != len(set(labels)):
            raise ValueError(f"profile dimension {dimension} contains duplicate labels")
        keys.update((dimension, label) for label in labels)
    return keys


def _profile(row: Mapping[str, Any]) -> Mapping[str, Sequence[str]]:
    target = row.get("target")
    if isinstance(target, Mapping):
        profile = target.get("user_profile")
    else:
        profile = row.get("gold_profile")
    if not isinstance(profile, Mapping):
        raise ValueError(f"{row.get('sample_id')}: Gold profile is missing")
    return profile


def _strata(row: Mapping[str, Any], labels: set[tuple[str, str]]) -> tuple[int, int, int]:
    label_count = len(labels)
    label_bin = 1 if label_count == 1 else 2 if label_count <= 3 else 3
    prompt = int(row.get("prompt_length", 0))
    history = int(row.get("history_message_count", 0))
    return label_bin, min(prompt // 256, 5), min(history // 4, 5)


def _stratified_fill(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    source: str,
    selected_ids: set[str],
    conversation_counts: Counter[str],
    conversation_cap: int,
) -> tuple[list[Mapping[str, Any]], int]:
    strata: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in selected_ids:
            continue
        labels = profile_label_keys(_profile(row))
        strata[_strata(row, labels)].append(row)
    for key, values in strata.items():
        values.sort(key=lambda row: tie_hash(seed, source, str(row["sample_id"])))
    keys = sorted(strata)
    positions = {key: 0 for key in keys}
    chosen: list[Mapping[str, Any]] = []
    relaxed_to = conversation_cap
    while len(chosen) < count:
        made_progress = False
        for key in keys:
            values = strata[key]
            while positions[key] < len(values):
                row = values[positions[key]]
                positions[key] += 1
                sample_id = str(row["sample_id"])
                conversation_id = str(row["conversation_id"])
                if sample_id in selected_ids:
                    continue
                if conversation_counts[conversation_id] >= relaxed_to:
                    continue
                selected_ids.add(sample_id)
                conversation_counts[conversation_id] += 1
                chosen.append(row)
                made_progress = True
                break
            if len(chosen) == count:
                break
        if len(chosen) == count:
            break
        if made_progress:
            continue
        if relaxed_to < 8:
            relaxed_to = 8
            continue
        if relaxed_to < 10**9:
            relaxed_to = 10**9
            continue
        break
    return chosen, relaxed_to


def select_candidates(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    total_count: int = 3000,
    seed: int = 42,
    source_counts: Mapping[str, int] | None = None,
    per_conversation_cap: int = 5,
    rare_label_cap: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a fixed Train/Dev sample set with bounded label and history strata."""
    if set(source_rows) != {"train", "dev"} or total_count <= 0:
        raise ValueError("Replay candidates require Train and Dev sources and a positive target size")
    requested = dict(source_counts or {"train": round(total_count * 0.9), "dev": total_count - round(total_count * 0.9)})
    if set(requested) != {"train", "dev"} or sum(requested.values()) != total_count:
        raise ValueError("candidate source quotas must sum to total_count")
    all_ids: set[str] = set()
    all_label_frequency: Counter[tuple[str, str]] = Counter()
    prepared: dict[str, list[Mapping[str, Any]]] = {}
    for source, rows in source_rows.items():
        prepared[source] = []
        for row in rows:
            sample_id = row.get("sample_id")
            conversation_id = row.get("conversation_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in all_ids:
                raise ValueError(f"candidate source has a missing or duplicate sample_id: {sample_id!r}")
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError(f"{sample_id}: conversation_id is missing")
            all_ids.add(sample_id)
            keys = profile_label_keys(_profile(row))
            all_label_frequency.update(keys)
            prepared[source].append(row)
    if total_count > len(all_ids):
        raise ValueError(f"candidate target {total_count} exceeds available rows {len(all_ids)}")

    selected_by_source: dict[str, list[Mapping[str, Any]]] = {}
    source_audit: dict[str, Any] = {}
    targets = {source: min(requested[source], len(prepared[source])) for source in ("train", "dev")}
    unallocated = total_count - sum(targets.values())
    # If one source is short, deterministically reassign its unused quota to the
    # other source before selection. The fallback is recorded in the audit.
    for source in ("train", "dev"):
        if unallocated <= 0:
            break
        extra = min(unallocated, len(prepared[source]) - targets[source])
        targets[source] += extra
        unallocated -= extra
    if unallocated:
        raise ValueError(f"candidate target {total_count} exceeds available rows across Train/Dev by {unallocated}")
    for source in ("train", "dev"):
        rows = prepared[source]
        target = targets[source]
        if target == 0:
            selected_by_source[source] = []
            source_audit[source] = {"available": len(rows), "selected": 0, "conversation_cap_relaxation": None}
            continue
        nonempty = [row for row in rows if profile_label_keys(_profile(row))]
        empty = [row for row in rows if not profile_label_keys(_profile(row))]
        wanted_nonempty = min(len(nonempty), round(target * 0.75))
        wanted_empty = min(len(empty), target - wanted_nonempty)
        wanted_nonempty += min(len(nonempty) - wanted_nonempty, target - wanted_nonempty - wanted_empty)
        selected_ids: set[str] = set()
        conversation_counts: Counter[str] = Counter()
        selected: list[Mapping[str, Any]] = []
        relaxed = per_conversation_cap

        # Cover up to ten distinct examples per label, with rare labels first.
        local_frequency = Counter(key for row in nonempty for key in profile_label_keys(_profile(row)))
        needs = {key: min(rare_label_cap, frequency) for key, frequency in local_frequency.items()}
        covered: Counter[tuple[str, str]] = Counter()
        labels_by_row = {str(row["sample_id"]): profile_label_keys(_profile(row)) for row in nonempty}
        ordered_labels = sorted(local_frequency, key=lambda key: (local_frequency[key], key[0], key[1]))
        for label_key in ordered_labels:
            while covered[label_key] < needs[label_key] and sum(
                1 for row in selected if profile_label_keys(_profile(row))
            ) < wanted_nonempty:
                candidates = [
                    row for row in nonempty
                    if str(row["sample_id"]) not in selected_ids
                    and label_key in labels_by_row[str(row["sample_id"])]
                    and conversation_counts[str(row["conversation_id"])] < relaxed
                ]
                if not candidates:
                    if relaxed < 8:
                        relaxed = 8
                        continue
                    if relaxed < 10**9:
                        relaxed = 10**9
                        continue
                    break

                def coverage_rank(row: Mapping[str, Any]) -> tuple[float, int, str]:
                    row_keys = labels_by_row[str(row["sample_id"])]
                    gain = sum(
                        min(1, max(0, needs[key] - covered[key])) / math.sqrt(max(local_frequency[key], 1))
                        for key in row_keys
                    )
                    dims = len({key[0] for key in row_keys})
                    return (-gain - 0.025 * len(row_keys) - 0.05 * dims, -len(row_keys), tie_hash(seed, source, str(row["sample_id"])))

                chosen = min(candidates, key=coverage_rank)
                selected.append(chosen)
                selected_ids.add(str(chosen["sample_id"]))
                conversation_counts[str(chosen["conversation_id"])] += 1
                covered.update(labels_by_row[str(chosen["sample_id"])])
        nonempty_fill, relaxed = _stratified_fill(
            nonempty,
            max(0, wanted_nonempty - len(selected)),
            seed=seed,
            source=source,
            selected_ids=selected_ids,
            conversation_counts=conversation_counts,
            conversation_cap=relaxed,
        )
        selected.extend(nonempty_fill)
        empty_fill, relaxed = _stratified_fill(
            empty,
            max(0, wanted_empty),
            seed=seed,
            source=source,
            selected_ids=selected_ids,
            conversation_counts=conversation_counts,
            conversation_cap=relaxed,
        )
        selected.extend(empty_fill)
        if len(selected) < target:
            remainder, relaxed = _stratified_fill(
                rows,
                target - len(selected),
                seed=seed,
                source=source,
                selected_ids=selected_ids,
                conversation_counts=conversation_counts,
                conversation_cap=relaxed,
            )
            selected.extend(remainder)
        if len(selected) != target:
            raise ValueError(f"could select only {len(selected)}/{target} candidates from {source}")
        selected_by_source[source] = selected
        source_audit[source] = {
            "available": len(rows),
            "requested": requested[source],
            "selected": len(selected),
            "reallocated_from_requested_quota": target - requested[source],
            "gold_nonempty": sum(bool(profile_label_keys(_profile(row))) for row in selected),
            "gold_empty": sum(not profile_label_keys(_profile(row)) for row in selected),
            "distinct_conversations": len({str(row["conversation_id"]) for row in selected}),
            "max_per_conversation": max((conversation_counts[cid] for cid in conversation_counts), default=0),
            "conversation_cap_initial": per_conversation_cap,
            "conversation_cap_final": relaxed,
            "label_coverage_target": {f"{dim}:{label}": count for (dim, label), count in sorted(needs.items())},
            "label_coverage_achieved": {
                f"{dim}:{label}": int(covered[(dim, label)]) for dim, label in sorted(needs)
            },
        }

    if sum(len(rows) for rows in selected_by_source.values()) != total_count:
        raise ValueError("source availability could not satisfy the frozen total candidate target")
    candidates = []
    for source in ("train", "dev"):
        values = selected_by_source[source]
        values.sort(key=lambda row: tie_hash(seed, source, str(row["sample_id"])))
        for row in values:
            candidates.append({
                "sample_id": str(row["sample_id"]),
                "source": source,
                "conversation_id": str(row["conversation_id"]),
                "prompt_length": int(row.get("prompt_length", 0)),
                "history_message_count": int(row.get("history_message_count", 0)),
                "gold_profile": {dim: list(_profile(row)[dim]) for dim in PROFILE_DIMS},
                "gold_nonempty": bool(profile_label_keys(_profile(row))),
                "candidate_tie_sha256": tie_hash(seed, source, str(row["sample_id"])),
            })
    return candidates, {
        "seed": seed,
        "requested_count": total_count,
        "selected_count": len(candidates),
        "source_counts_requested": requested,
        "source_counts_allocated": targets,
        "source_counts_actual": dict(Counter(row["source"] for row in candidates)),
        "overall_gold_nonempty": sum(row["gold_nonempty"] for row in candidates),
        "overall_gold_empty": sum(not row["gold_nonempty"] for row in candidates),
        "candidate_ids_sha256": canonical_json_sha256([row["sample_id"] for row in candidates]),
        "source_audit": source_audit,
        "train_dev_label_frequency": {f"{dim}:{label}": count for (dim, label), count in sorted(all_label_frequency.items())},
    }


def score_candidate_difficulty(
    candidates: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    label_frequency: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score strict parsed predictions against Train/Dev Gold in 33 dimension keys."""
    frequency = {}
    for key, count in label_frequency.items():
        dimension, label = key.split(":", 1)
        frequency[(dimension, label)] = int(count)
    fmax = max(frequency.values(), default=1)
    output: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    errors: dict[str, Counter[str]] = {dim: Counter() for dim in PROFILE_DIMS}
    empty_gold_fp_labels = 0
    nonempty_predictions = 0
    for row in candidates:
        sample_id = str(row["sample_id"])
        prediction = predictions.get(sample_id)
        if not isinstance(prediction, Mapping):
            raise ValueError(f"candidate prediction is missing for {sample_id}")
        valid = prediction.get("status") == "valid" and isinstance(prediction.get("parsed"), Mapping)
        gold = {dim: list(row["gold_profile"][dim]) for dim in PROFILE_DIMS}
        if valid:
            parsed_profile = prediction["parsed"].get("user_profile")
            if not isinstance(parsed_profile, Mapping):
                valid = False
            else:
                predicted = {dim: list(parsed_profile.get(dim, [])) for dim in PROFILE_DIMS}
                try:
                    predicted_keys = profile_label_keys(predicted)
                except (TypeError, ValueError):
                    valid = False
        if not valid:
            group = "invalid"
            predicted = None
            fn_keys: list[tuple[str, str]] = []
            fp_keys: list[tuple[str, str]] = []
            tp_keys: list[tuple[str, str]] = []
            fn_count = fp_count = tp_count = None
            difficulty = None
        else:
            gold_keys = profile_label_keys(gold)
            fn_set = gold_keys - predicted_keys
            fp_set = predicted_keys - gold_keys
            tp_set = gold_keys & predicted_keys
            fn_keys, fp_keys, tp_keys = sorted(fn_set), sorted(fp_set), sorted(tp_set)
            fn_count, fp_count, tp_count = len(fn_set), len(fp_set), len(tp_set)
            if fn_count and fp_count:
                group = "mixed"
            elif fn_count:
                group = "miss"
            elif fp_count:
                group = "extra"
            else:
                group = "clean"
            rarity = sum(min(2.0, math.sqrt(fmax / max(frequency.get(key, 1), 1))) for key in fn_set)
            difficulty = (
                2.0 * min(fn_count, 3)
                + 1.5 * min(fp_count, 3)
                + 0.5 * bool(fn_count and fp_count)
                + 0.25 * min(rarity, 2.0)
            )
            if any(predicted[dim] for dim in PROFILE_DIMS):
                nonempty_predictions += 1
            if not gold_keys:
                empty_gold_fp_labels += fp_count
            for dimension, label in fn_set:
                errors[dimension][f"FN:{label}"] += 1
            for dimension, label in fp_set:
                errors[dimension][f"FP:{label}"] += 1
        group_counts[group] += 1
        output.append({
            "sample_id": sample_id,
            "source": row["source"],
            "conversation_id": row["conversation_id"],
            "gold_profile": gold,
            "predicted_profile": predicted,
            "prediction_status": prediction.get("status"),
            "prediction_reason": prediction.get("reason"),
            "fn_count": fn_count,
            "fp_count": fp_count,
            "tp_count": tp_count,
            "false_negatives": [{"dimension": dim, "label": label} for dim, label in fn_keys],
            "false_positives": [{"dimension": dim, "label": label} for dim, label in fp_keys],
            "true_positives": [{"dimension": dim, "label": label} for dim, label in tp_keys],
            "difficulty": difficulty,
            "group": group,
            "duplicate_candidate_weight": 1,
            "candidate_tie_sha256": row["candidate_tie_sha256"],
        })
    output.sort(key=lambda row: (
        row["group"],
        -(float(row["difficulty"]) if row["difficulty"] is not None else -1.0),
        row["candidate_tie_sha256"],
    ))
    stats = {
        "candidate_count": len(output),
        "group_counts": {key: group_counts[key] for key in ("miss", "extra", "mixed", "clean", "invalid")},
        "valid_nonempty_prediction_count": nonempty_predictions,
        "empty_gold_false_positive_label_count": empty_gold_fp_labels,
        "distinct_conversations": len({row["conversation_id"] for row in output}),
        "label_error_counts": {
            dim: {key: value for key, value in sorted(errors[dim].items(), key=lambda item: (-item[1], item[0]))}
            for dim in PROFILE_DIMS
        },
        "top_label_errors": {
            dim: sorted(
                ({"error": key, "count": value} for key, value in errors[dim].items()),
                key=lambda item: (-item["count"], item["error"]),
            )[:10]
            for dim in PROFILE_DIMS
        },
    }
    return output, stats


def _rank_strata(rows: Sequence[Mapping[str, Any]], seed: int, phase: str, bucket: str) -> list[Mapping[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row.get("difficulty") or 0), str(row.get("candidate_tie_sha256", ""))))
    width = max(1, math.ceil(len(ranked) / 4))
    strata = [ranked[index * width:(index + 1) * width] for index in range(4)]
    result: list[Mapping[str, Any]] = []
    round_index = 0
    while any(round_index < len(values) for values in strata):
        for values in strata:
            if round_index < len(values):
                result.append(values[round_index])
        round_index += 1
    return result


def build_replay_schedule(
    hard_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    pilot_steps: int = 128,
    extension_steps: int = 256,
    accumulation: int = 4,
    dev_fraction_cap: float = 0.25,
    pilot_repeat_fraction: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze 384 steps of exact source-aware sample exposures."""
    if pilot_steps <= 0 or extension_steps <= 0 or accumulation != 4:
        raise ValueError("Replay requires positive staged steps and gradient accumulation of four")
    by_id = {str(row["sample_id"]): dict(row) for row in candidate_rows}
    if len(by_id) != len(candidate_rows):
        raise ValueError("candidate rows contain duplicate sample IDs")
    groups: dict[str, list[dict[str, Any]]] = {"miss": [], "extra": [], "maintain": []}
    for row in hard_rows:
        sample_id = str(row["sample_id"])
        if sample_id not in by_id:
            raise ValueError(f"hard sample {sample_id} is outside the frozen candidate set")
        item = dict(row)
        if item["group"] == "miss":
            primary = "miss"
        elif item["group"] == "extra":
            primary = "extra"
        elif item["group"] == "mixed":
            fn_severity = 2.0 * int(item["fn_count"])
            fp_severity = 1.5 * int(item["fp_count"])
            if fn_severity > fp_severity:
                primary = "miss"
            elif fp_severity > fn_severity:
                primary = "extra"
            else:
                primary = "miss" if int(hashlib.sha256(f"{seed}|{sample_id}".encode()).hexdigest(), 16) % 2 == 0 else "extra"
        else:
            primary = "maintain"
        if primary == "maintain":
            groups["maintain"].append(item)
        else:
            item["difficulty"] = float(item.get("difficulty") or 0.0)
            groups[primary].append(item)
    hard_ids = {str(row["sample_id"]) for bucket in ("miss", "extra") for row in groups[bucket]}
    # Maintain anchors are clean predictions first, then other valid candidates
    # that were not assigned to a hard bucket. Invalid predictions are excluded.
    maintain_ids = {str(row["sample_id"]) for row in groups["maintain"]}
    invalid_ids = {str(row["sample_id"]) for row in hard_rows if row["group"] == "invalid"}
    for row in candidate_rows:
        sample_id = str(row["sample_id"])
        if sample_id not in hard_ids | maintain_ids | invalid_ids:
            groups["maintain"].append({**dict(row), "difficulty": 0.0, "group": "clean"})
            maintain_ids.add(sample_id)

    def valid_group(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [row for row in rows if row.get("prediction_status", "valid") == "valid"]

    groups = {key: list(valid_group(value)) for key, value in groups.items()}
    for key in groups:
        groups[key] = _rank_strata(groups[key], seed, "all", key)
    if not groups["maintain"]:
        raise ValueError("mining_insufficient: no legal clean/representative anchors are available")

    stage_specs = [
        ("pilot", 1, pilot_steps, {"miss": 230, "extra": 128, "maintain": 154}, 2),
        ("extension", pilot_steps + 1, extension_steps, {"miss": 461, "extra": 256, "maintain": 307}, 4),
    ]
    selected_total: Counter[str] = Counter()
    schedule: list[dict[str, Any]] = []
    exposure_summary: Counter[str] = Counter()
    source_summary: Counter[str] = Counter()
    conversation_fallbacks = []
    bucket_deviations: dict[str, Any] = {}

    for phase_name, first_step, step_count, quotas, max_total in stage_specs:
        phase_exposures: dict[str, list[dict[str, Any]]] = {key: [] for key in quotas}
        phase_dev_cap = math.floor(step_count * accumulation * dev_fraction_cap)
        phase_dev_used = 0
        deviations = Counter()
        for bucket, quota in quotas.items():
            pool = list(groups[bucket])
            if not pool and bucket != "maintain":
                deviations[f"{bucket}_filled_from_other_hard"] += quota
                alternatives = [row for other in ("miss", "extra") if other != bucket for row in groups[other]]
                pool = _rank_strata(alternatives, seed, phase_name, bucket)
            if bucket == "maintain" and len(pool) < quota:
                supplemental = [row for row in candidate_rows if str(row["sample_id"]) not in hard_ids and str(row["sample_id"]) not in invalid_ids]
                pool.extend(row for row in _rank_strata(supplemental, seed, phase_name, bucket) if str(row["sample_id"]) not in {str(item["sample_id"]) for item in pool})
            if not pool:
                raise ValueError(f"mining_insufficient: Replay bucket {bucket} has no candidate pool")
            ordered = _rank_strata(pool, seed, phase_name, bucket)
            if phase_name == "pilot" and bucket in {"miss", "extra"}:
                requested_repeats = math.ceil(quota * pilot_repeat_fraction)
            else:
                requested_repeats = 0
            unique_needed = quota - requested_repeats
            usable = [row for row in ordered if selected_total[str(row["sample_id"])] < max_total]
            # Hold Dev usage below one quarter of each frozen stage. Favor Train,
            # then use Dev only when required by available hard examples.
            train_pool = [row for row in usable if by_id[str(row["sample_id"])]["source"] == "train"]
            dev_pool = [row for row in usable if by_id[str(row["sample_id"])]["source"] == "dev"]
            wanted_dev = min(len(dev_pool), max(0, round(unique_needed * 0.10)), phase_dev_cap - phase_dev_used)
            picked = train_pool[:max(0, unique_needed - wanted_dev)]
            if len(picked) < unique_needed - wanted_dev:
                wanted_dev = min(len(dev_pool), unique_needed - len(picked), phase_dev_cap - phase_dev_used)
            picked.extend(dev_pool[:wanted_dev])
            if len(picked) < unique_needed:
                extra_train = train_pool[len(picked):]
                picked.extend(extra_train[:unique_needed - len(picked)])
            if len(picked) < unique_needed:
                extra_dev = dev_pool[wanted_dev:]
                available_dev = phase_dev_cap - phase_dev_used - sum(by_id[str(row["sample_id"])]["source"] == "dev" for row in picked)
                picked.extend(extra_dev[:max(0, min(unique_needed - len(picked), available_dev))])
            if len(picked) < unique_needed:
                # If necessary, borrow from the other error bucket before anchors.
                alternatives = [row for other in ("miss", "extra") if other != bucket for row in groups[other]]
                used = {str(row["sample_id"]) for row in picked}
                for row in _rank_strata(alternatives, seed, phase_name, bucket):
                    sid = str(row["sample_id"])
                    source = by_id[sid]["source"]
                    if sid in used or selected_total[sid] >= max_total or (source == "dev" and phase_dev_used >= phase_dev_cap):
                        continue
                    picked.append(row)
                    used.add(sid)
                    deviations[f"{bucket}_borrowed_other_hard"] += 1
                    if len(picked) == unique_needed:
                        break
            if len(picked) < unique_needed:
                raise ValueError(f"mining_insufficient: {bucket} has {len(picked)} unique IDs for {unique_needed} exposures")
            rows_for_bucket = list(picked)
            starting_usage = {str(row["sample_id"]): selected_total[str(row["sample_id"])] for row in picked}
            for item in picked:
                sid = str(item["sample_id"])
                source = by_id[sid]["source"]
                if source == "dev":
                    phase_dev_used += 1
                selected_total[sid] += 1
            repeat_needed = quota - len(rows_for_bucket)
            if repeat_needed:
                eligible_repeats = [row for row in picked if selected_total[str(row["sample_id"])] < max_total]
                if len(eligible_repeats) < repeat_needed:
                    repeat_needed = len(eligible_repeats)
                    deviations[f"{bucket}_repeat_target_reduced"] += quota - len(rows_for_bucket) - repeat_needed
                for index in range(repeat_needed):
                    item = eligible_repeats[index % len(eligible_repeats)]
                    sid = str(item["sample_id"])
                    if by_id[sid]["source"] == "dev" and phase_dev_used >= phase_dev_cap:
                        continue
                    rows_for_bucket.append(item)
                    selected_total[sid] += 1
                    if by_id[sid]["source"] == "dev":
                        phase_dev_used += 1
            if len(rows_for_bucket) < quota:
                # Fill from remaining candidates of the same bucket without
                # exceeding per-sample total or Dev exposure limits.
                existing = Counter(str(row["sample_id"]) for row in rows_for_bucket)
                for item in ordered:
                    sid = str(item["sample_id"])
                    source = by_id[sid]["source"]
                    if selected_total[sid] >= max_total or (source == "dev" and phase_dev_used >= phase_dev_cap):
                        continue
                    rows_for_bucket.append(item)
                    selected_total[sid] += 1
                    existing[sid] += 1
                    if source == "dev":
                        phase_dev_used += 1
                    if len(rows_for_bucket) == quota:
                        break
            if len(rows_for_bucket) != quota:
                raise ValueError(f"mining_insufficient: could not satisfy {phase_name} {bucket} quota")
            phase_counts = Counter(str(row["sample_id"]) for row in rows_for_bucket)
            first_occurrence_order = []
            seen_for_bucket = set()
            for item in rows_for_bucket:
                sid = str(item["sample_id"])
                if sid not in seen_for_bucket:
                    first_occurrence_order.append(item)
                    seen_for_bucket.add(sid)
            exposure_order = list(first_occurrence_order)
            for occurrence_index in range(2, max(phase_counts.values(), default=1) + 1):
                exposure_order.extend(
                    row for row in first_occurrence_order
                    if phase_counts[str(row["sample_id"])] >= occurrence_index
                )
            for item in exposure_order:
                sid = str(item["sample_id"])
                occurrence = sum(1 for previous in phase_exposures[bucket] if previous["sample_id"] == sid) + 1
                phase_exposures[bucket].append({
                    "sample_id": sid,
                    "source": by_id[sid]["source"],
                    "conversation_id": by_id[sid]["conversation_id"],
                    "bucket": bucket,
                    "occurrence_in_schedule": starting_usage.get(sid, 0) + occurrence,
                    "gold_nonempty": by_id[sid]["gold_nonempty"],
                    "profile_dimensions": [dim for dim in PROFILE_DIMS if by_id[sid]["gold_profile"][dim]],
                    "profile_labels": sorted(
                        f"{dim}:{label}"
                        for dim in PROFILE_DIMS
                        for label in by_id[sid]["gold_profile"][dim]
                    ),
                    "history_message_count": by_id[sid]["history_message_count"],
                })
        if sum(map(len, phase_exposures.values())) != step_count * accumulation:
            raise ValueError(f"{phase_name} exposure count differs from its frozen step count")
        phase_dimensions = {
            dimension
            for items in phase_exposures.values()
            for row in items
            for dimension in row["profile_dimensions"]
        }
        if phase_dimensions != set(PROFILE_DIMS):
            raise ValueError(f"mining_insufficient: {phase_name} schedule does not cover all three profile dimensions")

        # Interleave buckets by their remaining quota, then place four unique
        # sample IDs in each optimizer step with a bounded conversation share.
        remaining = {bucket: len(items) for bucket, items in phase_exposures.items()}
        queues = {bucket: list(items) for bucket, items in phase_exposures.items()}
        offsets = Counter()
        chunk_conversations: Counter[str] = Counter()
        phase_dev_actual = 0
        for local_step in range(step_count):
            if local_step % 32 == 0:
                chunk_conversations.clear()
            step_items: list[dict[str, Any]] = []
            step_ids: set[str] = set()
            step_conversations: Counter[str] = Counter()
            step_bucket_counts = Counter()
            for slot in range(accumulation):
                active = [bucket for bucket in quotas if remaining[bucket] > 0]
                if not active:
                    raise ValueError(f"{phase_name}: schedule ran out of bucket exposures")
                bucket = max(active, key=lambda name: (remaining[name] / quotas[name], -list(quotas).index(name)))
                queue = queues[bucket]
                chosen_index = None
                for index, item in enumerate(queue):
                    if item["sample_id"] in step_ids:
                        continue
                    if chunk_conversations[item["conversation_id"]] + step_conversations[item["conversation_id"]] >= 4:
                        continue
                    if item["source"] == "dev" and phase_dev_actual >= phase_dev_cap:
                        continue
                    chosen_index = index
                    break
                if chosen_index is None:
                    # Explicitly relax the per-32-step conversation cap to six;
                    # never relax the unique-sample-per-step or source limits.
                    for index, item in enumerate(queue):
                        if item["sample_id"] not in step_ids and chunk_conversations[item["conversation_id"]] + step_conversations[item["conversation_id"]] < 6 and not (item["source"] == "dev" and phase_dev_actual >= phase_dev_cap):
                            chosen_index = index
                            conversation_fallbacks.append({"phase": phase_name, "step": first_step + local_step, "bucket": bucket, "fallback_cap": 6})
                            break
                if chosen_index is None:
                    raise ValueError(f"{phase_name}: cannot satisfy sample, conversation, or Dev exposure constraints")
                item = queue.pop(chosen_index)
                offsets[bucket] += 1
                remaining[bucket] -= 1
                step_ids.add(item["sample_id"])
                step_conversations[item["conversation_id"]] += 1
                step_bucket_counts[bucket] += 1
                if item["source"] == "dev":
                    phase_dev_actual += 1
                step_items.append({**item, "optimizer_step": first_step + local_step, "microbatch_slot": slot + 1})
            chunk_conversations.update(step_conversations)
            schedule.extend(step_items)
            for item in step_items:
                exposure_summary[item["bucket"]] += 1
                source_summary[item["source"]] += 1
        bucket_deviations[phase_name] = {
            "requested_quotas": quotas,
            "actual_quotas": dict(offsets),
            "dev_exposure_count": phase_dev_actual,
            "dev_exposure_cap": phase_dev_cap,
            "fallback_notes": dict(deviations),
        }

    pilot_hard = [row for row in schedule[:pilot_steps * accumulation] if row["bucket"] in {"miss", "extra"}]
    pilot_ids = Counter(row["sample_id"] for row in pilot_hard)
    pilot_second_exposures = sum(max(0, count - 1) for count in pilot_ids.values())
    pilot_repeat_rate = pilot_second_exposures / len(pilot_hard) if pilot_hard else 0.0
    if pilot_repeat_rate + 1e-12 < pilot_repeat_fraction:
        raise ValueError(f"mining_insufficient: pilot difficult repeat rate {pilot_repeat_rate:.4f} < {pilot_repeat_fraction:.4f}")
    counts = Counter(row["sample_id"] for row in schedule)
    if max(counts.values(), default=0) > 4:
        raise ValueError("Replay schedule exceeds four cumulative exposures per sample")
    if any(len({row["sample_id"] for row in schedule[index:index + accumulation]}) != accumulation for index in range(0, len(schedule), accumulation)):
        raise ValueError("Replay schedule duplicates a sample within an optimizer step")
    if sum(row["source"] == "dev" for row in schedule) > math.floor(len(schedule) * dev_fraction_cap):
        raise ValueError("Replay schedule exceeds the frozen Dev exposure cap")
    label_exposures = Counter(label for row in schedule for label in row["profile_labels"])
    history_counts = [int(row["history_message_count"]) for row in schedule]
    return schedule, {
        "seed": seed,
        "optimizer_steps": pilot_steps + extension_steps,
        "pilot_steps": pilot_steps,
        "extension_steps": extension_steps,
        "accumulation": accumulation,
        "total_exposures": len(schedule),
        "sample_exposure_counts": dict(sorted(counts.items())),
        "bucket_exposure_counts": dict(exposure_summary),
        "source_exposure_counts": dict(source_summary),
        "profile_dimension_exposure_counts": {
            dimension: sum(dimension in row["profile_dimensions"] for row in schedule)
            for dimension in PROFILE_DIMS
        },
        "profile_label_exposure_counts": dict(sorted(label_exposures.items())),
        "history_message_count_min": min(history_counts, default=0),
        "history_message_count_max": max(history_counts, default=0),
        "pilot_hard_exposures": len(pilot_hard),
        "pilot_second_exposures": pilot_second_exposures,
        "pilot_repeat_rate": pilot_repeat_rate,
        "max_sample_exposures": max(counts.values(), default=0),
        "conversation_fallbacks": conversation_fallbacks,
        "bucket_deviations": bucket_deviations,
        "schedule_sha256": canonical_json_sha256(schedule),
    }
