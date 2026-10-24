"""Stable conversation grouping and Train/Dev assignment for D0.3."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class _DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        first, second = sorted((root_left, root_right))
        self.parent[second] = first


def build_conversation_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link records sharing exact role/content turns or an exact legal history.

    Each record must contain ``conversation_id``, ``turns``, and
    ``legal_histories``. Fingerprints are based on canonical JSON values, never
    Python's process-randomized ``hash``.
    """
    conversation_ids = [record["conversation_id"] for record in records]
    if len(conversation_ids) != len(set(conversation_ids)):
        raise ValueError("conversation_id must be unique within a split")
    disjoint = _DisjointSet(conversation_ids)
    first_by_fingerprint: dict[tuple[str, str], str] = {}
    fingerprints_by_conversation: dict[str, dict[str, set[str]]] = {
        cid: {"full_turns": set(), "legal_history": set()} for cid in conversation_ids
    }

    for record in records:
        cid = record["conversation_id"]
        turn_payload = [{"role": turn["role"], "content": turn["content"]} for turn in record["turns"]]
        turn_fp = fingerprint(turn_payload)
        fingerprints_by_conversation[cid]["full_turns"].add(turn_fp)
        link_key = ("full_turns", turn_fp)
        if link_key in first_by_fingerprint:
            disjoint.union(cid, first_by_fingerprint[link_key])
        else:
            first_by_fingerprint[link_key] = cid

        for history in record["legal_histories"]:
            history_fp = fingerprint(history)
            fingerprints_by_conversation[cid]["legal_history"].add(history_fp)
            link_key = ("legal_history", history_fp)
            if link_key in first_by_fingerprint:
                disjoint.union(cid, first_by_fingerprint[link_key])
            else:
                first_by_fingerprint[link_key] = cid

    components: dict[str, list[str]] = defaultdict(list)
    for cid in conversation_ids:
        components[disjoint.find(cid)].append(cid)

    result: list[dict[str, Any]] = []
    for members in components.values():
        members = sorted(members)
        group_id = f"group_{fingerprint(members)}"
        full_fingerprints = sorted({
            fp for cid in members for fp in fingerprints_by_conversation[cid]["full_turns"]
        })
        history_fingerprints = sorted({
            fp for cid in members for fp in fingerprints_by_conversation[cid]["legal_history"]
        })
        result.append({
            "group_id": group_id,
            "conversation_ids": members,
            "conversation_count": len(members),
            "full_turn_fingerprints": full_fingerprints,
            "legal_history_fingerprints": history_fingerprints,
        })
    return sorted(result, key=lambda item: item["group_id"])


def _round_nearest_positive(value: float) -> int:
    return int(math.floor(value + 0.5))


def assign_train_dev(
    groups: list[dict[str, Any]],
    conversation_sample_counts: dict[str, int],
    conversation_emotion_counts: dict[str, Counter[str]],
    *,
    dev_ratio: float,
    seed: int,
    preserve_emotions: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign whole connected components to Dev by a deterministic greedy score."""
    if not 0.0 < dev_ratio < 0.5:
        raise ValueError("dev_ratio must be greater than 0 and less than 0.5")

    total_conversations = sum(group["conversation_count"] for group in groups)
    total_samples = sum(conversation_sample_counts.values())
    total_emotions: Counter[str] = Counter()
    group_stats: dict[str, dict[str, Any]] = {}
    for group in groups:
        sample_count = 0
        emotion_count: Counter[str] = Counter()
        for cid in group["conversation_ids"]:
            sample_count += conversation_sample_counts.get(cid, 0)
            emotion_count.update(conversation_emotion_counts.get(cid, Counter()))
        group_stats[group["group_id"]] = {
            "sample_count": sample_count,
            "emotion_counts": dict(sorted(emotion_count.items())),
        }
        total_emotions.update(emotion_count)

    target_conversations = _round_nearest_positive(total_conversations * dev_ratio)
    target_samples = _round_nearest_positive(total_samples * dev_ratio)
    target_emotions = {label: total_emotions[label] * dev_ratio for label in preserve_emotions}

    def score(conversations: int, samples: int, emotions: Counter[str]) -> float:
        value = ((conversations - target_conversations) / max(target_conversations, 1)) ** 2
        value += ((samples - target_samples) / max(target_samples, 1)) ** 2
        if preserve_emotions:
            value += sum(
                ((emotions[label] - target_emotions[label]) / max(target_emotions[label], 1.0)) ** 2
                for label in preserve_emotions
            ) / len(preserve_emotions)
        return value

    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}:{group['group_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected: set[str] = set()
    current_conversations = 0
    current_samples = 0
    current_emotions: Counter[str] = Counter()
    for group in ranked:
        stats = group_stats[group["group_id"]]
        after_emotions = current_emotions + Counter(stats["emotion_counts"])
        if score(
            current_conversations + group["conversation_count"],
            current_samples + stats["sample_count"],
            after_emotions,
        ) < score(current_conversations, current_samples, current_emotions):
            selected.add(group["group_id"])
            current_conversations += group["conversation_count"]
            current_samples += stats["sample_count"]
            current_emotions = after_emotions

    if not selected and ranked:
        selected.add(ranked[0]["group_id"])

    conversation_to_split: dict[str, str] = {}
    for group in groups:
        split = "dev" if group["group_id"] in selected else "train"
        for cid in group["conversation_ids"]:
            if cid in conversation_to_split:
                raise ValueError(f"conversation assigned more than once: {cid}")
            conversation_to_split[cid] = split

    actual_conversations = sum(
        group["conversation_count"] for group in groups if group["group_id"] in selected
    )
    actual_samples = sum(group_stats[group_id]["sample_count"] for group_id in selected)
    actual_emotions: Counter[str] = Counter()
    for group_id in selected:
        actual_emotions.update(group_stats[group_id]["emotion_counts"])

    group_entries = []
    for group in groups:
        group_id = group["group_id"]
        group_entries.append({
            **group,
            "split": "dev" if group_id in selected else "train",
            **group_stats[group_id],
        })
    manifest = {
        "algorithm": "sha256_ranked_greedy_group_assignment_v1",
        "grouping_rules": [
            "same exact full turns projected to role/content",
            "same exact legal history projected to role/content through target user turn",
        ],
        "seed": seed,
        "dev_ratio_target": dev_ratio,
        "target_conversation_count": target_conversations,
        "actual_dev_conversation_count": actual_conversations,
        "actual_dev_conversation_fraction": actual_conversations / total_conversations if total_conversations else 0.0,
        "total_conversation_count": total_conversations,
        "target_sample_count": target_samples,
        "actual_dev_sample_count": actual_samples,
        "actual_dev_sample_fraction": actual_samples / total_samples if total_samples else 0.0,
        "total_sample_count": total_samples,
        "preserved_emotions": preserve_emotions,
        "all_train_emotion_counts": dict(sorted(total_emotions.items())),
        "dev_emotion_counts": dict(sorted(actual_emotions.items())),
        "group_count": len(groups),
        "dev_group_count": len(selected),
        "max_group_conversation_count": max((group["conversation_count"] for group in groups), default=0),
        "max_group_sample_count": max((stats["sample_count"] for stats in group_stats.values()), default=0),
        "conversation_to_split": dict(sorted(conversation_to_split.items())),
        "groups": group_entries,
    }
    return conversation_to_split, manifest
