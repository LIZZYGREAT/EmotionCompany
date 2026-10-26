"""Local, offline evaluation helpers for E00/E01."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: JSONL 中不能有空行")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 解析失败: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: 每行必须是 JSON object")
            rows.append(row)
    return rows


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def load_submission_schema(root: Path) -> dict[str, Any]:
    path = root / "participant/templates/submission_schema.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def schema_enums(schema: dict[str, Any]) -> dict[str, list[str]]:
    properties = schema["properties"]
    profile = properties["user_profile"]["properties"]
    return {
        "emotion_label": [item for item in properties["emotion_label"]["enum"] if item],
        **{
            dimension: list(profile[dimension]["items"]["enum"])
            for dimension in ("personality_traits", "interests", "style")
        },
    }
