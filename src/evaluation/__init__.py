"""Local, offline evaluation helpers for E00/E01."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_loads(text: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(text, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: JSONL 中不能有空行")
            try:
                row = strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: 严格 JSON 解析失败: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: 每行必须是 JSON object")
            rows.append(row)
    return rows


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


def load_d03_target_validator(d03_manifest_path: Path, workspace_root: Path) -> tuple[Draft202012Validator, dict[str, Any]]:
    with d03_manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    input_facts = manifest.get("inputs", {}).get("train", {})
    schema_value = input_facts.get("schema_path") or manifest.get("effective_config", {}).get("inputs", {}).get("train", {}).get("schema")
    if not isinstance(schema_value, str) or not schema_value:
        raise ValueError("D0.3 Manifest 缺少 Train target Schema 路径")
    schema_path = Path(schema_value)
    schema_path = (schema_path if schema_path.is_absolute() else workspace_root / schema_path).resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"D0.3 Train Schema 缺失: {schema_path}")
    schema_sha = sha256_file(schema_path)
    expected_sha = input_facts.get("schema_sha256")
    if expected_sha is None:
        expected_sha = manifest.get("effective_config", {}).get("inputs", {}).get("train", {}).get("expected_schema_sha256")
    if not isinstance(expected_sha, str) or schema_sha != expected_sha:
        raise ValueError("D0.3 Train Schema SHA-256 与 Manifest 不符")
    full_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if "$defs" not in full_schema or "target" not in full_schema["$defs"]:
        raise ValueError("D0.3 Train Schema 缺少 $defs.target")
    target_schema = {
        "$schema": full_schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$defs": full_schema["$defs"],
        "$ref": "#/$defs/target",
    }
    Draft202012Validator.check_schema(target_schema)
    try:
        display_path = schema_path.relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        display_path = str(schema_path)
    facts = {
        "path": display_path,
        "sha256": schema_sha,
    }
    return Draft202012Validator(target_schema), facts
