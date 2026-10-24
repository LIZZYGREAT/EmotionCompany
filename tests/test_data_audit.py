import hashlib
import json
import shutil
from pathlib import Path

import yaml

from src.data.audit import run_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_EMPTY = {"personality_traits": [], "interests": [], "style": []}


def make_target(response: str, emotion: str = "neutral", profile=None, memory_refs=None):
    return {
        "emotion_label": emotion,
        "user_profile": PROFILE_EMPTY if profile is None else profile,
        "response_text": response,
        "memory_refs": [] if memory_refs is None else memory_refs,
    }


def make_point(target_id, response_id, response, emotion="neutral", profile=None, memory_refs=None):
    return {
        "target_user_turn_id": target_id,
        "assistant_response_turn_id": response_id,
        "target": make_target(response, emotion, profile, memory_refs),
    }


def make_row(split, number, turns, points):
    return {
        "conversation_id": f"{split}_conv_{number:06d}",
        "turns": turns,
        "prediction_points": points,
    }


def write_case(root: Path, train_lines, val_lines):
    (root / "data/train").mkdir(parents=True, exist_ok=True)
    (root / "data/val").mkdir(parents=True, exist_ok=True)
    for split, lines in (("train", train_lines), ("val", val_lines)):
        name = f"{split}_public"
        input_path = root / "data" / split / f"{name}.jsonl"
        if isinstance(lines, bytes):
            input_path.write_bytes(lines)
        else:
            input_path.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
        schema_source = PROJECT_ROOT / "data" / split / f"{name}.schema.json"
        shutil.copyfile(schema_source, input_path.with_name(f"{name}.schema.json"))
    config = {
        "inputs": {
            "train": "data/train/train_public.jsonl",
            "val": "data/val/val_public.jsonl",
        },
        "output": {
            "directory": "outputs/audit-test",
            "report": "docs/audit-test.md",
        },
        "checks": {
            "schema": True,
            "turn_references": True,
            "duplicates": True,
            "cross_split": True,
            "distributions": True,
        },
        "thresholds": {
            "short_response_chars": 5,
            "long_response_chars": 2000,
            "long_message_chars": 4000,
            "short_cross_split_chars": 100,
        },
    }
    config_path = root / "configs/data_audit.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


def by_code(result):
    return result["summary"]["issues"]["by_code"]


def test_non_monotonic_turn_ids_and_multiple_points_use_array_prefixes(tmp_path):
    turns = [
        {"turn_id": 20, "role": "user", "content": "你好🙂\n！"},
        {"turn_id": 3, "role": "assistant", "content": "第一条回复"},
        {"turn_id": 10, "role": "user", "content": "继续说"},
        {"turn_id": 2, "role": "assistant", "content": "第二条回复"},
    ]
    row = make_row("train", 1, turns, [
        make_point(20, 3, "第一条回复", "joy"),
        make_point(10, 2, "第二条回复", "sadness"),
    ])
    config = write_case(tmp_path, [row], [make_row("val", 1, turns[:2], [make_point(20, 3, "第一条回复")])])
    raw_before = (tmp_path / "data/train/train_public.jsonl").read_bytes()
    first = run_config(config, root=tmp_path)
    summary_bytes = (tmp_path / "outputs/audit-test/summary.json").read_bytes()
    issues_bytes = (tmp_path / "outputs/audit-test/issues.jsonl").read_bytes()
    second = run_config(config, root=tmp_path)
    assert first["summary"]["splits"]["train"]["counters"]["prediction_point_count"] == 2
    assert first["summary"]["splits"]["train"]["text_lengths_unicode_characters"]["legal_history_messages"]["max"] == 3
    assert by_code(first).get("RESPONSE_NOT_AFTER_TARGET", 0) == 0
    assert by_code(first).get("RESPONSE_NOT_FIRST_ASSISTANT", 0) == 0
    assert summary_bytes == (tmp_path / "outputs/audit-test/summary.json").read_bytes()
    assert issues_bytes == (tmp_path / "outputs/audit-test/issues.jsonl").read_bytes()
    assert raw_before == (tmp_path / "data/train/train_public.jsonl").read_bytes()
    assert first["manifest"]["inputs"]["train"]["sha256_before"] == first["manifest"]["inputs"]["train"]["sha256_after"]
    assert second["manifest"]["inputs"]["train"]["sha256_before"] == second["manifest"]["inputs"]["train"]["sha256_after"]


def test_malformed_rows_are_located_without_losing_following_rows(tmp_path):
    good = make_row("train", 1, [{"turn_id": 1, "role": "user", "content": "问"}, {"turn_id": 2, "role": "assistant", "content": "答"}], [make_point(1, 2, "答")])
    raw = b'{"conversation_id":\n[]\nnull\n{}\n{"conversation_id":"train_conv_000009","turns":"bad","prediction_points":null}\n' + (json.dumps(good, ensure_ascii=False) + "\n").encode("utf-8")
    val = make_row("val", 1, [{"turn_id": 1, "role": "user", "content": "问"}, {"turn_id": 2, "role": "assistant", "content": "答"}], [make_point(1, 2, "答")])
    config = write_case(tmp_path, raw, [val])
    result = run_config(config, root=tmp_path)
    train = result["summary"]["splits"]["train"]
    assert train["counters"]["physical_line_count"] == 6
    assert train["counters"]["unparseable_line_count"] == 1
    assert train["counters"]["parsed_object_count"] == 3
    assert train["counters"]["prediction_point_count"] == 1
    parse_issue = next(issue for issue in result["issues"] if issue["code"] == "FILE_PARSE")
    assert parse_issue["source_line"] == 1
    assert any(issue["code"] == "ROOT_SCHEMA" and issue["source_line"] == 2 for issue in result["issues"])
    assert any(issue["code"] == "ROOT_SCHEMA" and issue["source_line"] == 3 for issue in result["issues"])
    assert any(issue["code"] == "ROOT_SCHEMA" and issue["source_line"] == 4 for issue in result["issues"])
    assert any(issue["code"] == "TURN_SCHEMA" and issue["source_line"] == 5 for issue in result["issues"])
    assert any(issue["code"] == "POINT_SCHEMA" and issue["source_line"] == 5 for issue in result["issues"])


def test_reference_failures_whitespace_mismatch_and_duplicate_ids(tmp_path):
    turns = [
        {"turn_id": 1, "role": "user", "content": "起始"},
        {"turn_id": 2, "role": "assistant", "content": "r1"},
        {"turn_id": 3, "role": "user", "content": "目标"},
        {"turn_id": 4, "role": "assistant", "content": "first"},
        {"turn_id": 5, "role": "assistant", "content": "second"},
        {"turn_id": 6, "role": "user", "content": "末轮"},
        {"turn_id": 7, "role": "user", "content": "重复一"},
        {"turn_id": 7, "role": "user", "content": "重复二"},
    ]
    points = [
        make_point(99, 2, "r1"),
        make_point(2, 5, "second"),
        make_point(3, 2, "r1"),
        make_point(3, 5, "different"),
        make_point(3, 4, " first\n"),
        make_point(6, 6, "末轮"),
        make_point(6, 999, "missing"),
        make_point(7, 5, "second"),
    ]
    row = make_row("train", 1, turns, points)
    val = make_row("val", 1, turns[:2], [make_point(1, 2, "r1")])
    config = write_case(tmp_path, [row], [val])
    result = run_config(config, root=tmp_path)
    codes = by_code(result)
    assert codes["TARGET_TURN_NOT_FOUND"] == 1
    assert codes["TARGET_TURN_NOT_USER"] == 1
    assert codes["TARGET_TURN_ID_AMBIGUOUS"] == 1
    assert codes["RESPONSE_TURN_NOT_FOUND"] == 1
    assert codes["RESPONSE_TURN_NOT_ASSISTANT"] == 1
    assert codes["RESPONSE_NOT_AFTER_TARGET"] == 1
    assert codes["RESPONSE_NOT_FIRST_ASSISTANT"] == 1
    assert codes["RESPONSE_TEXT_MISMATCH"] == 2
    assert codes["DUPLICATE_TURN_ID"] == 1
    assert codes["DUPLICATE_TARGET_POINT"] == 2
    assert codes.get("SUPERVISION_LABEL_CONFLICT", 0) == 0
    mismatch = [issue["observed"]["comparison"] for issue in result["issues"] if issue["code"] == "RESPONSE_TEXT_MISMATCH"]
    assert sorted(mismatch) == ["different_text", "whitespace_only"]


def test_empty_points_empty_profiles_and_nonempty_memory_are_not_errors(tmp_path):
    no_points = make_row("train", 1, [{"turn_id": 1, "role": "user", "content": "无监督对话"}], [])
    with_memory = make_row("train", 2, [
        {"turn_id": 1, "role": "user", "content": "你好"},
        {"turn_id": 2, "role": "assistant", "content": "你好！"},
    ], [make_point(1, 2, "你好！", "neutral", PROFILE_EMPTY, ["mem_000001"])])
    val = make_row("val", 1, [{"turn_id": 1, "role": "user", "content": "问"}, {"turn_id": 2, "role": "assistant", "content": "答"}], [make_point(1, 2, "答")])
    config = write_case(tmp_path, [no_points, with_memory], [val])
    result = run_config(config, root=tmp_path)
    train = result["summary"]["splits"]["train"]
    assert train["counters"]["empty_points_conversation_count"] == 1
    assert train["profile"]["all_three_empty_points"] == 1
    assert train["profile"]["readable_points"] == 1
    assert train["memory_refs"]["nonempty_points"] == 1
    assert by_code(result).get("ROOT_SCHEMA", 0) == 0
    assert by_code(result).get("PROFILE_SCHEMA", 0) == 0
    assert next(issue for issue in result["issues"] if issue["code"] == "EMPTY_POINTS")["severity"] == "INFO"


def test_duplicate_content_cross_split_conflicts_and_different_replies(tmp_path):
    profile = {"personality_traits": ["sensitive"], "interests": ["music"], "style": ["brief"]}

    def row(split, number, reply, emotion="joy"):
        turns = [{"turn_id": 1, "role": "user", "content": "同一历史"}, {"turn_id": 2, "role": "assistant", "content": reply}]
        return make_row(split, number, turns, [make_point(1, 2, reply, emotion, profile)])

    train = [row("train", 1, "回复A"), row("train", 2, "回复A"), row("train", 3, "回复B"), row("train", 4, "回复D", "anger")]
    val = [row("val", 1, "回复A"), row("val", 2, "回复C", "sadness")]
    config = write_case(tmp_path, train, val)
    result = run_config(config, root=tmp_path)
    codes = by_code(result)
    assert codes["DUPLICATE_CONVERSATION_TURNS"] == 1
    assert codes["DUPLICATE_PREDICTION_POINT"] == 1
    assert codes["CROSS_SPLIT_DUPLICATE_PREDICTION_POINT"] == 1
    assert codes["CONFLICTING_SUPERVISION"] == 2
    assert codes["TRAIN_VAL_HISTORY_OVERLAP"] == 2
    assert codes["CROSS_SPLIT_SUPERVISION_CONFLICT"] == 2
    assert codes["TRAIN_VAL_TURNS_OVERLAP"] == 1


def test_boolean_ids_do_not_alias_integer_ids_and_text_is_never_rewritten(tmp_path):
    row = make_row("train", 1, [
        {"turn_id": 1, "role": "user", "content": "中文🙂\n！\x00"},
        {"turn_id": True, "role": "assistant", "content": "答"},
    ], [make_point(1, True, "答")])
    val = make_row("val", 1, [{"turn_id": 1, "role": "user", "content": "问"}, {"turn_id": 2, "role": "assistant", "content": "答"}], [make_point(1, 2, "答")])
    config = write_case(tmp_path, [row], [val])
    raw_path = tmp_path / "data/train/train_public.jsonl"
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    result = run_config(config, root=tmp_path)
    after = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert before == after
    assert by_code(result).get("DUPLICATE_TURN_ID", 0) == 0
    assert by_code(result).get("TARGET_TURN_NOT_FOUND", 0) == 0
    assert by_code(result)["TURN_SCHEMA"] >= 1  # JSON Schema rejects bool as an integer turn_id.
    assert by_code(result)["INVALID_TEXT_CHAR"] == 1


def test_whitespace_normalized_history_and_different_replies_do_not_imply_label_conflict(tmp_path):
    profile = {"personality_traits": ["sensitive"], "interests": ["music"], "style": ["brief"]}

    def row(split, number, user_text, reply):
        turns = [{"turn_id": 1, "role": "user", "content": user_text}, {"turn_id": 2, "role": "assistant", "content": reply}]
        return make_row(split, number, turns, [make_point(1, 2, reply, "neutral", profile)])

    train = [row("train", 1, "hello world", "reply A"), row("train", 2, "hello world", "reply B")]
    val = [row("val", 1, " hello   world\n", "reply C")]
    config = write_case(tmp_path, train, val)
    result = run_config(config, root=tmp_path)
    codes = by_code(result)
    assert codes.get("CONFLICTING_SUPERVISION", 0) == 0
    assert codes.get("CROSS_SPLIT_SUPERVISION_CONFLICT", 0) == 0
    normalized_issues = [issue for issue in result["issues"] if issue["code"] == "TRAIN_VAL_HISTORY_OVERLAP"]
    assert len(normalized_issues) == 1
    assert normalized_issues[0]["observed"]["match_kind"] == "whitespace_normalized_only"
    assert result["summary"]["duplicates_and_cross_split"]["normalized_history_only_overlap_validation_points"] == 1

