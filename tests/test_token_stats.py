"""Synthetic budget tests with the pinned, real local Qwen3 tokenizer."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

from src.data.token_stats import (
    _history_key,
    _plan_t1_selections,
    _candidate_grid,
    run_audit,
    select_recent_complete_suffix,
)
from src.shared.prompting import (
    apply_chat_template_batch,
    build_messages,
    encode_full_training_sequence,
    encode_inference_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
P0 = (ROOT / "configs/prompts/p0.txt").read_text(encoding="utf-8")
TRAIN_SCHEMA = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def qwen_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B", revision=REVISION,
            local_files_only=True, trust_remote_code=False,
        )
    except Exception as exc:  # pragma: no cover - expected only without the declared local cache
        pytest.skip(f"pinned local Qwen3 tokenizer unavailable: {type(exc).__name__}: {exc}")


def _sample(sample_id: str, history: list[dict[str, str]], response: str = "合成回复", *, emotion: str = "anxiety", nonempty: bool = False) -> dict:
    profile = {
        "personality_traits": ["sensitive"] if nonempty else [],
        "interests": [],
        "style": [],
    }
    return {
        "sample_id": sample_id,
        "conversation_id": f"conv_{sample_id.replace(':', '_')}",
        "source": {"split": "train", "source_line": 1, "prediction_point_index": 0},
        "history": history,
        "target": {
            "emotion_label": emotion,
            "user_profile": profile,
            "response_text": response,
            "memory_refs": ["mem_000001"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _audit_fixture(tmp_path: Path, tokenizer, *, repeat_run: bool = False):
    for path in ("data/train/train_public.schema.json", "data/val/val_public.schema.json", "participant/templates/submission_schema.json", "configs/prompts/p0.txt"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    (tmp_path / "participant/configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "participant/configs/inference_config.json").write_text(
        json.dumps({"generation": {"max_new_tokens": 100}}), encoding="utf-8"
    )

    train = [
        _sample("train:full", [{"role": "user", "content": "普通的当前问题。"}], "短的合成回答。"),
        _sample("train:trim", [
            {"role": "user", "content": "早期短问题"},
            {"role": "assistant", "content": "中间消息" * 2500},
            {"role": "user", "content": "当前短问题"},
        ], "裁剪后仍可用。", nonempty=True),
        _sample("train:overflow", [{"role": "user", "content": "当前问题"}], "长答案" * 1800, emotion="surprise"),
        _sample("train:floor", [{"role": "user", "content": "超长当前消息" * 1200}], "无法裁当前消息。"),
        _sample("train:longer", [
            {"role": "user", "content": "更早消息"},
            {"role": "assistant", "content": "最后一个较早消息" * 2500},
            {"role": "user", "content": "今天我有点难过"},
        ], "另一个合成短答。", emotion="sadness"),
    ]
    dev = [_sample("dev:one", [{"role": "user", "content": "Dev 当前问题"}], "Dev 合成回复。")]
    val = [_sample("val:one", [{"role": "user", "content": "Val 当前问题"}], "Val 合成回复。", emotion="joy")]
    normalized = tmp_path / "data/processed/v1"
    expected_outputs = []
    for split, rows in (("train", train), ("dev", dev), ("val", val)):
        for row in rows:
            row["source"]["split"] = "val" if split == "val" else "train"
        path = normalized / f"{split}.jsonl"
        _write_jsonl(path, rows)
        expected_outputs.append({
            "path": f"data/processed/v1/{split}.jsonl",
            "records": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest_path = tmp_path / "outputs/data_pipeline/v1/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"pipeline_version": "D0.3-v1", "policy_version": "D0.2-v1", "outputs": expected_outputs}), encoding="utf-8")

    config = {
        "version": "D0.4.2-v1",
        "inputs": {
            "normalized_dir": "data/processed/v1",
            "pipeline_manifest": "outputs/data_pipeline/v1/manifest.json",
            "train_schema": "data/train/train_public.schema.json",
            "val_schema": "data/val/val_public.schema.json",
            "submission_schema": "participant/templates/submission_schema.json",
        },
        "model": {"tokenizer_id": "Qwen/Qwen3-1.7B", "tokenizer_revision": REVISION, "local_files_only": True, "enable_thinking": False},
        "prompt": {"version": "P0", "text_path": "configs/prompts/p0.txt", "history_format": "native_chat_template", "completion_json": "compact", "memory_refs_policy": "empty_in_a0_completion", "profile_order": "official_schema_enum_order"},
        "context": {"history_strategy": "recent_first_complete_messages", "preserve_current_user_message": True, "keep_complete_completion": True},
        "simulation": {
            "candidates": [{"max_train_seq_length": 2048, "max_prompt_tokens": [1024]}],
            "percentile_method": "nearest_rank",
            "response_length_train_percentiles": [50, 90],
        },
        "deployment": {"reference_inference_config": "participant/configs/inference_config.json", "additional_generation_candidates": [], "serving_context_limits": []},
        "tokenizer": {"batch_size": 4, "verify_batch_with_scalar_samples": 4},
        "outputs": {"directory": "outputs/token_audit/v1", "report": "docs/data/D0.4.2_Report.md"},
        "validation": {"assert_inputs_unchanged": True, "unit_test_command": "python -m pytest tests/test_token_stats.py -q"},
    }
    config_path = tmp_path / "configs/data_token_audit.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    hashes_before = {path: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() for path in (
        "data/processed/v1/train.jsonl", "data/processed/v1/dev.jsonl", "data/processed/v1/val.jsonl", "outputs/data_pipeline/v1/manifest.json",
    )}
    return config_path, hashes_before, train + dev + val


def test_t01_t02_untruncated_counts_equal_direct_template_and_prefix(qwen_tokenizer):
    history = [{"role": "user", "content": "合成用户问题"}, {"role": "assistant", "content": "合成助手回复"}, {"role": "user", "content": "当前问题"}]
    messages = build_messages(history, P0)
    completion = '{"emotion_label":"anxiety","user_profile":{"personality_traits":[],"interests":[],"style":[]},"response_text":"合成回复","memory_refs":[]}'
    prompt_ids = encode_inference_prompt(qwen_tokenizer, messages)
    full_ids = encode_full_training_sequence(qwen_tokenizer, messages, completion)
    # The helper takes one consistent call mode, so compare each mode with direct encoding.
    assert apply_chat_template_batch(qwen_tokenizer, [messages], add_generation_prompt=True)[0] == prompt_ids
    assert apply_chat_template_batch(qwen_tokenizer, [[*messages, {"role": "assistant", "content": completion}]], add_generation_prompt=False)[0] == full_ids
    assert full_ids[:len(prompt_ids)] == prompt_ids
    assert len(full_ids) == len(prompt_ids) + len(full_ids[len(prompt_ids):])


def test_t03_t04_t05_t06_t08_t14_t1_suffix_is_contiguous_cached_and_target_independent(qwen_tokenizer):
    history = [
        {"role": "user", "content": "最早的一条很短消息"},
        {"role": "assistant", "content": "最后一个较早消息" * 700},
        {"role": "user", "content": "当前短消息"},
    ]
    current_len = len(encode_inference_prompt(qwen_tokenizer, build_messages([history[-1]], P0)))
    complete_len = len(encode_inference_prompt(qwen_tokenizer, build_messages(history, P0)))
    middle_pair_len = len(encode_inference_prompt(qwen_tokenizer, build_messages(history[1:], P0)))
    assert current_len < middle_pair_len < complete_len

    full = select_recent_complete_suffix(history, P0, complete_len, qwen_tokenizer)
    assert full["selected_history"] == history and not full["floor_exceeds_cap"]
    trimmed = select_recent_complete_suffix(history, P0, middle_pair_len, qwen_tokenizer)
    assert trimmed["selected_history"] == history[1:]
    no_skip = select_recent_complete_suffix(history, P0, middle_pair_len - 1, qwen_tokenizer)
    assert no_skip["selected_history"] == [history[-1]]
    floor = select_recent_complete_suffix(history, P0, current_len - 1, qwen_tokenizer)
    assert floor["floor_exceeds_cap"] and floor["selected_history"] == [history[-1]]

    # Cache-assisted planning uses only history, P0 and P; target/completion is absent from this API.
    current_key = _history_key([history[-1]])
    cache = {current_key: current_len}
    items = [
        {"sample_id": "a", "sample": {"history": history}},
        {"sample_id": "b", "sample": {"history": history}},
    ]
    for item in items:
        item["current_key"] = current_key
    choices, _ = _plan_t1_selections(items, P0, [middle_pair_len - 1, middle_pair_len], qwen_tokenizer, cache, 2)
    assert choices["a"] == choices["b"]
    assert choices["a"][middle_pair_len - 1]["selected_start_index"] == 2
    assert choices["a"][middle_pair_len]["selected_start_index"] == 1
    for cap, choice in choices["a"].items():
        direct = encode_inference_prompt(qwen_tokenizer, build_messages(history[choice["selected_start_index"]:], P0))
        assert len(direct) == choice["prompt_tokens"]


def test_t09_t10_t11_t12_full_audit_conserves_groups_is_deterministic_and_read_only(tmp_path, qwen_tokenizer):
    config_path, input_hashes_before, fixture_rows = _audit_fixture(tmp_path, qwen_tokenizer)
    first = run_audit(tmp_path, config_path)
    output_dir = tmp_path / "outputs/token_audit/v1"
    report = (tmp_path / "docs/data/D0.4.2_Report.md").read_text(encoding="utf-8")
    stats_first = (output_dir / "token_stats.json").read_bytes()
    jsonls_first = {
        name: (output_dir / name).read_bytes()
        for name in ("per_sample_lengths.jsonl", "budget_simulations.jsonl", "length_exceptions.jsonl")
    }
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert first["counts"] == {"train": 5, "dev": 1, "val": 1}
    assert first["input_hashes_unchanged"] is True
    assert manifest["input_hashes_unchanged_after_run"]

    simulation = json.loads(jsonls_first["budget_simulations.jsonl"].splitlines()[0])
    assert simulation["outcome"] in {"full_fit", "trimmed_fit", "prompt_floor_exceeds_cap", "completion_overflows_train_budget"}
    budget_summary = json.loads(stats_first)["budget_simulations"]["S2048_P1024"]
    assert budget_summary["counts_conserve"]
    assert sum(budget_summary[field] for field in (
        "n_full_fit", "n_trimmed_fit", "n_prompt_floor_exceeds_cap", "n_completion_overflows_train_budget"
    )) == 7
    assert budget_summary["n_trimmed_fit"] >= 2
    assert budget_summary["n_prompt_floor_exceeds_cap"] >= 1
    assert budget_summary["n_completion_overflows_train_budget"] >= 1
    assert "surprise" in json.loads(stats_first)["length_statistics"]["by_group"]["emotion_label"]
    assert "true" in json.loads(stats_first)["length_statistics"]["by_group"]["profile_all_empty"]
    assert report.count("P50/P90/P95/P99/Max") >= 1
    assert "not_assessable_K_unconfigured" in stats_first.decode("utf-8")

    lengths = [json.loads(line) for line in jsonls_first["per_sample_lengths.jsonl"].splitlines()]
    assert len(lengths) == len(fixture_rows)
    assert all("中间消息" not in line and "长答案" not in line for line in jsonls_first["per_sample_lengths.jsonl"].decode("utf-8").splitlines())
    for relative, expected in input_hashes_before.items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == expected

    # Re-running replaces the result transactionally; deterministic artifacts are byte-identical.
    run_audit(tmp_path, config_path)
    assert (output_dir / "token_stats.json").read_bytes() == stats_first
    for name, expected in jsonls_first.items():
        assert (output_dir / name).read_bytes() == expected


def test_t13_candidate_validation_and_missing_chat_template_fail_explicitly(qwen_tokenizer):
    with pytest.raises(ValueError, match="smaller than S"):
        _candidate_grid({"simulation": {"candidates": [{"max_train_seq_length": 512, "max_prompt_tokens": [512]}]}})
    changed = qwen_tokenizer
    original = changed.chat_template
    changed.chat_template = None
    with pytest.raises((ValueError, TypeError), match="chat_template|template"):
        apply_chat_template_batch(changed, [[{"role": "user", "content": "message"}]], add_generation_prompt=True)
    changed.chat_template = original

