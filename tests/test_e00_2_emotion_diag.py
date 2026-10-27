from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import e00_2_emotion_diag as diag


LABELS = diag._official_emotions()


class FakeTokenizer:
    eos_token_id = 151645

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        size = 10 + sum(len(message["content"]) for message in messages)
        return {"input_ids": list(range(size))}


@pytest.mark.parametrize("label", LABELS)
def test_every_official_emotion_is_a_strict_single_label(label):
    assert diag.validate_single_label(label, LABELS) == (label, None)
    assert diag.validate_single_label(f" \n{label}\t ", LABELS) == (label, None)


@pytest.mark.parametrize(
    ("raw_text", "stop_reason", "error"),
    [
        ("", "eos", "empty_output"),
        (None, "generation_failed", "raw_text_missing_or_non_string"),
        ("happiness", "eos", "non_enumerated_output"),
        ("快乐", "eos", "non_enumerated_output"),
        ('"joy"', "eos", "quoted_label"),
        ("{\"emotion_label\":\"joy\"}", "eos", "json_or_container_output"),
        ("```text\njoy\n```", "eos", "code_fence"),
        ("joy, sadness", "eos", "multiple_labels_or_explanation"),
        ("Emotion: joy", "eos", "label_with_extra_content"),
        ("helpless", "max_new_tokens", "truncated_at_generation_cap"),
    ],
)
def test_single_label_parser_rejects_wrappers_mappings_and_truncations(raw_text, stop_reason, error):
    assert diag.validate_single_label(raw_text, LABELS, stop_reason) == (None, error)


def test_diagnostic_prompt_enum_order_and_hash_are_frozen():
    config = diag._read_yaml(diag.ROOT / "configs/evaluation/e00_2_emotion_diag.yaml")
    prompt_path = diag._resolve(config["prompt_path"])
    raw = prompt_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config["prompt_sha256"]
    diag._validate_diag_prompt(raw.decode("utf-8"), LABELS)
    with pytest.raises(ValueError, match="枚举及顺序"):
        diag._validate_diag_prompt("x\n" + ", ".join(reversed(LABELS)), LABELS)


def _history_fixtures():
    tokenizer = FakeTokenizer()
    history = [
        {"role": "user", "content": "old-message-" * 5},
        {"role": "assistant", "content": "answer-" * 4},
        {"role": "user", "content": "current"},
    ]
    p0 = "P0 system prompt " * 3
    prompt = "emotion-only"
    cap = 100
    selection = diag.select_recent_complete_suffix(history, p0, cap, tokenizer)
    assert selection["status"] == "ok"
    assert selection["selected_history"] == history[-2:]
    perf = {
        "sample-1": {
            "original_history_messages": len(history),
            "retained_history_messages": len(selection["selected_history"]),
            "removed_history_messages": len(history) - len(selection["selected_history"]),
            "client_prompt_tokens": selection["prompt_tokens"],
        }
    }
    budget = {
        "sample-ref": {
            "original_history_message_count": len(history),
            "retained_history_message_count": len(selection["selected_history"]),
            "removed_history_message_count": len(history) - len(selection["selected_history"]),
            "selected_prompt_tokens": selection["prompt_tokens"],
        }
    }
    rows = [{"id": "sample-1", "history": history}]
    return tokenizer, history, p0, prompt, cap, perf, budget, rows


def test_shorter_diagnostic_prompt_reuses_exact_a0_history_without_expansion():
    tokenizer, history, p0, prompt, cap, perf, budget, rows = _history_fixtures()
    confirmed = diag.confirm_fixed_histories(
        rows, perf, budget, {"sample-1": "sample-ref"}, tokenizer, p0, prompt, cap
    )
    facts = confirmed["sample-1"]
    assert facts["selected_history"] == history[-2:]
    assert facts["retained_history_messages"] == 2
    assert facts["diag_prompt_tokens"] < facts["a0_prompt_tokens"]
    final_messages, options = tokenizer.calls[-1]
    assert final_messages[0] == {"role": "system", "content": prompt}
    assert [(message["role"], message["content"]) for message in final_messages[1:]] == [
        (message["role"], message["content"]) for message in history[-2:]
    ]
    assert options["enable_thinking"] is False
    assert final_messages[1]["reasoning_content"] == ""


@pytest.mark.parametrize("which", ["retained_history_messages", "client_prompt_tokens"])
def test_fixed_history_rejects_a0_performance_metadata_mismatch(which):
    tokenizer, _, p0, prompt, cap, perf, budget, rows = _history_fixtures()
    perf["sample-1"][which] += 1
    with pytest.raises(ValueError, match=which):
        diag.confirm_fixed_histories(
            rows, perf, budget, {"sample-1": "sample-ref"}, tokenizer, p0, prompt, cap
        )


def test_fixed_history_rejects_budget_or_id_mapping_mismatch():
    tokenizer, _, p0, prompt, cap, perf, budget, rows = _history_fixtures()
    budget["sample-ref"]["selected_prompt_tokens"] += 1
    with pytest.raises(ValueError, match="selected_prompt_tokens"):
        diag.confirm_fixed_histories(
            rows, perf, budget, {"sample-1": "sample-ref"}, tokenizer, p0, prompt, cap
        )
    tokenizer, _, p0, prompt, cap, perf, budget, rows = _history_fixtures()
    with pytest.raises(ValueError, match="Smoke ID map"):
        diag.confirm_fixed_histories(rows, perf, budget, {}, tokenizer, p0, prompt, cap)


def test_fixed_history_rejects_new_prompt_over_budget():
    tokenizer, _, p0, _, cap, perf, budget, rows = _history_fixtures()
    with pytest.raises(ValueError, match="超出"):
        diag.confirm_fixed_histories(
            rows, perf, budget, {"sample-1": "sample-ref"}, tokenizer, p0, "x" * 200, cap
        )


def test_run_input_schema_rejects_any_gold_or_extra_field():
    with pytest.raises(ValueError, match="只能包含 id/history"):
        diag.validate_unlabeled_inputs([{"id": "one", "history": [{"role": "user", "content": "x"}], "target": {}}])


def test_weight_and_quantization_gates_reject_wrong_identity_cpu_and_offload():
    with pytest.raises(ValueError, match="权重身份不符"):
        diag.validate_model_identity({"identity_status": "observed", "aggregate_sha256": "wrong"}, "expected")
    passed = {
        "cuda_available": True,
        "real_nf4_verified": True,
        "placement_verified": True,
        "offload_devices": [],
        "accelerate_offload_hook_modules": [],
    }
    with pytest.raises(RuntimeError, match="CUDA 不可用"):
        diag.validate_quantization_runtime(passed, False)
    with pytest.raises(RuntimeError, match="offload"):
        diag.validate_quantization_runtime({**passed, "offload_devices": ["cpu"]}, True)
    diag.validate_quantization_runtime(passed, True)


def test_scoring_keeps_invalid_outputs_in_twenty_row_denominator_and_uses_fixed_sixteen_classes():
    references = []
    raw_rows = []
    baseline = {}
    history = {}
    for index in range(20):
        sample_id = f"test_{index:06d}_t0"
        label = LABELS[index % len(LABELS)]
        references.append({"eval_id": sample_id, "target": {"emotion_label": label}})
        raw_rows.append({
            "id": sample_id,
            "raw_text": '"joy"' if index == 0 else label,
            "status": "success",
            "stop_reason": "eos",
        })
        baseline[sample_id] = {"id": sample_id, "emotion_label": label if index == 1 else ""}
        history[sample_id] = {
            "history_was_clipped": index < 4,
            "a0_prompt_tokens": 500,
            "diag_prompt_tokens": 300,
            "retained_history_messages": 2,
        }
    metrics, cases = diag.score_prediction_rows(raw_rows, references, baseline, history, LABELS)
    assert metrics["sample_count"] == 20
    assert metrics["strict_label_valid_count"] == 19
    assert metrics["invalid_count"] == 1
    assert metrics["correct_count"] == 19
    assert metrics["accuracy"] == pytest.approx(19 / 20)
    assert metrics["accuracy_when_strictly_valid"] == 1
    assert len(metrics["per_label"]) == 16
    assert metrics["diagnostic_confusion_matrix"]["labels"][-1] == "__invalid_or_empty_prediction__"
    assert metrics["history_groups"]["clipped"]["sample_count"] == 4
    assert metrics["history_groups"]["not_clipped"]["sample_count"] == 16
    assert "raw_text" not in cases[0]


def test_scoring_rejects_missing_or_duplicate_ids():
    rows = [{"id": "one", "raw_text": "joy"}]
    with pytest.raises(ValueError, match="ID 集合不一致"):
        diag.score_prediction_rows(rows, [], {"one": {}}, {"one": {}}, LABELS)
    with pytest.raises(ValueError, match="重复 id"):
        diag._index([{"id": "one"}, {"id": "one"}], "id", "Raw")


def test_run_with_unavailable_cuda_records_block_without_reading_references(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs/evaluation/e00_2/emotion_diag"
    monkeypatch.setattr(diag, "RUNS_ROOT", output_root)
    history = [{"role": "user", "content": "private conversation"}]
    facts = {
        "selected_history": history,
        "a0_prompt_tokens": 100,
        "diag_prompt_tokens": 50,
        "original_history_messages": 1,
        "retained_history_messages": 1,
        "removed_history_messages": 0,
        "history_was_clipped": False,
    }
    config = {
        "experiment_id": "E00.2-D",
        "prompt_version": "Pdiag-E1",
        "prompt_path": "unused.txt",
        "baseline_run_id": "E00.2-test",
        "references_path": "missing-gold-must-not-be-read.jsonl",
        "expected_model_weight_sha256": "expected",
        "generation": {"do_sample": False, "max_new_tokens": 64},
        "history_policy": "exact_E00.2_A0_greedy_selected_history",
        "prompt_budget": 1280,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    prompt = "只输出一个英文标签本身。\n" + ", ".join(LABELS)
    context = {
        "config": config,
        "labels": LABELS,
        "prompt": prompt,
        "prompt_sha256": "prompt-sha",
        "input_path": tmp_path / "input.jsonl",
        "input_sha256": "input-sha",
        "input_rows": [{"id": "sample-1", "history": history}],
        "histories": {"sample-1": facts},
        "baseline": {"run_id": "baseline"},
        "baseline_path": tmp_path / "baseline.json",
        "baseline_manifest_sha256": "baseline-sha",
        "baseline_predictions": {},
        "baseline_emotions": {},
        "evaluation_manifest_sha256": "eval-sha",
        "d042_manifest_sha256": "d042-sha",
        "tokenizer_facts": {"signature_sha256": "tokenizer-sha", "template_sha256": "template-sha"},
        "tokenizer": FakeTokenizer(),
        "model_path": tmp_path / "model",
        "model_index_sha256": "index-sha",
        "inference_config": {},
        "references_expected_sha256": "references-sha",
    }
    monkeypatch.setattr(diag, "load_preflight", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(diag, "_cuda_available", lambda: False)
    result = diag.run_diagnostic(config_path, "E00.2-D-emotion-test", output_root / "E00.2-D-emotion-test")
    assert result == 2
    run_dir = output_root / "E00.2-D-emotion-test"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "blocked_in_codex_env"
    assert manifest["references_sha256_expected"] == "references-sha"
    assert not (run_dir / "raw_emotion_predictions.jsonl").exists()
    assert not (run_dir / "metrics.json").exists()


def test_run_refuses_to_overwrite_an_existing_output_directory(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs/evaluation/e00_2/emotion_diag"
    existing = output_root / "E00.2-D-emotion-existing"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(diag, "RUNS_ROOT", output_root)
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        diag.run_diagnostic(tmp_path / "missing.yaml", "E00.2-D-emotion-existing", existing)
    assert marker.read_text(encoding="utf-8") == "keep"
