"""CPU Mock tests for explicit F3 inference in the official runner."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from participant import run_inference as runner


ROOT = Path(__file__).resolve().parents[1]
F3_CONFIG = ROOT / "participant/configs/e03_a6_f3_inference_config.json"


class FakeTokenizer:
    eos_token_id = 151645

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        assert enable_thinking is False
        text = "".join(
            f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>\n"
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(chr(int(value)) for value in ids)


def _f3(emotion: str = "anxiety", profile: str = "sensitive") -> str:
    return f"E={emotion}\nP={profile}\nI=study_exam\nS=direct\nR=我理解你现在的担心，我们可以一起理清最难的一步。"


def test_f3_runtime_protocol_is_explicit_and_locks_p0_selection():
    config = runner.load_config(F3_CONFIG)
    prompt, _schema, _enums, facts = runner.load_runtime_protocol(config)
    selection_prompt, selection_facts = runner.load_f3_selection_prompt(config)
    assert config["format_version"] == "F3"
    assert facts["version"] == "P_A6_F3"
    assert facts["selection_prompt"]["sha256"] == selection_facts["sha256"]
    assert "R=<完整回复" in prompt
    assert selection_prompt == (ROOT / "configs/prompts/p0.txt").read_text(encoding="utf-8")


def test_f3_runner_mock_writes_submission_report_raw_status_and_preserves_result_dir(monkeypatch, tmp_path):
    config = copy.deepcopy(runner.load_config(F3_CONFIG))
    config.update({
        "backend": "openai_compatible",
        "api_model": "mock-f3",
        "base_url": "http://127.0.0.1:8000/v1",
        "performance_rounds": 2,
    })
    config["context"]["max_prompt_tokens"] = 1160
    config["experiment_id"] = "E03-A6-F3-CPU-MOCK"
    tokenizer_facts = {
        "model_id": config["tokenizer"]["model_id"],
        "revision": config["tokenizer"]["revision"],
        "signature_sha256": config["tokenizer"]["expected_signature_sha256"],
        "template_sha256": config["tokenizer"]["expected_template_sha256"],
        "source": "synthetic fake tokenizer",
        "local_files_only": True,
    }
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "run_configuration_provenance", lambda: {"mock": True})
    monkeypatch.setattr(runner, "load_runtime_tokenizer", lambda *_args, **_kwargs: (FakeTokenizer(), tokenizer_facts))
    monkeypatch.setattr(runner, "resolve_raw_output_path", lambda *_args: tmp_path / "raw_f3.jsonl")

    returned_messages = []
    generations = iter([
        _f3(),
        _f3(profile="invented_label"),
    ])

    def mock_generation(_config, messages, _generation):
        returned_messages.append(messages)
        return next(generations), {
            "server_prompt_tokens": None,
            "server_completion_tokens": 32,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(runner, "_request_chat_completion_details", mock_generation)

    input_path = tmp_path / "input.jsonl"
    samples = [
        {
            "id": f"test_00000{index}_t1",
            "history": [
                {"role": "user", "content": "以前我也会担心很多。" * 5},
                {"role": "assistant", "content": "听起来这件事一直让你挂心。" * 5},
                {"role": "user", "content": "我现在很担心考试。"},
            ],
        }
        for index in (1, 2)
    ]
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
        encoding="utf-8",
    )
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "keep.txt").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(test_file=input_path, result_dir=result_dir),
    )

    runner.main()

    submission_rows = [json.loads(line) for line in (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in (tmp_path / "raw_f3.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (result_dir / "keep.txt").read_text(encoding="utf-8") == "preserve"
    assert [row["id"] for row in submission_rows] == [row["id"] for row in samples]
    assert submission_rows[0]["emotion_label"] == "anxiety"
    assert submission_rows[1]["emotion_label"] == ""
    assert all(row["memory_refs"] == [] for row in submission_rows)
    assert [message["role"] for message in returned_messages[0]] == ["system", "assistant", "user"]
    assert returned_messages[0][0]["content"] == (ROOT / "configs/prompts/p_a6_f3.txt").read_text(encoding="utf-8")
    assert report["format_version"] == "F3"
    assert report["protocol"]["completion_field_order"] == ["E", "P", "I", "S", "R"]
    assert report["raw_f3_protocol"]["strict_parse_success_count"] == 1
    assert report["raw_f3_protocol"]["strict_parse_success_rate_over_all_inputs"] == 0.5
    assert report["raw_f3_protocol"]["final_submission_schema_validated_count"] == 2
    assert report["failure_counts"]["invalid_profile_label"] == 1
    assert [row["f3_parse_status"] for row in raw_rows] == ["success", "invalid_profile_label"]
    assert raw_rows[1]["strict_format_parse_success"] is False
    assert report["sample_diagnostics"][0]["selection_prompt_tokens"] is not None


def test_f3_resume_identity_rejects_a_changed_format(tmp_path):
    config = runner.load_config(F3_CONFIG)
    test_file = tmp_path / "resume-input.jsonl"
    test_file.write_text('{"id":"test_000001_t1","history":[{"role":"user","content":"x"}]}\n', encoding="utf-8")
    result_dir = tmp_path / "resume"
    result_dir.mkdir()
    submission = result_dir / "submission.jsonl"
    raw = result_dir / "raw.jsonl"
    samples = runner.read_jsonl(test_file)
    completed, *_ = runner._prepare_resume_prefix(
        result_dir=result_dir,
        submission_path=submission,
        raw_path=raw,
        samples=samples,
        test_file=test_file,
        config=config,
        run_id="same-run",
        experiment_id="E03-A6-F3",
    )
    assert completed == 0
    changed = copy.deepcopy(config)
    changed["format_version"] = "F2"
    import pytest

    with pytest.raises(ValueError, match="resume input, model config, Run ID"):
        runner._prepare_resume_prefix(
            result_dir=result_dir,
            submission_path=submission,
            raw_path=raw,
            samples=samples,
            test_file=test_file,
            config=changed,
            run_id="same-run",
            experiment_id="E03-A6-F3",
        )
