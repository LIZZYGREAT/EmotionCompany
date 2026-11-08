from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from participant.final_dual_inference import _resume_identity, read_test_samples


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    content = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(content)
    return content


def _samples() -> list[dict]:
    return [
        {"id": "test_000001_t1", "history": [{"role": "user", "content": "one"}]},
        {"id": "test_000002_t2", "history": [{"role": "user", "content": "two"}]},
    ]


def test_read_test_samples_accepts_only_official_target_free_input(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(_samples()[0]) + "\n", encoding="utf-8")
    assert read_test_samples(path) == _samples()[:1]

    path.write_text(json.dumps({**_samples()[0], "target": {}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only id and history"):
        read_test_samples(path)


@pytest.mark.parametrize("sample_id", ["val:1", "test_1_t1", "test_000001_t1"])
def test_read_test_samples_rejects_bad_or_duplicate_ids(tmp_path: Path, sample_id: str) -> None:
    path = tmp_path / "input.jsonl"
    row = {"id": sample_id, "history": [{"role": "user", "content": "one"}]}
    if sample_id == "test_000001_t1":
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid or duplicate"):
            read_test_samples(path)
    else:
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid or duplicate"):
            read_test_samples(path)


def test_resume_trims_one_file_ahead_to_last_fsynced_progress(tmp_path: Path, monkeypatch) -> None:
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, _samples())
    config_path = tmp_path / "config.json"
    config_path.write_text('{"backend":"vllm"}\n', encoding="utf-8")
    lock_path = tmp_path / "model_lock.json"
    lock_path.write_text('{"version":"lock"}\n', encoding="utf-8")
    result_dir = tmp_path / "result"
    samples = _samples()

    completed, _, _, progress = _resume_identity(
        result_dir=result_dir,
        test_file=test_file,
        config_path=config_path,
        lock_path=lock_path,
        samples=samples,
        backend="vllm",
        resume_enabled=False,
    )
    assert completed == 0
    assert progress["status"] == "running"

    committed_submission = {"id": samples[0]["id"], "value": "first"}
    committed_raw = {"id": samples[0]["id"], "sample_end_to_end_ms": 12.5}
    submission_prefix = _write_jsonl(result_dir / "submission.jsonl", [committed_submission])
    raw_prefix = _write_jsonl(result_dir / "raw_outputs.jsonl", [committed_raw])
    progress.update(
        completed_samples=1,
        submission_prefix_sha256=hashlib.sha256(submission_prefix).hexdigest(),
        raw_prefix_sha256=hashlib.sha256(raw_prefix).hexdigest(),
    )
    (result_dir / "run_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    # Simulate interruption after the submission append but before the raw append/progress update.
    with (result_dir / "submission.jsonl").open("ab") as handle:
        handle.write(_write_jsonl(tmp_path / "extra.jsonl", [{"id": samples[1]["id"]}]))

    monkeypatch.delenv("EVALUATION_RUN_ID", raising=False)
    completed, submissions, raw_rows, resumed = _resume_identity(
        result_dir=result_dir,
        test_file=test_file,
        config_path=config_path,
        lock_path=lock_path,
        samples=samples,
        backend="vllm",
        resume_enabled=True,
    )

    assert completed == 1
    assert [row["id"] for row in submissions] == [samples[0]["id"]]
    assert [row["id"] for row in raw_rows] == [samples[0]["id"]]
    assert resumed["attempt"] == 2
    assert len((result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_resume_rejects_changed_input_identity(tmp_path: Path) -> None:
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, _samples())
    config_path = tmp_path / "config.json"
    config_path.write_text('{"backend":"vllm"}\n', encoding="utf-8")
    lock_path = tmp_path / "model_lock.json"
    lock_path.write_text('{"version":"lock"}\n', encoding="utf-8")
    result_dir = tmp_path / "result"
    _resume_identity(
        result_dir=result_dir,
        test_file=test_file,
        config_path=config_path,
        lock_path=lock_path,
        samples=_samples(),
        backend="vllm",
        resume_enabled=False,
    )
    test_file.write_text(test_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input.*changed"):
        _resume_identity(
            result_dir=result_dir,
            test_file=test_file,
            config_path=config_path,
            lock_path=lock_path,
            samples=_samples(),
            backend="vllm",
            resume_enabled=True,
        )
