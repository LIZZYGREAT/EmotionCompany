"""Create explicit non-model predictions to exercise real D0.3 Smoke scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant.run_inference import load_submission_schema
from src.evaluation import jsonl_bytes, read_jsonl, schema_enums, sha256_file


def _atomic_write(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_mock_run(input_path: Path, output_dir: Path, run_id: str, config_path: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    outputs_root = (ROOT / "outputs" / "evaluation").resolve()
    if not output_dir.is_relative_to(outputs_root):
        raise ValueError(f"Mock 验收产物只能写入 outputs/evaluation/: {output_dir}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Smoke 输入缺失: {input_path}")
    if not run_id.strip():
        raise ValueError("run_id 不能为空")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = ("submission.jsonl", "raw_predictions.jsonl", "performance_report.json")
    existing = [name for name in output_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Mock 输出目录已有文件，拒绝覆盖: {output_dir}: {existing}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema = load_submission_schema()
    validator = Draft202012Validator(schema)
    emotion = schema_enums(schema)["emotion_label"][0]
    input_rows = read_jsonl(input_path)
    seen_ids: set[str] = set()
    submissions: list[dict[str, Any]] = []
    raw_predictions: list[dict[str, Any]] = []
    for index, row in enumerate(input_rows, 1):
        if set(row) != {"id", "history"}:
            raise ValueError(f"Smoke 输入第 {index} 行字段无效")
        eval_id = row["id"]
        if not isinstance(eval_id, str) or not eval_id or eval_id in seen_ids:
            raise ValueError(f"Smoke 输入第 {index} 行 id 缺失或重复")
        seen_ids.add(eval_id)
        completion = {
            "emotion_label": emotion,
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "response_text": "这是用于评测流程校验的合成回复。",
            "memory_refs": [],
        }
        submission = {"id": eval_id, **completion}
        errors = list(validator.iter_errors(submission))
        if errors:
            raise ValueError(f"Mock Submission 不符合官方 Schema: {eval_id}: {errors[0].message}")
        submissions.append(submission)
        raw_predictions.append({
            "id": eval_id,
            "raw_text": json.dumps(completion, ensure_ascii=False, separators=(",", ":")),
            "status": "success",
            "prompt_tokens": None,
            "generated_tokens": None,
            "stop_reason": "mock_fixture",
            "latency_ms": None,
            "removed_history_messages": None,
            "backend": "mock",
            "error_type": None,
        })

    submission_path = output_dir / "submission.jsonl"
    raw_path = output_dir / "raw_predictions.jsonl"
    report_path = output_dir / "performance_report.json"
    submission_bytes = jsonl_bytes(submissions)
    raw_bytes = jsonl_bytes(raw_predictions)
    _atomic_write(submission_path, submission_bytes)
    _atomic_write(raw_path, raw_bytes)
    report = {
        "run_id": run_id,
        "experiment_id": "E00.1-mock-smoke-acceptance",
        "execution_mode": "mock_fixture",
        "backend": "mock",
        "model_id": None,
        "model_revision": None,
        "model_weight_identity": {
            "identity_status": "not_applicable",
            "fingerprint_status": "mock_fixture_no_model_loaded",
            "observed_weight_hash": None,
        },
        "hardware_label": "not_applicable_mock",
        "timing_scope": "Mock fixture only; no model inference or latency measurement",
        "test_file_sha256": sha256_file(input_path),
        "submission_sha256": sha256_file(submission_path),
        "raw_predictions_sha256": sha256_file(raw_path),
        "raw_predictions_path": str(raw_path),
        "raw_predictions_rows": len(raw_predictions),
        "samples": len(submissions),
        "submission_rows": len(submissions),
        "rounds": 0,
        "complete": False,
        "average_latency_ms": None,
        "median_latency_ms": None,
        "p95_latency_ms": None,
        "generation_config": None,
        "config_file": str(config_path.resolve()),
        "model_config_note": "The E00 model config is recorded as context; this run did not load it or generate predictions.",
    }
    _atomic_write(report_path, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="target-free Smoke input JSONL")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory under outputs/evaluation/")
    parser.add_argument("--run-id", required=True, help="unique mock run identifier")
    parser.add_argument("--config", default=ROOT / "configs/evaluation/e00.yaml", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_mock_run(args.input, args.output_dir, args.run_id, args.config)
    print(f"PASS: wrote {report['samples']} explicit Mock rows under {args.output_dir}")
    print("No model was loaded; these predictions are pipeline fixtures, not model quality evidence.")


if __name__ == "__main__":
    main()
