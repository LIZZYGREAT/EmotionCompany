#!/usr/bin/env python3
"""Create a small deterministic, synthetic E00 run using the official parser."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as runner
from src.evaluation import jsonl_bytes, load_submission_schema, schema_enums, sha256_file

def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fixture_target(emotion: str, response: str, *, profile: dict[str, list[str]] | None = None) -> dict[str, Any]:
    return {
        "emotion_label": emotion,
        "user_profile": profile or {"personality_traits": [], "interests": [], "style": []},
        "response_text": response,
        "memory_refs": [],
    }


def create_fixture(config: dict[str, Any], config_sha256: str, output_dir: Path) -> dict[str, Any]:
    schema = load_submission_schema(ROOT)
    enums = schema_enums(schema)
    examples = [
        {
            "history": [{"role": "user", "content": "我完成了一件开心的事🙂。"}],
            "target": _fixture_target("joy", "做得很好呀！"),
            "raw_text": json.dumps(_fixture_target("joy", "做得很好呀！"), ensure_ascii=False),
        },
        {
            "history": [{"role": "user", "content": "我有点害怕。"}],
            "target": _fixture_target("fear", "我会陪着你。", profile={"personality_traits": [], "interests": ["music"], "style": []}),
            "raw_text": "```json\n" + json.dumps(_fixture_target("fear", "我会陪着你。", profile={"personality_traits": [], "interests": ["music"], "style": []}), ensure_ascii=False) + "\n```",
        },
        {
            "history": [{"role": "user", "content": "最近状态还不错。"}],
            "target": _fixture_target("anxiety", "那真不错。", profile={"personality_traits": ["open"], "interests": [], "style": []}),
            "raw_text": json.dumps(_fixture_target("anxiety", "那真不错。", profile={"personality_traits": ["open"], "interests": [], "style": []}), ensure_ascii=False) + "\nMock diagnostic suffix",
        },
        {
            "history": [{"role": "user", "content": "我感觉有些难过。"}],
            "target": _fixture_target("sadness", "愿意说说发生了什么吗？", profile={"personality_traits": [], "interests": [], "style": ["brief"]}),
            "raw_text": "not json",
        },
    ]
    inference_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    id_map: list[dict[str, Any]] = []
    submission: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, 1):
        eval_id = f"test_{index:06d}_t0"
        sample_id = f"synthetic:fixture:{index:03d}"
        inference_rows.append({"id": eval_id, "history": example["history"]})
        references.append({"eval_id": eval_id, "sample_id": sample_id, "target": example["target"]})
        id_map.append({"eval_id": eval_id, "sample_id": sample_id, "source": {"split": "synthetic_fixture", "index": index}})
        prediction, _parse_failure, _invalid_label = runner.parse_prediction(eval_id, example["raw_text"], enums)
        runner.validate_submission_prediction(prediction, schema)
        submission.append(prediction)
        raw_rows.append({
            "id": eval_id,
            "raw_text": example["raw_text"],
            "status": "parse_failure" if example["raw_text"] == "not json" else "success",
            "prompt_tokens": None,
            "generated_tokens": None,
            "stop_reason": "mock_not_run",
            "latency_ms": None,
            "removed_history_messages": None,
            "backend": "mock",
            "error_type": None,
        })

    file_rows = {
        "inference_input.jsonl": inference_rows,
        "references.jsonl": references,
        "id_map.jsonl": id_map,
        "submission.jsonl": submission,
        "raw_predictions.jsonl": raw_rows,
    }
    payloads = {name: jsonl_bytes(rows) for name, rows in file_rows.items()}
    output_facts = {
        name: {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(file_rows[name]),
        }
        for name, data in payloads.items()
    }
    manifest = {
        "manifest_version": "E00.1-synthetic-fixture-v1",
        "experiment_id": "E00.1",
        "dataset_scope": "synthetic_fixture",
        "source": {"fixture": "fabricated public-safe rows; never used as model input outside this directory"},
        "config_sha256": config_sha256,
        "planned_config": {
            "model": config.get("model"),
            "quantization": config.get("quantization", {}).get("planned"),
            "generation": config.get("generation"),
        },
        "observed_runtime": {"backend": "mock", "real_model_loaded": False, "generation_run": False},
        "outputs": output_facts,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(f"Fixture 输出目录非空且没有本工具 Manifest，拒绝覆盖: {output_dir}")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("outputs") != output_facts or old.get("config_sha256") != config_sha256:
            raise FileExistsError(f"Fixture 输出目录含有其他数据，拒绝覆盖: {output_dir}")
    for name, data in payloads.items():
        _atomic_write(output_dir / name, data)
    run_metadata = {
        "run_id": "synthetic-e00-1",
        "experiment_id": "E00.1",
        "backend": "mock",
        "hardware_label": "not_applicable_synthetic_fixture",
        "timing_scope": "Synthetic fixture; no model inference or timing was run",
        "complete": False,
        "required_rounds": config.get("generation", {}).get("performance_rounds", 100),
        "rounds": 0,
        "average_latency_ms": None,
        "median_latency_ms": None,
        "p95_latency_ms": None,
        "latencies_ms": [],
        "test_file_sha256": output_facts["inference_input.jsonl"]["sha256"],
        "submission_sha256": output_facts["submission.jsonl"]["sha256"],
        "raw_predictions_sha256": output_facts["raw_predictions.jsonl"]["sha256"],
        "samples": len(examples),
        "submission_rows": len(examples),
        "submission_schema_validated": True,
        "model_id": "synthetic_fixture_only",
        "model_revision": None,
        "model_weight_identity": {"fingerprint_status": "not_applicable_synthetic_fixture"},
        "quantization": {"applied": False, "method": "not run"},
        "generation_config": {"not_run": True},
        "protocol": {"prompt": {"version": "P0", "sha256": config["protocol"]["prompt_sha256"]}},
    }
    _atomic_write(manifest_path, manifest_bytes)
    _atomic_write(output_dir / "performance_report.json", (json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成离线 synthetic/mock E00 评分验收样例")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest = create_fixture(
        config,
        hashlib.sha256(config_bytes).hexdigest(),
        args.output_dir.resolve(),
    )
    print(
        f"PASS: dataset_scope=synthetic_fixture backend=mock rows="
        f"{manifest['outputs']['references.jsonl']['rows']} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
