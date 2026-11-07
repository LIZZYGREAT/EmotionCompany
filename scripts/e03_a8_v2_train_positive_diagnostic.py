#!/usr/bin/env python3
"""Run an isolated A8-V2 diagnostic on already-consumed positive Train rows."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e03_a8_v2_staged_eval as frozen_eval
from scripts.e03_a8_staged_eval import _append_jsonl_fsynced, _generate_rows
from src.data.e03_a8_sft import canonical_sha256
from src.evaluation.e03_a6_f4_staged_metrics import selected_ids_sha256, sha256_file
from src.evaluation.e03_a8_v2_staged_metrics import score_a8_cohort
from src.shared.e03_a8_protocol import validate_a8_prompt_enum_order
from src.training.e03_a8_staged import _read_json, _read_jsonl, _write_json
from src.training.e03_a8_v2_staged import (
    DEFAULT_CONFIGS,
    _base_path,
    _load_config as _load_training_config,
    _resolve,
    _runtime_tokenizer,
)


def evaluate_consumed_train_positives(arm: str, run_id: str, step: int, count: int = 64) -> dict:
    if arm != "P" or step not in {512, 1024, 2048, 4048} or count != 64:
        raise ValueError("A8-V2 diagnostic requires P, a reviewed node, and exactly 64 positives")
    training_config, _ = _load_training_config(DEFAULT_CONFIGS[arm], arm)
    eval_config, _ = frozen_eval._load_eval_config()
    training_manifest, data_facts, training_root, checkpoint = frozen_eval._validate_run(
        training_config, run_id, step
    )
    dataset_manifest = _read_json(_resolve(training_config["data"]["dataset_manifest"]))
    normalized_path = _resolve(dataset_manifest["inputs"]["normalized_train"]["path"])
    schedule = _read_jsonl(_resolve(training_config["data"]["sample_order_path"]))
    consumed_ids = {
        sample_id
        for schedule_row in schedule[:step]
        for sample_id in schedule_row["sample_ids"]
    }
    selected_rows = []
    with normalized_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sample_id") not in consumed_ids:
                continue
            profile = row["target"]["user_profile"]
            if any(profile[dimension] for dimension in ("personality_traits", "interests", "style")):
                selected_rows.append(row)
                if len(selected_rows) == count:
                    break
    if len(selected_rows) != count:
        raise ValueError(f"P Step {step} has fewer than {count} consumed nonempty profiles")

    diagnostic_ids = [f"train-positive:{row['sample_id']}" for row in selected_rows]
    mapping = {
        diagnostic_id: {"sample_id": row["sample_id"], "conversation_id": row["conversation_id"]}
        for diagnostic_id, row in zip(diagnostic_ids, selected_rows)
    }
    cohorts = {
        "cohorts": {"train_positive64": {"mapping": mapping}},
        "union_count": count,
        "union_ids": diagnostic_ids,
        "union_ids_sha256": selected_ids_sha256(diagnostic_ids),
        "union_inputs": [
            {"id": diagnostic_id, "history": row["history"]}
            for diagnostic_id, row in zip(diagnostic_ids, selected_rows)
        ],
    }
    output_root = _resolve(eval_config["output_root"]) / "train_positive" / arm / run_id / f"step-{step:04d}"
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Train-positive diagnostic: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    p_prompt = (ROOT / "configs/prompts/p_a8_v2_profile.txt").read_text(encoding="utf-8")
    ep_prompt = (ROOT / "configs/prompts/p_a8_v2_emotion_profile.txt").read_text(encoding="utf-8")
    p0_path = _resolve(eval_config["protocol"]["p0_path"])
    schema = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))
    enums = validate_a8_prompt_enum_order(p_prompt, schema, "P")
    validate_a8_prompt_enum_order(ep_prompt, schema, "EP")
    tokenizer, tokenizer_facts = _runtime_tokenizer(training_config)
    model = None
    try:
        from src.training.training_sft import _load_train_model

        model, torch, model_facts = _load_train_model(
            training_config,
            _base_path(training_config),
            adapter_path=checkpoint,
            autocast_adapter_dtype=True,
        )
        model.eval()
        generated, generation_facts = _generate_rows(
            config=training_config,
            eval_config=eval_config,
            run_id=run_id,
            step=step,
            cohorts=cohorts,
            tokenizer=tokenizer,
            model=model,
            p_prompt=p_prompt,
            ep_prompt=ep_prompt,
            p0_prompt=p0_path.read_text(encoding="utf-8"),
            schema=schema,
        )
        prediction_path = output_root / "predictions.jsonl"
        _append_jsonl_fsynced(prediction_path, generated)
        prediction_sha = sha256_file(prediction_path)
        prediction_index = {row["id"]: row for row in generated}
        if set(prediction_index) != set(diagnostic_ids):
            raise RuntimeError("Train-positive prediction coverage differs from the fixed IDs")

        source_ids = [row["sample_id"] for row in selected_rows]
        generation_manifest = {
            "status": "predictions_frozen",
            "experiment_id": "E03-A8-V2",
            "diagnostic": "consumed_train_nonempty_profiles",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "consumed_optimizer_steps": step,
            "consumed_sample_count": step * int(training_config["training"]["gradient_accumulation_steps"]),
            "sample_count": count,
            "sample_ids_sha256": canonical_sha256(source_ids),
            "prediction_sha256": prediction_sha,
            "diagnostic_script_sha256": sha256_file(Path(__file__)),
            "checkpoint_manifest_sha256": training_manifest["checkpoint_manifest_sha256"],
            "references_read": False,
            "created_at_utc": frozen_eval._now(),
        }
        _write_json(output_root / "generation_manifest.json", generation_manifest, exclusive=True)
        references = {
            diagnostic_id: {"target": row["target"]}
            for diagnostic_id, row in zip(diagnostic_ids, selected_rows)
        }
        diagnostic_cohort = {
            "name": "train_positive64",
            "ids": diagnostic_ids,
            "ids_sha256": selected_ids_sha256(diagnostic_ids),
            "conversation_count": len({row["conversation_id"] for row in selected_rows}),
        }
        metrics = score_a8_cohort(diagnostic_cohort, references, prediction_index, enums, arm=arm)
        result = {
            "status": "complete",
            "experiment_id": "E03-A8-V2",
            "diagnostic": "consumed_train_nonempty_profiles",
            "arm": arm,
            "run_id": run_id,
            "checkpoint_step": step,
            "sample_count": count,
            "sample_ids_sha256": generation_manifest["sample_ids_sha256"],
            "prediction_sha256": prediction_sha,
            "checkpoint_manifest_sha256": generation_manifest["checkpoint_manifest_sha256"],
            "diagnostic_script_sha256": generation_manifest["diagnostic_script_sha256"],
            "references_read_after_prediction_freeze": True,
            "metrics": metrics,
            "generation": generation_facts,
            "tokenizer_facts": tokenizer_facts,
            "model_facts": model_facts,
            "created_at_utc": frozen_eval._now(),
        }
        result_path = output_root / "train_positive_evaluation.json"
        _write_json(result_path, result, exclusive=True)
        generation_manifest.update(
            status="complete",
            references_read_after_prediction_freeze=True,
            metrics_sha256=sha256_file(result_path),
        )
        _write_json(output_root / "generation_manifest.json", generation_manifest)
        return result
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("P",), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, choices=(512, 1024, 2048, 4048), required=True)
    args = parser.parse_args()
    result = evaluate_consumed_train_positives(args.arm, args.run_id, args.step)
    print(json.dumps({
        "status": result["status"],
        "arm": result["arm"],
        "step": result["checkpoint_step"],
        "sample_count": result["sample_count"],
        "prediction_sha256": result["prediction_sha256"],
        "metrics": result["metrics"]["profile_33_label"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
