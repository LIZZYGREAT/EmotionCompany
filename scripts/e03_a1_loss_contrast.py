#!/usr/bin/env python3
"""Run the isolated E03-A1 weighted-loss contrast stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import training_sft


DEFAULT_CONFIG = "configs/training/e03_a1_loss_contrast.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _apply_stage(config: dict[str, Any], stage: str, variant: str) -> dict[str, Any]:
    if variant not in {"token_mean", "profile_x2", "label_value_x2"}:
        raise ValueError(f"unknown loss variant: {variant}")
    key = "short" if stage == "short" else "full"
    budget = config["stage_budgets"][key]
    config["loss_contrast"]["variant"] = variant
    config["training_scope"] = "short_contrast" if stage == "short" else "full_contrast"
    for name in (
        "max_optimizer_steps", "max_training_seconds", "save_interval_steps",
        "evaluation_interval_steps", "parameter_check_interval_steps",
    ):
        config["training"][name] = budget[name]
    config["evaluation"]["sft_dev_loss_rows"] = budget["sft_dev_loss_rows"]
    return config


def verify_short_gate(
    candidate_root: Path,
    token_mean_root: Path,
    *,
    candidate_variant: str,
) -> dict[str, Any]:
    """Require matching, passed short Runs and the frozen Dev-loss thresholds."""
    if candidate_variant not in {"profile_x2", "label_value_x2"}:
        raise ValueError("full contrast can only follow an approved weighted short Run")
    candidate_root, token_mean_root = candidate_root.resolve(), token_mean_root.resolve()
    candidate_manifest = _read_json(candidate_root / "manifest.json")
    baseline_manifest = _read_json(token_mean_root / "manifest.json")
    expected = {
        "experiment_id": "E03-A1",
        "training_scope": "short_contrast",
        "status": "passed",
        "optimizer_steps": 128,
        "reload_check_status": "passed",
        "base_weights_unchanged": True,
    }
    for name, manifest in (("candidate", candidate_manifest), ("token_mean", baseline_manifest)):
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f"{name} short Run does not satisfy full-stage prerequisite: {field}")
    if candidate_manifest.get("loss_variant") != candidate_variant or baseline_manifest.get("loss_variant") != "token_mean":
        raise ValueError("short Run variants do not match the requested full-stage comparison")
    candidate_loss = _read_json(candidate_root / "loss_breakdown.json")
    baseline_loss = _read_json(token_mean_root / "loss_breakdown.json")
    candidate_map = _read_json(candidate_root / "loss_mapping_manifest.json")
    baseline_map = _read_json(token_mean_root / "loss_mapping_manifest.json")
    if candidate_loss.get("metric_scope") != "unweighted_teacher_forced_sft_dev" or baseline_loss.get("metric_scope") != candidate_loss.get("metric_scope"):
        raise ValueError("short Run is missing the unweighted SFT Dev loss breakdown")
    if candidate_loss.get("dev_rows") != 128 or baseline_loss.get("dev_rows") != 128:
        raise ValueError("short Run loss breakdown must use the frozen 128-row SFT Dev subset")
    if candidate_loss.get("sample_ids_sha256") != baseline_loss.get("sample_ids_sha256"):
        raise ValueError("short Runs evaluated different fixed SFT Dev IDs")
    if candidate_map.get("mapping_sha256") != baseline_map.get("mapping_sha256"):
        raise ValueError("short Runs used different A0/A0.1 token-region mappings")
    candidate_regions, baseline_regions = candidate_loss.get("regions", {}), baseline_loss.get("regions", {})
    try:
        candidate_total = float(candidate_regions["all_tokens"]["mean_ce"])
        baseline_total = float(baseline_regions["all_tokens"]["mean_ce"])
        candidate_label = float(candidate_regions["label_value"]["mean_ce"])
        baseline_label = float(baseline_regions["label_value"]["mean_ce"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("short Run loss breakdown is incomplete") from exc
    if min(candidate_total, baseline_total, candidate_label, baseline_label) <= 0:
        raise ValueError("short Run CE values must be positive")
    label_value_relative_improvement = (baseline_label - candidate_label) / baseline_label
    total_ce_relative_change = (candidate_total - baseline_total) / baseline_total
    passed = label_value_relative_improvement >= 0.02 and total_ce_relative_change <= 0.01
    result = {
        "status": "passed" if passed else "not_passed",
        "candidate_variant": candidate_variant,
        "candidate_short_run": str(candidate_root),
        "token_mean_short_run": str(token_mean_root),
        "dev_sample_ids_sha256": candidate_loss["sample_ids_sha256"],
        "mapping_sha256": candidate_map["mapping_sha256"],
        "label_value_ce_relative_improvement": label_value_relative_improvement,
        "total_ce_relative_change": total_ce_relative_change,
        "thresholds": {"label_value_ce_relative_improvement_min": 0.02, "total_ce_relative_change_max": 0.01},
    }
    if not passed:
        raise ValueError(f"short-stage Dev-loss gate did not pass: {json.dumps(result, ensure_ascii=False)}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("preflight", "train", "resume", "reload-check"), required=True)
    parser.add_argument("--stage", choices=("short", "full"), default="short")
    parser.add_argument("--variant", choices=("token_mean", "profile_x2", "label_value_x2"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-run-root")
    parser.add_argument("--run-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--short-run-root", help="Weighted short Run required before a full weighted Run")
    parser.add_argument("--token-mean-short-run-root", help="token_mean short Run required before a full weighted Run")
    parser.add_argument("--allow-e03-a1-training", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_path = training_sft.load_config(args.config)
    config = _apply_stage(config, args.stage, args.variant)
    training_sft._validate_staged_budget(config)
    training_sft.validate_training_mode(
        config, args.mode, allow_e03_a1_training=args.allow_e03_a1_training,
    )
    gate = None
    if args.stage == "full":
        if not args.short_run_root or not args.token_mean_short_run_root:
            raise ValueError("full stage requires both the weighted and token_mean short Run roots")
        gate = verify_short_gate(
            Path(args.short_run_root),
            Path(args.token_mean_short_run_root),
            candidate_variant=args.variant,
        )
    if args.mode == "preflight":
        if not args.run_id:
            raise ValueError("preflight requires --run-id")
        run_root = training_sft.execute_preflight(config, config_path, args.run_id)
    elif args.mode == "train":
        if not args.run_id or not args.preflight_run_root:
            raise ValueError("train requires --run-id and --preflight-run-root")
        run_root = training_sft.execute_staged_train(
            config, config_path, args.run_id, Path(args.preflight_run_root),
            allow_e03_a1_training=args.allow_e03_a1_training,
            stop_after_step=args.stop_after_step,
        )
    elif args.mode == "resume":
        if not args.run_root or not args.checkpoint:
            raise ValueError("resume requires --run-root and --checkpoint")
        run_root = training_sft.execute_staged_resume(
            config, config_path, Path(args.run_root), Path(args.checkpoint),
            allow_e03_a1_training=args.allow_e03_a1_training,
            stop_after_step=args.stop_after_step,
        )
    else:
        if not args.run_root:
            raise ValueError("reload-check requires --run-root")
        result = training_sft.execute_reload_check(config, Path(args.run_root))
        print(json.dumps({"reload_check": result, "short_gate": gate}, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "passed" else 2
    if gate is not None:
        (run_root / "short_stage_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_path = run_root / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["short_stage_gate"] = gate
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = _read_json(run_root / "manifest.json")
    print(json.dumps({"run_root": str(run_root), "status": manifest["status"], "loss_variant": args.variant, "stage": args.stage, "short_gate": gate}, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
