#!/usr/bin/env python3
"""Run isolated A8-V2 preflight, equivalence dry-run, or Step 512 training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.e03_a8_v2_staged import DEFAULT_CONFIGS, execute_continue, execute_gpu_dry_step, execute_start, prepare_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an E03-A8-V2 P/EP training stage or authorized P continuation.")
    parser.add_argument("--mode", choices=("preflight", "gpu-dry-run", "start", "continue"), required=True)
    parser.add_argument("--arm", choices=("P", "EP"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--dry-run-dir", type=Path)
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--allow-gpu-dry-run", action="store_true")
    parser.add_argument("--allow-first-stage-training", action="store_true")
    parser.add_argument("--allow-1024-continuation", action="store_true")
    parser.add_argument("--allow-2048-continuation", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config or DEFAULT_CONFIGS[args.arm]
    if args.mode == "preflight":
        result = prepare_preflight(config_path, args.run_id)
    elif args.mode == "gpu-dry-run":
        if args.preflight_dir is None:
            parser.error("gpu-dry-run requires --preflight-dir")
        result = execute_gpu_dry_step(config_path, args.run_id, args.preflight_dir, allow_gpu_dry_run=args.allow_gpu_dry_run)
    elif args.mode == "start":
        if args.preflight_dir is None or args.dry_run_dir is None or args.stop_at_step is None:
            parser.error("start requires --preflight-dir, --dry-run-dir, and --stop-at-step")
        result = execute_start(
            config_path,
            args.run_id,
            args.preflight_dir,
            args.dry_run_dir,
            stop_at_step=args.stop_at_step,
            allow_first_stage_training=args.allow_first_stage_training,
        )
    else:
        if args.preflight_dir is None or args.dry_run_dir is None or args.stop_at_step is None or args.parent_run_id is None:
            parser.error("continue requires --parent-run-id, --preflight-dir, --dry-run-dir, and --stop-at-step")
        result = execute_continue(
            config_path,
            args.run_id,
            args.parent_run_id,
            args.preflight_dir,
            args.dry_run_dir,
            stop_at_step=args.stop_at_step,
            allow_authorized_continuation=(
                args.stop_at_step == 1024 and args.allow_1024_continuation
                or args.stop_at_step == 2048 and args.allow_2048_continuation
            ),
        )
    print(json.dumps({"status": "completed", "path": str(result)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
