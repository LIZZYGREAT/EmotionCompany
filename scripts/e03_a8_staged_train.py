#!/usr/bin/env python3
"""CPU preflight, GPU dry-run, and bounded staged E03-A8 training CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.e03_a8_staged import (
    DEFAULT_CONFIGS,
    authorize_conditional_resume,
    execute_gpu_dry_step,
    execute_resume,
    execute_start,
    prepare_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded staged E03-A8 QLoRA training for one arm.")
    parser.add_argument("--mode", choices=("preflight", "gpu-dry-run", "start", "authorize", "resume"), required=True)
    parser.add_argument("--arm", choices=("P", "EP"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--dry-run-dir", type=Path)
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--from-step", type=int)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--allow-gpu-dry-run", action="store_true")
    parser.add_argument("--allow-first-stage-training", action="store_true")
    parser.add_argument("--allow-conditional-resume", action="store_true")
    parser.add_argument("--allow-conditional-authorization", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config or DEFAULT_CONFIGS[args.arm]
    if args.mode == "preflight":
        result = prepare_preflight(config_path, args.run_id)
    elif args.mode == "gpu-dry-run":
        if args.preflight_dir is None:
            parser.error("--preflight-dir is required for gpu-dry-run")
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
    elif args.mode == "authorize":
        if args.from_step is None:
            parser.error("authorize requires --from-step")
        result = authorize_conditional_resume(
            config_path,
            args.run_id,
            from_step=args.from_step,
            allow_conditional_authorization=args.allow_conditional_authorization,
        )
    else:
        if args.from_step is None or args.stop_at_step is None or args.approval_file is None:
            parser.error("resume requires --from-step, --stop-at-step, and --approval-file")
        result = execute_resume(
            config_path,
            args.run_id,
            from_step=args.from_step,
            stop_at_step=args.stop_at_step,
            approval_file=args.approval_file,
            allow_conditional_resume=args.allow_conditional_resume,
        )
    print(json.dumps({"status": "completed", "path": str(result)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
