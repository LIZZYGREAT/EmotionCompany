#!/usr/bin/env python3
"""Run the read-only E03-A0 Token or bounded CE diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.e03_a0 import PROJECT_ROOT, run_diagnosis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("token", "ce"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/diagnostics/e03_a0.yaml",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-cap", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--token-run-id",
        help="completed full-population CPU Token Run required before CE mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_dir = run_diagnosis(
            config_path=args.config,
            run_id=args.run_id,
            mode=args.mode,
            sample_cap=args.sample_cap,
            seed=args.seed,
            token_run_id=args.token_run_id,
        )
    except Exception as exc:
        print(f"E03-A0 {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"E03-A0 {args.mode} run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
