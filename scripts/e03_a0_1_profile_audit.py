#!/usr/bin/env python3
"""Run the read-only E03-A0.1 profile label and fine-grained supervision audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.e03_a0_1_profile import run_diagnosis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "ce"), default="cpu")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/diagnostics/e03_a0_1.yaml",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cpu-run-id",
        help="completed E03-A0.1 CPU Run required before optional CE",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_dir = run_diagnosis(
            config_path=args.config,
            run_id=args.run_id,
            mode=args.mode,
            cpu_run_id=args.cpu_run_id,
        )
    except Exception as exc:
        print(f"E03-A0.1 {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"E03-A0.1 {args.mode} run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
