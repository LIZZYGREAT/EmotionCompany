#!/usr/bin/env python3
"""Prepare and audit the isolated E03-A6 F3 SFT cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.e03_a6_sft import load_and_prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/e03_a6_f3.yaml",
        help="F3 preparation config (F3-only; refuses to overwrite existing output)",
    )
    args = parser.parse_args()
    report = load_and_prepare(args.config, ROOT)
    print(json.dumps({
        "status": report["status"],
        "dataset_dir": report["outputs"]["dataset_dir"],
        "train_count": report["outputs"]["train_count"],
        "dev_count": report["outputs"]["dev_count"],
        "tokenizer": report["tokenizer"],
        "full_training": report["full_training"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
