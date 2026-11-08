"""Build the replay training set from the manifest and the encoded 4665 train rows."""

from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()

    rows = {}
    with open(args.train, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["sample_id"]] = row

    instances = []
    missing = 0
    with open(args.manifest, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            sample_id = "train:" + item["key"]
            row = rows.get(sample_id)
            if row is None:
                missing += 1
                continue
            copies = int(item.get("copy", 1))
            for _ in range(copies):
                instances.append(row)

    random.Random(args.seed).shuffle(instances)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in instances:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("base_rows", len(rows), "instances", len(instances), "missing", missing)


if __name__ == "__main__":
    main()
