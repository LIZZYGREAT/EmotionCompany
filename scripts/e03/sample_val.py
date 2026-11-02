# -*- coding: utf-8 -*-
"""抽代理集：Val 900 段按 seed 整段抽 78 段，展开成点 id。"""
import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=78)
ap.add_argument("--seed", type=int, default=20261103)
args = ap.parse_args()

root = Path(args.root).expanduser().resolve()
out = Path(args.out).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in (root / "data" / "val" / "val_public.jsonl").open(encoding="utf-8")]
cids = [r["conversation_id"] for r in rows]
random.Random(args.seed).shuffle(cids)
picked = sorted(cids[: args.n])

point_ids = []
by_conv = {}
for r in rows:
    if r["conversation_id"] not in picked:
        continue
    ids = [f"{r['conversation_id']}_{p['target_user_turn_id']}" for p in r["prediction_points"]]
    by_conv[r["conversation_id"]] = ids
    point_ids.extend(ids)

payload = {
    "seed": args.seed,
    "conversations_total": len(rows),
    "conversations_picked": len(picked),
    "points": len(point_ids),
    "conversation_ids": picked,
    "point_ids": point_ids,
}
(out / "sample_ids.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: payload[k] for k in ("seed", "conversations_picked", "points")}, ensure_ascii=False))
