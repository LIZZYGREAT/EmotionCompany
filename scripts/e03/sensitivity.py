# -*- coding: utf-8 -*-
"""敏感性检查：四轮因 prompt 变长导致 T1 选择不同的样本，剔除后重算指标。"""
import json
import sys
from pathlib import Path

root = Path.home() / "projects/match"
sys.path.insert(0, str(root))
sys.path.insert(0, "/mnt/c/Users/sumiha/work/ops/e03")

from transformers import AutoTokenizer
from src.shared.prompting import build_messages, select_recent_complete_suffix
from metrics import parse_obj, check

run = root / "outputs/e03/p1-base-run1"
ids = set(json.loads((run / "sample_ids.json").read_text(encoding="utf-8"))["point_ids"])

val = root / "data/val/val_public.jsonl"
points = []
for line in val.open(encoding="utf-8"):
    row = json.loads(line)
    by = {t["turn_id"]: t for t in row["turns"]}
    for p in row["prediction_points"]:
        tid = p["target_user_turn_id"]
        pid = f"{row['conversation_id']}_{tid}"
        if pid not in ids:
            continue
        hist = [{"role": by[t]["role"], "content": by[t]["content"]}
                for t in sorted(t for t in by if t <= tid)]
        points.append((pid, hist))

tok = AutoTokenizer.from_pretrained(str(root / "participant/model"), trust_remote_code=True)
prompts = {n: (root / "configs/prompts" / f).read_text(encoding="utf-8")
           for n, f in [("p0", "p0.txt"), ("p1a", "p1-a.txt"), ("p1b", "p1-b.txt"), ("p1c", "p1-c.txt")]}

sel = {}
for name, text in prompts.items():
    for pid, hist in points:
        r = select_recent_complete_suffix(hist, text, 1280, tok)
        sel[(name, pid)] = r["selected_start_index"]

diff = {pid for pid, _ in points
        if any(sel[(n, pid)] != sel[("p0", pid)] for n in ("p1a", "p1b", "p1c"))}
print("points", len(points), "selection_diff", len(diff))

def metrics(recs, keep):
    n = len(keep)
    agg = {k: 0 for k in ("json", "fields", "types", "enum", "extra", "nonempty", "empty")}
    chars = 0
    for r in recs:
        if r["id"] not in keep:
            continue
        o = parse_obj(r["raw"])
        c = check(o)
        agg["json"] += o is not None
        agg["fields"] += c["fields"]
        agg["types"] += c["types"]
        agg["enum"] += c["enum"]
        agg["extra"] += c["extra"]
        agg["nonempty"] += c["profile_nonempty"]
        agg["empty"] += c["empty_response"]
        chars += c["chars"]
    return {k: round(v / n, 4) for k, v in agg.items()} | {"n": n, "mean_chars": round(chars / n, 1)}

keep = {pid for pid, _ in points} - diff
out = {}
for name in prompts:
    recs = [json.loads(l) for l in (run / "raw" / f"{name}.jsonl").open(encoding="utf-8")]
    out[name] = {"all": metrics(recs, {pid for pid, _ in points}), "excl": metrics(recs, keep)}
print(json.dumps({"selection_diff_points": len(diff), "metrics": out}, ensure_ascii=False, indent=2))
(run / "sensitivity.json").write_text(json.dumps(
    {"selection_diff_points": len(diff), "excluded_ids": sorted(diff), "metrics": out},
    ensure_ascii=False, indent=2), encoding="utf-8")
