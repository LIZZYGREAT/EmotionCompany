# -*- coding: utf-8 -*-
"""正式版截断修复器 v2：
- 句末标点位置需在正文后 30% 内才截；否则在截断处直接补引号收尾（保留半句）。
- 合法样本必须零改动（自检）。
- 每条记录 id、原始字数、截断位置、收尾方式。
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/c/Users/sumiha/work/ops/e03")
from metrics import parse_obj, EMO, DIMS


def repair(raw: str):
    """返回 (文本, 方式, 截断位置)。方式: none / sentence_cut / close_at_end。"""
    obj = parse_obj(raw)
    if obj is not None and raw.strip().endswith("}"):
        return raw, "none", None
    m = re.search(r'"response_text"\s*:\s*"', raw)
    if not m:
        return raw, "none", None
    head, body = raw[:m.end()], raw[m.end():]

    ends = [mm.end() for mm in re.finditer(r"[。！？!?]", body)
            if mm.start() > 0 and body[mm.start() - 1] != "\\"]
    cut = ends[-1] if ends else -1
    if cut > 0 and cut >= 0.7 * len(body):
        text = body[:cut].rstrip("\\")
        fixed = head + text + '", "memory_refs": []}'
        if parse_obj(fixed) is not None:
            return fixed, "sentence_cut", cut
    # 保底：保留正文，直接补引号收尾
    text = body.rstrip("\\")
    fixed = head + text + '", "memory_refs": []}'
    if parse_obj(fixed) is not None:
        return fixed, "close_at_end", len(body)
    return raw, "none", None


def valid_structured(obj):
    if not isinstance(obj, dict):
        return False
    if obj.get("emotion_label") not in EMO:
        return False
    if not isinstance(obj.get("response_text"), str):
        return False
    if not isinstance(obj.get("memory_refs"), list):
        return False
    prof = obj.get("user_profile")
    if not isinstance(prof, dict):
        return False
    for d, allowed in DIMS.items():
        v = prof.get(d)
        if not isinstance(v, list) or any(x not in allowed for x in v):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    report = []
    for f in sorted((run / "raw").glob("*.jsonl")):
        rows = [json.loads(l) for l in f.open(encoding="utf-8")]
        stat = {"round": f.stem, "n": len(rows), "bad_before": 0, "repaired": 0,
                "unchanged_valid": 0, "sentence_cut": 0, "close_at_end": 0,
                "valid_after": 0, "structured_after": 0}
        new_rows = []
        for r in rows:
            obj = parse_obj(r["raw"])
            bad = obj is None or not r["raw"].strip().endswith("}")
            if bad:
                stat["bad_before"] += 1
            new_raw, method, cut = repair(r["raw"])
            if method == "none" and not bad:
                stat["unchanged_valid"] += 1
            if method == "sentence_cut":
                stat["sentence_cut"] += 1
            if method == "close_at_end":
                stat["close_at_end"] += 1
            stat["repaired"] += method != "none"
            obj2 = parse_obj(new_raw)
            stat["valid_after"] += obj2 is not None
            stat["structured_after"] += valid_structured(obj2)
            if method != "none":
                report.append({"round": f.stem, "id": r["id"], "method": method,
                               "before_chars": len(r["raw"]), "after_chars": len(new_raw),
                               "cut_at": cut})
            rec = dict(r)
            rec["raw"] = new_raw
            rec["repair"] = {"applied": method != "none", "method": method,
                             "before_chars": len(r["raw"]), "after_chars": len(new_raw)}
            new_rows.append(rec)
        with (out / f"{f.stem}.jsonl").open("w", encoding="utf-8") as fh:
            for r in new_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps(stat, ensure_ascii=False))

    (out / "repair_matrix.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
