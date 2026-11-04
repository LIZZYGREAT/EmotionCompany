# -*- coding: utf-8 -*-
"""截断修复器（可进管线版）：无外部依赖，失败分支显式记录。

规则：
- 合法样本零改动。
- 句末标点落在正文后 30% 内则截到该句末并补齐收尾字段（sentence_cut）。
- 否则保留正文、在截断点补引号（close_at_end）。
- 补完仍解析失败记 failed，不静默当成"不需要修"。
"""
import argparse
import json
import re
from pathlib import Path

EMO = ["", "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
       "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
       "shame", "helplessness"]
DIMS = {
    "personality_traits": ["extroverted", "introverted", "open", "conservative",
                           "high_conscientiousness", "casual", "agreeable",
                           "assertive", "emotionally_stable", "sensitive"],
    "interests": ["study_exam", "programming_technology", "reading_writing",
                  "film_animation", "music", "games", "sports_fitness",
                  "travel_outdoor", "pets", "social", "career_development",
                  "art_design"],
    "style": ["brief", "detailed", "colloquial", "formal", "direct", "indirect",
              "humorous", "rational", "high_emotional_expression",
              "low_emotional_expression", "emoji_user"],
}


def parse_obj(text):
    t = text.strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    s = t.find("{")
    if s < 0:
        return None
    depth = 0
    instr = False
    esc = False
    for i in range(s, len(t)):
        c = t[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(t[s:i + 1])
                    return v if isinstance(v, dict) else None
                except Exception:
                    return None
    return None


def _valid_escape_tail(text: str) -> str:
    """去掉结尾不完整的 \\uXXXX 或孤立反斜杠，避免补引号后解析失败。"""
    text = text.rstrip("\\")
    m = re.search(r"\\u[0-9a-fA-F]{0,3}$", text)
    if m:
        text = text[: m.start()]
    return text


def repair(raw: str):
    """返回 (文本, 方式, 截断位置)。方式: none / sentence_cut / close_at_end / failed。"""
    obj = parse_obj(raw)
    if obj is not None and raw.strip().endswith("}"):
        return raw, "none", None
    m = re.search(r'"response_text"\s*:\s*"', raw)
    if not m:
        return raw, "failed", None
    head, body = raw[:m.end()], raw[m.end():]

    ends = [mm.end() for mm in re.finditer(r"[。！？!?]", body)
            if mm.start() > 0 and body[mm.start() - 1] != "\\"]
    cut = ends[-1] if ends else -1
    if cut > 0 and cut >= 0.7 * len(body):
        text = _valid_escape_tail(body[:cut])
        fixed = head + text + '", "memory_refs": []}'
        if parse_obj(fixed) is not None:
            return fixed, "sentence_cut", cut
    text = _valid_escape_tail(body)
    fixed = head + text + '", "memory_refs": []}'
    if parse_obj(fixed) is not None:
        return fixed, "close_at_end", len(body)
    return raw, "failed", None


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
                "failed": 0, "valid_after": 0, "structured_after": 0}
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
            if method == "failed":
                stat["failed"] += 1
            stat["repaired"] += method in ("sentence_cut", "close_at_end")
            obj2 = parse_obj(new_raw)
            stat["valid_after"] += obj2 is not None
            stat["structured_after"] += valid_structured(obj2)
            if method != "none":
                report.append({"round": f.stem, "id": r["id"], "method": method,
                               "before_chars": len(r["raw"]), "after_chars": len(new_raw),
                               "cut_at": cut})
            rec = dict(r)
            rec["raw"] = new_raw
            rec["repair"] = {"applied": method in ("sentence_cut", "close_at_end"),
                             "method": method, "before_chars": len(r["raw"]),
                             "after_chars": len(new_raw)}
            new_rows.append(rec)
        with (out / f"{f.stem}.jsonl").open("w", encoding="utf-8") as fh:
            for r in new_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps(stat, ensure_ascii=False))

    (out / "repair_matrix.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
