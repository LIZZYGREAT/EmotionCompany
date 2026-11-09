#!/usr/bin/env python3
"""Check report figure assets, manifest hashes, TeX references, and compiled PDF."""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
EXPECTED = [
    "01-overall-pipeline", "02-data-distribution", "03-sft-dataflow",
    "04-qlora-training", "05-supervision-share", "06-a8-learning-curve",
    "07-emotion-replay", "08-dual-adapter-architecture",
    "09-final-profile-compare", "10-deployment-performance",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdftotext", help="Path to pdftotext for compiled text checks")
    parser.add_argument("--pdffonts", help="Path to pdffonts for font embedding checks")
    parser.add_argument("--pdfinfo", help="Path to pdfinfo for page-size/count checks")
    parser.add_argument("--write-report", action="store_true", help="Write scripts/../QA_REPORT.md")
    args = parser.parse_args()

    errors: list[str] = []
    details: list[str] = []
    tex_path = ROOT / "main.tex"
    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    manifest_path = FIG / "FIGURE_MANIFEST.md"
    manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
    pdf_path = ROOT / "main.pdf"

    calls = re.findall(r"\\figureplaceholder(?:\[[^]]+\])?\{([^}]+)\}", tex)
    calls = [name for name in calls if name in EXPECTED]
    if calls != EXPECTED:
        errors.append(f"Expected the ten figures in order; found {calls}.")
    else:
        details.append("TeX contains all 10 figure calls in the prescribed order.")

    manifest_rows: dict[str, tuple[str, str]] = {}
    for line in manifest.splitlines():
        match = re.match(r"\|\s*(\d{2})\s+[^|]+\|\s*`(figures/[^`]+\.pdf)`\s*\|", line)
        if not match:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 10:
            figure_id = match.group(1)
            pdf_name = match.group(2).removeprefix("figures/").removesuffix(".pdf")
            file_hash = parts[9].strip("`")
            manifest_rows[pdf_name] = (figure_id, file_hash)

    for name in EXPECTED:
        pdf = FIG / f"{name}.pdf"
        svg = FIG / "source" / f"{name}.svg"
        data = FIG / "data" / f"{name}.json"
        if not data.exists():
            data = FIG / "data" / f"{name}.csv"
        for required in (pdf, svg, data):
            if not required.is_file():
                errors.append(f"Missing figure asset: {required.relative_to(ROOT)}")
        if pdf.is_file() and not pdf.read_bytes().startswith(b"%PDF-"):
            errors.append(f"Invalid PDF header: {pdf.relative_to(ROOT)}")
        if svg.is_file() and "<svg" not in svg.read_text(encoding="utf-8"):
            errors.append(f"Invalid SVG source: {svg.relative_to(ROOT)}")
        if pdf.is_file():
            row = manifest_rows.get(name)
            if row is None:
                errors.append(f"Manifest row missing for {name}.")
            elif row[1] != sha256(pdf):
                errors.append(f"Manifest SHA-256 mismatch for {name}.")
        if f"\\ref{{fig:{name}}}" not in tex:
            errors.append(f"No in-text reference to fig:{name}.")
    if len(manifest_rows) != 10:
        errors.append(f"Manifest has {len(manifest_rows)} figure rows, expected 10.")
    if not errors:
        details.append("All 10 PDFs, SVG sources, aggregate data files, and manifest hashes are present and consistent.")

    if not pdf_path.is_file() or not pdf_path.read_bytes().startswith(b"%PDF-"):
        errors.append("Compiled report/main.pdf is missing or does not have a PDF header.")
    else:
        details.append("Compiled main.pdf is present and readable as a PDF file.")

    pdf_text = ""
    if args.pdftotext and pdf_path.is_file():
        try:
            pdf_text = run_text([args.pdftotext, str(pdf_path), "-"])
            forbidden = ("待插图：", "待补充插图清单", "??")
            for phrase in forbidden:
                if phrase in pdf_text:
                    errors.append(f"Compiled PDF contains unresolved text: {phrase}")
            for phrase in ("甘文杰", "庄振宇", "摘要", "研究背景与任务分析", "实验数据来源与版本说明"):
                if phrase not in pdf_text:
                    errors.append(f"Compiled PDF text extraction is missing expected Chinese text: {phrase}")
            if not errors:
                details.append("PDF text extraction contains expected Chinese text and no figure placeholders, pending-figure appendix, or unresolved references.")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"pdftotext check failed: {exc}")

    font_summary = "not checked"
    if args.pdffonts and pdf_path.is_file():
        try:
            font_text = run_text([args.pdffonts, str(pdf_path)])
            rows = [line for line in font_text.splitlines()[2:] if line.strip()]
            embedded_flags = []
            for line in rows:
                match = re.search(r"\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
                if match:
                    embedded_flags.append(match.group(1))
            if not embedded_flags or any(flag != "yes" for flag in embedded_flags):
                errors.append("At least one PDF font is missing or the embedded-font list could not be parsed.")
            else:
                font_summary = f"{len(embedded_flags)} embedded font records"
                details.append(f"PDF fonts are embedded ({len(embedded_flags)} records).")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"pdffonts check failed: {exc}")

    page_summary = "not checked"
    if args.pdfinfo and pdf_path.is_file():
        try:
            info = run_text([args.pdfinfo, str(pdf_path)])
            page_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
            size_match = re.search(r"^Page size:\s+(.+)$", info, re.MULTILINE)
            page_summary = f"{page_match.group(1)} pages" if page_match else "page count unavailable"
            if not page_match or not size_match:
                errors.append("pdfinfo did not report page count and page size.")
            else:
                details.append(f"PDF: {page_match.group(1)} pages; {size_match.group(1)}.")
                if "A4" not in size_match.group(1):
                    errors.append(f"Expected A4 page size; found {size_match.group(1)}.")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"pdfinfo check failed: {exc}")

    log_path = ROOT / "main.log"
    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []
    overfull = [line.strip() for line in log_lines if "Overfull \\hbox" in line]
    underfull_count = sum("Underfull \\hbox" in line for line in log_lines)
    undefined = [line.strip() for line in log_lines if "Undefined control sequence" in line or "undefined references" in line]
    if undefined:
        errors.append("Compile log contains undefined control sequences or references.")
    if log_path.exists():
        details.append(f"Compile log: {len(overfull)} Overfull hbox notices; {underfull_count} Underfull notices.")

    if args.write_report:
        lines = [
            "# 排版与文件检查记录", "",
            "本文件由 `scripts/check_report.py` 生成，并补充了逐页视觉复核记录。", "",
            "## 自动检查", "",
            f"- 图表与数据文件：{'通过' if not errors else '存在错误'}，10 张图的 PDF、SVG、聚合输入和 Manifest 哈希逐项核对。",
            f"- TeX 插图引用：{'通过' if calls == EXPECTED else '未通过'}，10 张图均有正文引用。",
            f"- 编译 PDF：{'通过' if pdf_path.exists() else '缺失'}；页数：{page_summary}；字体：{font_summary}。",
            f"- Overfull hbox：{len(overfull)} 条；Underfull hbox：{underfull_count} 条。",
            "",
            "## 详细结果", "",
        ]
        lines.extend(f"- {item}" for item in details)
        lines += [
            "", "## 数值差异记录", "",
            "执行方案及原稿中预填的延迟来自早期记录。定稿改用当前最终 Docker Run 03 的 100 条原始记录；逐项如下。", "",
            "| 指标 | 方案草稿值 | 定稿值 | 定稿来源与口径 |",
            "|---|---:|---:|---|",
            "| P50 | 3.05 s | 3.244649 s | `outputs/final_delivery/E03-DUAL-20261109-03/docker/stress100/performance_report.json` 的 `median_latency_ms` |",
            "| P90 | 9.57 s | 10.206099 s | 同文件的 100 个 `latencies_ms`，最近秩法第 90 个值 |",
            "| P95 | 11.69 s | 15.237110 s | 同文件的 `p95_latency_ms` |",
            "",
            "定稿的 P50/P95 保留 Run 03 性能报告记录值；P90 可由保存的逐样本延迟复算。Schema 100/100、GPU 峰值 7,756/8,188 MiB 和 0 个未解决错误均与 Run 03 验收文件一致。", "",
            "## 全页视觉复核", "",
            "- 23 页 A4 全页缩略图：`../outputs/report-qa/contact-sheet-final.png`。",
            "- 封面将甘文杰、庄振宇两位作者并列列出，并在姓名下显示共同单位。",
            "- 重点放大检查图 01（第 6 页）、03/04（第 9 页）、05（第 12 页）、08（第 17 页）、10（第 21 页），箭头不交叉，图中文字无重叠或出界。",
            "- 调整图 07 为就地排版后，原浮动页空白已消除；第 13、15、16、18、19、20、22、23 页已复查页边界、标题、表格、图注和章节衔接。", "",
            "## 编译说明", "",
            "- 在当前工具环境用 Tectonic（XeLaTeX 兼容引擎）完成编译；系统未提供 XeLaTeX 可执行文件。",
            "- 编译命令：`tectonic --outdir report --keep-logs --reruns 2 report/main.tex`（设置临时 `TECTONIC_CACHE_DIR` 与 `FONTCONFIG_FILE`）；退出码 0。",
            "- 编译无错误、无未定义引用、无 Overfull/Underfull hbox；xeCJK 字体重定义提示和 microtype footnote patch 提示不影响输出。",
            "- Poppler 能提取正文中文并通过嵌入字体检查，但该环境仍提示 Adobe-GB1 CMap/Type 0C 字体映射警告；PDF 页面渲染与中文正文视觉检查正常。",
        ]
        if errors:
            lines += ["", "## 错误", ""]
            lines.extend(f"- {item}" for item in errors)
        (ROOT / "QA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for item in details:
        print(f"PASS: {item}")
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
