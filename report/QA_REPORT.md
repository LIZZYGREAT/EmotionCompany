# 排版与文件检查记录

本文件由 `scripts/check_report.py` 生成，并补充了逐页视觉复核记录。

## 自动检查

- 图表与数据文件：通过，10 张图的 PDF、SVG、聚合输入和 Manifest 哈希逐项核对。
- TeX 插图引用：通过，10 张图均有正文引用。
- 编译 PDF：通过；页数：23 pages；字体：34 embedded font records。
- Overfull hbox：0 条；Underfull hbox：0 条。

## 详细结果

- TeX contains all 10 figure calls in the prescribed order.
- All 10 PDFs, SVG sources, aggregate data files, and manifest hashes are present and consistent.
- Compiled main.pdf is present and readable as a PDF file.
- PDF text extraction contains expected Chinese text and no figure placeholders, pending-figure appendix, or unresolved references.
- PDF fonts are embedded (34 records).
- PDF: 23 pages; 595.28 x 841.89 pts (A4).
- Compile log: 0 Overfull hbox notices; 0 Underfull notices.

## 数值差异记录

执行方案及原稿中预填的延迟来自早期记录。定稿改用当前最终 Docker Run 03 的 100 条原始记录；逐项如下。

| 指标 | 方案草稿值 | 定稿值 | 定稿来源与口径 |
|---|---:|---:|---|
| P50 | 3.05 s | 3.244649 s | `outputs/final_delivery/E03-DUAL-20261109-03/docker/stress100/performance_report.json` 的 `median_latency_ms` |
| P90 | 9.57 s | 10.206099 s | 同文件的 100 个 `latencies_ms`，最近秩法第 90 个值 |
| P95 | 11.69 s | 15.237110 s | 同文件的 `p95_latency_ms` |

定稿的 P50/P95 保留 Run 03 性能报告记录值；P90 可由保存的逐样本延迟复算。Schema 100/100、GPU 峰值 7,756/8,188 MiB 和 0 个未解决错误均与 Run 03 验收文件一致。

## 全页视觉复核

- 23 页 A4 全页缩略图：`../outputs/report-qa/contact-sheet-final.png`。
- 封面将甘文杰、庄振宇两位作者并列列出，并在姓名下显示共同单位。
- 重点放大检查图 01（第 6 页）、03/04（第 9 页）、05（第 12 页）、08（第 17 页）、10（第 21 页），箭头不交叉，图中文字无重叠或出界。
- 调整图 07 为就地排版后，原浮动页空白已消除；第 13、15、16、18、19、20、22、23 页已复查页边界、标题、表格、图注和章节衔接。

## 编译说明

- 在当前工具环境用 Tectonic（XeLaTeX 兼容引擎）完成编译；系统未提供 XeLaTeX 可执行文件。
- 编译命令：`tectonic --outdir report --keep-logs --reruns 2 report/main.tex`（设置临时 `TECTONIC_CACHE_DIR` 与 `FONTCONFIG_FILE`）；退出码 0。
- 编译无错误、无未定义引用、无 Overfull/Underfull hbox；xeCJK 字体重定义提示和 microtype footnote patch 提示不影响输出。
- Poppler 能提取正文中文并通过嵌入字体检查，但该环境仍提示 Adobe-GB1 CMap/Type 0C 字体映射警告；PDF 页面渲染与中文正文视觉检查正常。
