# EmotionCompany 实验研究报告

报告入口为 `main.tex`，最终阅读版为 `main.pdf`。正文保留原有七章研究主线与实验结论；10 张图均由脚本从报告内的聚合数据文件生成。

## 目录

- `main.tex`：报告全文与 10 个插图引用。
- `main.pdf`：编译后的 A4 报告。
- `figures/*.pdf`：嵌入字体的矢量图。
- `figures/source/*.svg`：可编辑图稿源文件。
- `figures/data/`：绘图使用的 CSV/JSON 聚合数据，不含对话原文、完整预测或权重。
- `figures/FIGURE_MANIFEST.md`：图表来源、数据范围、指标口径与 PDF SHA-256。
- `scripts/generate_figures.py`：一次重建 10 张图和图表清单。
- `scripts/check_report.py`：检查图片、来源清单、引用、PDF 文本与字体嵌入。
- `QA_REPORT.md`：本次编译、排版和视觉检查记录。

## 生成图表与编译

需要 Python 3.10+、Matplotlib、FontTools、Fontconfig 中可见的 Noto Sans CJK SC，以及 XeLaTeX 和 `ctex` 等 TeX 宏包。绘图字体按 Fontconfig 查询，不需要复制操作系统字体到仓库。

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_figures.py
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
python3 scripts/check_report.py
```

编译命令在 `report/` 目录运行。当前工作环境没有 XeLaTeX，交付 PDF 使用 Tectonic 的 XeLaTeX 兼容引擎生成；正式使用时仍可按上面的 XeLaTeX 命令重建。编译不需要 `shell-escape`。

## 数据口径

- 图 02 的情绪分布来自原始 Train；画像空/非空统计来自有效 SFT Train，使用各自的分母。
- 图 06 的 Pilot160 与 Dev224 是独立固定集合，只连接各自真实评测节点。
- 图 07 展示同一 209 点切片的类别命中数与 Gold 支持数；anxiety 另行标注。
- 图 09 使用相同的 2,354 个 Val ID。E01-B 与 A8 的原生 Prompt 和输出协议不同，因此图表只描述候选结果，不代表单因素消融。
- 图 10 对应 Docker Run 03 的 100 条本地输入。P50/P95 为记录值；P90 从保存的原始延迟按最近秩法计算。该验收不等于官方隐藏测试成绩。

封面将甘文杰、庄振宇两位作者并列列出，并标注共同单位南开大学；姓名、单位和报告时间由 `main.tex` 中的宏维护。
