# E01-B Full：深度诊断与证据整理

> 审阅日期：2026-10-29<br>
> 训练 Run：`E01-B-FULL-20261028-133805`<br>
> Base / Adapter Dev Run：`E00.3-BASE-20261028-223105` / `E01-B-EVAL-20261028-223105`<br>
> 配对比较：`full-20261029-062541`<br>
> 本地只读诊断：`outputs/evaluation/e01_b/diagnostics/full-b-v4/`

## 一、范围、来源与证据等级

本报告整理已经完成的 Full B 训练与同一批完整 Dev 结果。没有重新训练、生成、推理或评分；只核验现有 Manifest/文件 Hash、ID 映射和已评分的聚合结果，再从既有数据与逐样本统计提取分布。未将对话、Raw 回复、Gold 明细或 ID 列表写入报告或 Git。

诊断脚本为 [scripts/e01_b_diagnose.py](../../scripts/e01_b_diagnose.py)，CPU 回归为 [tests/test_e01_b_diagnose.py](../../tests/test_e01_b_diagnose.py)。脚本验证训练配置、数据集、比较 Run、评分文件、References、Raw/Submission、逐样本比较表的 Hash/数量/ID 关系；分析汇总输出位于 Git 忽略目录。分类与文本质量指标直接引用已完成评分的 `metrics.json` / `comparison_summary.json`，没有复制新的 F1/BLEU 评分定义。

| 身份项 | 核验结果 |
|---|---|
| Full B Run | `passed` / `completed`；16,192 条 SFT Train、4,048 步、1 Epoch；Run Manifest SHA `9204b7cb374765438fd23054078d28992b240d50c4df8763cae0817f0d7752d7` |
| Base 与 Adapter Dev | 两个 Run 均为 `scored`；配对 Run 为 `completed`；各 1,833 条 |
| D0.3 Train / Dev | 16,312 / 1,833；Manifest SHA `605867b5ab0162de05ea9e396bffd574cd4029c5adc3c29d851f41ff73a96b8d`；Dev SHA `bfacf9cd4a0e069e50862f973a98a3d379215231b0797e94ac76c5a70c5fbd78` |
| SFT Dev Loss | 1,821 条 teacher-forced Dev；与完整 1,833 条生成评测不是同一个分母 |
| 配对 ID / 输入 / Gold | 两组均为相同 1,833 IDs；ID SHA `739a88b6e47e053a1a358c1ac0f119de474522d0fa471bdbe4a8bd86bac1f322`；无标签推理输入 SHA `5105be04135d30fb853231892b65475d2cd9a35064dc25f705106af53b1f73dd`；References SHA `570db280feaa0d7ec5356f9410826b8b0b515bbb62f7de290a3a112b9798c5a2` |
| 模型与协议 | Base SHA `005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5`；Adapter SHA `77b61e8a4caf46f54a8c813e98def488c7a8493fe891966e47df1cdac6df7d67`；P0/F2/T1、P=1280、G=1200、Greedy、NF4/BF16；无 CPU/disk offload |
| 版本身份边界 | Base 文件 Hash 已观测，但 Hub `revision_status=unverified_revision`。两份评测的 scorer commit 均记为 `45258594334aee078638b72b2392e23bc232954c`；Full B Manifest 没记录训练源码 Git commit 或源码文件 Hash，不能从当前 checkout 倒推 |
| 本轮分析身份 | 当时 HEAD `ccba9b3ebd0ffefdfd76b0aeddfc3c66118f0f6e`；脚本 Hash 与输入 Hash 记录在本机诊断 `manifest.json`。该 HEAD 不含本轮未提交的诊断脚本和文档修改 |

Full B Run Manifest、训练证据、Base/Adapter Metrics、Raw/Submission 与逐样本比较文件的哈希均记录在本机只读诊断目录 `outputs/evaluation/e01_b/diagnostics/full-b-v4/manifest.json`。该目录被 Git 忽略；本报告保存可审阅的聚合结果和统计口径。

## 二、总体结果

| 指标（分母均为 1,833，除非注明） | Base | Full B Adapter | 差值 / 说明 |
|---|---:|---:|---|
| Emotion Accuracy | 235/1,833 = 0.1282 | 1,198/1,833 = 0.6536 | +0.5254 |
| Emotion Macro-F1 | 0.0690 | 0.4972 | 16 个固定情绪类别等权平均 |
| Profile 33-label Micro-F1 | 0.0782 | 0.4223 | 依逐标签 TP/FP/FN 汇总 |
| Profile 33-label Macro-F1 | 0.0724 | 0.2700 | 33 个标签等权平均，仍有长尾弱类 |
| 三维 Profile Exact Match | 396/1,833 = 0.2160 | 588/1,833 = 0.3208 | Full B 仍低于 660/1,833 的全空预测基线 |
| Raw 严格 JSON | 846/1,833 = 0.4615 | 1,801/1,833 = 0.9825 | 原始模型文本解析为完整 JSON object |
| Raw A0 Schema | 17/1,833 = 0.0093 | 1,798/1,833 = 0.9809 | 严格字段、合法标签、非空回复和 `memory_refs=[]` |
| Submission Schema | 1.0000 | 1.0000 | Runner 归一化后的提交格式；不能代替 Raw 合法率 |
| 字符级 BLEU-4 | 2.3993 | 12.8212 | `sacrebleu 2.5.1`，百分比，char tokenize |
| 字符级 ROUGE-L | 0.1188 | 0.2603 | Unicode 字符 LCS F1；移除全部 Unicode whitespace |
| 空回复率 | 407/1,833 = 0.2220 | 32/1,833 = 0.0175 | 评分 Submission 中的空回复 |

上述质量和回复指标是 `local_proxy`，不是官方得分。配对情绪迁移来自现有比较器：两组同时正确 191、Base 独对 44、Adapter 独对 1,007、两组都错 591；合计 1,833。按 826 个会话簇、Seed 42、2,000 次 percentile bootstrap，Accuracy 差 `+0.525368`，95% 区间 `[+0.498346, +0.552044]`。这是同 ID 配对统计，不表示 Dev 等同隐蔽 Test。

## 三、训练过程与 SFT 泛化趋势

Full B 使用独立初始化的全新 LoRA（r=8、alpha=16、dropout=0；q/k/v/o、gate/up/down projections），主模型 NF4 量化并冻结，BF16 compute，AdamW，LR `1e-4` constant，Seed 42，Micro Batch 1、梯度累计 4。实际训练为 4,048/4,048 步和 16,192/16,192 样本；基础参数 SHA 未变化。训练耗时 `23,066` 秒（约 6 小时 24 分），输入 Token `17,228,707`，Completion Token `2,851,614`。GPU 为 RTX 4060 Laptop 8 GiB；训练、Trainer、SFT Dev 与 Checkpoint 阶段峰值 allocator allocated/reserved 为 `3.83/4.10 GB`（物理总显存 8.59 GB）。

逐步日志共有 4,048 条，Step 从 1 连续到 4,048；全步梯度有限。日志记录 204 次抽样参数更新检查，抽样到的 LoRA 更新均非零；`parameter_update_checked=false` 的其他步表示未做全量参数差分抽查，不能解读为零更新。首 10 步平均 Loss `2.9730`，末 10 步 `1.5974`；日志逐步 Loss 均值为 `1.6781`。SFT Dev Loss 每 256 步在 1,821 条 teacher-forced Dev 上计算，最佳和末次都在 step 4,048，为 `1.4280`。Train Loss、SFT Dev Loss、生成任务指标是不同测量，不互相替代。

| Optimizer Step 窗口 | 窗口平均 Train Loss | SFT Dev Loss（窗口末 Step） |
|---:|---:|---:|
| 1–256 | 1.9647 | 1.6609 |
| 257–512 | 1.8027 | 1.6069 |
| 513–768 | 1.7310 | 1.5659 |
| 769–1,024 | 1.7170 | 1.5392 |
| 1,025–1,280 | 1.7078 | 1.5233 |
| 1,281–1,536 | 1.6616 | 1.5073 |
| 1,537–1,792 | 1.6783 | 1.4932 |
| 1,793–2,048 | 1.6387 | 1.4821 |
| 2,049–2,304 | 1.6350 | 1.4739 |
| 2,305–2,560 | 1.6379 | 1.4636 |
| 2,561–2,816 | 1.6229 | 1.4550 |
| 2,817–3,072 | 1.6036 | 1.4492 |
| 3,073–3,328 | 1.6034 | 1.4445 |
| 3,329–3,584 | 1.6262 | 1.4386 |
| 3,585–3,840 | 1.5940 | 1.4323 |
| 3,841–4,048 | 1.6133 | 1.4280 |

保留的 Trainer Checkpoint 为 step 3,840（65.66 MB）和 step 4,048（65.74 MB）。最终 Adapter 独立重载检查通过，64 个确定性生成 Token IDs 完全一致，Base 权重 SHA 未变化。

## 四、情绪分类：类别支持、逐类结果与混淆

Train 的情绪分布明显偏斜：anxiety 5,483/16,312（33.62%）、relaxed 2,500（15.33%）、helplessness 1,381（8.47%）；care 61、surprise 56。Full Dev 1,833 条覆盖全部 16 类。下表的 P/R/F1 取自已完成 scorer 的 Adapter 指标；`Base→Full 命中`按 Gold 情绪类别统计两模型正确数，分母为该行 Dev Support。

| 情绪 | Train Support | Dev Support | Base→Full 命中 | Full 预测数 | Full Precision | Full Recall | Full F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| joy | 153 | 16 | 0→7 | 11 | 0.636 | 0.438 | 0.519 |
| gratitude | 628 | 61 | 4→34 | 40 | 0.850 | 0.557 | 0.673 |
| relaxed | 2,500 | 272 | 100→232 | 337 | 0.688 | 0.853 | 0.762 |
| care | 61 | 6 | 0→0 | 0 | 0.000 | 0.000 | 0.000 |
| pride | 201 | 28 | 1→9 | 17 | 0.529 | 0.321 | 0.400 |
| neutral | 568 | 68 | 6→20 | 39 | 0.513 | 0.294 | 0.374 |
| surprise | 56 | 6 | 0→2 | 3 | 0.667 | 0.333 | 0.444 |
| mixed | 834 | 93 | 1→5 | 10 | 0.500 | 0.054 | 0.097 |
| sadness | 953 | 105 | 39→49 | 94 | 0.521 | 0.467 | 0.492 |
| loneliness | 766 | 82 | 0→67 | 101 | 0.663 | 0.817 | 0.732 |
| anxiety | 5,483 | 601 | 83→518 | 797 | 0.650 | 0.862 | 0.741 |
| anger | 744 | 90 | 1→60 | 83 | 0.723 | 0.667 | 0.694 |
| fear | 1,230 | 141 | 0→57 | 65 | 0.877 | 0.404 | 0.553 |
| disgust | 102 | 16 | 0→3 | 5 | 0.600 | 0.188 | 0.286 |
| shame | 652 | 87 | 0→45 | 75 | 0.600 | 0.517 | 0.556 |
| helplessness | 1,381 | 161 | 0→90 | 124 | 0.726 | 0.559 | 0.632 |

Dev 全部 16 类的 correct 命中数均未低于 Base；但这不表示所有分类问题都已解决。care 为 0/6，mixed 仅 5/93（Recall 5.38%），disgust 为 3/16；care/surprise 的 Dev Support 仅 6，逐类比率不稳定。Train 多数类 anxiety 在 Dev 的多数类基线为 601/1,833（32.79%），低于 Adapter Accuracy 65.36%。

Adapter 的主要合法类别混淆对（从现有混淆矩阵、Gold 行/预测列读取）为：fear→anxiety 71、helplessness→anxiety 43、mixed→anxiety 43、shame→anxiety 30、gratitude→relaxed 21、mixed→relaxed 20、relaxed→anxiety 19、neutral→relaxed 18。误分为 anxiety 是可观察集中现象；其原因仍是 `hypothesis`，本报告不推断因果。Base 主要把 anxiety、helplessness、fear、shame、loneliness、anger、mixed 输出成 sadness；Base 还有 972 个情绪错误归于 Raw JSON 解析失败、3 个归于非法情绪值、623 个是合法标签错类。Adapter 的 635 个错误中 32 个来自 Raw JSON 解析失败，603 个是合法情绪错类。

## 五、画像：逐维、逐标签与全空基线

Full Adapter 的 33 标签 Micro-F1 为 `0.4223`，Macro-F1 为 `0.2700`；TP/FP/FN 合计 `1,130/1,271/1,821`。维度结果如下；非空 Gold Recall 定义为“Gold 该维非空且预测至少命中一个标签的样本数 / 该维非空 Gold 样本数”。

| 维度 | Gold 非空 | 预测非空 | Gold 非空 Recall | Micro-F1 | Macro-F1 | Exact Match | TP/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| personality_traits | 834/1,833 (45.50%) | 669/1,833 (36.50%) | 380/834 (45.56%) | 0.4189 | 0.1404 | 991/1,833 | 435/500/707 |
| interests | 593/1,833 (32.35%) | 514/1,833 (28.04%) | 309/593 (52.11%) | 0.5392 | 0.4836 | 1,356/1,833 | 313/222/313 |
| style | 1,022/1,833 (55.76%) | 847/1,833 (46.21%) | 374/1,022 (36.59%) | 0.3614 | 0.1548 | 866/1,833 | 382/549/801 |

Gold 三维同时为空有 `660/1,833=36.01%`。恒输出三个空数组会在这 660 条上 Exact Match；Full B 有 950 条三维预测全空，其中 523 条 Gold 也全空、427 条 Gold 非空被全部漏掉。另有 137 条 Gold 三维全空却预测出标签。Full B 总 Exact Match 588 条，其中 523 条是全空正确、65 条是非空画像完全正确，因此仍比恒空基线少 72 条。与此同时 Micro-F1 0.4223 表明逐标签有大量正确恢复；Exact Match 对任何一个漏标签或多标签都会整条判错，不应单独用于衡量多标签能力。

按逐样本 33 标签 TP/FP/FN 拆分的互斥结果：完全正确 588；全部 Gold 标签漏报（TP=FP=0 且 FN>0）427；部分命中但只有漏报 84；只有额外误报 274；同时有漏报和额外误报 460。分母合计 1,833。Pilot 160 条 Adapter 有 159/160 三维全空；Full B 完整 Dev 是 950/1,833。二者样本范围不同，只能说明观察到的输出分布变化，不构成同 ID 学习曲线。

下面逐标签 Support 来自已核验 D0.3 Train/Dev Gold；预测数与 TP/FP/FN/P/R/F1 来自既有 Full Dev `profile_metrics`。低 Support 标签的分数应连同分母阅读。

### personality_traits（10 标签）

| 标签 | Train Support | Dev Support | Pred | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extroverted | 36 | 1 | 0 | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 |
| introverted | 642 | 71 | 13 | 11 | 2 | 60 | 0.846 | 0.155 | 0.262 |
| open | 53 | 3 | 1 | 0 | 1 | 3 | 0.000 | 0.000 | 0.000 |
| conservative | 1,205 | 122 | 6 | 3 | 3 | 119 | 0.500 | 0.025 | 0.047 |
| high_conscientiousness | 1,931 | 250 | 251 | 98 | 153 | 152 | 0.390 | 0.392 | 0.391 |
| casual | 27 | 0 | 1 | 0 | 1 | 0 | 0.000 | 0.000 | 0.000 |
| agreeable | 289 | 48 | 11 | 2 | 9 | 46 | 0.182 | 0.042 | 0.068 |
| assertive | 596 | 65 | 16 | 4 | 12 | 61 | 0.250 | 0.062 | 0.099 |
| emotionally_stable | 296 | 39 | 0 | 0 | 0 | 39 | 0.000 | 0.000 | 0.000 |
| sensitive | 4,471 | 543 | 636 | 317 | 319 | 226 | 0.498 | 0.584 | 0.538 |

### interests（12 标签）

| 标签 | Train Support | Dev Support | Pred | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| study_exam | 651 | 56 | 72 | 29 | 43 | 27 | 0.403 | 0.518 | 0.453 |
| programming_technology | 73 | 7 | 0 | 0 | 0 | 7 | 0.000 | 0.000 | 0.000 |
| reading_writing | 271 | 28 | 27 | 15 | 12 | 13 | 0.556 | 0.536 | 0.545 |
| film_animation | 99 | 5 | 2 | 1 | 1 | 4 | 0.500 | 0.200 | 0.286 |
| music | 116 | 15 | 12 | 8 | 4 | 7 | 0.667 | 0.533 | 0.593 |
| games | 605 | 74 | 79 | 63 | 16 | 11 | 0.797 | 0.851 | 0.824 |
| sports_fitness | 538 | 68 | 80 | 43 | 37 | 25 | 0.537 | 0.632 | 0.581 |
| travel_outdoor | 130 | 14 | 18 | 9 | 9 | 5 | 0.500 | 0.643 | 0.563 |
| pets | 655 | 84 | 48 | 38 | 10 | 46 | 0.792 | 0.452 | 0.576 |
| social | 674 | 80 | 49 | 24 | 25 | 56 | 0.490 | 0.300 | 0.372 |
| career_development | 1,268 | 161 | 119 | 66 | 53 | 95 | 0.555 | 0.410 | 0.471 |
| art_design | 295 | 34 | 29 | 17 | 12 | 17 | 0.586 | 0.500 | 0.540 |

### style（11 标签）

| 标签 | Train Support | Dev Support | Pred | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brief | 1,017 | 119 | 75 | 28 | 47 | 91 | 0.373 | 0.235 | 0.289 |
| detailed | 949 | 100 | 53 | 12 | 41 | 88 | 0.226 | 0.120 | 0.157 |
| colloquial | 582 | 46 | 67 | 14 | 53 | 32 | 0.209 | 0.304 | 0.248 |
| formal | 112 | 13 | 0 | 0 | 0 | 13 | 0.000 | 0.000 | 0.000 |
| direct | 2,233 | 264 | 162 | 77 | 85 | 187 | 0.475 | 0.292 | 0.362 |
| indirect | 459 | 53 | 0 | 0 | 0 | 53 | 0.000 | 0.000 | 0.000 |
| humorous | 60 | 10 | 3 | 1 | 2 | 9 | 0.333 | 0.100 | 0.154 |
| rational | 3,733 | 483 | 569 | 249 | 320 | 234 | 0.438 | 0.516 | 0.473 |
| high_emotional_expression | 831 | 94 | 2 | 1 | 1 | 93 | 0.500 | 0.011 | 0.021 |
| low_emotional_expression | 3 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| emoji_user | 26 | 1 | 0 | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 |

`interests` 的 Macro-F1 最高且 Gold 非空 Recall 为 52.11%；`personality_traits` 和 `style` 的 Macro-F1 分别为 0.1404、0.1548，低于 Micro-F1。部分长尾标签在 Dev Support 为 0–3，不能据此稳定判断泛化；这些 Support 同样用于解释为何 overall Micro 会被高频标签主导。画像空预测、标签不平衡、联合任务目标竞争等都只是候选解释，本报告没有把它们写成已证实因果。

## 六、Raw 格式、回复、生成长度与延迟

| Full B Adapter Raw 层次 | 数量 / 分母 | 结果 |
|---|---:|---|
| 严格 JSON object | 1,801/1,833 | 32 条解析失败 |
| Raw A0 Schema | 1,798/1,833 | 另有 3 条可解析 JSON 的画像标签超出枚举 |
| Submission Schema | 1,833/1,833 | Runner 归一化层全部合法；35 条 Raw 未通过 A0 的情况不会因归一化而消失 |
| 达到 `G=1,200` | 29/1,833 = 1.58% | 29 条均因未终止字符串导致 JSON 截断 |
| 其余 3 条 JSON 解析失败 | 3/1,833 | 2 条含非法控制字符，1 条缺少逗号分隔符 |

Raw 失败分组的指标使用相应分组样本作分母；TP/FP/FN 为该组经过 Runner 归一化后的 33 标签总计，不代表 Raw A0 本身有效：

| 模型与 Raw 组 | 样本数 | 平均生成 Token | 平均延迟 | Emotion Accuracy | Profile TP/FP/FN |
|---|---:|---:|---:|---:|---:|
| Base 严格 JSON 失败 | 987 | 126.6 | 3.96 秒 | 15/987 = 1.52% | 21/183/1,626 |
| Base JSON 可解析、Raw A0 不合法 | 829 | 130.7 | 4.06 秒 | 26.42% | 155/1,225/1,133 |
| Base Raw A0 合法 | 17 | 99.1 | 3.06 秒 | 5.88% | 2/14/14 |
| Adapter 严格 JSON 失败 | 32 | 1,134.8 | 66.76 秒 | 0/32 | 0/0/44 |
| Adapter JSON 可解析、Raw A0 不合法 | 3 | 279.7 | 15.50 秒 | 1/3 | 3/3/7 |
| Adapter Raw A0 合法 | 1,798 | 180.9 | 10.48 秒 | 66.57% | 1,127/1,268/1,770 |

Adapter 生成 Token 分布（1,833 条）：均值 197.7，P50 125，P90 428，P95 640，最大 1,200；Base 均值 128.2，P50 119，P90 178，P95 223，最大 342。按 Adapter 生成长度分组的平均延迟依次为：1–256 Token（1,497 条）7.00 秒；257–512（193 条）20.46 秒；513–768（80 条）34.87 秒；769–1,199（34 条）51.53 秒；1,200（29 条）70.96 秒。Base 的 ≤256 Token 组为 1,784 条、均值 3.87 秒，257–512 组为 49 条、均值 8.35 秒。组间延迟随生成长度增加而上升是观察到的相关性；不能据此将 Adapter 总体延迟差异全部归因于模型参数本身。

从已评分 Submission 读取的回复字符长度：Base 均值 125.9、P50 89、P90 314、最大 703；Adapter 均值 217.4、P50 131、P90 532、最大 1,807（分母 1,833，含空回复）。字符级 BLEU-4、ROUGE-L 与空回复率见第二节。Scorer 仅保存整体 BLEU/ROUGE，未保存逐样本文本分数；本轮明确没有重新评分，因此“长度与 BLEU/ROUGE 的逐样本相关性”标记为 unavailable。BERTScore 为 `not_run`。

本机 Runner 的计时报告完成 `100/100` 轮：Base 平均/中位/P95 `4.169/3.550/7.616` 秒，Adapter `13.426/9.079/34.270` 秒，平均约慢 3.22 倍。另按全部 1,833 条逐样本延迟记录统计，Base 均值/中位/P95 `3.994/3.626/7.189` 秒，Adapter `11.474/7.442/36.446` 秒。两种统计对象不同；它们都是本机 RTX 4060 Laptop Transformers 计时，不是官方性能分数。官方性能仍 `not_run`。

## 七、已证实事实、假设与证据缺口

### 已证实（`verified_from_local_artifacts` / `derived_from_verified_artifacts`）

- D0.3 Train、Dev、评测输入、References、ID Map、Base/Adapter 输出及比较表的身份与计数完成本机哈希/ID 核对；Base/Adapter 使用相同 1,833 条 Dev。
- Full B 训练完成；训练日志连续、梯度全有限；日志中抽样的 204 次 LoRA 参数更新检查均为非零。
- Adapter standalone reload 通过、64 Token IDs 一致、基础权重 SHA 不变。
- 质量提升明显，但情绪错误仍集中于少数混淆对；画像 Micro/Macro、非空 Recall、空预测基线给出不同信息，33 个标签差距明显。
- Full B Adapter 的格式失败 35 条，其中 29 条触及 `G=1,200`，32 条 JSON parse failure、3 条画像枚举错误；Submission 归一化合法率 100% 不代表 Raw 合格。

### 解释线索（`hypothesis`）

情绪频率偏斜、情绪标签边界、画像稀疏度、联合 JSON 目标中字段学习竞争、输出长度与推理时间之间的关系，都可成为下一步审查问题；本轮统计不能证明其中任何一个是性能差异的因果来源。不得用本 Dev 的错误案例直接改 Prompt 或选阈值。

### 未运行或不可从现有聚合结果回答

| 项目 | 状态与边界 |
|---|---|
| BERTScore | `not_run`；本地模型、Revision 和固定实现未验收 |
| 官方性能 / 目标平台吞吐 | `not_run`；本地 Runner 100 轮不等价官方容器/服务 |
| vLLM Token ID 等价、Adapter 加载、服务端 K | `not_verified` |
| 不超过 50 GB 的离线 Docker 镜像与官方 `start.sh` | `not_verified` |
| 回复人工质量盲审 | `not_run` |
| 逐样本 BLEU/ROUGE 与长度相关性 | 评分产物只保存聚合分数；本轮遵守“不重新评分”边界，未计算 |
| Full B 训练代码 Git Commit | Run Manifest 未记录训练源码提交或文件 Hash；只能确认 Run/配置/数据/权重身份，不能倒推原提交 |
| Hub Revision | `unverified_revision`；本地权重文件 SHA 有观测值 |

## 八、交接

当前实验完成状态、完整阶段链、运行身份、复现入口与下一步边界见[唯一当前交接文档](../context/E01_当前状态与后续交接.md)。权威总体验收指标仍见[正式 SFT 与 Dev 对照验收报告](E01-B_正式SFT与Dev对照验收报告.md)。本报告不批准新训练、改 Prompt、改 Loss、改标签或部署方案；下一步先由团队审阅这些逐类/逐标签证据，再单独决定工作优先级。
