# E01-B Full SFT 与完整 Dev 配对验收报告

> 报告日期：2026-10-29<br>
> 训练 Run：`E01-B-FULL-20261028-133805`<br>
> Base Run：`E00.3-BASE-20261028-223105`<br>
> Adapter Run：`E01-B-EVAL-20261028-223105`<br>
> 配对比较：`full-20261029-062541`

## 1. 验收结论

E01-B Full 训练、最终 Adapter 独立重载，以及 1,833 条完整 Dev 的 Base/Adapter 生成、评分和配对比较均已完成。训练 Manifest、两个评测 Run 和配对比较状态分别为 `passed`、`scored` 和 `completed`。两组使用相同的 Dev ID 集；References 没有进入生成器。

Full B Adapter 在本地代理指标上显著优于未经微调的 Base：情绪 Accuracy 从 `0.1282` 提升到 `0.6536`，画像 33 标签 Micro-F1 从 `0.0782` 提升到 `0.4223`，Raw A0 Schema 合法率从 `0.0093` 提升到 `0.9809`。Pilot 阶段出现的近乎全空画像没有在 Full B 上重现。

所有质量指标均标记为 `local_proxy`，**不等同于官方比赛得分**。BERTScore 与官方性能口径为 `not_run`。本报告用于记录 Full B 实验验收、结果和局限，不批准任何 Prompt、Loss、模型或标签定义变更。

## 2. Full B 训练和重载

| 项目 | 实际结果 |
|---|---:|
| 初始化 | 固定 Base + 全新 LoRA；不继承 E01-A/Pilot Adapter |
| 训练样本 | 16,192 |
| Optimizer Steps / Epoch | 4,048 / 1 |
| 优化器 | AdamW，LR `1e-4`，constant scheduler，Seed 42，Micro Batch 1，梯度累计 4 |
| LoRA | r=8，alpha=16，dropout=0；q/k/v/o、gate/up/down projections |
| 量化 | bitsandbytes NF4，BF16 compute，double quant |
| Trainer 汇总 Train Loss | 1.678 |
| 首 10 步 / 末 10 步平均训练 Loss | 2.973 / 1.597 |
| 固定 SFT Dev Loss | Step 256：1.6609；Step 4048：1.4280 |
| SFT Dev Loss 样本数 | 1,821（teacher-forced eval only） |
| 已消耗输入 / Completion Token | 17,228,707 / 2,851,614 |
| 训练耗时（含 SFT Dev Loss 和 Checkpoint） | 23,066 秒（约 6 小时 24 分） |
| CUDA 峰值 allocated / reserved | 3.83 / 4.10 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GiB |
| 最终 Run 状态 | `passed`；`training_scope=full_train` |
| 最终独立 Adapter 重载 | `passed`；64 个 Greedy Token IDs 完全一致，Base SHA 未变 |

训练结束时的内置重载检查曾因训练与 Adapter 重载的量化 Base 精度准备路径不一致而失败。修复加载路径后，现有 Adapter 的独立重载检查通过；没有重跑训练。原始失败记录保留在 Run 的 `reload_check.json`，通过记录在 `reload_check_standalone.json`，Manifest 记录最终状态与检查历史。

训练身份指纹：

| 项目 | SHA-256 |
|---|---|
| 批准训练配置 | `25498728bb427806e281dda2a2790ecc2458755df23959c5464d078359f8781e` |
| D0.4.3 SFT Train | `072076d1a74fe1e29e89a598d2e6acee7188624de0738bfecfe3ea19267f1e79` |
| D0.4.3 SFT Dev | `6fdee702d434b01e2cac769bcdd97d5b984ce33273a484dcee9c20f1580e3e9f` |
| 训练样本选择 | `66b4115acf529ddd1e07ddf334a099fd9d1d75d68c2f96bb17b0fbb705aa64e2` |
| Base 权重 | `005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5` |
| Adapter 身份 Hash | `77b61e8a4caf46f54a8c813e98def488c7a8493fe891966e47df1cdac6df7d67` |
| Adapter 文件 Hash | `83d1ccb041b8b1c0403e7bcb6cafbf03d571801f23060e5f4fa7862f176ebc3d` |
| Adapter 配置 Hash | `05d578eb242868385839b2bc779c5cb88911973db766e2a793cb361b42aafafd` |

## 3. 配对评测范围与公平性

| 项目 | 结果 |
|---|---|
| 范围 | `full-dev`，每组 1,833 条 |
| Base / Adapter ID 集 SHA-256 | `739a88b6e47e053a1a358c1ac0f119de474522d0fa471bdbe4a8bd86bac1f322`（相同） |
| Dev Manifest SHA-256 | `36fe8a74249af9f602330927452918925b19b9ec1a22f11f0385f6ee1cb8573f` |
| 无标签推理输入 SHA-256 | `5105be04135d30fb853231892b65475d2cd9a35064dc25f705106af53b1f73dd` |
| Gold References SHA-256 | `570db280feaa0d7ec5356f9410826b8b0b515bbb62f7de290a3a112b9798c5a2` |
| ID Map SHA-256 | `fc677c688c0bd46d1851b3c3b37152d10a1f1972c55436e77946dab6ab8a1db9` |
| D0.3 Dev Manifest / Dev Data | `605867b5ab0162de05ea9e396bffd574cd4029c5adc3c29d851f41ff73a96b8d` / `bfacf9cd4a0e069e50862f973a98a3d379215231b0797e94ac76c5a70c5fbd78` |
| P0 SHA-256 | `d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2` |
| Tokenizer signature SHA-256 | `1662c9e99c6185441026a168b2dd6b09af103dc6dc3563b72f598411f14d1913` |
| Chat Template SHA-256 | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| Prompt 与解码 | P0/F2/T1，P=1280，G=1200，Greedy，`do_sample=false` |
| 量化 | NF4 / BF16 / double quant；运行时检查通过，无 offload |
| 生成器读取 References | Base 与 Adapter 均为 `false` |
| 评测完成状态 | Base `scored`；Adapter `scored`；Comparison `completed` |

两组模型权重和 Adapter 身份、输入 ID、协议、解码和评分器均通过比较器核验。Base 权重文件 Hash 已实测；本地报告仍将配置中的 Hub revision 标记为 `unverified_revision`，不把配置字符串当成远端 revision 的独立证明。

## 4. 主要质量结果

| 指标 | Base | Full B Adapter | Adapter − Base |
|---|---:|---:|---:|
| Emotion Accuracy | 0.128205（235/1,833） | **0.653573（1,198/1,833）** | **+0.525368** |
| Emotion Macro-F1 | 0.069040 | **0.497169** | +0.428129 |
| Profile 33-label Micro-F1 | 0.078225 | **0.422272** | +0.344047 |
| Profile 33-label Macro-F1 | 0.072443 | **0.270004** | +0.197562 |
| 三维 Profile Exact Match | 0.216039（396/1,833） | **0.320786（588/1,833）** | +0.104746 |
| Raw strict JSON | 0.461538（846/1,833） | **0.982542（1,801/1,833）** | +0.521004 |
| Raw A0 Schema | 0.009274（17/1,833） | **0.980906（1,798/1,833）** | +0.971631 |
| Submission Schema | 1.000000 | 1.000000 | 0 |
| 字符级 BLEU-4（百分比） | 2.3993 | **12.8212** | +10.4218 个百分点 |
| 字符级 ROUGE-L | 0.118824 | **0.260285** | +0.141461 |
| 空回复率 | 0.222040 | **0.017458** | -0.204583 |

情绪配对迁移为：Base 与 Adapter 同时正确 191 条；Base 正确而 Adapter 错误 44 条；Base 错误而 Adapter 正确 1,007 条；两者都错 591 条。按 826 个会话簇、Seed 42、2,000 次 Bootstrap 计算，Accuracy 配对差值 `+0.525368` 的 95% 区间为 `[+0.498346, +0.552043]`。

Adapter 各画像维度的 Micro-F1 为：`personality_traits=0.4189`、`interests=0.5392`、`style=0.3614`；对应 Macro-F1 为 `0.1404`、`0.4836`、`0.1548`。画像不再塌缩为空，但各维度的空预测率仍略高于 Gold 空值率，且 Macro-F1 明显低于 Micro-F1，说明尾部标签仍较弱。三维 Exact Match 的上升有 Micro-F1 与逐维指标支持，不能单独解释为画像整体已解决。

情绪方面，Adapter Accuracy 高于 Train 多数类基线 `0.327878`。仍需关注低频/难分类标签，例如 `care` 在 6 条 Gold 中没有命中，`mixed` 命中 5/93；整体 Macro-F1 低于 Accuracy，类别差异仍然明显。

## 5. Raw 输出归因与性能

Full B Adapter 的 1,833 条输出中，1,801 条通过严格 JSON 解析；1,798 条符合 Raw A0 Schema。32 条 JSON 解析失败，其中 29 条生成到达 `max_new_tokens=1200` 上限；另有 3 条可解析输出因画像字段不合法而不符合 A0。Runner 将这些失败行归一化后，最终 Submission Schema 合法率仍为 100%，因此不能用 Submission 合法率替代 Raw 格式指标。

本地 Runner 性能报告的 100/100 轮统计均已完成。Adapter 每条延迟均值/中位数/P95 为 `13.43 / 9.08 / 34.27` 秒；Base 为 `4.17 / 3.55 / 7.62` 秒。Adapter 平均延迟约为 Base 的 3.2 倍。计时范围为单样本 tokenize、generate、decode、parse 与 schema validation；模型加载和初始化不计入样本延迟。该计时来自本地 RTX 4060 Laptop 环境，不代表官方运行环境或官方性能口径；比较报告仍将官方性能标为 `not_run`。BERTScore 按冻结配置关闭，状态为 `not_run`。

## 6. 结论与后续

1. E01-B 的正式 Full 训练和公平完整 Dev 配对评测达到本阶段核心交付目标；这不是官方比赛成绩或最终部署批准。
2. 全量训练的画像表现不再是 Pilot 观察到的近全空输出。现有证据支持先记录为 E01-B 基线，不因单次 Dev 结果自动改 P0、标签语义或标准 Loss。
3. 后续如开展调优，应明确单变量假设、固定协议和预算，并避免把这次 Dev 的逐条案例直接用于反复调参；优先分析训练侧标签分布、输出长度与类别支持。
4. 需要权衡 Adapter 明显较慢的推理速度。29 条达到生成上限和 3 条非法画像是可继续诊断的工程问题，但应与模型质量实验分开。

完整产物（权重、预测、逐行案例、Runner 日志）留在 Git 忽略的 `outputs/`：

```text
outputs/training/E01-B-FULL-20261028-133805/
outputs/evaluation/e01_b/full/E00.3-BASE-20261028-223105/
outputs/evaluation/e01_b/full/E01-B-EVAL-20261028-223105/
outputs/evaluation/e01_b/comparison/full-20261029-062541/
```

本报告只保存聚合数据、实验身份、结论与局限，不包含逐条原始对话或生成文本。
