# E01-B 训练工程与 Pilot 验收报告

> 报告日期：2026-10-28
> 训练 Run：`E01-B-PILOT-20261028-092248`
> 评测范围：固定 160 条 `pilot-dev`；本报告不代表完整 Dev 泛化结果。

## 1. 验收结论

E01-B Pilot 训练、同 Run Checkpoint 恢复、Adapter 重载，以及 160 条 Base/Adapter 配对生成与离线评分均已完成。训练 Run 状态为 `passed`，两个评测 Run 状态均为 `scored`，配对比较状态为 `completed`。

Pilot Dev 上，Adapter 的情绪识别、Raw JSON/A0 格式合法率和回复本地代理指标均有改善；但画像 33 标签 Micro-F1 从 `0.0648` 降至 `0.0077`，几乎所有画像输出都为空。这是需要跟进的质量风险信号，不能由三维 Exact Match 的表面上升抵消。

本结果满足 Pilot 工程与诊断阶段的记录要求，**不等于批准 Full B，不等于完整 Dev 验收，也不构成官方比赛得分**。Pilot 质量指标按训练工程方案仅作为风险信号，不据此自动改变模型、Prompt、标签语义或训练损失。

## 2. Pilot 训练与恢复

| 项目 | 实际结果 |
|---|---:|
| 训练样本 | 512 |
| 最终 Optimizer Steps | 128 |
| 本次训练耗时（含 Dev Loss 与 Checkpoint） | 870.1 秒 |
| 已消耗输入 Token | 544,683 |
| 输入 Token 吞吐 | 626.0 Token/秒 |
| Pilot 序列长度范围 | 425–2,048 Token |
| Full Train 序列长度范围 | 425–2,048 Token |
| 首 10 步 / 末 10 步平均训练 Loss | 2.829 / 1.688 |
| 固定 256 条 SFT Dev Loss | Step 32：1.915；Step 128：1.779 |
| CUDA 峰值 allocated / reserved | 3.83 / 4.10 GB |
| 保留 Checkpoint | 128,367,460 字节 |
| 训练后磁盘可用空间 | 990,351,773,696 字节 |
| Adapter Reload Check | `passed` |
| Base 权重未改变 | 是；SHA-256 `005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5` |

训练 Manifest 记录了两次从同一 Run 的 `checkpoint-32` 恢复尝试，最终运行完成到 Step 128。恢复过程曾触发 LoRA 梯度检查异常；修复后的续训和最终重载检查均通过。最终 Adapter 权重 SHA-256 为 `f2fde06fcccbc668bceca74ff76fae56531f35c4291d37a1c3ea06de369374b2`，Adapter 配置 SHA-256 为 `b5c2bbe706e7e92f7a9eb16ecec477748fa4814758ce99a455adc630cf08bfe7`。

供复核的训练身份指纹：Pilot Manifest SHA-256 `f19fe4fa852b2ca60f75bcb7640677abaa66b8f80ecd3509b0f012192dea36be`；训练配置 SHA-256 `5c33502ead211096ca6de4fe0aa6365ab255b9172e0297951e8751447765a68c`；D0.4.3 SFT Train SHA-256 `072076d1a74fe1e29e89a598d2e6acee7188624de0738bfecfe3ea19267f1e79`；512 样本选择文件 SHA-256 `b53336224071914500aa91a4ad82c6f9e6334b11eb5a69231a6b4e20ba09314e`；Tokenizer 签名 SHA-256 `1662c9e99c6185441026a168b2dd6b09af103dc6dc3563b72f598411f14d1913`。

按输入 Token 吞吐线性外推，Full Train 一 Epoch 约需 `7.64` 小时。该数值没有计入长序列的非线性算力成本，也不能准确预测全量 SFT Dev Loss 与 Checkpoint 开销，**只能用于预算讨论，不是 Full B 批准或耗时保证**。

## 3. 配对评测身份与范围

| 项目 | 结果 |
|---|---|
| Base Run | `E01-B-PILOT-BASE-20261028-101136` |
| Adapter Run | `E01-B-PILOT-EVAL-20261028-101136` |
| Comparison Run | `pilot-20261028-101136` |
| 范围 | `pilot-dev`，两组均为同一批 160 个 ID |
| 推理是否读取 References | 两组均为 `false`；References 只用于后续评分 |
| 数据 Manifest SHA-256 | `d886f696a050a92f88e519a777323a030ccdd4855e7e4532eae89cbae5006559` |
| 无标签推理输入 SHA-256 | `63e1f14707cd66b1d9c41777aeabdb2a54d357fbe6109993507d33da4780162b` |
| Gold References SHA-256 | `b6ee4a702c915a5de8a95e2bb4f35f51b1592ab4d8fc91cc00f166460ed48955` |
| ID 集 SHA-256 | `98f24f1c1b41e1676618f4e557daa402b3cbb80686916ca818e1bdc1b89ae49c` |
| P0 SHA-256 | `d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2` |
| Base 权重 SHA-256 | `005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5` |
| 指标范围 | 全部是 `local_proxy` |
| BERTScore / 比较报告中的官方性能口径 | `not_run` / `not_run` |

Runner 对两组分别生成了 `submission.jsonl` 和 `performance_report.json`；文件格式检查通过。配对评测只改变是否启用 Adapter，比较程序完成了同 ID 离线评分。两组保存了独立配置快照（配置 Hash 因 Adapter 开关而不同）；对比器核对了模型、Tokenizer、Prompt、解码、数据 ID、Scorer 代码和运行环境。

## 4. 质量结果

| 指标 | Base | Adapter | 差值（Adapter − Base） |
|---|---:|---:|---:|
| Emotion Accuracy | 0.106250（17/160） | **0.406250（65/160）** | **+0.300000** |
| Emotion Macro-F1 | 0.052965 | **0.245617** | **+0.192652** |
| Profile 33 标签 Micro-F1 | **0.064838** | 0.007663 | -0.057175 |
| Profile 33 标签 Macro-F1 | **0.044269** | 0.001188 | -0.043081 |
| 三维 Profile Exact Match | 0.256250（41/160） | 0.375000（60/160） | +0.118750 |
| Raw strict JSON | 0.456250 | **0.912500** | **+0.456250** |
| Raw A0 Schema 合法率 | 0.012500 | **0.912500** | **+0.900000** |
| Submission Schema 合法率 | 1.000000 | 1.000000 | 0 |
| 本地 BLEU-4（字符级，百分比） | 2.4083 | **5.9667** | +3.5585 个百分点 |
| 本地 ROUGE-L | 0.114626 | **0.211950** | +0.097324 |
| 空回复率 | 0.243750 | **0.087500** | -0.156250 |

### 4.1 情绪识别

160 条配对迁移中，Base 与 Adapter 同时正确 12 条；Base 正确而 Adapter 错误 5 条；Base 错误而 Adapter 正确 53 条；两者都错 90 条。情绪 Accuracy 配对差值为 `+0.30`，按 147 个会话簇、Seed 42、2,000 次 Bootstrap 计算的 95% 区间为 `[+0.2169, +0.3782]`。区间显示该 Pilot 子集上的提升方向稳定，但样本范围仍只有 160 条。

### 4.2 画像退化与 Exact Match 解读

Adapter 的三个画像维度在各 160 条样本中都只预测出 **1 条非空画像**；对应 Gold 非空样本数分别为 `personality_traits` 73 条、`interests` 51 条、`style` 87 条。画像标签真阳性总数从 Base 的 13 降至 Adapter 的 1。

因此 Exact Match 从 41/160 升至 60/160 不能解释为画像变好：预测大量空画像会碰巧匹配 Gold 的空维度或全空组合。后续审阅要同时看 33 标签 Micro-F1、每维非空 Gold Recall、空预测率和逐类错误。

### 4.3 格式与回复

Adapter 原始输出的严格 JSON 解析率和 Raw A0 Schema 合法率均为 91.25%，Base 分别为 45.63% 和 1.25%。Submission Schema 两边都是 100%，但它经过 Runner 归一化，不能替代 Raw 格式合法率或模型字段正确性。

BLEU-4 和 ROUGE-L 是当前本地实现的回复代理指标，不是官方评分。Adapter 输出延迟高于 Base；单次评测的全体样本平均延迟分别约为 11.56 秒和 3.95 秒，中位数分别约为 5.52 秒和 3.58 秒。两个 Runner 报告含 100 轮本地延迟统计；配对比较仍将官方比赛性能口径标为 `not_run`，因此这里不将本地计时称为官方性能得分。

## 5. 画像数据分布与 Raw 输出归因

为判断空画像是否由 Pilot 选样造成，对 D0.3 Train/Dev、Pilot 512 个选中 ID 和 Pilot Dev 160 条 Gold 做了只读计数。Pilot 的画像非空比例与全量数据接近：Train 全空画像占 `37.27%`，Pilot Train 为 `38.48%`，完整 Dev 为 `36.01%`，Pilot Dev 为 `37.50%`。

| 数据范围 | 样本数 | 三维全空样本 | `personality_traits` 非空 | `interests` 非空 | `style` 非空 |
|---|---:|---:|---:|---:|---:|
| D0.3 Train | 16,312 | 6,079（37.27%） | 7,059 | 4,986 | 8,556 |
| Pilot Train 选样 | 512 | 197（38.48%） | 228 | 162 | 276 |
| D0.3 Dev | 1,833 | 660（36.01%） | 834 | 593 | 1,022 |
| Pilot Dev Gold | 160 | 60（37.50%） | 73 | 51 | 87 |

因此，当前证据不支持“Pilot 选样画像过度为空”作为主要解释；Adapter 每个画像维度仅预测一条非空值，与 Gold 的非空比例明显不符。

对 Adapter 的 160 条 Raw 与 Gold 按评测 ID 逐条对齐，160 个 ID 全部一一匹配，没有缺失或重复。严格解析和字段检查得到：

| Raw 输出归因 | 数量 | 说明 |
|---|---:|---|
| Strict JSON 解析且符合 Raw A0 Schema | 146 | 字段集合、情绪、画像维度与标签、非空回复和 `memory_refs=[]` 均合法 |
| 解析失败 | 14 | Runner 状态均为 `invalid_label`；全被归一化成空情绪/画像后，最终 Submission 仍符合 Schema |
| 解析失败且结束于 `max_new_tokens` | 11 | 达到 `G=1200`；其中 9 条为未结束字符串，另有 1 条非法转义、1 条非法控制字符 |
| 解析失败但以 EOS 结束 | 3 | 1 条非法转义，2 条 JSON 字符串含非法控制字符 |

最关键的是，在 **146 条合法 Raw A0 对象中，145 条三个画像数组都显式为 `[]`**；只有 1 条 Raw 输出了非空画像，且三个维度各输出一个标签：`personality_traits=sensitive` 命中 Gold，`interests=social` 与 `style=direct` 未命中。最终归一化的 160 条 Submission 中有 159 条三维画像全空；100 条 Gold 非空样本里，99 条得到全空画像，剩下 1 条也只命中一个标签。总标签计数为 TP=1、FP=2、FN=257。

这说明画像塌缩不主要来自 JSON Parser 把合法的非空画像解析成空：合法 Raw 的画像结构和标签均无 Schema 错误，模型本身就在绝大多数合法输出中选择输出空数组。14 条格式失败会再增加空字段和漏报；其中 11 条生成达到 Token 上限，另外 3 条虽以 EOS 结束仍有转义/控制字符错误。现有证据把问题定位到“有效生成中偏向空画像”与“少数长输出/字符串 JSON 格式失败”两部分，但不能单凭输出确定训练目标竞争、生成行为或其他因素各自的根因。

## 6. Completion-only 监督序列核查

本项核查只用 CPU 和 `participant/model/tokenizer.json` 的 Fast Tokenizer，没有加载模型权重或运行推理。按预编码 `input_ids`/`labels` 取出真实 Completion Token，到终端 EOS 前解码 JSON；再用同一 Tokenizer 编码并核对 Token IDs、JSON 字段及 D0.3 原始监督目标。

- D0.4.3 的 16,192 条 SFT Train 全部满足：Prompt 标签为 `-100`，Completion 标签与输入后缀逐 Token 相同，包含锁定 EOS `151645`，解码后的 JSON 与源目标情绪、画像标签和回复一致；没有发现序列化、错位或漏监督样本。
- 512 条 Pilot 选中样本也全部匹配；其 86,513 个 Completion 标签 Token 与训练 Run 记录的 `consumed_completion_tokens=86,513` 完全一致。
- Full Train 的 Completion 共 2,851,614 个受监督 Token；Pilot 共 86,513 个。按 Fast Tokenizer 的 offset，将每个 Token 分给字符重叠最多的顶层 JSON 字段块；若同一 Token 与多个字段块重叠相同，则计入 JSON 结构。字段块包含 key/value；Completion 总数另外包含 EOS 及其后的一个受监督空白 Token。
- Full Train 中 10,159 条目标含非空画像，共有 24,683 个画像标签值；Pilot 中 315 条目标含非空画像，共有 812 个画像标签值。画像监督数据并非缺失或只出现在极少数 Pilot 样本中。

| 监督字段/部分 | Full Train Token 数 | Full 占比 | 每条均值 | Pilot Token 数 | Pilot 占比 | Pilot 每条均值 |
|---|---:|---:|---:|---:|---:|---:|
| `response_text` 字段块 | 2,308,828 | **80.97%** | 142.59 | 69,284 | **80.09%** | 135.32 |
| `user_profile` 字段块 | 302,152 | 10.60% | 18.66 | 9,609 | 11.11% | 18.77 |
| `emotion_label` 字段块 | 94,908 | 3.33% | 5.86 | 3,012 | 3.48% | 5.88 |
| `memory_refs` 字段块 | 64,768 | 2.27% | 4.00 | 2,048 | 2.37% | 4.00 |
| 顶层 JSON 分隔结构 | 48,574 | 1.70% | 3.00 | 1,536 | 1.78% | 3.00 |
| 终端 EOS | 16,192 | 0.57% | 1.00 | 512 | 0.59% | 1.00 |
| EOS 后空白 Token | 16,192 | 0.57% | 1.00 | 512 | 0.59% | 1.00 |
| **全部 Completion** | **2,851,614** | **100%** | **176.11** | **86,513** | **100%** | **168.97** |

按相同 Tokenizer offsets 统计，回复文本字符串本身占 Full Train 总 Completion Token 的 `79.26%`（2,260,252 Token），画像标签字符串值占 `1.93%`（55,007 Token），情绪标签字符串值占 `1.06%`（30,140 Token）；Pilot 分别为 `78.31%`、`2.06%`、`1.11%`。这解释了为什么画像字段块总占比（还含字段名、维度名和 JSON 结构）高于标签值本身占比。

Pilot 和 Full Train 的字段 Token 占比很接近，因此 Pilot 选样的目标长度构成有代表性。回复字段块占 Completion Token 的约 81%，画像字段块约 11%；标签字符串本身占比更低。这是任务间 Token 曝光差异的线索，但不等于实际梯度占比：训练损失归一化方式、每条样本的平均损失和各 Token 梯度大小都会影响最终更新，不能单凭 Token 数认定画像退化的根因。

## 7. 产物位置与复现边界

真实模型、Checkpoints、训练日志、逐条输出、References 与预测均留在 Git 忽略目录：

```text
outputs/training/E01-B-PILOT-20261028-092248/
outputs/evaluation/e01_b/pilot/E01-B-PILOT-BASE-20261028-101136/
outputs/evaluation/e01_b/pilot/E01-B-PILOT-EVAL-20261028-101136/
outputs/evaluation/e01_b/comparison/pilot-20261028-101136/
```

配对汇总报告为 `outputs/evaluation/e01_b/comparison/pilot-20261028-101136/report.md`；本报告只保留聚合指标、身份指纹和解释，不复制逐 ID 原始输出或对话。

## 8. 下一阶段

1. 训练目标序列核查已完成：没有发现 Prompt Mask、Completion 对齐或监督目标序列化错误。当前证据提示回复 Token 占比远高于画像字段，但不足以单独证明这是画像塌缩的原因；暂不改 P0、标签定义或标准 Loss。
2. 如果目标仍是建立标准 E01-B 基线，按 [`docs/workflow.md`](../workflow.md) 的审批步骤人工冻结 Full B 总预算、学习率、Epoch 和保存/验证间隔，再用独立的新 LoRA 运行 16,192 条 Full Train。Full B 不继承 Pilot Adapter 或其优化器状态。
3. Full B 完成且重载通过后，另行决定并执行 1,833 条完整 Dev 的 Base 与 Adapter 生成，固定相同 ID、Prompt、Tokenizer、Greedy 解码与 Scorer，再作配对比较。
4. 根据完整 Dev 结果判断画像问题是否仍然存在，再决定是否设计单变量实验。Pilot 的 160 条结果不足以批准模型切换、Prompt 更改、额外分类头或加权损失。
