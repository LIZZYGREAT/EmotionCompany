# E03-A8：困难样本 Replay 自动挖掘、分阶段训练与回溯——Codex 执行任务

## 一、任务目标与执行边界

**目标**：在已有 A8 Profile-only 模型上进行一次完整的困难样本 Replay 实验。自动从 Train/Dev 挖掘模型容易预测错误的画像样本，集中重复训练，在固定验证集上检查收益；若小规模试验通过预设门槛，则自动继续扩大一段训练。全程无需中途等待用户审批，**但任何阶段失败必须停止扩展、保留结果，并确保原模型能立即恢复**。

只处理 A8 的 **P（Profile-only）**，不启动 EP、Response 训练，不修改当前三行 `Personality=...` / `Interests=...` / `Style=...` 生成协议、现有专用 Prompt、官方标签 Schema 或 Completion-only Loss。

**当前父模型（必须精确核验）**：

- 工作分支：`exp/e03-a8-generative-specialist`；制定本方案时 HEAD：`d55510aafb85feb319ccf7970c55fa84a9a8c94e`。执行前记录实际 Git HEAD、dirty 状态；在独立新分支实现，例如 `exp/e03-a8-hard-replay`。
- Base：`Qwen/Qwen3-1.7B`，NF4/BF16；Base 权重 SHA-256 `005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5`。
- 已完成 Train 4,048 优化步 + DevFT 456 优化步；父 Run：`E03-A8-DEVFT-20261108-02-PDEV-1EPOCH`。
- 父 Checkpoint：`outputs/training/E03-A8-DEVFT/P/E03-A8-DEVFT-20261108-02-PDEV-1EPOCH/trainer/checkpoint-0456/`；总优化步记为 **4,504**。
- 父 Checkpoint Manifest SHA-256（实验报告记录）：`79f59d95247771160198f7e5dcecfe05ea754f222d3a2236f72f0eeadceb9c79`；最终 Adapter 权重 SHA-256：`fa09260cb95f07e4fe25dc886c2d4ef31bd6bf65d3449fe0855370d76e7e851f`。运行时重新计算并核对。若实际目录缺失或 SHA 不匹配，停止，不替换为其他模型。
- 当前 Val 本地代理基线：2,354 条，画像 Micro-F1 **0.4678**，Precision **0.5156**，Recall **0.4280**，严格格式 2,352/2,354。此分数仅用于背景；自动门槛必须使用**同一冻结子集上的逐样本配对计分**。

约束：不能写入或覆盖父 Run、父 Checkpoint、既有 Val 预测文件、P4048、A8-V2 原训练数据；所有 Replay 产物独立放置。禁止使用 Val Gold 做困难样本发现、排序、采样、重采样或训练。**允许新程序一次执行整个流水线**，仅根据冻结门槛自动决定扩展或停止，绝不自动部署失败候选。

## 二、先保存好当前模型：备份与恢复硬门槛

**任何模型推理、困难样本挖掘或训练之前，先完成这一步。**

1. 读取父 `checkpoint-0456` 及其 `checkpoint_manifest.json`，核验 Adapter、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json`、所有文件大小与 SHA；核验父 Run/Arm、4,504 总步数、DevFT 456 步、1,821 条 Dev 的游标以及 Base/Prompt 身份。
2. 创建两份**真实、独立的文件副本**：
   - 长期归档：`~/model-archives/EmotionCompany/E03-A8/P4504-<parent-manifest-sha-prefix>/`（位于仓库 `outputs/` 目录之外）。
   - 本实验快照：`outputs/training/E03-A8-REPLAY/baselines/P4504-<parent-manifest-sha-prefix>/`。
   不使用指向父目录的软链或硬链接，不借助父 Run 目录的可写引用；复制后独立递归核验每个文件的 SHA。预检查磁盘可用容量；不足则阻止训练，不能删旧模型腾空间。
3. **确保 Base 也可找回**：`participant/model` 可能是指向外部 Hugging Face 缓存的软链接，因此长期归档必须额外完整复制一次其实际指向的 Qwen Base 配置、Tokenizer 和权重文件到 `~/model-archives/EmotionCompany/base/Qwen3-1.7B-<weight-sha-prefix>/`，逐文件核验且拒绝以软链接代替。若该 Base 已存在同 SHA 的完整独立归档，则复用并验证；磁盘不足则停止任务，不能留下只有 Adapter、却无法独立加载的“备份”。本实验快照中的 Adapter 以归档 Base 的文件身份进行绑定，无须为每个阶段重复复制 Base。
4. 归档目录另存 `identity.json`（父 Run、Checkpoint、Base 归档实际路径及 Hash、Prompt、Tokenizer、代码 Git SHA、依赖版本、逐文件哈希）和 `RESTORE.md`（明确从 Base 归档+父 Adapter 加载的命令、验证步骤及恢复训练的状态文件）。新增归档路径处理器只允许解析到预设归档根目录；**不要复用 A8 限制路径必须在仓库内的 `_resolve()`**。
5. **恢复冒烟测试**：从归档的 Base + Adapter 副本加载模型，并对至少 5 条固定的、只含 Prompt 的输入核对下一 Token logits/argmax 与原父模型一致（允许按真实 BF16/NF4 数值容差比较 logits，argmax 必须相同）；检查 Base SHA 未变。只有 `baseline_backup_gate=passed`，才能进入下一阶段。
6. 归档视为只读快照：Replay 训练器必须拒绝任何以父 Checkpoint、长期归档、基线快照作为输出路径的写入。执行结束后再次核验两份备份完整且与起始哈希一致。

每次 Replay 新 Run 也按“**Checkpoint 完整状态 → 独立重载 → 哈希清单 → 通过才可进入下一阶段**”保存。任何意外停止都不得覆盖上一个成功状态。最终仅写 `selection.json` 指向被选中的已有候选；不得直接覆盖父 Adapter 或更新正式推理入口。

## 三、困难样本挖掘：先决定候选，再用模型真正预测

### 1. 数据来源与可用样本

使用现有 A8 已编码的两份 P 数据，不重新构造 Prompt：

| 来源 | 文件 | 有效数量 |
|---|---|---:|
| Train | `outputs/data/e03_a8_v2/p_train.jsonl` | 16,192 |
| DevFT | `outputs/data/e03_a8_devft_v1/p_devft.jsonl` | 1,821 |
| 合计 | | **18,013** |

从 `data/processed/v1/{train,dev}.jsonl` 按 `sample_id` 提取 Gold 画像和 `conversation_id`；交叉检查已编码 Completion 解码后的字段与 Gold 完全一致。训练输入只能使用 `input_ids[:prompt_length]`，绝不把当前 Gold Completion、`target` 或现有预测塞进模型 Prompt。核验 Train/Dev ID 无重复，Prompt、EOS、Tokenizer、T1 历史及 33 个标签枚举匹配原 A8。发现不一致必须停止。

为限制挖掘推理量，先从 18,013 条中**确定性选取 3,000 条候选**，默认 Train 2,700 + Dev 300；若实际不足，记录原因并按预冻结规则从另一来源补齐，不可临时根据 Val 分数改比例。

### 2. 候选 3,000 条的确定性覆盖抽样

按 Seed=42，先从训练 Gold 和元数据构建索引（**此时尚未加载 GPU 模型**）：

- Gold 是否三维全空；非空维度集合；正标签数；33 标签频率；历史消息数 / Prompt Token 长度；`conversation_id`；Train/Dev 来源。
- 目标约 **75% 非空 Gold、25% 全空 Gold**，在 Train 与 Dev 内分别按源比例满足；这样能覆盖漏报与空标签误报，且不预先把全空样本全部丢弃。
- 非空部分使用“稀有标签优先覆盖 + 三维覆盖 + 多标签覆盖”的确定性贪心选样。对于每个官方标签，优先保证最多 `min(10, 当前来源可用正例数)` 个不同样本的覆盖，剩余名额按类别、标签数量和 Prompt 长度分层补足；**不可实现的覆盖目标记录为不足，不重复伪造数据**。
- 优先每个 `conversation_id` 至多选 5 个不同预测点，超出时换其他会话；若名额不足才按固定规则逐级放宽，记录放宽量。标签稀有程度使用 Train+Dev 内部频率，不使用 Val。
- 所有同分选择用 `sha256(seed|source|sample_id)` 排序决定；固定候选文件、完整 ID 顺序、源数据和算法 SHA。保存 `candidate_ids.jsonl`、`candidate_audit.json`，不可在模型预测后回头更换 3,000 条候选。

### 3. 父模型预测与困难度计算

只加载**归档核验过的 P4504 Adapter**，对上述候选执行原 A8 P Prompt 的 Greedy 生成，生成配置、EOS、解析器与既有 Val 评测保持一致。首先将预测 `candidate_predictions.jsonl` 及 SHA 冻结；之后读取候选 Gold 做差异统计。不可用 Teacher Forcing Loss 替代真实 `generate()` 的预测错误。

对第 `i` 个样本，在 **33 个官方标签空间**计算：

\[
FN_i = |G_i \setminus \widehat G_i|,\quad
FP_i = |\widehat G_i \setminus G_i|,\quad
TP_i = |G_i \cap \widehat G_i|.
\]

其中三维各自使用 `(dimension, label)` 作为唯一键，不能混淆相同字符串；严格解析失败单独记录 `invalid`，不能把失效输出当成模型主动预测三个 `-`。

在挖掘结果上打标签：

- `miss`：`FN>0, FP=0`，正确标签漏报；
- `extra`：`FP>0, FN=0`，错误标签误报；
- `mixed`：`FN>0, FP>0`，漏报与误报同时发生；
- `clean`：合法输出且 `FN=FP=0`，普通维持样本；
- `invalid`：解析失败，单列分析；仅允许少量进入困难池，且监督仍使用正确的原始 Gold Completion。

定义可复现的难度分数（用于**同一错误类型内部**排序，错误类型的配额由下一节决定）：

\[
H_i = 2\min(FN_i,3) + 1.5\min(FP_i,3)
     +0.5\mathbf{1}[FN_i>0\land FP_i>0]
     +0.25\min(R_i,2),
\]

其中 \(R_i=\sum_{j\in G_i\setminus\widehat G_i}\min\bigl(2,\sqrt{f_{\max}/\max(f_j,1)}\bigr)\) 表示**漏报标签的有限稀有度加分**，`f_j` 只来自 Train+Dev，禁止 Val 参与。若 `invalid`，不给虚构的 FP/FN 分值，单独排队；优先利用真实标签错误样本。

按 `H` 降序、再按确定性 SHA 排序，保存 `hard_samples.jsonl`，每行包含 sample_id、source、conversation_id、Gold/预测的结构化 33 标签、FN/FP、分维错误、difficulty、group、重复候选权重与来源哈希。逐样本 Gold 与历史文本只写入 Git 忽略的 `outputs/`；Git 仅提交统计摘要、代码和方案，不上传敏感内容。

**挖掘验收**：输出候选 3,000 条的 `miss/extra/mixed/clean/invalid` 计数，三维和 33 标签 FN/FP Top 排名，空 Gold 误报数、实际非空预测率、按会话去重数量。至少能在合法预测中找到两类错误和足够样本供 512 次曝光；若不足，不伪造困难样本，标记 `mining_insufficient` 并停止训练。只在 Train+Dev 上分析，不能用 Val 错误清单补足。

## 四、自动冻结 Replay 样本表与曝光顺序

**Replay 是重复消费困难样本的 Gold Completion**。将原有 `input_ids/labels` 直接用于 Completion-only SFT，Prompt/答案不变；所有重复仅体现在 sample_id 在 Replay 调度中多次出现。**不修改 Gold、不产生伪标签、不额外调大字段 Loss 权重。**

首先一次性预生成并冻结 **384 个优化步、1,536 次样本曝光** 的 `replay_schedule.jsonl`。其中前 128 步为 Pilot，后 256 步为扩展段，后续是否执行只由固定质量门槛决定，不因 Pilot 结果改抽样表。两段都使用以下初始配额：

| 桶 | 曝光占比 | Pilot 512 次曝光（精确整数） | 样本选择来源 |
|---|---:|---:|---|
| 漏报重点 | 45% | 230 | `miss` + `mixed` 中 FN 较重的样本 |
| 误报重点 | 25% | 128 | `extra` + `mixed` 中 FP 较重的样本；包含空 Gold 误报 |
| 普通维持 | 30% | 154 | `clean` 及其他代表性 Train/Dev 样本，包含非空与全空 |

`mixed` 只归入一个主桶，按 `2FN` 与 `1.5FP` 较大者归类；相等按固定 SHA 决定，避免配额重复占用。`invalid` 上限为困难曝光的 5%，没有足够真实错误前不优先纳入。实际各维度份额依据**挖掘出的 Train+Dev 错误**确定，防止少数标签吞掉全部预算。

冻结调度必须同时满足：

1. `micro_batch=1`、`gradient_accumulation=4`；每 4 个曝光构成一个优化步。Pilot 恰好 128 步/512 曝光；扩展恰好新增 256 步/1,024 曝光。
2. 困难样本必须**真正出现重复**：Pilot 困难曝光中的至少 20% 为同一 sample_id 的第二次出现；Pilot 单一样本最多 2 次，扩展后累计最多 4 次，普通锚点单阶段优先只用 1 次。数量不足则先降低重复要求并明确记录，不因无法满足而复制单个样本几十遍。
3. 同一优化步中同一 `sample_id` 不重复；尽量避免四条样本来自同一 `conversation_id`。每段 32 步限制每个会话的曝光占比；无法满足时按确定性回退规则记录，不得悄悄取消限制。
4. Train/Dev 原样本均可选，但不能因为 Dev 是最近续训数据就让 Replay 极度偏向 Dev；按实际样本池来源至少报告曝光比例，并设置 Dev 总曝光不超过 25%。
5. 同桶内依据 `H` 排名分层轮转，不是只重复前 10 条；覆盖至少三个画像维度、常见与低频标签、长短历史；`clean` 从未用于困难排序的普通候选中固定抽取，不按 Val 成绩筛选。
6. 保存每条样本的被消费次数、所属桶、Gold 非空、会话、排程位置，以及整个调度的 Hash。Pilot 和扩展共用这一份不可变 Schedule。

推荐实现：先在每个桶构建按难度分层的唯一 ID 队列；用 **先轮转抽取一次、再给高优先级困难样本有限第二次曝光** 的规则补足配额；每一步做确定性冲突消解，再冻结调度。若某桶不足，按 `mixed → 其他困难桶 → clean` 的预设顺序补充并记录实际偏差，不能从 Val 借样本。

## 五、自动执行：基线 → Pilot → 质量门禁 → 扩展 → 最终选择

### 1. 训练条件

- 从**只读基线快照**加载 P4504 的 LoRA、AdamW、Scheduler 和 RNG 检查通过；训练写入独立 Replay Run。
- 继续使用 `src/training/e03_a8_completion_loss.py::completion_only_forward_loss`，`SFTDataCollator`、NF4/BF16、LoRA `r=8/alpha=16`、Microbatch 1、累积 4。仍对原 Gold Completion（含 EOS）计算 Token CE。
- AdamW 从父 Checkpoint 恢复动量与 step 计数，Replay LR 初值 **`2e-5`**，weight_decay 0、constant scheduler。**加载优化器/调度器后显式设置所有 param_group LR 与 scheduler base_lrs，做一个无更新验证，保证首个 Replay optimizer.step() 真正使用 `2e-5` 而非被恢复状态重置为 `1e-4`**；记录 `parent_global_step=4504` 和 `replay_step` 两套步数。
- 第 128、256、384 个 Replay 步保存完整训练状态；每个检查点包括 Adapter、Optimizer、Scheduler、RNG、sample cursor、训练参数、源数据与调度 Hash、分段耗时、完整逐文件哈希。每次保存后用新模型实例独立重载并比较 LoRA Tensor SHA 和固定 Prompt 下一 Token；失败则不训练下一阶段。
- 每 32 步缓冲写入 Ledger；避免每步的密集 GPU→CPU 同步；记录 GPU/P50/P90、Loss、非空/全空与三维样本曝光。Checkpoint/Manifest 原子写，拒绝覆盖。

### 2. Val 固定分组与结果对比

本次 **只把 Val 用作质量评测**，不用于挖掘、Gold 重构或 Replay 调度。沿用已冻结的 2,354 条 target-free Val 输入，`conversation_id` 分组，先冻结 `val_gate` 与 `val_confirm` 两个**按会话互斥**的 ID 集合：

- 优先把 A2 既往探索的 **78 段会话**单列为 `historical_exposed`，不用于自动扩大门槛；不只排除 209 个点，因为同会话历史可能重叠。
- 其余会话按 `sha256(42|conversation_id)` 确定性排序，约 25% 会话给 `val_gate`，其余给 `val_confirm`；要求两集合都具有充分的预测点、Gold 标签和覆盖三维的正例，无法满足则停止且报告，不用标签内容反向修改划分。
- 对 `val_gate`、`val_confirm` 固定 ID/会话 SHA，禁止因成绩调整。已有完整 Val 结果曾被查看，所有结果定位为**本地代理评测**，不宣称新增独立官方盲测。
- 基线预测直接复用现有已冻结的 P4504 Val 结果：`outputs/evaluation/e03_a8_devft_v1/E03-A8-DEVFT-20261108-02-PDEV-1EPOCH/predictions.jsonl`，核验 SHA-256 `493b9a973a45b893b6dc134872c1bd0b2df06ba49b1319ef92c47652c7b6faac` 和样本 ID。如果缺失，用备份 P4504 对完全相同的 target-free 输入**重新预测并单独冻结**，确保比较协议一致。
- Pilot 之后，只在 `val_gate` 对父模型和 Replay128 计算 33 标签 Micro-F1、Precision、Recall、TP/FP/FN、三维 F1、严格格式/EOS。用**同一样本的真实 Greedy 输出**计分，不使用训练 Loss 判断效果。

### 3. Pilot（Replay Step 1–128）与自动扩展条件

先训练 128 步（512 次曝光），保存 `checkpoint-replay-0128`、`pilot_eval.json` 和全部文件 SHA。仅同时满足下列条件，自动追加 256 步：

- 所有工程检查、Checkpoint 独立重载、源/调度身份与 EOS 检查通过；Pilot 严格格式解析率 **≥99%**。
- `val_gate` Micro-F1 相对同子集父模型 **提升至少 0.003（绝对值）**，且 `ΔTP>0`。
- Precision 相对父模型下降 **不超过 0.02（绝对值）**；三维任一维度 Micro-F1 下降 **不超过 0.02**。
- 与父模型相比新增 FP 不超过 `1.5 × max(新增 TP,1)`（这里新增 TP 指同集 TP 数量净增长）。

如果任意条件不满足，记录 `pilot_gate=failed`、失败指标与本次 `Replay128` 的所有产物，**自动停止**；`selection.json` 继续指向只读 P4504 基线。不允许改阈值、改样本表再在同一 Val Gate 上不断尝试，不能把失败模型覆盖到正式入口。

### 4. 通过后扩展（Replay Step 129–384）

若 Pilot 通过：

1. 从 Replay128 的 **Adapter + AdamW + Scheduler + RNG + Cursor** 完整恢复，按冻结 Schedule 续训 256 步；不得从父 P4504 重启训练、不得重新挖掘或重排。
2. 在 Replay Step 256 保存中间完整 Checkpoint；最终 Step 384 保存完整 Checkpoint。分别进行独立重载并验 SHA。中途异常保留最近已通过的 Checkpoint，Run 标记为 `interrupted` 或 `failed`。
3. **仅在 Step 384 完成后**，对 `val_confirm` 进行一次父模型 vs Replay384 的配对评测；可额外报告 `val_gate` 变化，但不能据此反复修改规则。最后可在完整 Val 上报告只读汇总，明确其中包含 `historical_exposed`。
4. 最终候选通过条件：`val_confirm` Micro-F1 至少高于同集父模型 **0.003**，Precision 下降不超过 **0.02**，三维任一 F1 下降不超过 **0.02**，严格解析率 ≥99%，Checkpoint 核验全部通过。若不通过，仍保存 Replay128/256/384，最终 `selection.json` 保持 P4504 基线。
5. 若通过，`selection.json` 指向 `Replay384` 的不可变 Checkpoint，并记录相对于 P4504 的所有分子分母增减。**此操作只是登记本实验选出的候选，不自动改写 `participant/`、不覆盖父 Adapter。**

该自动门槛用于控制额外训练开销，改善是否稳健仍须以实际数值和样本分布判断；绝不将刚好跨过阈值描述成统计显著结论。

## 六、建议代码落点与主要伪代码

在新实验分支中增加：

```text
configs/experiments/e03_a8_replay.yaml
src/data/e03_a8_replay.py                 # 候选抽样、逐样本画像差异、难度、曝光调度
scripts/e03_a8_replay.py                 # backup / mine / train / eval / run-all / restore-check
src/evaluation/e03_a8_replay_metrics.py   # 配对差值、门禁、会话分组
tests/test_e03_a8_replay.py              # 回归测试
docs/experiments/E0.3-A8_Replay实验结果.md
outputs/training/E03-A8-REPLAY/...        # Git 忽略
outputs/evaluation/e03_a8_replay/...      # Git 忽略
```

实现尽量直接复用：`scripts/e03_a8_devft.py` 的父模型加载/全状态恢复与 Val 冻结入口、`src/training/e03_a8_completion_loss.py` 的 Selective Completion Loss、`src/data/sft_collator.py`、`src/shared/e03_a8_protocol.py`、现有 33 标签 `profile_metrics` 和会话 Bootstrap。不得复制出不一致的新 Prompt、评分口径或 Tokenizer。

```python
# run-all；所有步骤幂等：只能校验并续接已确认的同身份部分，禁止覆盖
identity = resolve_and_verify_parent(P4504_checkpoint)
backup = create_two_independent_verified_copies(identity)
assert backup.restore_smoke_passed

sources = validate_train_dev_encoded_rows_and_gold()
candidate_ids = frozen_stratified_select(sources, n=3000, seed=42)
parent_preds = greedy_predict_from_backup(candidate_ids, prompt_only=True)
freeze_predictions_and_hash(parent_preds)
hardness = score_gold_miss_extra_mixed(candidate_ids, parent_preds, sources.gold)
freeze_hardness_and_schedule(hardness, replay_steps=384, accumulation=4)

val_splits = freeze_conversation_disjoint_val_splits(seed=42)
parent_val = verify_frozen_parent_val_predictions(val_splits)

pilot = resume_from_archive_and_train(schedule.steps[1:128], lr=2e-5)
verify_and_freeze_checkpoint(pilot, replay_step=128)
pilot_gate = evaluate_and_gate(parent_val, pilot, val_splits.gate)

if not pilot_gate.passed:
    choose_candidate("P4504", reason=pilot_gate.reasons)
else:
    mid = resume_and_train(pilot, schedule.steps[129:256])
    verify_and_freeze_checkpoint(mid, replay_step=256)
    final = resume_and_train(mid, schedule.steps[257:384])
    verify_and_freeze_checkpoint(final, replay_step=384)
    final_gate = evaluate_and_gate(parent_val, final, val_splits.confirm)
    choose_candidate("Replay384" if final_gate.passed else "P4504",
                     reason=final_gate.reasons)

verify_backup_unchanged()
write_aggregate_report_and_restore_commands()
```

> 伪代码 `steps[1:128]` 表示**包含两端**的优化步区间，代码实现中要使用明确的 `from_step_exclusive` / `to_step_inclusive`，不能因为 Python 切片索引造成丢步或重步。

建议 CLI：

```bash
.venv/bin/python scripts/e03_a8_replay.py run-all \
  --config configs/experiments/e03_a8_replay.yaml \
  --run-id E03-A8-REPLAY-<unique-id>

.venv/bin/python scripts/e03_a8_replay.py restore-check \
  --manifest <snapshot-or-checkpoint-manifest.json>
```

`run-all` 必须支持异常中断后的 `resume`：只允许在 Run ID、父模型 SHA、代码、配置、候选、预测、调度、Checkpoint 等哈希全部一致时继续缺失步骤。所有完成阶段原地只读；哈希不一致拒绝恢复，新建另一 Run。不得为了重跑而删除旧输出。

## 七、交付验收与结果报告

### 必须覆盖的测试

1. **备份恢复**：独立副本 SHA 完全一致；备份模型加载输出一致；父 Checkpoint 和备份在全过程中不变；输出路径不能指向原始/归档模型目录。
2. **样本隔离**：候选与 Replay 的 sample_id 只属于 Train/Dev；Val ID 永远不进入 `candidate_ids.jsonl` / `hard_samples.jsonl` / `replay_schedule.jsonl`。历史裁剪一致，训练 Prompt 不含 Gold Completion。
3. **困难度与选样**：合成 FN-only、FP-only、mixed、clean、invalid 样本得到正确分组、H 分数和顺序；多维标签 key 不冲突；稀有标签权重受限；样本重复、桶配额、会话上限与确定性 Seed 可复现。
4. **训练正确性**：Token CE 的 Prompt Mask/EOS/shift 不变；从 P4504 恢复 AdamW 动量/step；新 LR 在 `optimizer.step` 实际为 `2e-5`；Step128/256/384 的完整状态保存后独立重载可恢复；模拟 Step200 中断能从 Step128 恢复且不重复消费前 128 步。
5. **门禁与回退**：构造 F1 提升但 FP 暴涨、F1 降低、解析率不达标、Checkpoint 哈希损坏等情况，自动停止且 `selection.json` 指向 P4504；当结果全部通过才允许扩展并选择 Replay384。
6. **评分防混用**：同集、同 ID、同维度计算 TP/FP/FN 与 F1；Val 会话分组不交叉；Val Gold 不能进入挖掘/训练流程；生成文件先冻结，再读取 Gold 计分。
7. 提交前执行 `python -m compileall -q scripts src tests`、定向 `pytest`、已有 A8 相关 CPU 回归、`git diff --check`；GitHub CI 应通过。GPU 训练不得因本地权限问题静默降级为 CPU 或被假结果替代。

### 每个阶段必须落盘的文件

| 阶段 | 最小必需产物 | 恢复方式 |
|---|---|---|
| Baseline | 原父状态核验、双份完整副本、Manifest、`RESTORE.md` | 直接从 P4504 快照加载 |
| Mining | `candidate_ids.jsonl`、`candidate_predictions.jsonl`、`hard_samples.jsonl`、候选/预测 SHA、错误分布统计 | 不重新猜选样；按冻结文件复跑 |
| Schedule | `replay_schedule.jsonl`、每样本曝光次数、源/Gold/配置 Hash | 直接按 Schedule 的 cursor 恢复 |
| Replay128 | 完整 Checkpoint、Optimizer/Scheduler/RNG、训练 Ledger、`pilot_eval.json`、门禁结果 | 可独立加载 Pilot 模型 |
| Replay256 | 完整 Checkpoint + 独立重载证据 | 可从 256 步继续 |
| Replay384 | 完整 Checkpoint + `final_eval.json` + 逐维变化 | 可独立加载扩展模型 |
| 终态 | `selection.json`、总报告、所有阶段的路径/SHA、命令及回滚检查 | 按选择记录回到 P4504 或 Replay384 |

结果报告必须有：父 P4504、Replay128、Replay384（如果实际训练到）的同集指标；每阶段独立预测数、TP/FP/FN、Micro/Macro-F1、Precision/Recall、Personality/Interests/Style 分维指标、严格解析率；困难样本的四大类型数量、按标签错误统计、实际重复曝光数、总训练步数/耗时、Checkpoint SHA、质量门禁判断和最终指向的候选。未运行的阶段写 `not_run`，不能补造结果。

**一次性完成的定义**：Codex 实施代码并通过测试后，运行一次 `run-all`。它完成父模型归档、固定候选挖掘、Replay128、自动评测；若达标自动训练至 Replay384 并作最终确认；若不达标自动停止。无论最终结果如何，**原 P4504 和每一个实际完成阶段的完整 Checkpoint 必须仍然存在、可验证、可回退**。不得自动修改正式参赛推理入口。
