# EmotionalCompany

「数字人综合情感陪伴对话模型」赛道的建模与工程仓库。项目包含数据审计与构建、QLoRA 训练、离线评测，以及赛事推理入口。当前实现面向文本多轮对话，不包含语音、视觉或数字人形象模块。


## 任务与模型方案

给定截至目标用户消息的对话历史，系统一次生成结构化 JSON，完成三项任务：

- 从 16 个官方情绪类别中预测一个主要情绪；
- 预测 `personality_traits`、`interests`、`style` 三个维度的用户画像标签，共 33 个候选标签；
- 生成中文情感陪伴回复。

首轮采用单一 `Qwen/Qwen3-1.7B` 指令模型、4-bit NF4 QLoRA 和 Completion-only SFT。基础模型冻结，只训练独立 LoRA 参数；推理时在 Non-thinking 模式下生成完整 JSON。暂不引入额外分类器、字段加权损失、RAG 或其他模型。官方自动评分包括情绪、画像、回复和性能；专家评审还关注方案、训练记录、实验设计与可复现性。本地指标用于实验分析，不能直接当作官方最终分数。

训练与推理共用冻结协议：

| 协议 | 作用 | 当前版本 |
|---|---|---|
| P：Prompt | 定义任务、标签和 JSON 输出要求 | P0：`configs/prompts/p0.txt` |
| F：Format | 使用原生 `system/user/assistant` 消息和 Qwen Chat Template | F2 |
| T：Truncation | 在 Token 预算内保留最近连续完整历史，并保留最后一条用户消息 | T1 |

实验配置中常见的 `S=2048`、`P=1280`、`G=1200` 是当前数据和实验方案使用的预算；部署上下文长度 `K` 仍需在目标服务中核验，不能把这些值视为所有部署环境都已验收。

## 数据与执行流程

```text
官方 Train / Val
    → D0.1 数据审计与 D0.2 处理政策
    → D0.3 规范化、去重与数据划分
    → D0.4.2 Token 长度审计
    → D0.4.3 Completion-only 监督数据与 Batch 验收
    → E01 QLoRA 训练
    → 使用隔离的 Dev 输入进行生成与配对评分
    → 通过赛事入口输出提交文件和性能报告
```

当前数据账本：D0.3 为 Train 16,312、Dev 1,833、Val 2,354；D0.4.3 在 `S=2048/P=1280` 配置下生成有效 SFT Train 16,192 和 SFT Dev 1,821。Val 不进入 SFT 训练。Dev 的 SFT Loss 输入与生成质量评测输入分开管理；生成阶段只读取无标签 `id+history`，评分阶段才读取 Gold。

## 当前进度

| 阶段 | 状态 | 结果边界 |
|---|---|---|
| D0 数据链 | 已完成阶段验收 | 审计、清洗/划分、Token 审计和预编码 SFT 数据均有版本化报告；处理后数据为本机可重建产物。 |
| E00 评测与诊断 | 小样本阶段已完成 | 真实模型 Greedy/Sampling 与单字段情绪诊断均使用 20 条 Smoke；不是 1,833 条完整 Dev 或官方性能结果。 |
| E01-A 训练 | 工程验收通过 | 真实 QLoRA 完成 120 步并通过 Adapter 重载；8 条训练样本的拟合不能代表泛化，20 条 Dev Smoke 质量指标偏低。 |
| E01-B Pilot | 训练与恢复通过 | Run `E01-B-PILOT-20261028-092248` 完成 512 条、128 步训练，从同一 Run 的 step-32 Checkpoint 恢复；Adapter 重载通过，Base 权重未变。固定 256 条 SFT Dev Loss 从 step 32 的 1.915 降至 step 128 的 1.779。 |
| E01-B Pilot Dev 生成评测 | 已完成（160 条诊断） | Emotion Accuracy 从 0.106 升至 0.406，Raw A0 从 0.013 升至 0.913；Profile Micro-F1 从 0.0648 降至 0.0077，画像几乎恒为空。仅为 Pilot 风险信号，不是完整 Dev 结论；见 [Pilot 验收报告](docs/experiments/E01-B_训练工程与Pilot验收报告.md)。 |
| E01-B Full 训练 | 已完成 | Run `E01-B-FULL-20261028-133805`：16,192 条有效 Train、4,048 步、1 Epoch；独立 Adapter 重载最终通过，Base 权重未变。 |
| E01-B 完整 Dev 对照 | 已完成（1,833 条） | Base/Adapter 使用相同 D0.3 Dev IDs。情绪 Accuracy 0.1282→0.6536，画像 33 标签 Micro-F1 0.0782→0.4223，Raw A0 Schema 0.0093→0.9809；均为本地代理指标。见[正式验收报告](docs/experiments/E01-B_正式SFT与Dev对照验收报告.md)和[深度诊断](docs/experiments/E01-B_Full_深度诊断与证据整理.md)。 |
| 部署 / 官方指标 | 未完成 | BERTScore、官方性能、目标服务 vLLM/K/Token 等价、官方 `start.sh` 和 ≤50GB 离线 Docker 均未验收。 |

Pilot 的训练 Loss 和 SFT Dev Loss 只说明模型在 Completion-only 训练目标上的变化，不替代情绪、画像、Raw JSON/Schema、回复和性能评测。Pilot Run 的 Manifest、日志、Checkpoint 和权重保存在忽略目录 `outputs/training/`，以上摘要来自执行机上的真实结果，不是 Git 中的实验产物。

Full B 与完整 Dev 对照已有实际 Run；README 中的下一阶段是**结果审阅与部署验收决策**，不是再次启动 Full B。实际 Run 路径、诊断命令和未来新 Run 的资源门禁/执行配方见 [`docs/workflow.md`](docs/workflow.md) 与[当前状态交接](docs/context/E01_当前状态与后续交接.md)。

## 仓库结构

```text
configs/                  数据、训练、评测和 Prompt 配置
data/                     官方数据输入与本机生成的数据；原始数据受保护
participant/              赛事推理入口、参考实现、Schema 和配置
src/data/                 审计、划分、Token 分析、SFT 数据和 Collator
src/shared/               训练与推理共用的消息、Prompt 和历史选择逻辑
src/training/             QLoRA 训练与 Checkpoint/Resume 流程
src/evaluation/            Dev 输入准备、评分与指标
scripts/                  Smoke、诊断、训练审批和 E01-B 评测命令
tests/                    协议、数据、评分、训练和恢复回归
docs/                     架构、数据审计、实验报告、运行手册与协作规范
outputs/                  本机预检、Checkpoint、预测和报告；Git 忽略
```

`data/processed/`、`outputs/`、`participant/model/` 和 Hugging Face 缓存不由 Git 管理。新机器不能仅凭克隆仓库推断已具备原始数据、处理后数据或模型权重；请按数据来源、哈希和环境指南准备并核验。

## 开发与运行

### 开发环境和回归

项目已验证的开发环境及 GPU/依赖验收步骤见[完整环境配置与复现指南](docs/完整环境配置与复现指南.md)。在依赖和本机数据准备完成后，从仓库根目录运行：

```bash
source .venv/bin/activate
python -m pip check
python -m pytest tests -q
```

完整数据重建、Pilot、Full B 门禁和 Dev 评测命令统一维护在 [`docs/workflow.md`](docs/workflow.md)，README 不替代各阶段的运行合同。

### 赛事推理接口

赛事平台调用入口必须保持为：

```bash
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

`start.sh` 固定激活 `conda_env`，接收测试 JSONL 路径和结果目录两个位置参数，并生成：

- `RESULT_DIR/submission.jsonl`：符合官方提交 Schema 的逐样本预测；
- `RESULT_DIR/performance_report.json`：本次推理的运行信息与性能摘要。

入口和输出文件名不能改变。完整配置、后端、错误处理和本地诊断说明见 [`participant/README.md`](participant/README.md)。

## 重要边界

- 先阅读[赛事文件边界与运行契约](docs/competition/赛事文件边界与运行契约.md)。`data/train/train_public.jsonl`、`data/val/val_public.jsonl`、官方 Schema、`configs/prompts/p0.txt` 及既有 P0/F2/T1 与 D0.3 划分均受保护。若确需修改，必须说明来源、建立版本记录并提供回归覆盖。
- `participant/run_inference.py`、对应配置和依赖是可编辑的参考实现；改动需保留官方入口、输出 Schema 和默认运行行为，并通过协议回归。
- 不提交真实数据、模型权重或缓存、生成预测、Checkpoint、Raw 输出及 `outputs/` 内容。
- 方案语义变更要建立新协议或数据版本；可证明的 Bug 修复需增加针对性回归。
- Git 分支、明确暂存、提交、PR 和 CI 流程见 [`AGENTS.md`](AGENTS.md) 与[双人 Git 协作规范](docs/knowledge/双人Git协作与PR操作规范.md)。
- 专家评审所需的技术方案、实验设计与报告、平台训练/调优记录，应按当期赛事通知随镜像整理交付。
- 交付镜像需支持离线推理并满足 50 GB 上限；模型后端、Adapter 加载和实际性能仍须在目标镜像中验证。

## 文档导航

| 想了解 | 文档 |
|---|---|
| 唯一当前阶段入口与后续交接 | [E01 当前状态与后续交接](docs/context/E01_当前状态与后续交接.md) |
| A0 架构、建模任务与实验原则 | [A0 技术架构与实验方案](docs/task/A0_技术架构与实验方案.md) |
| 分阶段命令、运行产物和人工门禁 | [项目工作流](docs/workflow.md) |
| 原始数据审计、处理政策与数据构建 | [D0.1 报告](docs/data/D0.1_数据质量审计报告.md)、[D0.3 报告](docs/data/D0.3_数据构建报告.md)、[D0.4.3 报告](docs/data/D0.4.3_监督数据构建与Batch验收报告.md)及其引用文档 |
| E00 基线背景与阶段总结 | [E00 阶段总结与 E01 交接](docs/experiments/E00_阶段总结与E01交接.md) |
| E01-A 真实训练和 Dev Smoke 结果 | [E01-A 验收报告](docs/experiments/E01-A_训练可学习性验收报告.md) |
| E01-B Pilot 训练和 160 条配对评测结果 | [E01-B Pilot 验收报告](docs/experiments/E01-B_训练工程与Pilot验收报告.md) |
| E01-B Full 与 1,833 条 Dev 权威汇总 | [E01-B 正式验收报告](docs/experiments/E01-B_正式SFT与Dev对照验收报告.md) |
| E01-B 逐类情绪、逐标签画像、Raw 与延迟诊断 | [E01-B Full 深度诊断](docs/experiments/E01-B_Full_深度诊断与证据整理.md) |
| E01-B 训练工程和公平评测方案 | [训练工程完善方案](docs/task/E0.1-B_训练工程完善任务方案.md)、[分阶段训练与公平评测方案](docs/task/E0.1-B_分阶段训练与公平评测执行方案.md) |
| 开发环境、Git/PR 和 P/F/T 协议 | [环境复现指南](docs/完整环境配置与复现指南.md)、[协作规范](docs/knowledge/双人Git协作与PR操作规范.md)、[PFT 笔记](docs/knowledge/PFT_输入协议设计与版本演进笔记.md) |
