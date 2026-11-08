# E01-B Train+Dev 与 A8 P4504 同集 Val 最终对照与收尾

> 状态：`completed_local_proxy`。完成日期：2026-11-08。执行分支：`feat/e01-b-traindev-val`。

## 结论

E01-B 从 Full Step 4048 恢复完整训练状态，在 1,821 条符合原 SFT 预算的 Dev 数据上续训 456 步，达到 Global Step 4504。训练覆盖完整、四个全状态 Checkpoint 均通过 Hash 验证，终点 Adapter 独立重载通过。

在同一组 2,354 个冻结 Val ID 上，A8 P4504 的 33 标签画像 Micro-F1 为 **0.4678**，E01-B Train+Dev 4504 为 **0.4014**。E01-B 的 Recall 较高（0.4796 对 0.4280），但 Precision 较低（0.3451 对 0.5156），FP 多 1,585。按 `conversation_id` 配对重采样的 E01-B−A8 Micro-F1 差值为 `-0.0664`，95% 区间 `[-0.0866, -0.0466]`。

这是两个完整候选方案在本地 Val 上的比较，不是隔离 Prompt 差异后的权重消融：E01-B 使用 P0/F2/T1 并生成完整 JSON；A8 P4504 使用其原生三行画像 Prompt。**保留 E01-B Train+Dev 4504 和 A8 P4504 两套模型，不按单项分数自动选型或删除模型。** A8 的本地画像结果较高，但本报告不代表官方 Test 成绩，也不自动改变正式提交基线或推理入口。

## 来源、数据与归档身份

| 对象 | 实际来源与身份 |
|---|---|
| E01-B 父 Run | `E01-B-FULL-20261028-133805`，Run Manifest SHA-256 `9204b7cb374765438fd23054078d28992b240d50c4df8763cae0817f0d7752d7` |
| 父 Checkpoint | `trainer/checkpoint-4048`，Checkpoint Manifest SHA-256 `125996b4d00d391dbb32dd3297578a72dfdf7679f44658eff16ae63be94ec1f5`，优化器步数 4048 |
| DevFT 数据 | `data/processed/sft/v1/dev.jsonl`，1,821 个唯一 SFT ID；D0.3 Dev 共 1,833 条，12 条依 D0.4.2 规则以 `completion_overflows_train_budget` 排除 |
| 数据来源 | D0.3 Manifest `605867b5ab0162de05ea9e396bffd574cd4029c5adc3c29d851f41ff73a96b8d`；SFT Dev SHA-256 `6fdee702d434b01e2cac769bcdd97d5b984ce33273a484dcee9c20f1580e3e9f`；D0.4.2 Manifest `dd2ef98b33081a1606c431e34bede875cfb41217550ddcc6c27f01a5a5ee5210` |
| A8 候选 | Run `E03-A8-DEVFT-20261108-02-PDEV-1EPOCH`，Step 4504；Checkpoint Manifest SHA-256 `79f59d95247771160198f7e5dcecfe05ea754f222d3a2236f72f0eeadceb9c79` |
| Base | Qwen3-1.7B，本地权重 SHA-256 `005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5` |
| 冻结 Val 输入 | A8 已冻结的 target-free `id + history` 输入与 ID Map，各 2,354 条；目标与参考文本仅在两套预测冻结之后打开 |

训练前已将 E01-B Step 4048 与 A8 P4504 归档到忽略目录 `outputs/final_candidates/E01-B-TRAIN4048/`、`outputs/final_candidates/A8-P4504/`。两个归档均由副本构成，包含 Adapter、来源清单与 SHA256SUMS；E01-B 保留完整 Step 4048 Checkpoint，A8 保留完整 Step 456 Checkpoint。两套归档均在 GPU 上独立加载通过，原始 Run 未被覆盖。

另将 E01-B 原始 Run Adapter 与归档副本分别加载，在同一个冻结样本上做确定性探针：两者 Adapter identity SHA 均为 `77b61e8a4caf46f54a8c813e98def488c7a8493fe891966e47df1cdac6df7d67`，Prompt Token IDs SHA 均为 `4255a054e002ead816df9cff339471ac6ba198875b1ac6d50141d9ab829e9395`，解码生成文本 SHA 均为 `a83f6bf566c5571a48e8b4991bf311e4e25f7c2782bf2be6dc9d9a5e98871b7f`。该探针比对的是解码文本身份，不据此声称生成 Token IDs 逐位相同。

## E01-B DevFT 执行结果

| 项目 | 实际结果 |
|---|---|
| Run | `E01-B-DEVFT-FINAL-20261108-02-TRAINDEV-1EPOCH` |
| 训练目录 | `outputs/training/E01-B-DEVFT/E01-B-DEVFT-FINAL-20261108-02-TRAINDEV-1EPOCH/` |
| 训练状态 | `passed`；4048 → 4504，共 456 个新增 Optimizer Steps |
| 样本计划 | Seed 42，1,821 个唯一 Dev ID 不放回；group size 4；最后一步只含 1 条样本并按实际组大小归一化 |
| 协议 | 原 E01-B P0/F2/T1、完整 JSON Completion、`P=1280`、`S=2048`；未读取 Val Gold |
| Optimizer / Scheduler | 恢复原 AdamW 的 2 个参数组 `[392, 0]`、392 份 step-4048 动量状态；LR `1e-4`、weight decay `0`、clip `1.0`；恢复 constant Scheduler 并前进至 `last_epoch=4504` |
| 可训练参数 | 392 个 LoRA 张量、8,716,288 个 BF16 参数；冻结 Base 前后 Hash 相同 |
| 终点重载 | 独立 Adapter 重载通过；固定 Prompt 生成 Token IDs 精确一致；NF4/BF16、无 offload |
| 训练耗时 | 4,249 秒；平均每新增步 GPU 计时 9.17 秒，中位数 8.58 秒，P90 12.85 秒 |
| 新 Adapter 权重身份 | `05979dd1876d306b904638f52cb5becef7ce67cfb8dfc658b5582eb85111b4b1` |
| Dev ID 顺序 SHA-256 | `6c54885540070f5afe38232a610e7cdb111a2abac1f62ffb85b52c3c28f8b72a` |
| 消费 ID SHA-256 | `c2efc0a9c9c1fe023f98936a5e67af27c35eb692ef1075124da3977a1b33b2d4` |

四个节点均包含 Adapter、Optimizer、Scheduler、RNG、Trainer State、Checkpoint Manifest 和已消费 ID Hash，并复核通过：

| DevFT Step | Global Step | 文件清单 SHA-256 |
|---:|---:|---|
| 128 | 4176 | `ce84241f076686edb6dfbb3d8e29e6dc5d25070fb5c96d9ec8f01f7a3442201e` |
| 256 | 4304 | `8382743f380dc520d8d190dcb9b78915b20c1af85ba55e3eee1b6c3a894c578f` |
| 384 | 4432 | `328a81acf2aab3baf6853df366cc8174b13799d7ede38770bf17065b42843f7c` |
| 456 | 4504 | `bb5d80bf513a0c5816266f9f47f98ee9eeacd0e024c7b2a230eeaab17378d6ea` |

## Val 评测与同集结果

32 条固定样本的完整生成与前缀截停 **32/32 逐 Token 前缀和解析标签一致**，因此全量画像评分使用 P0 的 JSON 画像前缀。E01-B 对 2,354 个冻结 Val ID 均有预测，其中 2 条遇到非法标签后按规则运行完整生成回退并保留记录；其余 2,352 条以前缀结果评分。无样本从固定分母中移除。

另在稳定 Hash 选出的 160 条固定 Val ID 上完成完整 E01-B JSON 生成。该子集只用于完整 JSON、Raw A0 和回复 BLEU/ROUGE，不外推为全 Val 回复质量。

| 指标 | E01-B Train+Dev 4504 | A8 P4504 | 解释 |
|---|---:|---:|---|
| Val 实际预测数 | 2,354 | 2,354 | 同一冻结 ID、Gold 与固定分母 |
| 画像 33 标签 Micro-F1 | 0.4014 | **0.4678** | 共同主指标 |
| 画像 Precision / Recall | 0.3451 / 0.4796 | **0.5156** / 0.4280 | E01-B Recall 高，A8 Precision 高 |
| 画像 TP / FP / FN | 1,496 / 2,839 / 1,623 | 1,335 / 1,254 / 1,784 | E01-B 多 161 TP，同时多 1,585 FP |
| 画像 33 标签 Macro-F1 | 0.2532 | 0.3128 | A8 较高 |
| 三维 Exact Match | 19.50% | 37.72% | 三个画像维度均完全匹配的样本比例 |
| Personality F1 | 0.3770 | 0.4479 | 逐维 Micro-F1 |
| Interests F1 | 0.5099 | 0.6736 | 逐维 Micro-F1 |
| Style F1 | 0.3790 | 0.3894 | 逐维 Micro-F1 |
| 情绪 Accuracy / Macro-F1 | 73.49% / 0.5994 | 不适用 | A8 原生输出不含情绪 |
| 完整 JSON / Raw A0 Schema | 159/160 / 159/160 | 不适用 | E01-B 固定完整 JSON 子集 |
| 回复 BLEU-4 / ROUGE-L | 11.5393 / 0.2795 | 不适用 | E01-B 固定 160 条，本地字符级指标 |
| E01-B 前缀生成耗时 | 平均 2.025 秒/条 | — | 覆盖 2,354 条画像预测 |
| A8 原生生成耗时 | — | 平均 0.949 秒/条 | 输出任务和长度不同，不是受控速度对照 |

160 条完整 JSON 的 E01-B 生成平均延迟为 7.96 秒/条，P50 6.10 秒，P95 18.89 秒；平均生成 128.39 Token，最长达到 1,200 Token 上限。159/160 条以 EOS 结束，1 条触及生成上限并导致严格 JSON 解析失败。回复非空率为 159/160。

画像逐样本标签命中变化如下：E01-B 独有命中 469，A8 独有命中 308，两者共同命中 1,027，两者共同漏报 1,315；Gold 正标签样本—标签对共 3,119。按 900 个 conversation cluster、Seed 42、2,000 次配对 Bootstrap，E01-B Micro-F1 95% 区间为 `[0.3782, 0.4239]`，A8 为 `[0.4407, 0.4928]`，差值区间为 `[-0.0866, -0.0466]`。该区间是探索性本地估计，不用于官方泛化结论。

Raw A0 Schema 检查要求完整 JSON 字段、合法标签、非空回复及 `memory_refs=[]`，结果为 159/160。报告中官方提交 Schema 在原始 Val 评测 ID 上为 0/160，是因为 Val ID 形如 `val:...`，而官方 ID 正则只接受 `test_...`；仅为结构诊断时将 ID 换成符合测试格式的合成值后，160/160 通过。该结果不代表这 160 行是可直接提交的 Test 预测。

**A8 的 Val Micro-F1 较高，E01-B Train+Dev 的召回和命中数较高但误报更多。** 两者使用不同原生 Prompt 与输出任务；A8 已使用 D0.3 Dev 续训，Val 中有 209 个点曾被 A2 探索，其他点也可能共享对话上下文。本报告是模型选择用的本地代理对照，不等于官方独立 Test 评分。

## 执行中的异常与处理

首次准备的 Run `E01-B-DEVFT-FINAL-20261108-01-TRAINDEV-1EPOCH` 在首个训练步前被 Trainer 初始化拒绝：父 TrainingArguments 保留了 `eval_strategy=steps`，临时优化器工厂未传 eval dataset。失败 Manifest 保留在本地。修复只将临时 Trainer 工厂的评测策略设为 `no`，以继续使用它重建原 AdamW 参数分组；优化器状态、Scheduler、训练目标和 Dev 协议均未改变。使用新 Run ID `02` 后，456 步训练全部通过。

首次评分尝试在预测冻结后因缺少 `Draft202012Validator` 导入而退出；补齐导入后只重跑 CPU 评分，预测文件 SHA 未变化，最终评分状态为 `scored_after_prediction_freeze`。

## 产物、复现命令与回归

本机忽略目录中保留完整证据：

- E01-B Train+Dev 4504 Run：`outputs/training/E01-B-DEVFT/E01-B-DEVFT-FINAL-20261108-02-TRAINDEV-1EPOCH/`
- 可直接加载的终点 Adapter：`outputs/training/E01-B-DEVFT/E01-B-DEVFT-FINAL-20261108-02-TRAINDEV-1EPOCH/adapter/`
- 归档的 E01-B 4048 与 A8 P4504：`outputs/final_candidates/E01-B-TRAIN4048/`、`outputs/final_candidates/A8-P4504/`
- Val 生成、画像预测、固定 160 条完整 JSON、评分与 Manifest：`outputs/evaluation/e01_b_final_val/E01-B-TRAINDEV-4504-VAL-20261108-01/`

本次执行命令：

```bash
.venv/bin/python scripts/e01_b_devft_final.py --phase prepare
.venv/bin/python scripts/e01_b_devft_final.py --phase train
.venv/bin/python scripts/e01_b_final_val.py --phase generate
.venv/bin/python scripts/e01_b_final_val.py --phase score
```

脚本拒绝覆盖已有 Run；重新生成 Val 预测需指定新的 `--run-id`，并继续使用锁定的模型与输入配置。

验证结果：`python -m compileall -q scripts src tests` 完成；新增 DevFT/Val 单测与 E01-B、A8、推理协议和评分 CPU 回归 **115 passed**。21 条既有依赖警告为 NVML 不可用、PyTorch JIT 弃用及 PEFT `Conv1D fan_in_fan_out` 提示；测试均通过。

原始模型权重、训练 Checkpoint、预测与评分明细都保存在 Git 忽略的 `outputs/`，不纳入提交。官方数据、Schema、P0 Prompt、正式入口和提交输出协议均未修改。
