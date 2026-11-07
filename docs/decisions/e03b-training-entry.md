# E03-B-P2 双字段稠密半训练入口

日期：2026-11-07
分支：`chore/e03-b-p2-twofield`

## 背景

双模型拆分的第一半：只输出 `emotion` 和 `response`，格式固定为 `emotion=xxx;response=...`。画像半不在本次范围。

## 为什么新开入口

仓库原有 `src/training/training_sft.py` 的 `_training_rows` 绑定 D0.4.3 清单：校验 `version=D0.4.3-v1`、`dataset_dir=data/processed/sft/v1`、`d03_manifest`/`d042_manifest` 的 sha、`normalized_train` 的行数，并在此后组装训练行。本次数据是新编码的 E03B 数据集，没有这些清单；跳过那一段会断数据链，所以新增独立分支 `_training_rows_e03b`，D0.4.3 路径一字未改。

## 入口校验（新分支自带）

- `version=E03B-v1`，`dataset_dir=data/processed/sft/e03b`；
- `files` 中 train/dev 各有 `sha256` 与 `records`，逐行核对行数；
- `config.profile.max_train_seq_length/max_prompt_tokens` 与配置一致（1536/1280）；
- 逐行检查 `prompt_length` 合法、prompt 段全 `-100`、completion 段与 `input_ids` 对齐且含 `151645`。

## 训练口径

- 行数 18,657，4 样本/步，一 epoch 4,665 步；四档 512/1024/2048/4665。
- 总长 1536、prompt 上限 1280、超长整轮丢历史（T1）。
- NF4 双量化、compute bf16、LoRA r8 alpha16、lr 1e-4 恒定、micro batch 1、梯度累积 4。
- 1536 是工程取舍（独立脚本显存管理），不是 8G 跑不了 2048；E01-B 在 2048 下峰值 3.83G/4.10G。

## 与 E01-B 的可比性

E01-B 的 16,192 条是 18,657 的严格子集：D0.3 账本 18657 = 4 隔离 + 508 去重 + 16312 train + 1833 dev，D0.4.3 再扣 120 条超长得 16192。我们多训 2465 点，其中 1833 是 E01-B 的 dev，其 Dev 65.36 与 209 点不可比；只比 209 点 `val_public`。

## 结果

三档 209 点：512 情绪 59.81 / 2048 71.77 / 4665 72.25，门槛 72.68，最终 4665 判 FAIL（差 0.43 点）；格式均 209/209，BLEU/ROUGE/长度均过线。详见 `outputs/training/E03B-RUN-NOTE.txt` 与 `e03-b/results/E03-B_稠密半四档汇总.md`。
