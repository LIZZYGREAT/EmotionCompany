# E03-B 难例回放微调实验报告

> 日期：2026-11-08
> 起点：ckpt-4665 adapter（`e03b-run1536-4665`）
> 评测集：同一 209 点（val_public 抽 78 段会话 / 209 点，seed 20261103），未参与训练
> 生成：greedy，P=1280，G=1200
> 门槛与解析口径：开训前冻结，见 `../../README.md` 与 `replay_handoff.md`

## 1. 目的

ckpt-4665 的情绪准确率 0.7225（151/209），差门槛 1 条。错误集中在标签先验导致的
决策边界：mixed 20 条只认对 1 条、sadness 12 条认对 2 条、care 5 条全判 anxiety、
disgust 零预测。继续加步数（2048→4665 只多对 1 条）已经改不动，因此只改训练数据
构成，做难例回放，其余配置不变。

## 2. 数据

清单 `replay_manifest.jsonl`，seed 20261108，由 `tools/e03b_build_replay.py` 从官方
`train_e03b.jsonl` 生成。

| role | 情绪 | 重复 | 唯一点 | 实例 |
|---|---|---:|---:|---:|
| hard | care | ×4 | 79 | 316 |
| hard | disgust | ×4 | 125 | 500 |
| hard | mixed | ×2 | 945 | 1,890 |
| hard | sadness | ×2 | 1,085 | 2,170 |
| normal | 其余各类随机 | ×1 | 4,000 | 4,000 |

合计 6,234 唯一点 / 8,876 实例；normal 中自然含 1,585 条 anxiety（占实例 18%），
不额外加，用于压住不塌缩。

## 3. 训练

除数据外与 4665 同配置：LoRA r=8 / alpha=16 / dropout=0（q/k/v/o + gate/up/down），
NF4 double quant，bf16 compute，LR 1e-4 constant，seed 42，micro batch 1、梯度累计 4，
训练总长 1536，prompt 上限 1280。8,876 实例约 2,219 optimizer steps，每 128 步存盘。

**档位偏差（需备案）**：计划评 step 1,024 / 1,792 / 2,219，但 `save_total_limit=10`
在训练过程中自动删除了 1,024 与 896，故早档改用 **1,152** 顶替。1,152 不是计划内档位，
评的是同一套固定配置，未做任何中途调整。

## 4. 结果

| 档位 | 格式 | 情绪 | BLEU-4 | ROUGE-L F | 长度比 | 门槛 |
|---|---:|---:|---:|---:|---:|---|
| ckpt-4665（起点） | 209/209 | 0.7225 (151) | 0.1270 | 0.3018 | 1.0262 | 4/5 |
| replay-1152 | 209/209 | **0.7321 (153)** | 0.1192 | 0.2981 | 1.0930 | **5/5** |
| replay-1792 | 209/209 | 0.7225 (151) | 0.1146 | 0.3030 | 1.1719 | 4/5 |
| replay-2219 | 209/209 | 0.7273 (152) | 0.1169 | 0.3051 | 1.1881 | 5/5 |

门槛：格式 ≥0.99；情绪 ≥0.7268；BLEU-4 ≥0.1081；ROUGE-L F ≥0.2869；长度比 ∈[0.9,1.2]。

## 5. 逐类 recall

| 情绪 | gold n | 4665 | 1152 | 1792 | 2219 |
|---|---:|---:|---:|---:|---:|
| anxiety | 64 | 58 | 49 | 47 | 51 |
| relaxed | 32 | 28 | 26 | 26 | 22 |
| mixed | 20 | 1 | 12 | 12 | 11 |
| gratitude | 16 | 14 | 13 | 13 | 11 |
| shame | 13 | 10 | 8 | 10 | 10 |
| helplessness | 12 | 11 | 9 | 11 | 11 |
| sadness | 12 | 2 | 11 | 5 | 8 |
| anger | 11 | 9 | 9 | 8 | 10 |
| fear | 10 | 6 | 7 | 10 | 9 |
| loneliness | 7 | 7 | 4 | 5 | 5 |
| care | 5 | 0 | 1 | 1 | 1 |
| neutral | 3 | 3 | 2 | 1 | 1 |
| joy | 2 | 2 | 0 | 0 | 0 |
| disgust | 2 | 0 | 2 | 2 | 2 |
| 合计 | 209 | 151 | 153 | 151 | 152 |

预测数（相对 gold 的过判）：1152 的 mixed 预测 26 / gold 20，sadness 18 / gold 12；
1792 的 fear 预测 22 / gold 10；2219 的 fear 18 / gold 10。三类难例都出现了
“边界外扩”，靠多判换召回。

## 6. 变化结构

1152 与 4665 的配对变化是 28 条翻对、26 条翻错，净 +2。翻对集中在 hard 类
（mixed、sadness、care、disgust），翻错集中在原先把稳的近邻类（anxiety −9、
loneliness 7→4、joy 2→0、shame −2、helplessness −2）。

1792 是 24 翻对 / 24 翻错，净 0：anxiety 掉到 47，fear 过判到 22 条，sadness 回落到 5。
2219 是 28 / 27，净 +1：mixed 11、sadness 8、anxiety 51。

结论：回放确实把目标类的决策边界推过去了，但推的方式是整体边界外扩，
不是在原位修准。三档之间的 1–2 条差异主要来自边界落在哪一侧，噪声成分大。

## 6.1 独立复验（201 点新切片）

209 点上的 +2 统计上不显著（McNemar p=0.892，配对 bootstrap 95% 区间
[-0.057, +0.077]）。为验证它是否只是同批点上的高点，另建一套互不重叠的点：
val_public 剩余 822 段会话中按 seed 20261108 抽 80 段，展开 201 点
（`points_fresh201.jsonl`，与 209 点零重叠）。

| 档位 | 格式 | 情绪 | BLEU-4 | ROUGE-L F | 长度比 |
|---|---:|---:|---:|---:|---:|
| replay-1152 | 201/201 | **0.7413 (149)** | 0.1338 | 0.3144 | 0.9893 |
| ckpt-4665 | 201/201 | 0.6766 (136) | 0.1267 | 0.3058 | 0.9311 |

配对：1152 翻对 28、翻错 15，净 +13；McNemar 精确 p=0.066，配对 bootstrap
95% 区间 [+0.000, +0.129]。

翻对结构与回放目标一致：sadness 2→15（gold 15 全中）、mixed 1→9、fear 5→9；
代价为 anxiety 62→55、neutral 6→2。两批点合并 410 点，1152 302/410=0.7366，
4665 287/410=0.7000，净 +15（McNemar p=0.155），两批方向一致。

注意：4665 在 209 点 0.7225、在 201 点 0.6766，切片间绝对分差可达 4–5 个点，
跨切片只能比配对差值，不能比绝对值。

## 7. 选档

按开训前冻结的规则：格式 ≥0.99 为硬条件；在通过者里取情绪准确率最高；同分取
BLEU-4 高。通过者为 1152（0.7321）与 2219（0.7273），**选 1152**。

风险记录：

1. 209 点上 1152 相对 4665 只净胜 2 条，单看这批点证据不充分；新增的 201 点
   独立切片给出 +13（p=0.066，区间下沿贴 0），两批方向一致，合并 410 点为 +15
   （p=0.155）。结论从“正好高出 1 条门槛”升级为“方向可复现、量级有一致区间”，
   但仍未达到稳健显著，不能宣传为确定更优。
2. 209 点上 1152 的 BLEU-4 与 ROUGE-L 略低于 4665（0.1192/0.2981 对
   0.1270/0.3018）；新切片上反而更高（0.1338/0.3144 对 0.1267/0.3058），
   说明文本差异在两批间不稳定，不构成明确代价。
3. 长度比都在区间内：209 上 1152 为 1.0930、4665 为 1.0262（后者更贴 1）；
   201 上 1152 为 0.9893、4665 为 0.9311（前者更贴 1）。2219 的 1.1881 已贴近
   1.2 上限，若换分布评测可能越界。
4. 官方最终成绩只由测试集决定，本报告任何数字都不代表官方得分。

## 8. 产物

- adapter：`outputs/training/e03b-replay-<steps>/adapter`（训练侧）；
  1152 与 4665 的 adapter zip 另存服务器 `../backup/`（sha256 已核对）
- 预测（209 点）：`replay_1152_pred_209.jsonl`、`replay_1792_pred_209.jsonl`、
  `replay_2219_pred_209.jsonl`
- 门槛：`replay_1152_gate.txt`、`replay_1792_gate.txt`、`replay_2219_gate.txt`
- 复验（201 点）：`points_fresh201.jsonl`、`replay_1152_pred_fresh201.jsonl`、
  `ckpt4665_pred_fresh201.jsonl`
- 清单与交接：`../replay_manifest.jsonl`、`../replay_handoff.md`

复现：

```
python3 tools/e03b_perclass.py --gold <points_209> --pred results/replay_1152_pred_209.jsonl \
    --ref results/ckpt4665_pred_209.jsonl
python3 tools/e03b_compare.py --gold <points_209> \
    --base base209.jsonl --new results/replay_1152_pred_209.jsonl
python3 tools/e03b_build_eval_points.py --val val_e03b.jsonl.gz \
    --exclude <sample_ids.json> --convs 80 --seed 20261108 \
    --out results/points_fresh201.jsonl
python3 tools/e03b_paired.py --gold results/points_fresh201.jsonl \
    --a results/replay_1152_pred_fresh201.jsonl \
    --b results/ckpt4665_pred_fresh201.jsonl --name-a 1152 --name-b 4665
```

## 9. 未完成

画像半未参与本轮，仍由训练侧单独产出。合并脚本按 id 拼接两半，稠密半换档不影响
画像半，无需重跑。
