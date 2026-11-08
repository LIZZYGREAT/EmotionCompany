# E03 双 Adapter 最终集成、vLLM 与 Docker 验收

执行日期：2026-11-08；任务分支：`feat/final-dual-adapter-delivery`。

当前结论：**阶段 A Transformers、阶段 B vLLM 双 LoRA、阶段 C 官方入口验收通过。阶段 D 已构建、导出并回导 Docker 镜像；首次离线 GPU Smoke 通过，但回导后的第二次 Smoke 有 1 条 E03-B 回复命中 1200-token 上限，入口返回失败。整体仍为 `partial`，不得标记为 `ready_for_submission`。**

## 阶段状态

| 阶段 | 状态 | 证据 / 边界 |
|---|---|---|
| A. 资产锁定、Transformers 双 Adapter Smoke | `passed` | Base、Tokenizer、Prompt、LoRA 配置与候选文件身份均核验；20 条双路生成、Schema 和两 Adapter 切换等价性通过。 |
| B. vLLM 0.11.1 双 LoRA Smoke | `passed` | 单服务先跑 10 条、通过后跑同一冻结集 20 条；双别名、Token IDs、格式、Schema、显存及延迟门禁通过。 |
| C. 官方 `start.sh` 双路入口 | `passed` | 保持两个位置参数；显式选用 vLLM 后真实入口完成 20 条双路生成，两个官方输出文件及报告均验证。 |
| D. Docker 离线 GPU Smoke、镜像导出/回导 | `not_passed` | 镜像构建、保存、SHA-256、回导和回导后 GPU 执行均已完成；回导后 `test_000005_t0` 的 E03-B 生成达到 1200 tokens，`performance_report.status=failed`。 |
| 官方评分、完整 Val、提交 | `not_run` | Smoke 不读取 Gold、不计算质量分数，不代表赛事提交就绪。 |

## 冻结资产与身份

候选根目录为 `outputs/final_candidates`。预检只读原始 Adapter；E03-B 的旧机器 Base 路径仅在 `outputs/final_delivery/E03-DUAL-20261108-02/deployment/adapters/emotion-response/adapter_config.json` 部署副本中改为本机 Base 路径。

| 身份 | SHA-256 / 事实 |
|---|---|
| Base | `Qwen/Qwen3-1.7B`，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；实物权重指纹 `005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5`。 |
| Tokenizer / Chat Template | Signature `1662c9e99c6185441026a168b2dd6b09af103dc6dc3563b72f598411f14d1913`；Template `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`；EOS `151645`。 |
| A8-P4504 Adapter | 权重 `4f4bea7c738109b2b90502350abbc8bb37e98de08c62031f497d3516b2aa1787`；配置 `2530a87d84ab3c0ce9555947d573c0bbf24e75dcc23e42b4608ea59f93755835`；归档 Run `E03-A8-DEVFT-20261108-02-PDEV-1EPOCH`，optimizer step `4504`。本地权重和配置与归档 Manifest 一致。 |
| A8-ER / Replay-1152 Adapter | 权重 `b3db90b9eff9204f3b5717fc52e33481feb4149571c5fdad9238b47642e62b3b`；配置 `477f4418ee14f4c0ad02c98d7cb4d3cd1153f50c4ecc1be2164db052e9f01c29`；候选目录与 `checkpoint-1152` 中权重、配置逐字节相同；`trainer_state.json` 为 replay-local step `1152/2219`。来源 Run Note 记录从 checkpoint-4665 续训，Replay 报告选用 1152。 |
| Prompt | P0 `d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2`；A8 P `0cf6e196b5d4ac4005ea2cc2b454ec058ee24592fb8bb0a1c172ca464dba3cdb`；A8 EP 预算 Prompt `832aa10ad57626b38d45247fd6715a7aac7b0c7c838e817c6a0206d50a498a94`；E03-B `630e842b31defc42968110d35154ab20c007041bd8b75583072bdd51e68a9c35`。 |

Replay-1152 的 Prompt 和训练配置来自 `origin/chore/e03-b-p2-twofield`，来源 commit 为 `f10582fd3db4525e86770939dcd1ee798cf9e8e1`。该分支的 P2 Replay 报告、Run Note、双字段训练配置和原始 E03-B Prompt 已按文件选择性纳入任务分支，没有合并其余分支改动。

锁定文件位于 `outputs/final_delivery/E03-DUAL-20261108-02/model_lock.json`，SHA-256 为 `97b21bc9d64b509e9703e91c5ec2236f25ebba1da57b0d5f136ac5ea77ff092f`。资产预检摘要位于同目录 `asset_preflight.json`，状态为 `passed`。

## 阶段 A：Transformers Smoke

### 输入

冻结输入为 `outputs/final_delivery/E03-DUAL-20261108-02/transformers_smoke20/input/smoke20_input.jsonl`，SHA-256 为 `61ddba9c9dedd790e7315d99fec14225c48d4ef510d202d2bb19346f1e0be7ee`。20 条包含 3 条本地测试输入和 17 条已有的 target-free Val 输入；所有记录仅包含 `id` 与 `history`，ID 映射至官方 `test_XXXXXX_tN` 格式。没有读取 Gold 或 References，也没有运行质量评分。历史长度为 1–49 条消息，10 条不少于 8 条历史消息，4 条包含英文词汇。

### 验收结果

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| A8 画像严格解析 | 20/20 | ≥19/20 |
| E03-B 情绪/回复严格解析 | 20/20 | ≥19/20 |
| 官方 Submission Schema | 20/20 | 20/20 |
| A8 单独加载 vs 双 Adapter 切换生成 Token IDs | 5/5 完全相同 | 5/5 |
| E03-B 单独加载 vs 双 Adapter 切换生成 Token IDs | 5/5 完全相同 | 5/5 |
| 双 Adapter 反向顺序重复切换 | 两路各 5/5 完全相同 | 两路各 5/5 |
| 达到生成上限 | A8 0/20 达到 256；E03-B 0/20 达到 1200 | 不得出现静默截断 |
| OOM / 崩溃 | 0 | 0 |

设备为 NVIDIA GeForce RTX 4060 Laptop GPU。A8 与 E03-B 独立加载耗时分别为 6.89 秒和 3.01 秒；双 Adapter 加载耗时 2.70 秒。20 条主生成的单路延迟中位数 / P95：A8 为 0.81 / 1.39 秒，E03-B 为 3.59 / 25.80 秒。显存峰值为 2.30 GB allocated、2.47 GB reserved。Transformers 双路主生成加 5 条反向切换复测总耗时为 241.35 秒；该总耗时包含复测，不作为官方逐样本延迟指标。

预测和原始输出分别为 `transformers_smoke20/predictions.jsonl` 与 `raw_outputs.jsonl`；SHA-256 分别为 `55ca262998a5657da9d76a9ff6f2e46644e09d1e43ceca6a1d076332c9547801` 与 `20d73cc43990cbaa15267c2aa67159c859339a734715f731dc6529363f347964`。独立复核确认输入、预测、原始输出均为 20 条且 ID 顺序一致、唯一，逐条通过官方 Schema。

原 Smoke `manifest.json` 的 `git_head` 字段沿用了预检锁创建时的 HEAD。为保留原始报告且准确绑定实际执行代码，未改写原 Manifest；另存 `transformers_smoke20/code_identity.json`，记录执行 commit `9df4be89440607b3bdf1ad6767dc3eeacf0c502f`、代码文件 SHA 和原 Manifest SHA。该补充记录确认运行时没有本任务代码的未提交变更。

阶段 A 实现与来源锁定提交为 `f1f29fa`；CUDA 显存统计兼容性修复提交为 `9df4be8`。新增协议/运行时 CPU 回归为 11/11 通过；相关 Python 文件 `compileall` 通过，提交前 `git diff --cached --check` 通过。

## 阶段 B：vLLM 0.11.1 双 LoRA Smoke

实现提交为 `ae5e46c`。服务只启动一次，先完成 `smoke10` 门禁，再使用相同进程完成 `smoke20`；两阶段合计 60 次串行 Chat Completion 请求。vLLM 官方 0.11.1 支持用 `--lora-modules name=path` 注册命名 LoRA，并通过请求的 `model` 名称路由，实际使用的命令与别名已记录在 Smoke Manifest 和服务日志中。[vLLM 0.11.1 LoRA 文档](https://docs.vllm.ai/en/v0.11.1/features/lora/)

实际运行参数：Base `Qwen/Qwen3-1.7B` 本地锁定目录、BF16、`max_model_len=2560`、`max_num_seqs=1`、`max_num_batched_tokens=2560`、`gpu_memory_utilization=0.8`、`max_lora_rank=8`、`max_loras=1`、`max_cpu_loras=2`；注册 `qwen-base`、`profile`、`emotion-response`。两个 Adapter 均由锁定的部署副本加载，原候选资产未修改。

| 门禁 | Smoke 10 | Smoke 20 | 结果 |
|---|---:|---:|---|
| Profile 严格解析 | 10/10 | 20/20 | 通过 |
| E03-B 情绪/回复严格解析 | 10/10 | 20/20 | 通过 |
| 官方 Schema | 10/10 | 20/20 | 通过 |
| 服务端 Prompt Token IDs 精确匹配 | 两路各 10/10 | 两路各 20/20 | 通过 |
| LoRA 注册与调用 | `profile`、`emotion-response` 均注册并各请求 10 次 | 两别名各请求 20 次 | 通过 |
| 生成上限命中 / 路由失败 / OOM | 0 / 0 / 0 | 0 / 0 / 0 | 通过 |

vLLM 初始化耗时 57.47 秒。GPU 为 NVIDIA GeForce RTX 4060 Laptop，采样峰值占用 7508/8188 MiB，服务退出码为 0。显存峰值约为设备容量的 91.7%；启动日志另有 WSL pin-memory 性能提示及 CPU swap 配置提示，实际 60 次请求均完成且没有 OOM。请求延迟中位数 / P95：Smoke 20 的 Profile 为 0.306 / 0.504 秒，E03-B 为 1.547 / 7.387 秒；每条样本从两路 Prompt 构造开始，串行完成两路生成与官方输出组装的端到端时间为 1.903 / 8.504 秒。Smoke 10 的端到端中位数 / P95 为 2.799 / 10.815 秒。以上为 20 条工程 Smoke，不代表完整 Val 或质量评估。

证据位于 `outputs/final_delivery/E03-DUAL-20261108-02/vllm_smoke/`：`smoke10/manifest.json`、`smoke20/manifest.json`、两份预测与原始输出、聚合 `manifest.json` 和 `vllm_server.log`。独立复核确认 10/20 行数、ID 顺序与唯一性、100% Schema、两路 Prompt Token IDs 及报告哈希一致；日志没有 OOM 或崩溃标记。

## 阶段 C：官方 `start.sh` 双路入口

本地入口实现新增于 `participant/final_dual_inference.py`，并在 `participant/start.sh` 中根据独立双 Adapter 配置的 `mode=dual_adapter` 路由；入口原有 Conda `conda_env` 激活和恰好两个位置参数保持不变。运行显式设置 `EVALUATION_BACKEND=vllm`，部署配置原有 `backend=transformers` 保留，未改写已锁定配置或模型身份。实际调用：

```bash
EVALUATION_INFERENCE_CONFIG=configs/deployment/final_dual.yaml \
EVALUATION_BACKEND=vllm \
EVALUATION_RUN_ID=E03-DUAL-20261108-02-OFFICIAL-START-20 \
bash participant/start.sh \
  outputs/final_delivery/E03-DUAL-20261108-02/transformers_smoke20/input/smoke20_input.jsonl \
  outputs/evaluation/final_dual/E03-DUAL-20261108-02-official-start-vllm20
```

入口退出码为 0；20/20 输入 ID 顺序完整且唯一，官方 Schema 20/20。`profile` 与 `emotion-response` 别名各完成 20 次请求，两路服务端 Prompt Token IDs 各 20/20 精确匹配；两路严格格式均 20/20，路由/生成失败为 0，生成上限命中为 0。性能报告测量范围是逐样本 Prompt 构造、两次串行生成、解析、Schema 校验及输出追加；服务初始化单独计时 38.87 秒。20 条逐样本端到端 P50/P95 为 1.892 / 8.984 秒，GPU 峰值 7664/8188 MiB。输出 `submission.jsonl` SHA-256 为 `807844152a8de8f3750cc247453b5f0112be8b5700e58945db0c2f74ddea90cc`；`raw_outputs.jsonl` SHA-256 为 `9f5c8a91de342394459866ca26621191273cc455c70d0f026070f69b4b9d5169`。完整报告及逐样本原始输出位于 `outputs/evaluation/final_dual/E03-DUAL-20261108-02-official-start-vllm20/`。

阶段 C 入口实现及回归提交为 `c955eff`。运行报告保留原始 `git_head=38e2c27`；同目录 `code_identity.json` 记录性能报告 SHA-256 `91beba084680c6fc6b6dd050277b30c249d3036c0449a51d01196fbfe784a754`、提交 `c955eff6269b8bd7dbe96b8c86fccc667879feef` 和对应执行文件哈希，没有重写运行报告。

官方入口修复了一个锁结构读取错误：锁中的 Base 指纹位于 `base.observed.observed_weight_hash`。首次运行在模型服务启动前停止且结果目录无输出文件；修正后重新核验实际 Base 与冻结 SHA 一致并完整通过阶段 C 门禁。

## 阶段 D：Docker 离线 GPU Smoke、镜像导出与回导

Docker Desktop 的数据根目录仍位于 E 盘数据盘：`E:\DockerData\DockerDesktopWSL\disk\docker_data.vhdx`；Docker Engine 内部报告的数据根目录为 `/var/lib/docker`。镜像 tag 在回导测试前先移除，再由归档重新加载。镜像与 tar 均位于 Git 忽略目录，不提交仓库。

| 项 | 结果 |
|---|---|
| 镜像 tag | `emotionalcompany/final-dual:E03-DUAL-20261108-02` |
| Image ID / RepoDigest | `sha256:42465c1eb5870221d2c224def8492adb9e09ec44152741523831a6777b18e186` |
| 架构 / 层数 | `linux/amd64` / 40 层 |
| Inspect 虚拟大小 | 构建后 `31,955,841,193` bytes；回导后 `31,955,808,425` bytes；两者均低于 `50,000,000,000` bytes 门槛 |
| Save 归档 | `outputs/final_delivery/E03-DUAL-20261108-02/docker/image.tar`，`11,768,250,368` bytes |
| 归档 SHA-256 | `4bd10c5c00e071212c7c49b9be7a1e579a7da104fe56a3ce11a8b27274575e9d`；回导后复核通过 |
| 回导身份 | Image ID、RepoDigest、tag 和 40 个 RootFS layer digest 均一致；Inspect `Size` 相差 `32,768` bytes |
| 构建来源 | Dockerfile SHA-256 `f4b4f1f14ed71c393bfad88e83b3f42502c87baa6d5dae5feb8ad35ba6efe3bf`；镜像代码 commit `a7cfd11b044cccc34aaafffac763bf28e4d4af06` |

### 首次离线 GPU Smoke

首次容器运行使用 `--gpus all --network none --shm-size 1g` 和正式两参数入口。容器 GPU 预检确认 NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB）；单 Base、`profile` 与 `emotion-response` 三个模型 ID 正确注册，两路 alias 各请求 20 次。输入和输出共 20 条，官方 Schema 20/20，两路 Prompt token IDs 各 20/20 精确匹配，两路严格格式各 20/20，路由失败与生成上限命中均为 0。峰值显存 `7373/8188 MiB`，无 OOM/崩溃。输出验证记录在忽略文件 `docker/docker-smoke-verify.json`，状态 `passed_offline_dual_adapter_docker_smoke`。

宿主官方 vLLM 与 Docker 首次 Smoke 的画像、情绪标签、memory_refs、ID 和解析/格式计数相同；生成回复文本只有 `7/20` 条逐字相同。为区分 Docker 差异与服务端复现差异，宿主官方入口在相同输入上又独立运行一次：宿主两次结果之间也只有 `7/20` 条完整输出相同，画像和情绪标签仍均为 `20/20` 相同，格式均为 `20/20`。vLLM 官方说明在线 `vllm serve` 不保证跨运行的逐字复现；因此复核器要求稳定字段、Prompt token IDs、路由和格式指标一致，并单独记录回复文本相同比例。[vLLM reproducibility notes](https://docs.vllm.ai/en/v0.11.2/usage/reproducibility/)

### 回导后 Smoke 与停止门禁

回导后的容器再次完整生成 20 条、Schema 20/20、两路 Prompt token IDs 各 20/20、格式各 20/20、路由失败为 0；但 `test_000005_t0` 的 E03-B 生成用了 `1200` 个 completion tokens 并以 `finish_reason=length` 结束。性能报告记录 `generation_cap_hits={"profile":0,"emotion_response":1}`、`status=failed`；官方入口最终返回退出码 1。因此阶段 D **不通过**，不能把首次 Smoke 的成功当作回导门禁的替代。

为诊断截断和回复差异，使用 `VLLM_BATCH_INVARIANT=1` 在宿主重复运行两次：两次均零 cap 且格式 20/20，但两次之间仅 6/20 条完整输出相同；vLLM 日志同时记录它将 Attention backend 强制改为 `FLASH_ATTN`。由于这种设置仍未达到重复输出一致且改变了数值执行路径，本轮没有将它纳入镜像。

本阶段保留 `docker/image_info.json`、`docker/image-inspect.json`、`docker/image-inspect-after-load.json`、build/smoke 日志、前后两次 Smoke 输出及 `docker/docker-comparison.json`。锁定的 E03-B 上限为 1200 tokens；继续之前需要针对该上限/停止策略制定新版本并重新评估，不能静默改写冻结配置。完整 Val、官方评分及提交仍未执行。

## 后续门禁

需要先解决并按新版本复验 E03-B 生成上限门禁，才可将阶段 D 标记为通过。完整 Val、官方评分及提交仍为 `not_run`。
