# EmotionCompany 最终镜像：官方 README v2 运行契约审查与修订清单

## 一、审查范围和结论

基准：用户提供的官方 `参赛者本地推理工具 v2` README（89 行），尤其是“唯一运行命令”板块。

仓库：`LIZZYGREAT/EmotionCompany`，`feat/final-dual-adapter-delivery`，审查 HEAD：`e890b9d2f5f70a38d818f102e8792bade891f03d`。

审查了 `docker/final_dual/Dockerfile`、`docker/final_dual/entrypoint.sh`、`participant/start.sh`、`participant/final_dual_inference.py`、`src/inference/final_dual_protocol.py`、`participant/run_inference.py`、输出 Schema、现有测试和 E03 交付报告；未直接访问服务器或重新执行本机 Docker/GPU 测试。因此，下面的代码缺陷可确认，无法断言哪项造成了平台实际失败。

**结论：尚未完全遵循官方 v2 的输出目录与异常恢复约定。优先修复镜像入口命令兼容性、结果覆盖规则和将内部评分门槛误当作推理成功条件的问题。**

| 官方要求 | 当前实现 | 审查 |
|---|---|---|
| `bash /root/participant/start.sh TEST_FILE RESULT_DIR`，恰好两个位置参数 | `participant/start.sh` 保留固定路径及双参数；Docker 的自定义 ENTRYPOINT 仅接受直接传入两个文件路径 | 内层符合，外层存在调用形式风险 |
| `conda_env` 环境 | `start.sh` 初始化 Conda；Dockerfile/entrypoint 设置固定环境 | 基本符合 |
| transformers 或 OpenAI 兼容 vLLM 后端 | 双模型使用 `transformers` 或 vLLM 的 OpenAI API | 功能符合；内部 `backend=vllm` 是自定义配置值，不是官方接口违规 |
| RESULT_DIR 允许预先存在并覆盖本次的两份结果文件 | `docker/final_dual/entrypoint.sh` 发现既有结果文件直接 `exit 1`；`_resume_identity()` 也拒绝已经存在的报告/提交 | **明确不符合** |
| 保留 RESULT_DIR 其他文件 | 当前运行维护 `raw_outputs.jsonl`、`run_progress.json` 等诊断文件 | 应将诊断文件与可能已有的其他文件隔离，不删改陌生文件 |
| 全部样本应输出合法提交，预测错误只计该项失败 | 末尾 `gate_passed` 把两路严格格式成功率 >= 95% 等内部 Smoke 门槛作为 `return 0/1` 条件 | **明确不符合** |
| 无法解析模型输出时尽量回收文本回复 | E03-B 输出缺少 `;response=` 时，`parse_emotion_response_output()` 清空回复 | **兜底语义不完整**；双字段原生协议须做等价适配 |
| 输出 `submission.jsonl` + `performance_report.json` | 两份文件都已生成并通过本地 Schema 验证 | 符合 |
| 测试全量，性能只统计前 100 条 | `index <= 100` 记录端到端时间，遍历全输入 | 正常情况下符合，不能推断平台允许的总耗时 |

## 二、确定需要修改的地方

### P0-A：容器入口兼容官方调用

- 文件：`docker/final_dual/entrypoint.sh`、`docker/final_dual/Dockerfile`。
- 已存在 `ENTRYPOINT ["/usr/local/bin/emotionalcompany-final-dual-entrypoint"]`，包装脚本只接受 `argc=2`。如果官方实际使用 `docker run IMAGE bash /root/participant/start.sh TEST_FILE RESULT_DIR` 而没有覆盖 ENTRYPOINT，参数数量会变成 4 并立即失败。
- 保留 `docker run IMAGE TEST_FILE RESULT_DIR` 的现有模式，同时支持 `docker run IMAGE bash /root/participant/start.sh TEST_FILE RESULT_DIR`、`docker run IMAGE /root/participant/start.sh TEST_FILE RESULT_DIR`；三种形式最终都执行同一内部启动脚本，拒绝其他未授权命令。还要验证 `docker run --entrypoint /bin/bash IMAGE /root/participant/start.sh TEST_FILE RESULT_DIR`。
- 官方 README 只规定容器内命令，没有确认平台实际如何使用 Docker ENTRYPOINT，因此这里只认定为**明确可复现的兼容性风险**，不认定已发生。

### P0-B：RESULT_DIR 允许已有结果并安全覆盖

- 文件：`docker/final_dual/entrypoint.sh`、`participant/final_dual_inference.py::_resume_identity()`。
- 官方 README 要求覆盖当次的 `submission.jsonl`、`performance_report.json`，保留其他原有文件。默认官方运行时使用“新任务”模式；可以先向同目录的唯一临时文件写入，再原子替换两个官方产物。
- 本地 Resume 是显式 opt-in，绝不自动恢复上次官方任务。诊断文件使用唯一 Run ID 子目录或不冲突的路径；不能清空 RESULT_DIR、删除陌生文件，也不能静默沿用上次的 `run_progress.json`。
- 核心测试：`RESULT_DIR` 预先存在，内有旧 `submission.jsonl`、旧 `performance_report.json` 及 `keep.txt`；调用一次后新两份文件覆盖成功、`keep.txt` 内容与 SHA 完全不变、退出码正常。再次调用相同目录也成功。

### P0-C：拆开工程退出码与模型预测质量门禁

- 文件：`participant/final_dual_inference.py` 末尾 `gate_passed` / `report.status` / `return`。
- 正式推理完成条件：全部输入 ID 恰好写出一次、所有提交行符合官方 Schema、官方两份文件成功落盘，并且没有导致无法完成任务的工程性致命错误。
- **格式解析率、未命中标签、普通逐条预测错误**：记为预测失败/质量警告，并按 v2 填充空字段；不再作为整次 `exit 1` 的条件。硬错误如模型文件缺失或身份错误、服务整体退出、无法写输出、缺失大量输入、严重路由/Token 身份异常，仍需返回非零并记录。
- 不删除内部工程门禁，而是将其分离为 `inference_success` 与 `diagnostic_quality_gate`，在报告中分别存档。
- 核心测试：构造 20 条输入，3 条 ER 原始格式不合法但回退为官方 Schema 合法的提交；在服务和文件正常的情况下，应输出 20 条、回退计数为 3、进程退出码为 0；再测试资产缺失、全局服务退出与写盘失败必须退出非零。

### P1-D：补足 E03-B 输出文本的安全兜底

- 文件：`src/inference/final_dual_protocol.py::parse_emotion_response_output()`。
- 已识别 `emotion=标签;response=正文` 时按原训练协议解析，保留分号和换行。
- 无法识别 `;response=` 时，目前 `response_text=""`。v2 明确建议在模型结构格式异常时尽量保留可用回复文本。可按冻结的、确定性兜底规则去除 `<think>...</think>` 与 Markdown 围栏，然后尝试回收 `response=` 后面的可见文本；仅在无法证明是合理回复时回落为空字符串，避免把格式指令或残缺控制标记塞入正式回复。
- 对历史 A8 Profile 解析失败仍回空列表、E03-B 情绪无效仍回空字符串；不虚构标签、不过度宽松地解析画像。
- 原训练 Prompt、模型权重、原生生成协议不变；这是后处理的格式异常兜底。

### P1-E：核查输入读取的过严约束

- 文件：`participant/final_dual_inference.py::read_test_samples()`。
- 当前强制每条顶层键集**恰好**等于 `{"id","history"}`，且每条历史消息只允许 `role/content/turn_id`，最后一条必须是 user。现有单元测试还固定要求拒绝额外 `target` 字段。
- 官方 v2 README 只明确传入测试 JSONL 路径，未给出每条测试记录必须只有这两个键的声明；因此当前实现不能凭 README 判定完全兼容未知隐藏数据结构。
- 应找到**官方发布的测试输入样例/Schema**做独立校验；如果测试记录允许额外非目标元数据，按白名单提取 `id/history`，安全忽略不进入 Prompt 的无关字段。绝不把隐藏 Gold、target、References 或非历史字段传给模型。未拿到官方依据前不应无差别放宽。

### P1-F：大批量吞吐检查

- 文件：`participant/final_dual_inference.py` 逐样本写入环节。
- 当前每条都有两次 `fsync()`，重读并重新哈希累计 `submission.jsonl` 与 `raw_outputs.jsonl`，随总行数增长造成额外的平方级累计读取工作。
- 保留故障恢复语义，建议使用增量 Hash 状态、固定频率的持久化和 final 文件整体校验；优化后对 100、500、1000 条输入做实际计时/峰值内存验证，确保结果全覆盖及 Resume 仍正确。
- 官方 README 未指定总体时间上限，所以本项列为可测量性能风险，不能把它断言为本次平台失败根因。

## 三、必须补的验证，不再拿 20/100 条“质量全成功”替代契约测试

当前测试大量覆盖了模型身份、20 条 Smoke、接口和异常重试，但尚缺这些关键情形：

| 用例 | 验收行为 |
|---|---|
| Docker 直接传两路径、Docker 传完整 `bash start.sh` 命令、显式 `--entrypoint /bin/bash` | 三种调用方式均可成功产生两份官方产物 |
| RESULT_DIR 预存在旧两份结果及无关文件 | 新结果成功覆盖，其他文件原字节保留 |
| 20 条里 3 条模型格式解析失败 | 20 条输出符合 Schema、失败计入计数、退出码 0 |
| 模型身份不正确、服务整体崩溃、写盘失败 | 非零退出码和明确日志；不冒充有效完成 |
| E03-B 输出缺分隔符/非法情绪/空回复 | 按文档定义的逐字段兜底，其他有效字段保留 |
| 真实官方输入格式边界 | 与官方真实样例对齐，不自行认定未知字段一定非法 |
| 100/500/1000 条或可承受的实际长批量 | 全量输出、仅前 100 条性能、无平方级 I/O 放大到不可接受 |

## 四、修复交付要求

先只修改官方入口、解析后处理和输出报告/诊断逻辑，**不得重新训练或修改 Qwen Base、A8 P4504、E03-B Replay-1152、四份 Prompt 或已有评分 Schema**。

完成修复后，依次：CPU 单测 + 官方命令回归 → 宿主 vLLM 20 条和 100 条 → 新 Docker 构建 → `docker save` / `docker load` → 离线 GPU 官方命令与预存在结果目录测试。由于镜像对推理代码 SHA 做了锁定，更新文件后必须重新生成运行身份锁、Docker 镜像及 tar 的 SHA；不得仅在旧镜像中热替换脚本后交付。

**证据边界**：当前本地 Docker v1.1 报告确实记录 20/100 条推理通过，不能代替上述官方契约回归；本次平台“推理阶段失败”缺少服务端日志，目前没有证据证明具体失败节点。修复并通过所有门禁后再考虑使用剩余提交次数。
