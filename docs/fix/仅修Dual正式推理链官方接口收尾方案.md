# EmotionCompany：Dual 正式推理链简化与最终 Docker 交付修订方案

> **Codex 执行任务**。以 `feat/final-dual-adapter-delivery` 分支为基础（方案审查参考 HEAD `5a9f5370657564652c19adca3c5187d570fbeb1b`，执行前重新记录 HEAD）。本轮仅修官方入口调用及过度检查，不改模型、训练、Prompt、双 Adapter 选择和生成参数。最终 Docker 必须包含 `report/main.pdf`。不再跑全量 Val、不再做预测结果一致性/准确率门槛。

## 一、修订边界与真实调用链

**禁止直接以用户上传的官方约 465 行 `run_inference.py` 覆盖仓库的 `participant/run_inference.py`。** 当前仓库该文件兼作公共库，被数据预处理、训练、诊断和 Dual 模块导入，直接替换会破坏其他工作流。真正的官方双 Adapter 执行链为：

```text
bash /root/participant/start.sh TEST_FILE RESULT_DIR
  └─ mode=dual_adapter
       └─ python /root/participant/final_dual_inference.py TEST_FILE RESULT_DIR
            ├─ src/inference/final_dual_protocol.py
            ├─ src/inference/final_dual_vllm.py
            ├─ src/inference/final_dual_runtime.py
            └─ participant/run_inference.py（当前仅复用少量公共函数）
```

`participant/start.sh` 的单模型分支仍需要 `participant/run_inference.py`，这次**不修改、不替换、不回退**它。用户上传的官方 `README.md` / `run_inference.py` 用作**行为契约基准**，而不是直接覆盖项目文件：官方只要求传入两个参数、读取 ID/历史、逐样本预测、输出 `submission.jsonl` 和 `performance_report.json`，后者记录本轮前 100 条耗时；原始 ID 原样回填，单条预测解析失败不导致整批退出。

**边界**：只允许修改 Dual 运行器及其直接依赖、Dual 模式的 `start.sh`/Docker 启动、镜像打包脚本和与之对应的少量测试。已有 E01-B/A8/E03-B 训练脚本与旧模型保持原样。

## 二、Dual 模式只保留最小推理流程

### 1. `participant/final_dual_inference.py`：去除实验门禁

将官方默认执行路径收敛为：

```python
config = load_dual_config()                 # 确定后端、模型/Adapter/Prompt 路径
samples = read_input_jsonl(TEST_FILE)       # 只要求 id 为字符串、history 为数组
ensure_result_dir(RESULT_DIR)               # 允许已存在，不动其他文件
runtime = start_or_connect_vllm(config)     # 单 Base + 两份 LoRA

for index, sample in enumerate(samples, 1):
    started = monotonic()
    profile_raw = try_generate("profile", build_a8_messages(sample["history"]))
    er_raw = try_generate("emotion-response", build_e03b_messages(sample["history"]))
    prediction = assemble_with_field_fallback(sample["id"], profile_raw, er_raw)
    write_jsonl_line(submission, prediction)
    if index <= 100:
        latencies_ms.append((monotonic() - started) * 1000)

write_official_performance_report(...)
return 0  # 只在完整处理并成功写出两个官方文件时返回 0
```

需要删除/退出**正式执行路径**的逻辑：

- `_verify_runtime_code_identity()`、Git SHA/运行时代码 SHA、Base/Adapter/Prompt/Tokenizer 指纹匹配、`model_lock.json` 与配置文件的哈希互锁；只保留必要路径存在、模型可加载、服务可用。`model_lock.json` 可留作离线元数据，不能作为运行必需门禁。
- `prompt_token_ids` 精确一致性、服务返回统计字段与本地计数比较、质量门槛、`format_threshold`、候选重复文本检测触发的拒绝、隐藏结果与历史参考输出对比。统计字段如存在可记录，不应影响结果与退出码。
- 官方默认运行不走 `_resume_identity()`、事务输出目录、`run_progress.json`、逐行原始输出哈希、恢复身份比较、最终发布前的整文件 SHA 回验。现有本地诊断功能若需保留，须明确标为**非官方可选模式**，不得在官方调用中执行。
- 不要把到达 `max_new_tokens`、ER `finish_reason=length`、字段格式解析失败计为整批失败；使用已生成文本进行字段级兜底。避免因质量判断反复生成大段回复；原生成参数保持原样。

保留：两路冻结的历史裁剪和 Prompt、一个 Base 两份 LoRA 的路由、固定输出字段、GPU 实际加载失败/文件不可读/输出写入失败等**真正工程错误**的异常反馈。对单条请求失败可记录错误并使用该路字段空值；如果服务整体不可用，应明确返回失败，不能生成全空伪成功结果。

### 2. 输入读取与 ID 完全放开

`final_dual_inference.py::read_test_samples()` 当前同时有 `set(row)=={"id","history"}` 和 `OFFICIAL_ID.fullmatch()`；两个都要修改：

```python
if not isinstance(row, dict):
    raise ValueError("input row must be a JSON object")
if not isinstance(row.get("id"), str) or not isinstance(row.get("history"), list):
    raise ValueError("input row needs string id and array history")
samples.append({"id": row["id"], "history": row["history"]})
```

允许多余顶层字段；ID 为不透明字符串，不匹配正则、不给前缀限制、不改名；读取时可保留重复 ID/明显无效 JSON 的必要错误提醒。历史消息只提取 `role/content` 供原 Prompt 代码使用，忽略不影响推理的其他元数据，**不因 `turn_id` 等附加字段触发拒绝**；保留生成输入所必需的角色/内容有效性要求。

最小回归输入必须包含：`test1_000001_t5`、`test_000001_t0`、带额外顶层元数据和历史消息额外字段的记录。输出 ID 与输入逐字一致。

### 3. `src/inference/final_dual_protocol.py`：字段级兜底，取消运行期 Schema 拒绝

保留 `parse_specialist_output()` 和 E03-B 的 `emotion=...;response=...` 解析；只取正式需要的情绪/回复及画像字段。

- A8 格式非法 → `user_profile={"personality_traits":[],"interests":[],"style":[]}`。
- E03-B 解析失败 → 可可靠提取的 `emotion_label` 或 `response_text` 尽量保留，无法提取的字段分别置 `""`。
- `memory_refs=[]`；`id` 原样回填。
- **移除** `assemble_submission()` 内的 `Draft202012Validator(...).validate(submission)`；也移除 Dual 主循环和完成状态中“必须全部通过该 Schema 才可退出 0”的判断。无需生成一份新的严苛 Schema 替代它。
- 离线测试如继续使用 `participant/templates/submission_schema.json`，必须删除其 `id.pattern`，仅保留 `"id":{"type":"string"}`，以免测试工具错误拒绝官方实际 ID；其他字段结构/枚举保留。

### 4. `src/inference/final_dual_vllm.py`：仅使用必需的 Chat Completion 响应

当前 `chat_completion()` 强制检查返回的 `prompt_token_ids`、`usage.prompt_tokens`、`usage.completion_tokens`，并把 Token ID 不一致视为异常；全部移除硬门禁。请求只需指定 `model`（`profile` 或 `emotion-response`）、`messages`、生成参数，读取 `choices[0].message.content`。对于 `usage`、`finish_reason`、`prompt_token_ids` 等非必需字段仅做可选统计：缺失时使用 `None`，不得清空已经收到的文本。可去掉 `return_token_ids: true`。

保留 HTTP 连接/请求超时和没有任何有效 content 时的**单路请求错误处理**；不能在模型已返回正常文本的情况下因诊断信息缺失抛异常。

### 5. Tokenizer：不动共享库，Dual 自己走普通加载

当前 `final_dual_inference.py::_verify_assets()` 使用：

```python
from participant import run_inference as runner
base_identity = runner._weight_identity(...)
tokenizer, tokenizer_identity = runner.load_runtime_tokenizer(...)
```

而 `load_runtime_tokenizer()` 函数本身**必需** `files_sha256`、`expected_template_sha256` 和 `expected_signature_sha256`；简单删除这些参数会报 `KeyError`。

**优先做法**：Dual 侧直接用：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    local_base_model_dir, local_files_only=True, trust_remote_code=False
)
```

并将其用于现有 A8/E03-B Prompt 编码，保留 `enable_thinking=False` 等冻结训练协议。这样完全不必改 `participant/run_inference.py`、训练/诊断代码及其 `load_runtime_tokenizer()` 定义。若有其他公用辅助函数被 Dual 用到，逐一确认是否必要；不要因为减少哈希检查而把整个公共库删掉。

### 6. 输出写入与退出码

官方运行应允许 `RESULT_DIR` 预先存在，**只覆盖** `submission.jsonl`、`performance_report.json`，保留目录中其他文件不变。按样本逐行写出提交，性能只记录本轮前 `min(100, N)` 条的整体端到端耗时；不强制推理结果合法率、F1 或格式通过率。统计可保留总数、完成数、失败数和失败 ID。

退出码 **0**：所有输入都已处理，两份官方文件成功生成且可读取；即使部分样本字段空值也算完成。退出码 **非 0**：无法解析输入文件、基本模型/服务无法启动、输出目录不可写或程序中断导致结果不完整等实际工程故障。

## 三、`start.sh` 与 Docker ENTRYPOINT 的最小调用链

保留 `start.sh` 的官方两个参数及 `conda_env` 激活，Dual 分支继续：

```bash
exec "$PYTHON_CMD" "$SCRIPT_DIR/final_dual_inference.py" "$TEST_FILE" "$RESULT_DIR"
```

不要让平台先运行自己修改的 `run_inference.py`。Docker 包装脚本只负责必要的参数透传，不再强制执行 `nvidia-smi` 命令、探针哈希检查或运行时代码指纹一致性；实际 vLLM 初始化能够访问 GPU 才是必要条件。保留必要的失败日志输出到 stderr，便于在缺少平台日志时复现。

如果 Docker 仍使用自定义 ENTRYPOINT，确保：

- `docker run IMAGE TEST_FILE RESULT_DIR` 可以调用同一 `start.sh`。
- `docker run IMAGE bash /root/participant/start.sh TEST_FILE RESULT_DIR` 不因包装脚本误判退出。
- 官方在已启动容器内直接调用 `bash /root/participant/start.sh TEST_FILE RESULT_DIR` 也可运行。

无参数容器是否需要常驻应根据主办方真实调用方式确定；不能以假设替代唯一规定的脚本命令。

## 四、镜像必须加入 `report/main.pdf`

**用户新增的明确交付要求：** 下一次导出参赛镜像时必须把仓库本地 `report/main.pdf` 放入镜像中的 **`/root/report/main.pdf`**。当前远端 Git 文件树未看到该文件，因此在本地执行时先确认实际存在，不要凭空创建空白 PDF。

实施至少修改三处：

1. `scripts/final_dual_docker_package.py` 的 `COPY_FILES` 增加 `Path("report/main.pdf")`。`prepare_context()` 在复制阶段检查源文件存在、大小大于 0、文件头以 `%PDF-` 开始，并将 SHA-256 和字节数记录到 `package_manifest.json`。PDF 无需进行内容重写，也不进入推理代码。
2. `docker/final_dual/Dockerfile` 新增：

   ```dockerfile
   COPY report/main.pdf /root/report/main.pdf
   ```

3. 采用 Overlay 增量镜像时也需包含相同 `COPY report/main.pdf /root/report/main.pdf`，不能只在全量构建 Dockerfile 加 PDF。最终镜像内用 `test -s /root/report/main.pdf` 验证存在，并与本地 `report/main.pdf` 比对 SHA-256。

如 `report/main.pdf` 尚未在本机仓库对应目录，不允许静默忽略、使用其他 PDF 替代或假装打包成功；停止**打包**阶段并报告缺失路径。原模型资产保留不变。

## 五、打包脚本也必须减负，并复用旧镜像层

当前 `scripts/final_dual_docker_package.py` 的 `check_static()` → `verify_previous_gates()` → `prepare_context()` 会要求历史 20/100 条预测报告、旧源码 SHA、Prompt Token ID 一致、格式率和 Replay 统计；这些与本次简化目标冲突。

- **删除正式打包的旧实验 Gate 依赖**，只检查需要打包的 Base、两份 Adapter、四份冻结 Prompt、Dual 推理代码、`report/main.pdf` 等必需文件存在/非空；无需先重跑历史实验来解锁构建。
- `verify_output()` 只检查镜像运行确实生成两份官方文件、提交行数与输入相同、ID 原样回填；不比较旧实验结果、模型生成内容、Token IDs、95% 原生格式率和预测统计。
- 为了避免重新下载安装依赖，优先从本机**已有的最近一次可运行完整镜像**创建 Overlay Dockerfile：`FROM <已存在的完整镜像标识>`，仅 `COPY` 本轮变动的 Python/Shell/配置以及 `report/main.pdf`。不要因代码更新执行 `conda create`、`pip install`、`docker pull`、`docker build --no-cache`。旧镜像若仅在本机 tar 中，先 `docker load`；只有旧镜像无法取得时才回退全量构建。
- 若最终基础镜像本身包含旧的 `model_lock`、`package_manifest`，可以留作归档元数据，但新 Dual 正式入口不能根据旧 SHA 拒绝运行。避免因新代码与旧锁不匹配发生新的启动失败。
- 正式提交仍需 `docker save` 导出**完整镜像**（不只是 Overlay 层），记录镜像 ID、Tar SHA-256、大小（官方 50GB 上限）。

## 六、最小验收：只看接口能否运行

不要重新做全量 Val、预测结果对照、F1 评测或同一输出的逐 Token 等价性。Codex 只需完成以下有明确用途的测试：

1. **CPU 无模型协议测试**：`test1_000001_t5` 等不规则 ID 和附加字段可读；单条两路模拟原始输出可组装固定五字段 JSON；任一路格式坏时仍完成写入。
2. **模拟 vLLM 返回值**：只返回 `choices[0].message.content`、没有 `usage` 或 `prompt_token_ids` 时，客户端仍能提取正文并继续。
3. **实际官方命令小样本**：用 2–5 条输入执行 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`，确认退出码 0、两个文件存在、JSONL 行数与输入相等、ID 完全保持；预置结果目录含有其他文件时不被删除。
4. **镜像核验**：离线 Docker 中按相同命令执行 2–5 条；确认 Base 与两 Adapter 正常加载、容器不尝试联网、`/root/report/main.pdf` 存在且 Hash 正确；保存命令、退出码、stderr 和镜像 SHA。

只有工程错误才导致测试失败；不设置画像准确率、情绪正确率、生成长度或严格原生格式成功率的门槛。最后交付一份简短收尾报告：修改文件、实际执行命令及退出码、两个产物路径、PDF SHA、镜像 tag/tar SHA 和下一次上传文件的精确路径。

**验收边界：仅对齐官方接口与交付文件。** 不改变两个模型，不碰 `participant/run_inference.py` 的公共函数，不增加新的训练或评测工作。
