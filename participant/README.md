# 参赛者本地推理工具 v3

本工具支持两种推理后端，推理与计时均在同一次样本遍历中完成：

- `transformers`：直接通过 Hugging Face Transformers 加载本地模型；
- `openai_compatible`：调用实现 Qwen Chat Template 扩展的本地 OpenAI 兼容
  `chat/completions` 接口，例如 vLLM。

Transformers 模式会根据本地 `config.json` 自动选择纯文本 CausalLM 或图文模型加载器。

## 准备

1. 安装依赖：`python3 -m pip install -r requirements.txt`
2. 在 `configs/inference_config.json` 中选择 `backend`，并配置模型目录或本地 Base URL。
3. 正式计时前填写配置中的 `hardware_label`。相对模型路径以本工具目录为基准。部署包还必须包含仓库根目录的 `configs/prompts/p0.txt` 和 `src/shared/prompting.py`；Tokenizer 按配置锁定的 revision、文件指纹和 Chat Template 签名从本地读取，不会运行时联网下载。

## 冻结输入协议和预算

两种后端共用仓库 P0、原生多轮 `system/user/assistant` 消息和 T1 最近连续完整历史选择。测试 `id` 只用于输出回填，不会进入模型 Prompt。默认 `P=1280` 是 `local_smoke_candidate_P1280` 临时候选，`G=1200` 是参考生成上限，不代表最终训练或生成配置。`K` 默认未配置，报告会将服务上下文兼容性记为 `not_verified`；配置 K 后，runner 和启动脚本都会检查 `P+G≤K`，vLLM 启动时同时传 `--max-model-len K`。

可通过 `EVALUATION_MAX_PROMPT_TOKENS`、`EVALUATION_PROMPT_BUDGET_PROFILE` 和 `EVALUATION_SERVING_CONTEXT_LIMIT` 覆盖 P、档位名称及 K。配置来源会写入 `performance_report.json`。真实服务端 Token ID 等价性仍需使用最终 vLLM 版本和模型实测，Mock 请求与 `prompt_tokens` 数量一致不能代替该验收。

## Transformers 本地目录模式

配置示例：

```json
{
  "backend": "transformers",
  "model_path": "./model"
}
```

## 统一启动入口

仓库提供了统一入口，它根据 `configs/inference_config.json` 的 `backend` 自动选择流程：

- `transformers`：直接通过 Transformers 加载本地模型并运行测试；
- `openai_compatible`：启动 vLLM、等待服务就绪、运行测试，然后自动关闭服务。

执行命令：

```bash
bash start.sh TEST_FILE RESULT_DIR
```

只需要传入测试文件和结果目录。也可通过 `EVALUATION_BACKEND` 临时覆盖配置文件中的
后端。vLLM 模式下，模型默认放在 `participant/model`，默认服务地址为
`http://127.0.0.1:8000/v1`，默认服务模型名为模型目录名。

服务配置集中在 `start.sh` 顶部，Tensor Parallel 默认单卡 `1`；多卡时用
`VLLM_TENSOR_PARALLEL_SIZE` 覆盖。也可使用 `VLLM_MODEL_DIR`、`VLLM_HOST`、`VLLM_PORT`、
`VLLM_SERVED_MODEL_NAME`、`VLLM_API_KEY` 和 `VLLM_START_TIMEOUT` 覆盖默认值。额外的
vLLM 参数可直接写入顶部的 `VLLM_ARGS` 数组。脚本默认启用 Hugging Face 离线模式；
完整服务日志会保留在脚本输出的临时日志路径。若需要单独调用已有服务，也可使用以下
推理配置：

```json
{
  "backend": "openai_compatible",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_model": "local-model",
  "api_key": "",
  "request_timeout_seconds": 600,
  "api_options": {}
}
```

`base_url` 可以填写 API 根地址，也可以直接填写完整的
`http://127.0.0.1:8000/v1/chat/completions`。无需鉴权的本地服务将 `api_key` 留空即可。
`generation.max_new_tokens` 会自动转换为接口参数 `max_tokens`；`api_options` 可用于传递
`response_format` 等服务端扩展参数，但不能覆盖 `model`、`messages`、`max_tokens` 或
`chat_template_kwargs`。Qwen3 vLLM 请求固定包含 `chat_template_kwargs.enable_thinking=false`。
若历史 assistant 文本含字面 `</think>`，默认会显式阻止 API 请求；只有确认目标服务支持 Qwen
模板的 `reasoning_content` 扩展后，才将 `openai_compatible.pass_reasoning_content_field` 设为
`true`。该扩展行为需要在最终服务端版本验证。

也可不修改配置文件，使用环境变量切换：

```bash
EVALUATION_BACKEND=openai_compatible \
EVALUATION_BASE_URL=http://127.0.0.1:8000/v1 \
EVALUATION_API_MODEL=local-model \
python3 run_inference.py TEST_FILE RESULT_DIR
```

可用覆盖变量为 `EVALUATION_BACKEND`、`EVALUATION_BASE_URL`、
`EVALUATION_API_MODEL`、`EVALUATION_API_KEY`、`EVALUATION_MODEL_PATH`、
`EVALUATION_HARDWARE_LABEL`、`EVALUATION_TOKENIZER_PATH`、`EVALUATION_MAX_PROMPT_TOKENS`、
`EVALUATION_PROMPT_BUDGET_PROFILE` 和 `EVALUATION_SERVING_CONTEXT_LIMIT`。API Key 只用于请求头，不会写入结果文件。

## 唯一运行命令

```bash
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

主办方只会运行该命令，并传入上述两个位置参数，确保脚本位置无更改：

- `TEST_FILE`：具体的待评测 `.jsonl` 文件，例如 `/data/test_public.jsonl`。
- `RESULT_DIR`：结果目录；允许已存在且包含其他文件。本次运行会覆盖其中的
  `submission.jsonl` 和 `performance_report.json`，其他文件保持不变。

固定产物为：

- `RESULT_DIR/submission.jsonl`：逐条预测结果；
- `RESULT_DIR/performance_report.json`：同轮端到端耗时及摘要。

脚本会对测试文件中的全部数据推理并生成完整提交，但性能成绩只统计前 100 条数据的
端到端耗时。

模型 JSON 中某个字段缺失或格式不合规时，脚本不会中断整批推理：文本和情绪字段写为
空字符串，画像子项和记忆引用写为空数组。空值作为该项预测失败参与评分，不会获得该项
命中分。模型输出无法解析为 JSON object 时，脚本会尝试提取 `response_text`；若无法
单独提取，则将去除思考标签和代码围栏后的完整原始输出作为回复文本，其余字段置空并
继续下一条。失败数量和样本 ID 记录在性能报告中。
