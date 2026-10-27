# EmotionCompany：P / F / T 输入协议设计与版本演进笔记

> **文档性质**：项目知识笔记与协议设计索引，而非新的实现任务书。<br>
> **适用范围**：EmotionCompany / A0 单模型联合预测，训练数据构造、E00/E01 推理与后续方案实验。<br>
> **当前已实现的组合**：`P0 + F2 + T1`。<br>
> **维护原则**：本文解释和记录协议，不取代仓库中的实际 Prompt、Tokenizer 配置、代码和测试。

## 一、先建立整体认识

模型接收到一段对话后，至少面临三个不同问题：

1. **做什么？** 当前任务是普通聊天，还是识别情绪、推断画像并生成结构化回复？
2. **怎样读取对话？** 如何区分系统指令、用户消息、助手消息，以及开始生成的位置？
3. **对话太长怎么办？** 在有限的 Token 预算内保留哪些历史内容？

为区分这三种职责，我们在项目内使用三个方案系列：

| 系列 | 项目内含义 | 主要职责 | 当前方案 |
|---|---|---|---|
| **P — Prompt** | 任务提示词方案 | 规定模型任务、标签范围和输出要求 | **P0** |
| **F — Format** | 对话格式方案 | 将任务指令和多轮历史组织为模型认可的消息结构 | **F2** |
| **T — Truncation** | 历史截断/选择方案 | 根据输入预算选择可保留的历史消息 | **T1** |

> **注意**：`P0`、`F2`、`T1` 是本项目自行约定的版本代号，不是 Qwen、Transformers 或学术界统一的术语。它们代表三个可以分别设计、分别测试的策略维度。

### 1. 一条样本从哪里开始

D0.3 已把官方数据整理成预测点。对单条预测点，至少涉及：

- `history`：截至当前用户消息为止，允许模型查看的连续聊天记录。
- `target`：训练数据中给出的正确情绪、画像和回复。

示例（人为构造，不是官方原文）：

```python
history = [
    {"role": "user", "content": "最近考试越来越近了。"},
    {"role": "assistant", "content": "你现在最担心什么？"},
    {"role": "user", "content": "我一直担心自己考不好。"},
]
```

模型预测时只拥有**任务说明与历史**；正确 `target` 只能在训练构造监督答案时使用，不能混入 Prompt。

### 2. P / F / T 在流程中的位置

```text
D0.3 样本：history + target
           │
           ├── P：取得已确定的任务指令
           │
           ├── T：在固定输入预算内选择 history 的连续后缀
           │     └── 每次测长度都调用 F + 实际 Tokenizer
           │
           ├── F：将 P + 被选择的多轮消息编码为 Prompt Token IDs
           │
           ├── 训练：另将 target 编成完整 JSON Completion
           │         → input_ids / labels / attention_mask
           │
           └── 推理：只输入相同的 Prompt Token IDs
                     → 模型自行生成 JSON
```

这张图展示**逻辑职责**，不是说 T 执行完毕后 F 才第一次运行。T 判断一组消息是否超长时，已经需要调用 F 和 Tokenizer。

---

## 二、P 系列：规定模型要完成什么任务

### 1. P 是什么

**Prompt** 是给模型的任务条件。在项目中，P 系列主要定义固定的 **System Prompt（系统提示词）**，不包含每条样本不断变化的对话 History。

没有任务说明时，语言模型可能直接继续聊天；但比赛需要同时产生：

- 最后一条用户消息的**主要情绪**（16 个合法类别中的一个）；
- 根据截至当前对话证据推断的**用户画像**（性格、兴趣、交流风格三个维度）；
- 面向当前用户消息的**自然语言回复**。

模型内部 Completion 使用固定 JSON 字段：

```json
{
  "emotion_label": "anxiety",
  "user_profile": {
    "personality_traits": [],
    "interests": ["study_exam"],
    "style": []
  },
  "response_text": "听起来考试让你承受了不少压力。",
  "memory_refs": []
}
```

`id` 由推理程序从官方测试输入取得并回填，不要求模型生成；A0 暂将 `memory_refs` 固定为 `[]`。正常预测情绪只使用 16 个类别，官方提交 Schema 的空字符串只是错误回退哨兵，不作为第 17 个训练类别。

### 2. 当前方案 P0

**P0 = 简洁任务型 System Prompt。** 核心要求是：

1. 说明情绪识别、画像推断、回复生成三项任务，以及情绪识别的对象是**最后一条用户消息**。
2. 明确合法情绪/画像标签与数组维度，不允许自行发明标签。
3. 画像判断依赖已有对话证据；证据不足的维度允许空数组。
4. 要求中文、贴合语境的回复；只输出单一合法 JSON，不附加 Markdown 或解释。
5. 采用 A0 已确定的 JSON 字段、顺序与 `memory_refs=[]` 规则。

项目中**唯一权威的 P0 文本**是：

```text
configs/prompts/p0.txt
```

当前冻结文本的 SHA-256：

```text
d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2
```

本文只是解释 P0 的作用，**不要从本笔记复制一份相似但不完全一致的 Prompt 当作第二套权威定义**。

### 3. 后续可能的 P1

`P1` 是**未来可设计的候选**，目前未冻结、未实现。可能的研究方向包括：

- 为容易混淆的情绪标签补充简短判定规则；
- 为画像输出制定更明确的证据标准；
- 提出可检验的回复质量约束（如简洁度、共情与建议的平衡）。

这些都可能影响 Token 开销、输出内容及评分。若设计 P1，应**保留 P0 不变**，单独测试增益，不要悄悄覆盖原基线。

---

## 三、F 系列：规定多轮消息如何交给模型

### 1. F 要解决什么

History 本质上是一串具有角色的消息。模型需要正确区分：

- `system`：比赛任务说明；
- `user`：用户说的话；
- `assistant`：之前助手的回复；
- 最后一个 `user` 之后：应由助手生成新结果。

**Chat Template（聊天模板）**负责将上述消息转成适合特定语言模型的文本/特殊 Token 序列。

### 2. F1 与 F2 的区别

**F1：把所有历史压成一段普通字符串**（此前讨论过的非推荐方案）。

```text
user: 最近考试越来越近了。
assistant: 你现在最担心什么？
user: 我一直担心自己考不好。
```

如果再将整段字符串作为一条 `user` 消息，模型看到的是**一条用户消息中嵌入了一些模拟角色文字**，而不是原生多轮对话结构。此前旧 runner 曾接近这种做法。

**F2：保留原生多轮角色消息**（当前已采用）。

```python
messages = [
    {"role": "system", "content": P0_TEXT},
    {"role": "user", "content": "最近考试越来越近了。"},
    {"role": "assistant", "content": "你现在最担心什么？"},
    {"role": "user", "content": "我一直担心自己考不好。"},
]
```

随后调用 Qwen Tokenizer 自带的 `apply_chat_template`，由模型原生模板插入角色边界和生成开始标记，**不手写 `<|im_start|>` 等特殊标记**。

### 3. 当前 F2 的关键约束

- 模型：`Qwen/Qwen3-1.7B`；模板与 Tokenizer 使用 D0.4.2 锁定的版本。
- `enable_thinking=False`：使用 Qwen3 **Non-thinking** 模式。
- 推理时 `add_generation_prompt=True`：在序列末尾建立 Assistant 即将生成内容的起点。
- 训练时完整 Assistant Completion 需通过**同一模板**编码，且 `full_ids[:len(prompt_ids)] == prompt_ids`。
- 特别处理历史助手消息中作为普通文本出现的 `</think>`，避免 Qwen 模板误判为推理标记；项目已有共享实现，不自行再发明另一套处理方式。
- 训练和正式 runner 必须共用同源消息构造及 Tokenizer/模板约束。

对应代码：

```text
src/shared/prompting.py
  build_messages(...)
  encode_inference_prompt(...)
  encode_full_training_sequence(...)
```

推理入口：

```text
participant/run_inference.py
```

### 4. 后续可能的 F3

`F3` 仅代表**未来可选的新格式方案代号**，不是现有功能。比如未来需要尝试其他合法的消息表示或模型/模板适配，应先明确：

- 是否改变角色边界或消息层级；
- 是否影响 Token ID、上下文长度与生成边界；
- 是否还能保证训练/推理一致；
- 是否需要重新执行 Token 长度审计与监督数据构建。

更换 F 往往会改变模型实际看到的 Token 序列，因此不能只在 runner 中单独替换。

---

## 四、T 系列：在输入预算内选择历史

### 1. T 要解决什么

随着对话增加，历史可能远长于本地训练或服务允许的输入长度。T 系列决定**保留什么历史信息**，而不是决定模型要做什么任务。

先区分几个容易混淆的变量：

| 记号 | 含义 |
|---|---|
| `P0` | **Prompt 方案版本名**，不是数字上限 |
| `P` | **Prompt Token 长度上限**（例如 `P=1280`） |
| `S` | **训练完整序列上限**，Prompt + Completion（例如 `S=2048`） |
| `G` | 推理最多新生成多少 Token（`max_new_tokens`） |
| `K` | 实际推理服务的可用上下文窗口 |

对训练样本需满足：

\[
L_{\mathrm{prompt}} \le P, \qquad
L_{\mathrm{prompt}} + L_{\mathrm{completion}} \le S.
\]

在推理部署中还需检查 `P+G≤K`，但 **S 与 K 不是一个概念**。

### 2. 当前方案 T1

**T1 = 最近连续完整消息后缀策略。**

给定 P0、完整 History、固定 `P` 和真实 Qwen Tokenizer：

1. 先保留当前预测点的**最后一条用户消息**，并计算包含 P0 与 F2 标记的完整 Prompt Token 数。
2. 若仅此最小集合已超过 P，返回 `prompt_floor_exceeds_cap`；不静默裁断当前用户消息。
3. 若未超过 P，尝试向前加入**紧邻的上一条完整消息**，通过 F2 + Tokenizer 重新计算真实长度。
4. 加入后仍满足 P，则保留；否则停止，不跳过这条过长消息去挑选更早的短消息。
5. 最后保留原始顺序不变的**连续历史后缀**。

伪代码：

```python
selected = [history[-1]]  # 当前用户消息
if token_length(F2(P0, selected)) > P:
    return "prompt_floor_exceeds_cap"

for i in range(len(history) - 2, -1, -1):
    candidate = history[i:]
    if token_length(F2(P0, candidate)) > P:
        break
    selected = candidate
return selected
```

上面的 `F2(...)` 是说明职责的伪代码，不是仓库实际函数签名。实际实现是：

```text
src/shared/prompting.py
  select_recent_complete_suffix(...)
```

### 3. 为什么不能使用正确答案的长度决定历史裁剪？

训练时已经知道标准 Completion 多长，推理时却不知道模型未来会生成多长。

如果训练时根据标准答案的实际长度额外删去历史，而推理时仅按 P 选历史，就会导致相同预测点在训练和推理时看到不同输入。因此：

- **T1 只依赖 Prompt / History / 固定 P / 模板与 Tokenizer**；
- 完整监督能否放入训练长度 `S`，应在选好历史后另行检查；
- 即使 Prompt 符合 P，只要完整 JSON Completion 使总长超过 S，就把该样本列为**当前预算下不可用**，不能截断标准答案，也不能改写 D0.3 原始数据。

### 4. 后续可能的 T2

`T2` 是尚未设计的候选代号。以后可以研究不同的历史选择方式，例如摘要、结构化记忆或其他策略，但这些方法会改变可见信息，有可能增加额外计算和信息失真风险。

若考虑 T2，必须先定义其输入、输出、训练/推理是否同源，以及是否使用未来消息或正确答案等**禁止泄漏的内容**。T2 不因听起来更智能就自动优于 T1；要通过同条件评测比较。

---

## 五、三个方案如何相互约束

三者虽可独立编号，但不能独立随意上线。

```text
                   P0：任务规则与合法输出
                            │
 D0.3 完整 History ───→ T1：决定保留哪些消息
                            │  每次测长调用 F2
                            ▼
                   F2：原生 Chat Template
                            │
                            ▼
                     Prompt Token IDs
                       /          \
                      /            \
        训练：追加标准 Completion     推理：直接生成 Completion
        并对它构建监督 labels         （无正确答案）
```

关键一致性约束：

1. **任务一致**：训练与推理不能采用不同的 P 文本或标签集合。
2. **格式一致**：不能训练使用 F2、官方 runner 却退回拼字符串的 F1。
3. **历史一致**：同一 P/T/F 组合与同一 Prompt 上限下，训练和推理的历史选择必须一致。
4. **Token 一致**：训练 Prompt 前缀 Token IDs 必须和推理 Prompt Token IDs 一致。
5. **目标隔离**：`target` 只能用于生成训练 Completion，不能影响 History 选择或泄漏进入 Prompt。

---

## 六、后续扩展：版本与实验管理建议

### 1. 版本名的含义

目前的基线可以写作：

```text
P0 / F2 / T1
```

未来可能出现（**仅为命名示例，尚未设计或实现**）：

```text
P1 / F2 / T1     # 只换 System Prompt
P0 / F3 / T1     # 只换对话格式
P0 / F2 / T2     # 只换历史选择策略
```

编号用于记录**方案身份与版本演进**，不暗示更高编号一定更好。不要重命名 P0/F2/T1，也不要在同名版本下修改其语义。

### 2. 做对照实验时的要求

| 对照 | 改变什么 | 尽量保持什么不变 | 特别检查 |
|---|---|---|---|
| P0 → P1 | 任务说明 | F2、T1、模型、数据、推理配置 | 标签口径、Prompt Token 开销、文本指标 |
| F2 → F3 | 对话编码表示 | P0、T1、模型、数据 | Tokenizer/Template、前缀、特殊标记 |
| T1 → T2 | 历史选择 | P0、F2、模型、数据 | 未来信息泄漏、历史保留、公平比较 |

原则上一次改变一个主要维度。若确实联合调整，要标记为联合实验，**不能把效果归因于单独一个 P、F 或 T**。

### 3. 一次新方案至少应记录什么

建议用一条简洁的协议记录（不必开发庞大的版本管理系统）：

```yaml
protocol:
  prompt_version: P0
  format_version: F2
  truncation_version: T1
  prompt_file: configs/prompts/p0.txt
  prompt_sha256: d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2
  tokenizer: Qwen/Qwen3-1.7B
  enable_thinking: false
context:
  max_prompt_tokens: 1280  # 示例：某一次具名候选配置，非最终冻结值
```

除此以外，应保留实际 Tokenizer 修订号/签名、输入数据版本、实验 ID、生成预算、测试结论及评测记录。以上 YAML 是说明应记录哪些字段的**示例**，不要求改写现有仓库配置结构。

当 P、F、T 任一方案改变时，应检查是否需要重新执行 D0.4.2 Token 审计、D0.4.3 监督构建及 runner 协议测试。不能沿用旧配置的 Token 长度统计冒充新方案结果。

### 4. 当前项目的权威入口

| 内容 | 文件/模块 |
|---|---|
| 冻结的 P0 文本 | `configs/prompts/p0.txt` |
| P0/F2/T1 共享规则 | `src/shared/prompting.py` |
| Token 审计配置及实现 | `configs/data_token_audit.yaml`、`src/data/token_stats.py` |
| SFT 数据构建配置及实现 | `configs/sft_data.yaml`、`src/data/sft_dataset.py` |
| 比赛推理入口 | `participant/run_inference.py` |
| 推理配置 | `participant/configs/inference_config.json` |
| 协议与监督测试 | `tests/test_prompting.py`、`tests/test_inference_protocol.py`、`tests/test_sft_dataset.py` |

本文只作为**概念和演进规则的索引**。代码与配置的实际行为以相应版本的仓库及验收结果为准。

---

## 七、最简总结

- **P：做什么**——规定任务、标签与输出约束；目前采用 **P0**。
- **F：怎么表达对话**——将 System/User/Assistant 转换为模型可识别的原生多轮序列；目前采用 **F2**。
- **T：看哪些历史**——在固定 Prompt Token 预算内保留连续的最近消息；目前采用 **T1**。

**三者组合定义了模型实际看到的输入。** D0.4.3 再基于这个输入追加标准答案、生成 Completion-only Labels；E00/E01 和正式推理应与其保持一致。以后设计新方案时，只替换被研究的维度，保留可复现的旧基线，并通过真实实验判断收益。
