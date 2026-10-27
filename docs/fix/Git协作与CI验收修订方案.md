# EmotionCompany：Git 协作规范与 CI 验收修订方案

> 类型：交给 Codex 的**针对性修订任务**，非新一轮架构重做。
> 审查对象：`LIZZYGREAT/EmotionCompany`，分支 `e00-1-evaluation-infrastructure`，检查时 HEAD `61b88a115cc1e926f84f43f848111efb78e8ef9d`。
> 目标：补齐 GitHub Actions 的真实可运行条件、验证协作门禁、清理剩余文档事实偏差。

## 一、现场事实（先复核后修改）

1. 已有协作 SOP：`docs/knowledge/双人Git协作与PR操作规范.md`；`AGENTS.md` 已引用；已配置 `.github/PULL_REQUEST_TEMPLATE.md` 和 `.github/workflows/ci.yml`。
2. 在方案基线 `61b88a1`，`unit-cpu` 已运行七个测试模块；其中 `tests/test_competition_contract.py` 和 `tests/test_data_audit.py` 读取 `data/train/train_public.schema.json`、`data/val/val_public.schema.json`。当时 `.gitignore` 忽略整个 `data/train/`、`data/val/`，这两个 Schema 未跟踪，干净 Runner 无法获得它们。此缺口已由提交 `3b7b0cf` 修复。
3. 在方案基线 `61b88a1`，开发分支领先 `main` 31 个提交；E00 尚未合入 `main`。不允许直接强推/重设 `main`。本次执行前已 fetch，`origin/main` 仍为该分支祖先。
4. 原始文档 `docs/task/A0_技术架构与实验方案.md` 的目录树出现 `requirements-lock.txt`，但当前 Git 跟踪文件中无此文件；如在其他说明中把它写成已随仓库提供，须纠正。不能编造或从本机全局 `pip freeze` 随意生成承诺可复现的锁文件。

## 二、修订任务 A：解决 CI 官方 Schema 缺失问题（首要）

**优先方案（仅在确认赛事资料可作为项目仓库代码/Schema 分发时）**：允许 Git 版本管理官方两份非数据内容的 Schema，仍忽略全部原始 Train/Val JSONL。

- 修改 `.gitignore` 的匹配粒度，仅给 `data/train/train_public.schema.json` 和 `data/val/val_public.schema.json` 开白名单；不得放开整个目录或允许原始 `.jsonl`、私人数据进入 Git。
- 使用已有本地官方 Schema 文件，不自行按内容推测并重写。提交前按 `tests/test_competition_contract.py` 中锁定的 SHA-256 核对，必须与历史版本**字节完全一致**；如果本地缺失，不能从报告反向编造。
- 精确 `git add` 两个 Schema 和 `.gitignore`，审查 `git diff --cached --name-only`，不得提交训练/验证 JSONL。
- 保持 `tests/test_competition_contract.py` 的原有 SHA 验证，以及 `test_data_audit.py` 的真实 Schema 夹具使用方式。


## 三、修订任务 B：CI 触发与首次验证

- 保留现有 CPU-only 原则：`unit-cpu` 不下载模型权重、不读取本地 `outputs/`、不执行真实 CUDA/NF4 训练、不要求官方 JSONL 数据。
- 复核 `ci.yml` 的所有测试和导入依赖是否可在 fresh checkout + fresh Python 3.10 + 声明的最小依赖下运行。必要时补充**最少**已固定版本依赖；先跑再决定，禁止为省事安装整套 GPU 训练依赖。
- 按用户补充要求，每次分支 push 都运行测试；保留 PR → `main` 触发，不取消前一次 push 的运行。
- 保持 Actions `permissions: contents: read` 和稳定 Job 名 `unit-cpu`；保留 Bash 语法检查、Python compileall 和测试。push 检查 before/after SHA；PR 检查 base/head SHA；新分支首次 push 从 `origin/main` 到 head 检查，避免依赖不明确的 `HEAD^`。
- 在任务分支创建真正的 **PR → main**；以 GitHub Actions 页面返回的绿色结果作为 CI 已通过的唯一证据。**推送分支会运行测试，但不等于 PR 验收。**
- 记录 PR 链接、commit SHA、Actions run 链接、job 名、全部 step 结论；如果 Codex 环境没有 GitHub Actions/写权限，明确交付管理员操作步骤并标记“未验收”，禁止虚构运行记录。

## 五、修订任务 c：文档轻量事实校准

1. `AGENTS.md`：保持简短索引，避免与 SOP 维护两份详尽且易漂移的指令；Required Check 名和当前 CI 实现一致。
2. `docs/workflow.md` 或协作 SOP：添加一段可复制的“第一次 PR CI / branch protection 验收”步骤，并说明 GitHub CI 不证明 GPU、数据与模型依赖可用。
3. `docs/task/A0_技术架构与实验方案.md`：修订 `requirements-lock.txt` 的已存在/已验收表述与实际文件树不一致的问题；不得随意改动 A0 的历史决策部分。如果此处只是原始规划目录树而非当前状态，明确标注“规划示例”即可。
4. 历史任务书 `docs/task/双人Git分支协作_Main保护_PR与GitHubCI_实施规范.md` 作为历史方案保留，无需把它全文再复制到 SOP；如引用历史 26 提交，注明为当时快照，现阶段 main 尚未合并。

## 六、验收命令与期望结果

在工作区确认本任务分支、不覆盖他人改动后执行：

```bash
git status --short
git fetch origin --prune
git branch --show-current

# 若采用“版本管理官方 Schema”方案，下列两文件应可被 Git 跟踪
git ls-files data/train/train_public.schema.json data/val/val_public.schema.json

# 原始数据必须仍被 Git 忽略（返回码 0）
git check-ignore -q data/train/train_public.jsonl
git check-ignore -q data/val/val_public.jsonl

.venv/bin/python -m compileall -q src scripts participant tests
bash -n participant/start.sh
.venv/bin/python -m pytest -q \
  tests/test_competition_contract.py \
  tests/test_data_audit.py \
  tests/test_data_pipeline.py \
  tests/test_e00_metrics.py \
  tests/test_e00_runner.py \
  tests/test_e00_2_diagnose.py \
  tests/test_e00_2_emotion_diag.py

git diff --check
git status --short
```

**最终验收须同时满足**：fresh checkout 的 PR `unit-cpu` 真实通过；另一位队友的批准与 Required Check 均能在真实 PR 阻断/放行；无数据、权重或产物泄露；协作 SOP 与 GitHub 实际设置一致。

**任务边界**：先修 CI 并做首次 PR，再启用服务端保护，最后审查 E00 旧分支整合到 `main`。不自动强推、不自作主张合并、不执行模型实验。

## 七、执行记录（2026-10-27）

- **A：完成并提交** `3b7b0cf`。原有两份 Schema 文件字节未改，SHA-256 与测试锁定值一致；仅加入 Git 白名单，训练/验证 JSONL 仍被忽略。
- **B：完成本地实现并提交** `f03931c`。所有分支 push 和面向 `main` 的 PR 均运行 `unit-cpu`；去除会取消较早 push 测试的并发取消配置。Transformers 计时测试改用合成 torch/Transformers 模块，使 Required CPU 测试不依赖 GPU 全家桶。
- **B：本地全新 Python 3.10 虚拟环境验收**：安装 CI 中四个固定依赖后，`compileall`、`bash -n participant/start.sh` 和七个测试模块全部通过（106 passed）；YAML 结构检查通过。全 PR 差异检查发现的历史 Markdown 硬换行已改写为 `<br>`，并清理了多余 EOF 空行，保留原有渲染语义。
- **C：完成**。已将 A0 目录树标为规划示例，纠正环境指南中不存在的 lock 文件描述，并标注历史 26/40 计数和 `main` 未合入状态；本方案记录随本阶段提交。
- **GitHub 分支 push 验收：通过**。分支 `ci/git-ci-validation`，SHA `7c9960ee8ba625e15d6212aa9100e92d8cf41ad2`；[Actions run 40444610952](https://github.com/LIZZYGREAT/EmotionCompany/actions/runs/40444610952)，事件 `push`，Job `unit-cpu`，结论 `success`。`Set up job`、`Checkout`、`Set up Python`、`Install minimal pinned CI dependencies`、`Check syntax and pushed diff whitespace`、`Run CPU synthetic regression suite`、`Post Set up Python`、`Post Checkout`、`Complete job` 均通过；Runner 上七个测试模块为 **106 passed**。
- **PR / Ruleset 验收：未完成**。GitHub PR 创建接口返回 `403 Resource not accessible by integration`；没有绕过权限改用其他凭据。管理员可在 [GitHub 创建 PR](https://github.com/LIZZYGREAT/EmotionCompany/pull/new/ci/git-ci-validation)，目标 `main`、来源 `ci/git-ci-validation`。Rulesets 查询结果为空；`main` branch-protection 查询同样因集成权限返回 403。因此尚无 PR 事件 CI、Required Check 绑定、队友审批或合并门禁阻断/放行证据，最终验收仍未完成。
