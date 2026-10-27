# EmotionCompany：双人 Git 分支协作规范、`main` 保护与 GitHub CI 实施方案

> **文档定位**：同时作为两名队员日常遵守的仓库协作规范，以及交给 Codex 的一次性治理实施任务。先建立规则与 CI，再进入 E01；**本文件不是要求立即训练模型、修改评测逻辑或部署 Docker**。<br>
> **仓库**：`LIZZYGREAT/EmotionCompany`。<br>
> **核查基线**：2026-10-27，远端 `e00-1-evaluation-infrastructure` HEAD `eff89062cc07d11e141ef2eae1069d83156940a7`，`main` HEAD `7b225eb8b99…`；实施前以 GitHub 和本地 Git 最新状态为准。<br>
> **重要**：下面的 GitHub 设置需仓库管理员在网页操作。本方案不代表这些设置已经开启。

## 一、目标、现状和协作原则

### 1. 目前的仓库状态

核查结果：

| 项目 | 当前状态 | 需要的措施 |
|---|---|---|
| 默认分支 | `main` | 继续作为唯一稳定集成分支 |
| 分支 | `main`、`e00-1-evaluation-infrastructure` | E01 起使用短期任务分支 |
| `main` 保护 | GitHub Branch API 显示 `protected=false`；未检索到仓库 Rulesets | 建立并启用一套 `main` Ruleset |
| 自动 CI | 目前没有 `.github/workflows/` | 新增最小 CPU CI |
| E00 开发成果 | 现有 E00 分支比 `main` 领先 **26 个提交、40 个变更文件** | 进行一次可审查的 PR 整合；禁止直接/强制推送 `main` |
| 官方比赛契约 | `AGENTS.md` 与 `docs/competition/赛事文件边界与运行契约.md` 已有保护规则 | 新规范必须兼容，不另造冲突规则 |
| 本地资源 | 原始数据、生成结果、模型缓存、CUDA GPU 不应依赖 GitHub CI | CI 只执行可自包含的 CPU / 合成测试 |

> **可验证性说明**：分支、提交、Rulesets 和工作流目录来自仓库检查；GitHub 设置是否经过人工修改应在实施后重新核对。此次没有替队伍修改任何仓库或网页设置。

### 2. 三条基本原则

1. **`main` 永远保持可追溯、可复现、通过合并门禁的状态**。即使当前模型效果还不理想，也不把进行到一半的训练实验直接推入 `main`。
2. **两人互相审核，而非各自维护两个长期主干**。每项工作有明确负责人；代码、配置、实验结论经过 PR 才进入 `main`。
3. **以真实验收代替形式性检查**。CI 检查不依赖数据或 GPU 的回归；真实模型训练、推理与平台验收由实际运行者提供可核验日志、命令和指纹。两者不能互相冒充。

## 二、统一分支模型

### 1. 分支职责

```text
main  （稳定/受保护，唯一共同基线）
 ├── ci/main-protection                （CI 与 GitHub 规则）
 ├── docs/e00-stage-handoff           （阶段总结与文档维护）
 ├── feat/e01-qlora-preflight         （E01 真模型训练预检）
 ├── feat/e01-small-fit               （E01 小样本可学习性验证）
 └── fix/<specific-problem>           （明确可复现的修复）

任务分支 ──提交──> PR ──对方审核 + CI──> squash merge──> main
```

上述分支名是示例，不要求预先全部创建。

| 分支类型 | 规则 |
|---|---|
| `main` | 禁止直接 `push`、强推、`reset` 后推送；任何变更均通过 PR |
| `feat/<task>` | 新功能、训练/评测实验的程序实现；一项任务一个分支 |
| `fix/<issue>` | 明确 Bug 和回归修复，不混入新功能 |
| `docs/<task>` | 文档、实验报告、交接；不暗中修改运行行为 |
| `ci/<task>` / `chore/<task>` | CI 和仓库维护，不夹带训练调参 |
| 旧 `e00-1-evaluation-infrastructure` | 作为已发生的历史工作分支处理，完成整合前不要删除或强制改写 |

**不设永久 `develop`、`dev`、双人 `work` 分支。** 两人规模不需要第二个长期集成分支；存在依赖关系时让后续任务短期从同一工作分支派生，或使用明确的 stacked PR，并在前序 PR 合并后更新后续 PR 基线。优先保持简单。

### 2. 创建任务分支

```bash
# 每次开始新任务，先检查并获取远端状态
git status --short
git fetch origin --prune

git switch main
git pull --ff-only origin main

git switch -c feat/e01-qlora-preflight
# 开发 / 测试 / 提交
git push -u origin feat/e01-qlora-preflight
```

**要求**：新任务默认从最新 `origin/main` 创建，不从已合并、长期陈旧的 E00 分支继续派生。开启新任务前，检查工作区是否有未提交改动；不允许 `git reset --hard` 清理队友的未保存成果。

### 3. 当前任务分支跟进 `main` 的方式

本项目默认推荐**在自己的、尚未交接给队友协作的任务分支上 `rebase`**，保持 PR 提交历史清晰；如任务分支已被两人共同提交或其他分支依赖，则优先讨论后使用 `merge origin/main`，避免无意改写别人的提交。

```bash
git fetch origin
# 确保自己在任务分支，并且工作区干净
git rebase origin/main
# 若该任务分支之前已推送，且确实仅本人维护：
git push --force-with-lease origin feat/e01-qlora-preflight
```

`--force-with-lease` **只允许用于确认归属清楚的个人任务分支**，不是日常必须操作，更不能用于 `main`。如果远端有新的队友提交、收到 lease 拒绝或不清楚历史关系，**停止操作、先 `fetch` 并核对日志**，禁止改用裸 `--force`。

发生冲突时先解决冲突、运行测试，再继续 `git rebase --continue`；不清楚冲突处理逻辑时使用 `git rebase --abort` 撤销本轮重放，而不是盲目 `reset`。

### 4. 提交与暂存规则

- 小步提交：一个提交表达一个可理解的更改；不要把训练数据、日志、大型生成文件夹一起提交。
- 格式建议：`feat(e01): ...`、`fix(evaluation): ...`、`docs(e00): ...`、`test(ci): ...`、`chore(repo): ...`。
- 默认使用 `git add <明确文件路径>`，**不要无检查地运行 `git add .`**。
- 提交前检查：`git diff --check`、`git diff --cached --stat`、`git diff --cached`、`git status --short`。
- 永不提交：官方原始数据、`data/processed/`、`outputs/`、`participant/model`、`.env`、Token/API Key、Checkpoint、权重文件、`.venv/`。
- 参赛官方文件与 P0/F2/T1/D0.3 历史协议的修改仍遵守 `AGENTS.md` 与赛事保护文档；此规则优先于一般代码改动习惯。

## 三、Pull Request：从开发到合并

### 1. 必须执行的流程

1. **作者**从 `main` 创建任务分支，开发时完成本地相关测试与产物核查。
2. 推送远端任务分支，创建 **PR：`base=main`，`compare=<task branch>`**。
3. PR 描述写清目标、文件范围、行为变化、测试命令与实际结果、风险、必要的实验/日志链接；如果仅有环境阻断，明确标记 `BLOCKED`，不可填写“通过”。
4. **另一名队员**审查 PR 的代码、设计含义、模型/数据版本、参数和边界；不只是点击 Approve。PR 作者不能通过自我审核满足 1 人批准规则。
5. CI 的 Required Checks 全绿，审查意见已处理后合并。新提交导致原审批过时时，需要重新获得对方审批。
6. **优先采用 Squash and merge**：一次 PR 成为 `main` 上一个有主题、可追踪的提交。合并后双方 `git fetch` 并同步 `main`，清理已无依赖的任务分支。

### 2. PR 必填信息

建议创建 `.github/PULL_REQUEST_TEMPLATE.md`，模板包含以下内容：

```md
## 本次目标
- 解决什么问题／属于哪个阶段？

## 变更范围
- 核心代码／配置／测试／文档分别改了什么？
- 是否涉及赛事保护文件、模型身份、数据划分或 P0/F2/T1？

## 验证记录
- [ ] 本地相关测试：命令 + 实际结果
- [ ] GitHub CI：通过 / 失败 / 尚未运行
- [ ] 如涉及真实 GPU：硬件、Run ID、模型/数据/配置指纹、结果路径

## 风险与未完成事项
- 当前问题、未验证项目、后续依赖

## 审核检查
- [ ] 无敏感数据、缓存、权重或 generated outputs 入库
- [ ] 未改变官方入口及提交 Schema，或变更已通过专项审查
- [ ] 运行结果与文档表述一致
```

尽量保持 PR 小而单一；如果 E00 历史整合这种大 PR 不可避免，应附阶段报告和各模块检查清单，不能以“大多数代码已做完”为由跳过审核。

### 3. 什么情形不能合并？

- Required CI 失败、结果未知、被手动跳过，或 GitHub 显示检查仍在等待。
- 审核者尚未批准，或其审批因后续提交已过时。
- 版本化报告写“模型效果已验证”，却只有 Mock、合成测试或被阻断的 GPU 环境。
- 引入不受控模型/数据变更、泄漏标签、推理模板漂移；或发布未验证的性能数字。
- Git 状态存在意外文件、冲突处理不清楚或修改了不属于本任务的用户/队友文件。

### 4. 紧急修复

从最新 `main` 创建 `fix/<issue>`，仍使用 PR、对方复核与 CI。不要把“紧急”当作绕过合并规则的日常通道。确需仓库管理员临时调整保护策略时，须在 PR/Issue 中写明原因、影响、执行者与恢复操作，完成后立即恢复；禁止默认开放永久 bypass。

## 四、GitHub `main` 分支保护：推荐设置

### 1. 配置前置条件

- `main` 已经是默认分支，保留不变。
- 仓库管理员先确认队友拥有适当的仓库 **Write** 权限（用于提交分支、发起 PR 与审核）。协作者加入须由其自行接受邀请。
- 建议使用 **Repository Ruleset** 而不是同时建立多套重复的 Branch protection 规则；GitHub 可以叠加规则，重复配置容易造成“明明通过却无法合并”。
- 本仓库目前是公开仓库，GitHub Free 支持公开仓库 Rulesets/受保护分支。不同权限/计划下的 UI 以实际 GitHub 页面为准。

### 2. 管理员实际操作步骤

打开仓库 `Settings → Rules → Rulesets → New ruleset → New branch ruleset`，建议：

| 设置 | 建议值 | 原因 |
|---|---|---|
| Ruleset Name | `protect-main` | 名称清楚 |
| Enforcement | `Active` | 真正拦截违规操作；启用前先确认 CI Job 已出现并成功 |
| Target branches | `main`（精确匹配） | 仅限制稳定分支，不限制个人开发分支 |
| Restrict deletions | 开启 | 禁止误删 `main` |
| Block force pushes | 开启 | 禁止重写主干历史 |
| Require a pull request before merging | 开启 | 所有正式变更必须通过 PR |
| Required approvals | **1** | 两人团队必须交叉审核 |
| Dismiss stale pull request approvals | 开启 | 审核后若继续改代码，旧审批作废 |
| Require conversation resolution | 开启（如 UI 提供） | 合并前处理审核意见 |
| Require status checks to pass | 开启 | CI 不过不能合并 |
| Required status check | **实际首次成功出现的唯一 CI Job 名称**，例如 `unit-cpu` | 防止绑定错误 check 名导致永远 pending |
| Require linear history | 开启 | 配合只允许 Squash merge，保持主干提交线性 |
| Require branches to be up to date before merging | 建议开启 | 避免 PR 在过时的主干上通过测试；两人项目维护成本可接受 |
| Bypass actors | 原则上不添加 | 避免管理员/机器人跳过规则 |
| Require signed commits / merge queue / deployments | 初期不开启 | 目前没有签名基建、发布队列或部署 CI，避免不必要门槛 |

**注意**：Ruleset UI 提供的选项可能有所不同；如果 `Require conversation resolution` 等设置在实际 Ruleset 中不可用，确认 GitHub 支持的替代配置，不要同时创建相互冲突的保护规则。保护不等于仓库 owner 永远无权修改设置，但日常流程必须遵守。

在 `Settings → General → Pull Requests`：

- 开启 **Allow squash merging**；团队约定统一使用它。
- 可关闭 Merge commit 与 Rebase merge 两种按钮，防止操作分歧；若保留也不得随意使用。
- 建议开启 **Automatically delete head branches**，但删除前必须确认无下游任务依赖；自动删除不意味着删除本地分支。

### 3. 正确的启用顺序（防止锁死仓库）

**不要在尚未存在成功 CI Job 时，直接填一个想象的 Required Check 名称。**

1. 新增 CI 工作流并通过一个真实 PR 触发，确认 Job 名称及结果确实出现在 GitHub Checks。
2. 解决首次 CI 的真实失败，确认 CI 不依赖本地 `outputs/`、官方原始数据或 CUDA。
3. 由管理员配置 `protect-main` Ruleset，绑定已出现的 check 并启用。
4. 用一次新的 PR 验证：无审核、CI 红灯时无法合并；CI 绿灯且队友批准后才能合并。
5. 记录设置截图或手工核验表到仓库文档；不要把访问令牌或后台截图中的敏感信息提交。

## 五、GitHub Actions CI：保持轻量、可复现

### 1. CI 到底负责什么

| CI 自动检查（Required） | 用户本机/队友机器验收（不放在 GitHub Required CI） |
|---|---|
| Python 语法、Shell 语法 | CUDA/NF4/真实权重加载 |
| 不依赖真实数据的合成数据回归 | 真实 QLoRA forward/backward/Adapter 更新 |
| 官方入口和提交 Schema 契约测试 | D0.4.2 全量 Token Audit |
| 基础评测数学逻辑与 Parser 回归 | 真实 20/100/1833 样本推理、延迟/显存 |
| PR 合并前的必要代码门禁 | vLLM/Docker/赛事平台完整离线验收 |

CI 的绿色代表**已选 CPU 级代码门禁通过**，不是模型效果达标，也不是比赛所有环境已验收。

### 2. 推荐最小工作流文件

由 Codex 增加 `.github/workflows/ci.yml`，首次版本建议如下。以下是**拟实施配置**，并未在 GitHub Runner 上实测；Codex 应在首个 PR 的 GitHub Actions 中验证后，按必要的最小差异修正依赖与测试选择。

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  unit-cpu:
    name: unit-cpu
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 2

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install minimal CI dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install \
            'pytest==9.1.1' \
            'PyYAML==6.0.3' \
            'jsonschema==4.26.0' \
            'sacrebleu==2.5.1'

      - name: Syntax and whitespace
        run: |
          python -m compileall -q src scripts participant tests
          bash -n participant/start.sh
          git diff --check HEAD^ HEAD

      - name: CPU synthetic regression
        run: |
          python -m pytest -q \
            tests/test_competition_contract.py \
            tests/test_data_audit.py \
            tests/test_data_pipeline.py \
            tests/test_e00_metrics.py \
            tests/test_e00_runner.py \
            tests/test_e00_2_diagnose.py \
            tests/test_e00_2_emotion_diag.py
```

**部署前必须注意：**

- 上述候选测试目录基于当前 E00 分支选择；**CI 具体用例必须在首次 PR 的纯净 Runner 中验证**。部分测试如需额外依赖，应补充最小、明确、固定版本依赖，而不是直接安装 `requirements.txt` 的 CUDA 全家桶。
- 如果某个测试真正依赖 Hugging Face 本地模型、`data/processed/`、`outputs/`、GPU，则应移出 Required CPU Job、使用 Mock/合成测试代替，并在 PR 写清原因；不能用静默 `skip` 伪装通过。
- Required 工作流**不使用 paths / paths-ignore 筛选**，避免 PR 没有触发但 Required Check 永远 Pending。
- Job 名称保持唯一、稳定（此处 `unit-cpu`），变更名称时同步更新 Ruleset 中的 Required Check。
- 不使用 `pull_request_target` 来执行 PR 代码，不给 `GITHUB_TOKEN` 写权限，不在 CI 中配置 HF Token 或赛事密钥；公开仓库尤其需要注意来自 fork 的 PR。
- `actions/checkout` / `setup-python` 的正式版本应在首次接入时再按官方维护状态确认；有严格供应链要求可固定到经审核的完整 commit SHA。
- 先保证 **一个 Required Job 真正可靠**，以后可增加轻量 `lint`、受控依赖兼容测试；不要一次建立冗长 CI 矩阵。

### 3. GPU / 数据专属检查：人工 Gate

涉及真实模型/训练的 PR，除 CI 外作者必须附：

- 当前分支/提交 SHA、命令、Python 与关键依赖版本、GPU；
- 数据/Prompt/Tokenizer/权重身份，Run ID、配置 SHA、样本数量；
- 实际日志摘要、退出码、性能与准确率，说明是不是 Mock；
- 未完成的环境兼容性项目，例如 vLLM 模板等价或 100 轮延迟。

审核者需确认“报告中的 PASS”有相应可重复的证据。CI 不允许下载私有数据来替代这个 Gate。

## 六、现有 E00 分支安全整合到 `main`

**这是一个一次性的特殊 PR，不表示以后继续使用超长开发分支。**

### 1. 先检查远端关系和未合并的工作

```bash
git fetch origin --prune
git status --short
git log --oneline origin/main..origin/e00-1-evaluation-infrastructure
git diff --stat origin/main...origin/e00-1-evaluation-infrastructure
git merge-base --is-ancestor origin/main origin/e00-1-evaluation-infrastructure
```

截至文档编写时，E00 分支领先 `main` 26 个提交，`main` 没有额外领先提交。上述关系在正式整合前要重新验证，**不能假设未来仍相同**。另外，先确认队友是否在该分支上还有本地未提交/未推送的工作。

### 2. 推荐落地路线

1. 让 Codex 在明确的协作/CI 任务分支上增加仓库协作规范、PR 模板和 CI 文件；考虑当前 E00 尚未进入 `main`，可以先从 E00 分支开短期 `ci/main-protection`，再将其合并回 E00 分支，或明确将其作为 E00 整合 PR 的组成提交。务必保证 PR 目标最终仍为 `main`，不要误把阶段分支当长期主干。
2. 先开 `e00-1-evaluation-infrastructure → main` 的**整合 PR**。按 D0.3 / D0.4、E00.1、E00.2、E00.2-D、文档、官方协议的顺序提供审查指引；这 40 个文件里不应夹带未归档的真实输出。
3. 在 PR 中运行和修复 CI；由另一位队友审核通过。**保护启用前两位成员口头/文档承诺不直接 push `main`**，防止窗口期绕过流程。
4. 管理员按第四节启用 Ruleset 并验证合并门禁，然后采用 Squash merge 将整个 E00 阶段并入 `main`。
5. 两位成员执行 `git fetch origin --prune`、`git switch main`、`git pull --ff-only origin main`；以后从新 `main` 创建 E01 分支。
6. **若使用 Squash merge**，E00 原来的 26 个提交不会以原来的 SHA 逐个出现在 `main` 历史中；新的 E01 分支应直接基于合并后的 `main`，不要无审查地把旧开发分支再次合并一次。确认旧分支无下游依赖后再删除。

> 管理员可在首次 PR 上观察 CI 后启用 Ruleset；若由于 1 人批准规则导致首次 PR 无法合并，应由真正的另一位成员审核，而不是关闭审批要求。

## 七、两人日常操作清单与故障处理

### 1. 开工 / PR / 合并后

| 时点 | 作者执行 | 审核者执行 |
|---|---|---|
| 开工前 | `git status`、`git fetch`、从最新 `main` 开任务分支 | 明确本次目标与边界 |
| 完成开发 | 相关测试 + `git diff --check` + 精确暂存 + commit/push | 无需提前干预 |
| 发起 PR | 填说明、标注测试/阻断和实验身份 | 审查代码、数据/模型身份、修改边界与风险 |
| CI 失败 | 查看日志定位失败，在原任务分支提交修复 | 检查修复没有顺手改变无关行为 |
| CI 通过 | 等待审查通过，不自行跳过 | 处理讨论并 Approve |
| Squash 合并 | 更新本地 `main`，仅清理已确认无依赖的分支 | 同步 `main` 并更新自己的待开发分支 |

### 2. 常见错误与正确处理

- **`non-fast-forward` / push 被拒绝**：先 `git fetch` 并检查分叉，不使用 `--force`；决定是 rebase 自己的任务分支还是 merge 队友的变更。
- **`main` 直接 push 被拒绝**：表示保护正在发挥作用；把变更转到任务分支，走 PR。不要去临时关闭保护。
- **Required CI 一直 Pending**：核对 Ruleset 所选 Check 名与实际 Job 名；看 Actions 是否触发、工作流路径/条件是否跳过，而不是通过绕过规则合并。
- **PR 经 rebase 后审批失效**：请另一名成员重新审核；这属于预期行为。
- **合并后旧分支显示仍落后 `main`**：Squash 导致的历史差异正常；新任务直接从 `main` 开，不为了“修复分支图”向 `main` 强推。
- **CI 无法访问 `data/processed` 或真实模型**：说明测试分层不合理；为 Required CPU Job 添加小型合成 fixture，真实依赖转至本地 GPU Gate。
- **环境或外部包导致全仓测试失败**：不隐瞒。在 PR 里记录失败用例、命令、环境、可复现情况；只有 CI 范围内的失败修复或有依据调整 CI 后，才允许 Required Check 变绿。

## 八、交给 Codex 的一次性实施任务

### 1. 允许修改的文件

```text
新增：docs/collaboration/Git_双人协作规范.md   # 将本文件的规则部分整理为仓库正式规范
新增：.github/PULL_REQUEST_TEMPLATE.md
新增：.github/workflows/ci.yml
修改：AGENTS.md                             # 只增加正式协作规范的链接与 PR/CI 基本要求
修改：docs/workflow.md                      # 增加简短的日常分支/PR/CI 入口链接
修改：README.md                             # 如有必要，只添加协作规范入口，不替换赛事 README
```

可以将本任务书放入 `docs/task/` 作为阶段性执行记录；**正式长期规则应以 `docs/collaboration/Git_双人协作规范.md` 为唯一权威版本**，避免多份文档互相矛盾。

禁止本任务修改赛事数据、`participant/start.sh`、`participant/run_inference.py`、P0/F2/T1、D0.3、历史实验报告、模型权重或训练代码。不要增加 GitHub Release、CD、Docker 发布、自动模型训练或 GPU Matrix CI。

### 2. 具体执行顺序

1. 记录 `git status --short`、当前分支/HEAD/remote；核对 GitHub `main` 的实际保护和 Ruleset 状态。
2. 新增正式规范、PR 模板、单个 CPU CI 工作流；CI 的依赖与测试选择必须考虑干净 GitHub Runner 缺少真实数据/GPU。运行适用的本地回归并检查 YAML 语法。
3. 验证工作流可被 GitHub 识别；通过正式 PR 测试 CI Job 名称、持续时间及状态。**不能把只在本机运行过的 CI 模板写成 GitHub 已验收**。
4. 由仓库管理员手动建立 Ruleset。Codex 只提供逐项设置说明；没有权限时不得声称已配置成功。
5. 验证 “无审批/红 CI → 禁止合并；经队友 Approve、绿 CI → 允许 Squash” 的门禁；记录 UI 核验结果，E00 历史整合按第六节操作。
6. 精确暂存本任务的文件，提交到任务分支，开 PR；不擅自合并、不删除队友分支、不 force push `main`。

### 3. 本地与 CI 验收标准

```bash
# 代码与脚本门禁（在具备本项目依赖的开发环境中）
python3 -m compileall -q src scripts participant tests
bash -n participant/start.sh
python3 -m pytest -q \
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

**验收通过的定义**：

- 正式规范讲清任务分支、PR 互审、main 保护、rebase/merge/force-with-lease、测试责任和禁入 Git 文件；
- `ci.yml` 在真实的 GitHub 托管 Runner 上通过一次 PR 校验；Required Job 真实存在；
- `main` Ruleset 状态为 **Active**，准确匹配 `main`，启用 PR、1 位他人批准、成功 Status Check、禁止强推/删除；
- 人工核实阻断条件有效，Squash 合并规则清楚；
- E00 26 提交整合路径明确，**未发生未经审核的直接合并/强推**；
- 所有未执行项目（例如 GitHub 管理员设置、GPU 真实训练）明确标注为未完成。

## 九、官方依据

- GitHub Rulesets 使用范围与设置：<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets>
- 创建 Ruleset：<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository>
- Required checks、PR approvals 等规则：<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets>
- 受保护分支与 Check 名约束：<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches>
- Required status check 排错：<https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks>

---

**最终目标**：`main` 成为有规则、有审核、有自动 CPU 门禁的稳定基线；队员各自的实验和训练代码在任务分支内进行。合并至 `main` 时，代码正确性、实验可追溯性和赛事接口约束均有明确责任人及可核验记录。
