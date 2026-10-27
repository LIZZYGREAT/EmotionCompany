# 双人 Git 协作与 PR 操作规范

## 共同约定

- `main` 是共同稳定分支；工作通过短期任务分支和 PR 集成。
- 分支按任务命名：`feat/<task>`、`fix/<issue>`、`docs/<task>`、`ci/<task>` 或 `chore/<task>`。一项工作一个分支，不另建长期 `develop` / `dev` 分支。
- 作者负责实现、相关检查、提交和 PR；另一位队友负责审查。作者不以自审代替队友批准。
- 不直接向 `main` 推送，不对 `main` 强推或重写历史；不使用 `git reset --hard` 清理他人改动。
- 不提交密钥、个人配置、模型/数据大文件、生成产物和其他被仓库规则排除的文件。

## 开始新任务

先确认本地没有待处理的改动，再同步远端并从最新 `main` 建分支(根据实际情况来，不一定是main分支)：

```bash
git status --short
git fetch origin --prune
git switch main
git pull --ff-only origin main
git switch -c feat/example-task
```

把 `example-task` 换成简短、能说明本次工作的分支名。

如果工作区不干净，先确认改动归属并妥善提交或移交，再开始切分支。不要丢弃或覆盖不清楚归属的文件。

## 开发与提交

保持提交小而清晰；一个提交只表达一个可审查的变化。修改 Markdown 后、暂存前，从仓库根目录手动运行空白修复脚本：

```bash
python3 .github/scripts/fix_whitespace.py --changed
git diff --check
```

如只需处理一个文件，可用 `python3 .github/scripts/fix_whitespace.py --file "docs/xxx.md"`。脚本只改本次变更的 Markdown；遇到 Markdown 双空格硬换行或部分暂存文件时会停止，需先人工处理。

提交前查看差异、运行相关检查，并只暂存本任务文件：

```bash
git diff --check
git status --short
git add path/to/changed_file
git diff --cached --stat
git diff --cached
git status --short
git commit -m "feat(scope): summary"
```

不要无检查地执行 `git add .` 或 `git add -A`。提交信息使用简洁类型前缀，例如 `feat(...)`、`fix(...)`、`docs(...)`、`test(...)`、`chore(...)`。

### 每个本地仓库启用 pre-commit Hook（一次）

先检查当前仓库是否已有自定义 Hook 路径：

```bash
git config --local --get core.hooksPath
```

只有该命令没有输出且返回非零时，才执行：

```bash
git config --local core.hooksPath .github/hooks
```

如果已经有值，先确认原配置用途，不要直接覆盖。启用后，`git commit` 会自动执行暂存 Markdown 修复及 `git diff --cached --check`；部分暂存、硬换行等无法安全自动修复时会阻止提交并提示处理方式。

## 同步 `main`/其他开发分支

个人维护、未交接给队友的任务分支，默认用 rebase 跟进最新 `main`：

```bash
git fetch origin
git switch task-branch-name
git status --short
git rebase origin/main
```

共同维护的分支，或已有其他分支依赖的分支，先与队友沟通，再选择 `git merge origin/main`，避免改写对方提交。

如果 rebase 冲突，先逐项解决并运行相关检查，再执行 `git rebase --continue`；无法确认处理方式时执行 `git rebase --abort`，回到 rebase 前状态。

只有在确认分支由自己独立维护、且刚刚有意 rebase 已推送的分支时，才可执行：

```bash
git push --force-with-lease origin task-branch-name
```

不使用裸 `--force`；任何情况下都不对 `main` 使用强推。

## 推送并发起 PR

```bash
git push -u origin task-branch-name
```

在 GitHub 创建 PR，目标分支设为 `main`/其他目标的分支，来源为当前任务分支。PR 说明应写明：

- 本次目标和改动范围；
- 本地检查的命令和实际结果；
- 尚未验证的内容、失败或环境阻断；
- 可能的影响和需要审查的边界。

可按仓库模板 `.github/PULL_REQUEST_TEMPLATE.md` 填写，避免遗漏验证和风险信息。

## 首次 CI 与 `main` 保护验收

1. 从最新 `main` 创建任务分支并推送。`.github/workflows/ci.yml` 会在每次分支 push 后运行 `unit-cpu`；发起目标为 `main` 的 PR 时也会运行该检查。每次 push 的运行会各自保留，不会因后续 push 而取消。
2. 等待 GitHub Actions 中该 PR 的 `unit-cpu` 检查完成。先在真实 PR 上确认工作流成功，并记下 GitHub Checks 显示的准确检查名；本地 pytest 通过不能替代这一步。
3. 管理员再为 `main` 启用 Ruleset：要求 PR、至少 1 位队友批准、上述准确名称的 CI 检查成功，并禁止删除和强推。审批后有新提交时要求重新审批，合并使用 Squash。
4. 确认规则确实阻止未获批准或 CI 未通过的 PR，并允许已获队友批准且 CI 通过的 PR 合并。不要为了完成验收而合并未经队友审查的 PR。

`unit-cpu` 只验证 CPU 合成回归和代码检查；它不验证 GPU、原始数据、模型权重或真实训练/推理环境。

审查者应实际检查差异、行为和风险，指出需要修改的内容并确认已解决。新的提交使旧审批失效时，由另一位队友重新审查。

## 合并与收尾

PR 合并前应满足：

1. 另一位队友已批准；
2. Required CI 检查已通过；
3. 所有审查讨论均已处理；
4. PR 中没有误入的无关文件或生成产物。

使用 **Squash and merge** 合并。合并后同步本地 `main`：

```bash
git fetch origin --prune
git switch main
git pull --ff-only origin main
```

确认任务分支没有下游依赖后，才删除本地或远端分支。Squash 合并后旧任务分支的提交图可能与 `main` 不同；新任务从最新 `main` 创建，不把已合并的旧分支再次合入。

## `main` 合并门禁

仓库管理员在 GitHub 为 `main` 配置一条 Active Ruleset，至少要求 PR、1 位队友批准、Required CI 通过，并禁止删除和强推。若审批后又有新提交，应要求重新审批；合并方式统一使用 Squash。

首次将 CI 设为 Required 前，先让工作流在一个真实 PR 上成功运行，再将 GitHub Checks 中实际出现的 Job 名绑定为 Required Check。本仓库工作流的预期稳定 Job 名为 `unit-cpu`；确认 GitHub Checks 中确实出现后再绑定。不要提前填写不存在的检查名，以免 PR 永久等待。
