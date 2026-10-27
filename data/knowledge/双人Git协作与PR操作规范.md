# 双人 Git 协作与 PR 操作规范

## 共同约定

- `main` 是共同稳定分支；工作通过短期任务分支和 PR 集成。
- 分支按任务命名：`feat/<task>`、`fix/<issue>`、`docs/<task>`、`ci/<task>` 或 `chore/<task>`。一项工作一个分支，不另建长期 `develop` / `dev` 分支。
- 作者负责实现、相关检查、提交和 PR；另一位队友负责审查。作者不以自审代替队友批准。
- 不直接向 `main` 推送，不对 `main` 强推或重写历史；不使用 `git reset --hard` 清理他人改动。
- 不提交密钥、个人配置、模型/数据大文件、生成产物和其他被仓库规则排除的文件。

## 开始新任务

先确认本地没有待处理的改动，再同步远端并从最新 `main` 建分支：

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

保持提交小而清晰；一个提交只表达一个可审查的变化。提交前查看差异、运行相关检查，并只暂存本任务文件：

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

## 同步 `main`

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

在 GitHub 创建 PR，目标分支设为 `main`，来源为当前任务分支。PR 说明应写明：

- 本次目标和改动范围；
- 本地检查的命令和实际结果；
- 尚未验证的内容、失败或环境阻断；
- 可能的影响和需要审查的边界。

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

首次将 CI 设为 Required 前，先让工作流在一个真实 PR 上成功运行，再将 GitHub Checks 中实际出现的 Job 名绑定为 Required Check。不要提前填写不存在的检查名，以免 PR 永久等待。

## 常见故障处理

| 情况 | 处理 |
|---|---|
| 推送提示 non-fast-forward | `git fetch` 后检查双方提交，再决定个人分支 rebase 或协商 merge；不要直接强推。 |
| 推送 `main` 被拒绝 | 转到任务分支并创建 PR，不关闭保护规则。 |
| Required CI 一直 Pending | 核对工作流是否触发，以及 Ruleset 绑定的检查名是否与实际 Job 名完全相同。 |
| rebase 后审批失效 | 请另一位队友重新审查并批准。 |
| CI 失败 | 阅读失败步骤和日志，在原任务分支提交针对性修复，再等待 CI 重新运行。 |
| 合并后旧分支仍落后 `main` | 从最新 `main` 创建后续任务分支；不要为修正旧分支图而重写 `main` 历史。 |
