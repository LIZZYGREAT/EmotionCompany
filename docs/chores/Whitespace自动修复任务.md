# EmotionCompany：Whitespace 自动修复任务

## 一、目标与边界

为仓库增加**本地可手动调用、Agent 可复用、Git 提交前自动执行**的空白字符修复工具，减少 `.github/workflows/ci.yml` 中 `git diff --check` 失败。

- 工具及 Hook 统一放在 `.github/`，不侵入 `src/`、`scripts/`、`participant/` 等业务目录。
- 保留现有 CI 的 `git diff --check`，CI 只负责检查，不在远端自动修改、提交或推送文件。
- **默认只自动修复本次改动的 Markdown 文件**（`.md`、`.markdown`）；其他类型的错误由 CI/检查结果指出，不做可能改变代码或数据语义的批量修复。
- 遵守现有 `AGENTS.md` 和双人 Git/PR 规范：不修改受保护数据或配置，不使用 `git add .`、`git add -A`，不直接操作 `main`。

## 二、实施清单

- [x] **新增 `.github/scripts/fix_whitespace.py`**（Python 标准库，无新依赖）。提供以下模式：
  - `python3 .github/scripts/fix_whitespace.py --changed`：修复工作区本次新增/修改的 Markdown 文件，供 Agent 修改完成后、`git add` 前及用户日常手动运行。
  - `python3 .github/scripts/fix_whitespace.py --staged`：修复已暂存的 Markdown 文件，供提交 Hook 调用；只对实际修改过的文件执行 `git add -- <path>`，不暂存其他文件。
  - `python3 .github/scripts/fix_whitespace.py --file "docs/xxx.md"`：手动修复指定文件。
- [x] **修复范围与安全规则**：清理行尾无意义的空格/Tab、文件结尾多余空行；保证文本文件恰好以一个换行符结束。跳过删除文件、符号链接、非 Markdown 文件和受保护路径。保留 UTF-8/BOM 与 LF/CRLF 风格，避免整文件无关改写；遇到不支持的编码会停止且不修改。
- [x] **防止误改**：如已暂存文件还存在未暂存的修改（包括 `git add -p` 的部分暂存场景），Hook 不得自动覆盖工作区或扩大暂存范围；应报出具体文件并提示先手动处理。遇到有语义的 Markdown 行尾双空格硬换行，不应静默改变渲染效果；要求显式处理后再提交。
- [x] **新增 `.github/hooks/pre-commit`**：提交前执行 `--staged`，随后运行 `git diff --cached --check`；修复成功则正常提交，无法安全修复则阻止提交并给出命令。兼容 Windows Git Bash 与 WSL/Linux。
- [x] **加入一次性启用说明**：每个开发者/Agent 的本地仓库执行 `git config --local core.hooksPath .github/hooks`。启用前检查已有 `core.hooksPath`，不要直接覆盖其他既有 Hook 配置；Hook 需要随仓库提交并具备可执行权限。
- [x] **更新 `AGENTS.md`**：要求 Agent 在修改文件后先执行 `--changed`，再显式 `git add <文件路径>`；`git commit` 触发 Hook。禁止 `--no-verify` 绕过检查。仅补充必要规则，不创建额外 `SKILL.md`。
- [x] **同步协作文档**：在 `docs/knowledge/双人Git协作与PR操作规范.md` 增加手动调用与首次启用命令，避免维护重复的长篇说明；不需要重写现有 CI。

## 三、验证与验收

- [x] 在临时测试文件中构造 `trailing whitespace`、文件末尾多余空行、缺少最终换行符；执行 `--changed` 后修复，重复执行不再产生改动。
- [x] 对已暂存 Markdown 测试 `git commit` 自动修复；确认没有把其他文件或未暂存改动带入提交。针对部分暂存场景验证安全拒绝。
- [x] 验证中文路径、空格文件名、LF/CRLF、正常 Markdown 文档与有意硬换行；不修改无关文件、不破坏正文。
- [x] 运行 `git diff --check` 和 `git diff --cached --check` 均无错误（如仍有非 Markdown 问题，应明确报告、人工修复）。
- [ ] 在任务分支推送并确认现有 `unit-cpu` CI 通过；不调整 `main` 分支保护或 Required Check。

## 四、交付要求

仅提交修复脚本、Hook 及必要的 `AGENTS.md`/协作文档修改；提供一次手动运行、一次自动提交修复、一次安全拒绝的实际验证结果。**不要全仓库格式化，不要顺带修改实验、模型、赛事协议或业务代码。**

## 五、执行记录

- `python3 .github/scripts/fix_whitespace.py --changed` 已在仓库根目录运行；本次没有需要清理的 Markdown 文件，未暂存任何文件。
- 启用前确认本仓库 `core.hooksPath` 未设置后，已通过 `git config --local core.hooksPath .github/hooks` 配置本地 Hook；这项配置仅保存在本机 `.git/config`。
- 临时仓库验证通过：`--file` 支持中文及空格路径，保留 UTF-8 BOM 和 CRLF，清理尾随空格/Tab、末尾空行并补齐最终换行，而且重复执行幂等；`--changed` 清理新增 Markdown 但不暂存；硬换行安全拒绝，符号链接、受保护路径、删除路径和非 Markdown 文件均跳过。
- 临时 Git 仓库的真实 `git commit` 验证通过：Hook 自动修复已暂存 Markdown，提交中没有带入其他未暂存文件；部分暂存时 Hook 阻止提交，HEAD、索引和工作区内容保持不变。
- `python3 -m py_compile .github/scripts/fix_whitespace.py`、`bash -n .github/hooks/pre-commit` 和 `git diff --check` 均通过。推送及 `unit-cpu` CI 结果将在完成后补记。
