# CI 测试覆盖清单

| 层级 | 执行位置 | 覆盖内容 | 不证明什么 |
|---|---|---|---|
| Syntax / whitespace | .github/workflows/ci.yml 的 unit-cpu | compileall、Shell 语法、推送差异空白检查 | 不覆盖模型运行、训练或 CUDA |
| 轻依赖 CPU 合成测试 | required unit-cpu | 显式列出的协议、数据工具、A3 独立纯 Python 诊断、E03 画像前缀与 A4 历史画像对齐、会话簇稳定性指标、候选评测配置/CLI 门禁，以及输入身份、会话切分和资源账本测试 | 不包含真实数据逐行审计、A0/A0.1/A1 或 A4 训练合同测试（导入链涉及 Torch），也不覆盖 GPU |
| Torch CPU 合成测试 | 本地 .venv 门禁；不属于 unit-cpu | A0/A0.1/A1、A3 短程训练路径、A4 训练配置/采样/预编码/优化器合同（`tests/test_e03_a4_stability.py`）和其他依赖 Torch 的合成回归；按任务执行方案运行聚焦测试 | 未执行的测试必须记为 not_run；不能由 unit-cpu 成功推断 |
| 真实模型 GPU 验收 | 本地受控命令与 Run Manifest | 真实 Token 前缀一致性、训练/推理设备、显存与耗时 | GitHub CPU CI 永远不能替代 |

更新 unit-cpu 测试清单时，保留 Job 名称 unit-cpu，不要把 Torch/CUDA 或本机模型资产装入轻依赖 Job。A4 的 `test_e03_a4_profile_reconcile.py`、`test_e03_a4_stability_metrics.py`、`test_e03_a4_stability_eval.py` 已进入显式 CI 清单；`test_e03_a4_stability.py` 保留在本地 Torch 环境。CI 是否通过以 GitHub Actions 实际检查为准，本地 pytest 结果单独记录。
