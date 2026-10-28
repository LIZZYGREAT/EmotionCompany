# Repository Workflow

Run the data stages from the repository root. The reports, configs, source, and tests are versioned; reproducible machine outputs under `outputs/` and normalized datasets under `data/processed/` are local artifacts.

## Current progress and handoff

D0 through E00.2-D have completed their documented data, protocol, and small-sample diagnostic scopes. E01-A Run `E01-A-20261027-223703` passed its 120-step engineering and reload checks. Its fixed 20-row Dev Smoke completed with verified Base/Adapter identity, but low local metrics: emotion accuracy 0.10 (equal to the Train majority baseline), raw A0 schema validity 0.35, and profile-tag Micro-F1 0.00. This Smoke is diagnostic and does not establish generalization. E01-B training and E00.3 full Dev inference remain unrun; see [`E01-A 训练可学习性验收报告`](experiments/E01-A_训练可学习性验收报告.md) before any budget decision. Earlier E00 evidence and handoff context are in [`docs/experiments/E00_阶段总结与E01交接.md`](experiments/E00_阶段总结与E01交接.md); future experiments remain candidates in [`docs/experiments/后续调优候选与实验决策.md`](experiments/后续调优候选与实验决策.md).

The team Git branch, PR review, merge, and `main` protection process is documented in [`docs/knowledge/双人Git协作与PR操作规范.md`](knowledge/双人Git协作与PR操作规范.md). `.github/workflows/ci.yml` runs `unit-cpu` on every branch push and PR to `main`; configure Required Checks only after confirming the exact successful check name in GitHub.

## Environment setup

```bash
cd ~/projects/EmotionalCompany
source .venv/bin/activate

# Only when creating or repairing the existing CUDA environment:
python -m pip install 'torch==2.10.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python scripts/smoke_inference.py
```

## D0.1 — Raw data quality audit

Run the synthetic audit tests, then audit the official Train and Val inputs:

```bash
python -m pytest tests/test_data_audit.py -q
python -m src.data.audit --config configs/data_audit.yaml
```

Review `docs/data/D0.1_数据质量审计报告.md` and `outputs/data_audit/d0_1/{manifest.json,summary.json,issues.jsonl}`. Confirm the manifest records the source hashes and that the before/after input SHA-256 values match. Do not treat detected data issues as a command failure; use the report to decide the D0.2 policy.

## D0.2 — Data handling policy

D0.2 is a reviewed policy step, not a separate data-processing command. Read the D0.1 report, record each keep/quarantine/deduplicate/review rule in `docs/data/D0.2_数据处理政策.md`, and confirm the policy version and choices in `configs/data_pipeline.yaml`. Keep ambiguous cases visible; do not silently rewrite source records.

## D0.3 — Normalize, deduplicate, and split

Run D0.1 regression tests with the D0.3 pipeline tests, then build the normalized prediction-point datasets:

```bash
python -m pytest tests/test_data_audit.py tests/test_data_pipeline.py -q
python -m src.data.prepare --config configs/data_pipeline.yaml
```

Review `docs/data/D0.3_数据构建报告.md` and `outputs/data_pipeline/v1/manifest.json`. Check the count ledger, split/leakage checks, output hashes, and `input_sha256_unchanged`. The current D0.3 baseline is Train 16,312 / Dev 1,833 / Val 2,354; use the current manifest as the source of truth if a later data version changes those counts.

## D0.4.2 — Exact Token audit and T1 budget simulation

Run the configured command from the repository root:

```bash
python -m src.data.token_stats --config configs/data_token_audit.yaml
```

The command first runs the configured D0.1/D0.3/D0.4 regression suite, then loads the pinned local Qwen3 tokenizer and measures all normalized samples. Review `docs/data/D0.4.2_Token长度审计报告.md` and `outputs/token_audit/v1/manifest.json`. The output directory contains `token_stats.json`, `per_sample_lengths.jsonl`, `budget_simulations.jsonl`, `length_exceptions.jsonl`, and three SVG summaries. Confirm all split counts match the D0.3 manifest, input hashes are unchanged, and each S/P outcome count conserves the total sample count.

D0.4.2 is a measurement stage only. It does not select final S/P/G/K values, truncate completions, rewrite D0.3 data, or start training/D0.4.3. Serving K and final generation budget G remain separate deployment decisions.

### Acceptance revision and inference protocol alignment

The participant runner loads the single P0 source and shared F2/T1 implementation. It pins the Qwen3-1.7B tokenizer revision, tokenizer file hashes, template hash, and EOS ID, then uses the exact template-produced input IDs in Transformers. The API branch sends the same native message roles and explicitly sets `chat_template_kwargs.enable_thinking=false`. The test ID is copied to output only and does not enter the prompt.

Reproduce the acceptance checks and full audit from the repository root:

```bash
sha256sum configs/prompts/p0.txt
python -m pytest tests/test_inference_protocol.py -q
python -m pytest tests -q
python -m src.data.token_stats --config configs/data_token_audit.yaml
bash -n participant/start.sh
```

The audit command runs the configured regression list, including `tests/test_inference_protocol.py`, before rebuilding the 20,499-sample token and budget artifacts. The current run records 50 tests passed and unchanged input hashes. Protocol tests use the pinned real local tokenizer plus Mock Generate/HTTP; they do not prove real model generation or server template equivalence.

Run the participant entry point after providing the model and service configuration:

```bash
bash participant/start.sh TEST_FILE RESULT_DIR
```

Current defaults are provisional: `P=1280` (`local_smoke_candidate_P1280`), `G=1200` reference, `K=null` (`not_verified`). Set `EVALUATION_MAX_PROMPT_TOKENS`, `EVALUATION_PROMPT_BUDGET_PROFILE`, and `EVALUATION_SERVING_CONTEXT_LIMIT` for a named run. If K is configured, startup and runner both reject `P+G>K`; vLLM then receives `--max-model-len K`. Tensor Parallel defaults to 1 and can be set with `VLLM_TENSOR_PARALLEL_SIZE`. The earlier E00.0.1 acceptance check preceded the local model mount and real GPU run. For E00.2, `participant/model` now resolves to the recorded Qwen3-1.7B snapshot, and the real Transformers/NF4/BF16/CUDA run passed the archived identity and device-placement checks. `K` and vLLM server-side prompt/token equivalence remain unverified; see `docs/experiments/E00.2_真实模型Smoke与解码对照报告.md` for the run record.

## D0.4.3 — Completion-only SFT data and Trainer Batch acceptance

Build the named candidate from the repository root:

```bash
python -m pytest tests/test_sft_dataset.py -q
python -m pytest tests -q
python -m src.data.sft_dataset --config configs/sft_data.yaml
```

The build command also runs the two pytest commands configured in `configs/sft_data.yaml` before reading the locked D0.3/D0.4.2 inputs. It then constructs pre-tokenized Train/Dev records with the shared P0/T1/Tokenizer path, checks every sample against the D0.4.2 budget ledger, verifies right-padded Batch sizes 1/2/4, and passes actual candidate rows through the pinned TRL `SFTTrainer` DataLoaders. The tiny GPT-2 structure used for the Trainer check is randomly initialized locally; no model forward, training loop, or weight update runs.

Review `docs/data/D0.4.3_监督数据构建与Batch验收报告.md` and the local `outputs/sft_data/v1/{manifest.json,summary.json,batch_validation.json,excluded_samples.jsonl}`. The pre-tokenized data are local under `data/processed/sft/v1/{train,dev}.jsonl`; Val is cross-checked against D0.4.2 but never emitted as SFT training data. The profile `local_smoke_S2048_P1280` is a data acceptance candidate, not the final E01 S/P/G/K configuration. Do not commit generated JSONL or `outputs/` artifacts.

## E00.1 — Evaluation infrastructure

Install the evaluation-only dependency set in the existing project environment. It does not change the official participant runtime dependencies:

```bash
python -m pip install -r requirements-eval.txt
python -m pip check
```

Build target-free evaluation input from the full D0.3 Dev set (1,833 rows), plus physically separate references and a deterministic 20-row smoke subset:

```bash
python -m pytest tests/test_e00_prepare.py tests/test_e00_metrics.py tests/test_e00_runner.py tests/test_e00_smoke_scoring.py tests/test_competition_contract.py -q
python -m src.evaluation.prepare_dev --config configs/evaluation/e00.yaml
```

The default output is `outputs/evaluation/e00_1/dev_adapter_v2/`. `inference_input.jsonl` contains only `id` and `history`; `references.jsonl` contains `target`; `id_map.jsonl` links the local evaluation IDs to D0.3 `sample_id`s. The generated IDs such as `test_000001_t0` are local evaluation IDs, not official hidden-test IDs. The builder verifies the D0.3 Dev hash/count, D0.3 Manifest, official target Schema, unique IDs, source provenance and output hashes. Its adapter fingerprint covers the source contract, ID rule, adapter version and Smoke seed/count; changing generation, quantization or scoring options does not change the dataset identity. Existing output directories are not overwritten when the source data or adapter conditions differ. Missing D0.3 files or manifests are errors; no substitute data is generated.

Run the offline synthetic acceptance and score its artifacts independently:

```bash
python scripts/e00_mock_acceptance.py \
  --config configs/evaluation/e00.yaml \
  --output-dir outputs/evaluation/e00_1/fixtures_v2

python -m src.evaluation.score \
  --config configs/evaluation/e00.yaml \
  --references outputs/evaluation/e00_1/fixtures_v2/references.jsonl \
  --submission outputs/evaluation/e00_1/fixtures_v2/submission.jsonl \
  --raw outputs/evaluation/e00_1/fixtures_v2/raw_predictions.jsonl \
  --output-dir outputs/evaluation/e00_1/fixtures_v2/scored
```

This fixture uses fabricated rows and the existing participant parser/schema; it is labeled `synthetic_fixture` / `mock` and is not an E00 model result. Scoring checks complete ID-set equality, duplicate IDs, evaluation-manifest hashes, input hash, run ID, target-free input structure and the official submission Schema before computing metrics. The scorer writes `metrics.json`, `report.md` and ID-only `case_errors.jsonl`. Outputs and raw conversations stay under ignored `outputs/`.

To exercise scoring with the real D0.3 Smoke IDs without loading a model, generate explicit Mock predictions using only the target-free Smoke input (the helper never reads References):

```bash
RUN_ID=E00.1-mock-smoke-20261026
python scripts/e00_mock_smoke.py \
  --input outputs/evaluation/e00_1/dev_adapter_v2/smoke_input.jsonl \
  --output-dir "outputs/evaluation/e00_1/mock_smoke_v2/$RUN_ID" \
  --run-id "$RUN_ID"

python -m src.evaluation.score \
  --config configs/evaluation/e00.yaml \
  --references outputs/evaluation/e00_1/dev_adapter_v2/smoke_references.jsonl \
  --submission "outputs/evaluation/e00_1/mock_smoke_v2/$RUN_ID/submission.jsonl" \
  --raw "outputs/evaluation/e00_1/mock_smoke_v2/$RUN_ID/raw_predictions.jsonl" \
  --output-dir "outputs/evaluation/e00_1/mock_smoke_v2/$RUN_ID/scored"
```

The output report is marked `execution_mode=mock_fixture`; its scores validate only the alignment and scoring pipeline, not a model. Use a new run ID/output directory for each execution.

## E00.2 — Real model Smoke and decoding comparison

E00.2 uses the original `Qwen/Qwen3-1.7B` weights through the ignored local mount `participant/model`. It must resolve to a complete, locally cached snapshot containing `config.json`, `model.safetensors.index.json`, and every non-empty indexed shard. Do not copy weights into the repository or let the runner download them. The inference runner requires CUDA, loaded `bitsandbytes.nn.Linear4bit` modules using NF4/BF16, and no CPU/disk device-map offload before generation.

The versioned profile is `configs/evaluation/e00_2_profiles.yaml`. Render-only mode creates independent Greedy and Sampling runner JSON/evaluation YAML snapshots under an ignored run directory, records both config SHA-256 values, and refuses to overwrite an existing run ID:

```bash
python scripts/e00_2_smoke.py --run-id "E00.2-config-$(date +%Y%m%d-%H%M%S)" --render-only
```

After activating the local CUDA `.venv` and providing the verified model mount, execute a fresh run with this command. It renders snapshots, runs a one-row real NF4 preflight, then runs and scores both independent 20-row profiles against the unchanged E00.1 Smoke references. CUDA, NF4, identity, ID-alignment, or scorer failure stops the run and is recorded in the local `run_manifest.json`:

```bash
cd ~/projects/EmotionalCompany
source .venv/bin/activate
python scripts/e00_2_smoke.py --run-id "E00.2-$(date +%Y%m%d-%H%M%S)" --execute
```

Greedy uses `do_sample=false`; Sampling uses `do_sample=true, temperature=0.7, top_p=0.8, top_k=20` with base seed 42 and a stable per-ID seed. Runner reports include effective runner/evaluation config hashes, model shard identity, actual NF4 module/device-map evidence, model-load time, CUDA allocator usage, cold first-sample timing, and per-sample diagnostics. Raw model text and per-case comparison files stay under ignored `outputs/evaluation/e00_2/`; the tracked report contains no conversation text. Twenty Smoke rows are exploratory and cannot establish a generally better decoding strategy. The E00.2 report records the original decision not to proceed directly to E00.3; full Dev timing remains an open decision, not an automatic next run. Current evidence and E01 handoff are summarized in `docs/experiments/E00_阶段总结与E01交接.md`.

After a completed real Smoke, run the offline diagnosis against its immutable artifacts:

```bash
python scripts/e00_2_diagnose.py \
  --run-root outputs/evaluation/e00_2/E00.2-20261026-220635 \
  --output-dir outputs/evaluation/e00_2/diagnostics/E00.2-20261026-220635
```

Use a new output directory for each run; the CLI refuses to overwrite an existing one. It verifies the Manifest and source hashes, joins Raw/Submission/Performance/References by ID, reconciles historical Metrics, checks local model-mount metadata without rehashing multi-GB shards, and compares Smoke truncation with D0.4.2. It reads existing artifacts only, does not load a model or modify inference behavior, and writes its detailed outputs under ignored `outputs/`. The tracked analysis is `docs/experiments/E00.2_错误归因与后续决策.md`; it contains field counts and conclusions, not Raw completions or conversation text.

## E00.2-D — Single-field emotion diagnosis

This is a separate 20-row diagnostic, not an A0 result or a full Dev evaluation. It uses the independent `Pdiag-E1` prompt and the exact history suffix selected by the archived A0 Greedy P0/T1 run. The preflight checks every row against A0 `performance_report.json` and the D0.4.2 `S=2048/P=1280` ledger, then verifies the new prompt stays within P=1280. The `run` command accepts only `id` and `history`; it never reads References. The `score` command reads References, applies strict single-label matching, counts invalid outputs in the 20-row denominator, and compares the same IDs with A0 Greedy.

Run the focused checks before using a CUDA host:

```bash
python -m pip check
python -m pytest tests/test_e00_2_emotion_diag.py -q
python -m pytest tests/test_e00_2_runtime.py tests/test_e00_2_profiles.py tests/test_e00_2_diagnose.py tests/test_inference_protocol.py tests/test_e00_metrics.py -q
bash -n participant/start.sh
```

On the local WSL machine with CUDA available and the verified model mount, create one fresh run and score it only after the run Manifest reports `real_run_completed`:

```bash
set -e
RUN_ID="E00.2-D-emotion-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="outputs/evaluation/e00_2/emotion_diag/$RUN_ID"
python scripts/e00_2_emotion_diag.py run \
  --config configs/evaluation/e00_2_emotion_diag.yaml \
  --run-id "$RUN_ID" \
  --output-dir "$RUN_DIR"

python - "$RUN_DIR/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("status") != "real_run_completed":
    raise SystemExit(f"score skipped: run status is {manifest.get('status')}")
PY

python scripts/e00_2_emotion_diag.py score --run-dir "$RUN_DIR"
```

The runner requires actual NF4/BF16/CUDA placement with no CPU/disk offload and the archived weight identity; it does not fall back to CPU. If CUDA is unavailable, it writes a `blocked_in_codex_env` Manifest and does not create predictions. A blocked run ID is immutable; use a new one on a later attempt. Raw predictions and per-case results stay under ignored `outputs/`; only the concise report is tracked at `docs/experiments/E00.2-D_单字段情绪诊断报告.md`. The completed run `E00.2-D-emotion-003` scored 15/20 strict legal labels and 0/20 correct; its weight SHA matched the archived identity, and the report records that the configured revision itself remains unverified by the Manifest. See the experiment report for full metrics. The 20-row result is descriptive only because the prompt and output contract changed together, so it cannot establish causality or replace the A0 baseline.

## E01-A — QLoRA engineering acceptance

The implementation uses `configs/training/e01_a.yaml` for the independent one-step proof and 8-row Micro-overfit. `configs/training/e01_b.yaml` remains a Full B draft; its training gate requires an independently approved config and an explicit launch flag. The separate Pilot and Full B commands are documented below. All baseline configs require empty `init_adapter` and `resume_from_checkpoint` fields; same-Run checkpoint resume is selected explicitly on the CLI.

Run CPU/synthetic regression first, then create a new, immutable preflight Run ID:

```bash
python -m pytest tests/test_e01_training.py tests/test_inference_protocol.py tests/test_e00_metrics.py -q
python -m src.training.training_sft \
  --config configs/training/e01_a.yaml \
  --mode preflight \
  --run-id "E01-A-preflight-$(date +%Y%m%d-%H%M%S)"
```

The preflight checks the D0.3/D0.4.3 manifests and hashes, SFT counts and Labels, P0/Tokenizer fingerprints, local Base aggregate SHA, CUDA/BF16 visibility, and the deterministic Train-only sample selection. It writes `outputs/training/<run-id>/preflight.json` and `manifest.json`. If CUDA/BF16 is unavailable, the status is explicitly blocked and no GPU work should start. Preflight may return exit code 0 for a blocked status, so inspect the JSON and proceed only when `status` is `passed`.

After a fresh preflight passes, prove one real TRL optimizer update using an independent Run ID. Only a passing `step_proof.json` can gate Micro-overfit:

```bash
STEP_RUN_ID="E01-STEP-$(date +%Y%m%d-%H%M%S)"
STEP_PROOF="outputs/training/$STEP_RUN_ID/step_proof.json"
A_RUN_ID="E01-A-$(date +%Y%m%d-%H%M%S)"
if python -m src.training.training_sft \
  --config configs/training/e01_a.yaml \
  --mode one-step \
  --run-id "$STEP_RUN_ID"; then
  if [ -s "$STEP_PROOF" ] && python -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "passed" else 1)' "$STEP_PROOF"; then
    python -m src.training.training_sft \
      --config configs/training/e01_a.yaml \
      --mode train \
      --run-id "$A_RUN_ID" \
      --one-step-proof "$STEP_PROOF"
  else
    echo "one-step proof missing or not passed; train was not started" >&2
    false
  fi
else
  echo "one-step failed; train was not started" >&2
  false
fi
```

The train mode starts a new Base + LoRA initialization, records optimizer-step losses/gradients/parameter updates, and runs Train-fitting diagnostics without placing Gold in generation prompts. It saves only this Run's Adapter and Adapter snapshots. A fixed Greedy probe is generated before release and after Base + Adapter reload; both Token IDs and Base SHA are checked. `train_log.jsonl` records each optimizer step, `progress.json` is atomically updated after each step, and Adapter snapshots are saved at the evaluation interval. Once training and Adapter save complete, `training_evidence.json` and the running Manifest are written before releasing the GPU model; reload evidence is then recorded before the Manifest is finalized. These artifacts preserve progress and weights if later finalization fails. Full optimizer/scheduler-state checkpointing and exact resume are not enabled, so a mid-run interruption cannot currently restore Adam state. Run train only after `one-step` exits successfully and its `step_proof.json` has `status: passed`; an interactive shell can continue to the next pasted command after a failed command. Afterward run the separate reload verifier and fixed 20-row E00.1 Smoke scorer:

```bash
python -m src.training.training_sft \
  --config configs/training/e01_a.yaml \
  --mode reload-check \
  --run-root outputs/training/E01-A-<run-id>

python scripts/e01_evaluate.py \
  --run-root outputs/training/E01-A-<run-id> \
  --scope train-and-dev-smoke \
  --attempt-id "dev-smoke-$(date +%Y%m%d-%H%M%S)"
```

The evaluation attempt writes to `outputs/training/E01-A-<run-id>/evaluation/dev_smoke/<attempt-id>/`. It records a `running` `run_manifest.json` before launching inference and updates it with inference/scoring status. A new attempt ID gives each retry an independent output directory; the CLI also generates a unique ID if `--attempt-id` is omitted. Raw training generations, predictions, runner reports, and scores stay under ignored `outputs/training/`. Do not commit those artifacts or use E01-A training fit as Dev generalization evidence. Review [`E01-A 训练可学习性验收报告`](experiments/E01-A_训练可学习性验收报告.md) before setting E01-B's Epoch/Steps, resource budget, and explicit start decision.

## E01-B — 分阶段训练与公平 Dev 评测

实现次序为「训练工程完善 → Pilot 及恢复验收 → 人工冻结 Full B 预算 → 独立 Full B → 单独许可的 Base/Adapter Dev 推理 → 离线配对比较」。[训练工程完善方案](task/E0.1-B_训练工程完善任务方案.md)是[分阶段训练与公平评测方案](task/E0.1-B_分阶段训练与公平评测执行方案.md)的前置工程；完成工程不依赖 Full B 的结果，也不需要改写两份任务方案。实验验收报告待真实产物审阅后填写；本节提供实际程序和运行合同。

### 1. 环境与工程回归

在仓库根目录、同一个 Bash 会话中分段执行。每次实验使用新 Run ID；恢复时显式指定原 Run 和真实 Checkpoint，不搜索「最新 Adapter」。

```bash
source .venv/bin/activate
set -euo pipefail
python -m pip check
python -m pytest tests/test_e01_b_training.py tests/test_e01_b_evaluate.py tests/test_inference_resume.py tests/test_e01_training.py tests/test_e01_evaluate.py -q
```

CPU 回归包含本地随机小模型的真实 Trainer 暂停/恢复，对照连续训练的 LoRA 参数、LR、数据顺序和监督 Token 计数；这些是工程机制的证据。赛事模型的 NF4/BF16/CUDA、显存和质量证据来自下面实际运行的产物。

### 2. Pilot：512 条 Train，先暂停再恢复

```bash
E01_PILOT_RUN_ID="E01-B-PILOT-$(date +%Y%m%d-%H%M%S)"
bash scripts/e01_b_pilot.sh "$E01_PILOT_RUN_ID"
```

该程序只执行 Pilot：独立 preflight → 新 Base + 新 LoRA → 32 步保存完整 Checkpoint 并暂停 → 从同一个 Run 的 `trainer/checkpoint-32` 恢复到 128 步 → 保存 Adapter、释放/重载核对 → 打印资源审阅摘要。每个环节失败立即停止，不解锁或启动 Full B。Pilot 配置集中于 `configs/training/e01_b_pilot.yaml`：512 条样本覆盖长度十分位、全部 Train 情绪及空/非空画像，包含最长序列；训练批次按 Seed 随机打乱。SFT Dev Loss 使用独立、固定的 256 条 Dev，Micro Batch / Eval Batch 均为 1，累计 4；Trainer 时间预算为两小时，含训练中的验证和 Checkpoint，按 Step 边界检查。

主要产物位于 `outputs/training/$E01_PILOT_RUN_ID/`：`manifest.json`、`selected_samples.json`、`train_log.jsonl`、`eval_log.jsonl`、`progress.json`、`trainer/checkpoint-*/checkpoint_manifest.json`、`training_evidence.json`、`adapter/` 和 `reload_check.json`。Checkpoint 同时保存 Adapter、Optimizer、Scheduler、Trainer State 和 RNG；保留最近两个 Checkpoint。仅首步、固定间隔及末步比较全部 LoRA 参数，常规 Step 不复制全部参数到 CPU。

可另行复核最终 Adapter 重载，并重新打印审阅摘要：

```bash
python -m src.training.training_sft \
  --config configs/training/e01_b_pilot.yaml --mode reload-check \
  --run-root "outputs/training/$E01_PILOT_RUN_ID"
python scripts/e01_b_config.py review \
  --pilot-run-root "outputs/training/$E01_PILOT_RUN_ID"
```

`review` 只读产物，打印实测长序列范围、Loss、Token 吞吐、显存、Checkpoint 占用和磁盘余量。其 Full B 时长按实测 Token 吞吐估算，不能消除长序列计算成本、全量 Dev Loss 与保存间隔带来的误差，也不构成批准。

### 3. 可选的 Pilot Dev 160 条质量诊断

先用原 E00 适配器构建独立的 160 条 Smoke，固定 Seed 和样本 ID，覆盖情绪、画像和历史长度特征。完整 D0.3 Dev 身份仍为 1,833 条；160 条只作为 Pilot 诊断，不用于正式泛化结论。推理子进程只收到 `id+history` 文件；References 由后续评分读取。

```bash
python -m src.evaluation.prepare_dev --config configs/evaluation/e01_b_pilot.yaml
E01_PILOT_BASE_ID="E01-B-PILOT-BASE-$(date +%Y%m%d-%H%M%S)"
E01_PILOT_EVAL_ID="E01-B-PILOT-EVAL-$(date +%Y%m%d-%H%M%S)"
python scripts/e01_b_evaluate.py --scope pilot-dev --model base \
  --run-id "$E01_PILOT_BASE_ID" --hardware-label local-rtx4060-laptop-8gb
python scripts/e01_b_evaluate.py --scope pilot-dev --model adapter \
  --training-run-root "outputs/training/$E01_PILOT_RUN_ID" \
  --run-id "$E01_PILOT_EVAL_ID" --hardware-label local-rtx4060-laptop-8gb
python scripts/e01_b_evaluate.py --scope compare \
  --base-run-root "outputs/evaluation/e01_b/pilot/$E01_PILOT_BASE_ID" \
  --adapter-run-root "outputs/evaluation/e01_b/pilot/$E01_PILOT_EVAL_ID" \
  --output-dir "outputs/evaluation/e01_b/comparison/pilot-$(date +%Y%m%d-%H%M%S)"
```

### 4. 审阅 Pilot 后显式批准 Full B 配置

只有审阅真实 Pilot 长序列、显存、恢复结果、时长估算和磁盘后再执行这一段。下面学习率 `1e-4`、1 Epoch、每 256 步保存/验证是候选示例；操作人应根据 Pilot 明确选择预算。脚本要求操作人名字、时间预算及 `--approve-full-training`，生成新的批准快照，记录日期、范围、Pilot Manifest SHA 和配置内容 SHA；不会启动训练。仓库中的 `configs/training/e01_b.yaml` 继续保持 `draft_manual_gate`。

```bash
read -r -p 'Full B 批准人：' E01_APPROVED_BY
read -r -p '根据 Pilot 批准的 Full B Trainer 总时间上限（小时）：' E01_FULL_GPU_HOURS
E01_FULL_CONFIG="outputs/training/configs/E01-B-approved-$(date +%Y%m%d-%H%M%S).yaml"
python scripts/e01_b_config.py approve \
  --pilot-run-root "outputs/training/$E01_PILOT_RUN_ID" \
  --output-config "$E01_FULL_CONFIG" --approved-by "$E01_APPROVED_BY" \
  --learning-rate 0.0001 --epochs 1 --save-steps 256 --eval-steps 256 \
  --max-training-hours "$E01_FULL_GPU_HOURS" --approve-full-training
```

Full B 同时检查 `approved_for_full_training`、批准配置 Hash、已审阅 Pilot 的身份和 `--allow-full-training`。修改批准快照的预算会使批准 Hash 失效；不得静默改变现有 Run 的配置。Full B 只用 Pilot 的审阅记录，不读取其 Adapter 或 Optimizer。16,192 条 Train、Batch 1、累计 4、1 Epoch 的候选为 4,048 个 Optimizer Steps，启动时再次与实际 DataLoader 调度核对。

### 5. Full B 独立训练

```bash
E01_FULL_RUN_ID="E01-B-FULL-$(date +%Y%m%d-%H%M%S)"
E01_FULL_PREFLIGHT_ID="${E01_FULL_RUN_ID}-preflight"
python -m src.training.training_sft --config "$E01_FULL_CONFIG" \
  --mode preflight --run-id "$E01_FULL_PREFLIGHT_ID"
python -m src.training.training_sft --config "$E01_FULL_CONFIG" \
  --mode train --run-id "$E01_FULL_RUN_ID" \
  --preflight-run-root "outputs/training/$E01_FULL_PREFLIGHT_ID" \
  --allow-full-training
python -m src.training.training_sft --config "$E01_FULL_CONFIG" \
  --mode reload-check --run-root "outputs/training/$E01_FULL_RUN_ID" \
  --allow-full-training
```

训练 Manifest 的 `training_scope=full_train`，全部 16,192 条有效 Train 更新 LoRA；1,821 条 SFT Dev 只用于 Teacher-forced Loss。最终 Adapter 与基础文件指纹、数据与协议、LR、实际样本/Token 数、Checkpoint 和重载结果分别保存。生成评测不会随训练成功自动启动。

如果训练中断，仅从原 Run 的明确 Checkpoint 恢复；不能给已完成 Run 续训。选择实际存在、完整且属于该 Run 的最后一个 Checkpoint，替换下方输入步骤。中断后超出 Checkpoint 的日志会先保存至 `interrupted_attempts/`，再恢复标准日志的 Step 连续性；不会重复计入丢失的更新。冻结的累计 Trainer 时间预算继续生效。

```bash
read -r -p '原 Full B Run 中实际 Checkpoint 的 Step：' E01_CHECKPOINT_STEP
python -m src.training.training_sft --config "$E01_FULL_CONFIG" \
  --mode resume --run-root "outputs/training/$E01_FULL_RUN_ID" \
  --checkpoint "outputs/training/$E01_FULL_RUN_ID/trainer/checkpoint-$E01_CHECKPOINT_STEP" \
  --allow-full-training
```

### 6. Full Dev Base 与 B 分别执行，再离线配对

Full B 完成且重载通过后，单独决定这两次 GPU 推理的资源许可。完整输入复用 E00.1 的 1,833 条无标签 Dev；如果本机尚未生成，则先执行 `python -m src.evaluation.prepare_dev --config configs/evaluation/e00.yaml`。两组均使用同一 Base/Tokenizer、P0/F2/T1、P=1280、G=1200、Greedy、NF4/BF16 和评分器，只改变 Adapter 是否启用。

```bash
E01_BASE_DEV_ID="E00.3-BASE-$(date +%Y%m%d-%H%M%S)"
E01_ADAPTER_DEV_ID="E01-B-EVAL-$(date +%Y%m%d-%H%M%S)"
python scripts/e01_b_evaluate.py --scope full-dev --model base \
  --run-id "$E01_BASE_DEV_ID" --hardware-label local-rtx4060-laptop-8gb \
  --allow-full-dev
python scripts/e01_b_evaluate.py --scope full-dev --model adapter \
  --training-run-root "outputs/training/$E01_FULL_RUN_ID" \
  --run-id "$E01_ADAPTER_DEV_ID" --hardware-label local-rtx4060-laptop-8gb \
  --allow-full-dev
python scripts/e01_b_evaluate.py --scope compare \
  --base-run-root "outputs/evaluation/e01_b/full/$E01_BASE_DEV_ID" \
  --adapter-run-root "outputs/evaluation/e01_b/full/$E01_ADAPTER_DEV_ID" \
  --output-dir "outputs/evaluation/e01_b/comparison/full-$(date +%Y%m%d-%H%M%S)"
```

每次生成前持久化 `run_manifest.json`，随后逐条提交 Raw/Submission 和带内容 Hash 的 `runner/run_progress.json`，同时在终端及 `runner.log` 输出进度。生成异常会停止；已经提交的前缀保留。发生中断时，使用完全相同的原命令追加 `--resume`（Full Dev 仍需 `--allow-full-dev`）；配置、代码/依赖、数据、Base/Adapter 身份必须一致。恢复会截去上次未提交的尾行，再从下一个 ID 生成。已评分 Run 不允许覆盖；新实验或重新开始用新 Run ID。`--render-only` 可只冻结并查看配置，再用同一命令追加 `--resume` 启动。

`compare` 只读取既有预测并复用 E00 Scorer 重新核验与评分，不训练、不生成。它要求两组完整 ID/Reference Hash、协议、解码、硬件标签、代码/依赖一致，拒绝 Pilot/Full 混比和部分完成结果。产物 `comparison_summary.json`、`report.md`、`comparison_cases.jsonl` 汇总情绪逐类指标/混淆矩阵、画像 33 标签及逐维/非空 Gold Recall、Raw 严格 JSON/A0 Schema 与错误分布、Submission Schema、BLEU-4/ROUGE-L、Token/延迟/显存及配对迁移；情绪差值附固定 Seed 的会话簇配对 Bootstrap 区间。逐 ID 表只含 ID 和诊断统计。

BERTScore 按冻结配置为 `not_run`；官方性能口径为 `not_run`。恢复后的计时包含多个模型加载后的首条冷启动，不能作为单次连续官方性能结果。空画像会抬高 Exact Match，必须结合非空 Gold Recall 与标签 Micro-F1 解读。所有权重、Raw、预测和报告产物仍在忽略目录；真实结果审阅后再填写两份方案要求的版本化验收报告。

## Regression suite

Run the complete current data and prompt/token audit suite with:

```bash
python -m pytest \
  tests/test_data_audit.py \
  tests/test_data_pipeline.py \
  tests/test_prompting.py \
  tests/test_token_stats.py \
  tests/test_inference_protocol.py -q
```

Or run the whole current test tree with `python -m pytest tests -q`.

## Commit checkpoints

Commit each completed stage after its tests and artifact checks. Include the tracked implementation/config/tests and the corresponding concise report or workflow update. Do not add `outputs/`, `data/processed/`, model files, or task-plan documents unless a task explicitly asks to version them.
