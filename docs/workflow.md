# Repository Workflow

Run the data stages from the repository root. The reports, configs, source, and tests are versioned; reproducible machine outputs under `outputs/` and normalized datasets under `data/processed/` are local artifacts.

## Current progress and handoff

D0 through E00.2-D have completed their documented data, protocol, and small-sample diagnostic scopes. E00.3 full Dev inference and E01 real QLoRA training have not run; D0.4.3 validates encoded data and the Trainer DataLoader, not a real model update. The current evidence, outstanding decisions, and recommended E01 micro-overfit starting point are in [`docs/experiments/E00_阶段总结与E01交接.md`](experiments/E00_阶段总结与E01交接.md); future experiments remain candidates in [`docs/experiments/后续调优候选与实验决策.md`](experiments/后续调优候选与实验决策.md).

The team Git branch, PR review, merge, and `main` protection process is documented in [`docs/knowledge/双人Git协作与PR操作规范.md`](knowledge/双人Git协作与PR操作规范.md). The CPU CI workflow is `.github/workflows/ci.yml`; use its actual successful GitHub Check name when configuring Required Checks.

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
  --output-dir outputs/evaluation/e00_1/fixtures_v1

python -m src.evaluation.score \
  --config configs/evaluation/e00.yaml \
  --references outputs/evaluation/e00_1/fixtures_v1/references.jsonl \
  --submission outputs/evaluation/e00_1/fixtures_v1/submission.jsonl \
  --raw outputs/evaluation/e00_1/fixtures_v1/raw_predictions.jsonl \
  --output-dir outputs/evaluation/e00_1/fixtures_v1/scored
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
