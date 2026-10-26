# Repository Workflow

Run the data stages from the repository root. The reports, configs, source, and tests are versioned; reproducible machine outputs under `outputs/` and normalized datasets under `data/processed/` are local artifacts.

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

Current defaults are provisional: `P=1280` (`local_smoke_candidate_P1280`), `G=1200` reference, `K=null` (`not_verified`). Set `EVALUATION_MAX_PROMPT_TOKENS`, `EVALUATION_PROMPT_BUDGET_PROFILE`, and `EVALUATION_SERVING_CONTEXT_LIMIT` for a named run. If K is configured, startup and runner both reject `P+G>K`; vLLM then receives `--max-model-len K`. Tensor Parallel defaults to 1 and can be set with `VLLM_TENSOR_PARALLEL_SIZE`. In this workspace there is no `participant/model` directory or `vllm` executable, so real Transformers generation, server prompt/token equivalence, K, and hardware feasibility remain unverified.

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

For a later E00.2 real-model smoke, use a fresh run directory and a new raw output path for every run. Raw logging is opt-in and refuses to overwrite an existing file. NF4 is also opt-in; the E00.1 implementation does not load Qwen or verify GPU execution:

First inspect the local cache. Tokenizer cache files do not prove model weights are complete:

```bash
ls -ld participant/model 2>/dev/null || true
ls -lah ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/ 2>/dev/null || true
find ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B \\
  -name '*.safetensors' -o -name config.json 2>/dev/null
```

Only after confirming an actual snapshot directory containing `config.json` and every weight shard, point the runner at that measured path. If that snapshot's tokenizer does not match the frozen tokenizer fingerprint, set a separately verified tokenizer directory. Keep local-only loading and the Chat Template fingerprint check enabled; do not download a model as part of E00.1.

Set these only to paths observed on the machine; the following values are placeholders, not expected cache locations:

```bash
export EVALUATION_MODEL_PATH='/absolute/path/confirmed/above/snapshots/<actual-revision>'
# Optional, only if a separate tokenizer directory matches the frozen fingerprint:
# export EVALUATION_TOKENIZER_PATH='/absolute/path/to/verified/tokenizer'
```

```bash
RUN_ID=E00.2-smoke-001
EVALUATION_EXPERIMENT_ID=E00.2 \
EVALUATION_RUN_ID="$RUN_ID" \
EVALUATION_ENABLE_NF4=1 \
EVALUATION_RAW_OUTPUT_PATH="$PWD/outputs/evaluation/$RUN_ID/raw_predictions.jsonl" \
bash participant/start.sh \
  outputs/evaluation/e00_1/dev_adapter_v2/smoke_input.jsonl \
  "outputs/evaluation/$RUN_ID/runner"

python -m src.evaluation.score \
  --config configs/evaluation/e00.yaml \
  --references outputs/evaluation/e00_1/dev_adapter_v2/smoke_references.jsonl \
  --submission "outputs/evaluation/$RUN_ID/runner/submission.jsonl" \
  --raw "outputs/evaluation/$RUN_ID/raw_predictions.jsonl" \
  --output-dir "outputs/evaluation/$RUN_ID/scored"
```

Use the full `inference_input.jsonl` / `references.jsonl` pair for the later 1,833-row Dev run. The scorer automatically selects the matching full or Smoke ID map and checks the exact Smoke membership from the Manifest. It rejects a real D0.3 run whose model ID, P0/Tokenizer/Chat Template fingerprint, P/G settings, T1 protocol or NF4 configuration do not match the current experiment config. The data adapter fingerprint remains stable across those runtime/scoring changes, while each scored output records the full config fingerprint to prevent confusing results. The 100-round performance report remains incomplete for a smoke subset; this is expected. Do not interpret the mock fixture as speed or quality evidence.

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
