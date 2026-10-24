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
