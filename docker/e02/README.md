# E02.2-C Docker package

## State

The CPU package preflight, pinned runtime files, image definition, official
entrypoint wrapper, and output verifier are prepared. A local fixed image was
built and passed a 10-request offline GPU Smoke on 2026-11-02. This establishes
local container feasibility. The final image virtual-size check, tar export and
SHA, clean-environment import, and official submission remain deferred. `check`
does not read or hash model files. `prepare` performs the full Base/Adapter
verification and copies roughly 4 GB of Base weights into the ignored
`outputs/` tree.

The package keeps the competition entrypoint
`bash /root/participant/start.sh TEST_FILE RESULT_DIR` and its
`submission.jsonl` / `performance_report.json` outputs. It contains only the
runtime files, the frozen Base/Adapter/Tokenizer, P0 and shared protocol. Raw
Train/Dev/Val data, targets, predictions, `.venv`, HF cache, and Git metadata do
not enter the image context.

### 2026-11-02 local build and offline GPU Smoke

Run artifacts are under
`outputs/deployment/e02/E02.2-C-DOCKER-FIX-20261102T134557Z/`. The retained
BuildKit log shows the fixed image built successfully with
`build-essential`; the packaged Dockerfile SHA-256 is
`c9979eecb75456f2a5d4e1dab16245042221bd9d06595c9346018811eb463c61`, matching
the current workspace file. The local performance report records vLLM
`0.11.1`, configured `K=2560`, 10/10 successful requests, 10 Submission rows,
and a valid Submission Schema. The owner's WSL `verify-output` result was
`passed_output_contract_and_local_identity`; its JSON was not retained in the
Run directory, so that status is a user-supplied local observation. The report
keeps `server_weight_identity=unverified` and does not observe exact server
Prompt Token IDs for this Docker run. The separate E02.2-A Run provides the
12/12 exact Prompt Token ID result.

The initial compiler failure was reported by the owner, while the retained
first-run directory contains only its Docker build log. The fixed build log
directly confirms installation of `build-essential`. The retained Smoke
report has no image-size field and the Run directory has no image-inspect
result, so the fixed image's virtual size is not recorded here. Do not reuse the pre-fix
`31,852,848,477` byte value. Tar export/SHA, import in a clean environment, and
official submission remain deferred.

## CPU-only preflight

From the repository root:

```bash
./.venv/bin/python scripts/e02_docker_package.py check
./.venv/bin/python -m pytest tests/test_e02_docker_package.py -q
bash -n docker/e02/entrypoint.sh
```

The environment lock follows the successful E02 vLLM environment:
Python `3.10.22`, vLLM `0.11.1`, PyTorch `2.9.0+cu128`, Transformers
`4.57.6`, and Tokenizers `0.22.2`. vLLM 0.11.1's versioned installation guide
documents its CUDA 12.8 wheel path: [vLLM 0.11.1 GPU installation](https://docs.vllm.ai/en/v0.11.1/getting_started/installation/gpu/).

## Prepare assets, build, and run the GPU Smoke

These commands hash and copy the local Base and Adapter. They require enough
disk space for the package context and a Docker engine with NVIDIA GPU support.
Image build downloads the pinned base and package artifacts; the resulting
container is configured to run inference without network access.

```bash
cd ~/projects/EmotionalCompany
source .venv/bin/activate

RUN_ID="E02.2-C-DOCKER-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$PWD/outputs/deployment/e02/$RUN_ID"
CONTEXT_DIR="$RUN_ROOT/context"
IMAGE="emotionalcompany/e02:$RUN_ID"
TEST_FILE="$PWD/outputs/evaluation/e02/E02.1-VLLM-FULLB-20261101T070852Z/test_input.jsonl"
RESULT_DIR="$RUN_ROOT/docker-smoke-result"

python scripts/e02_docker_package.py check
python scripts/e02_docker_package.py prepare \
  --base-model-dir participant/model \
  --adapter-dir outputs/training/E01-B-FULL-20261028-133805/adapter \
  --output-dir "$CONTEXT_DIR"

docker build \
  --file docker/e02/Dockerfile \
  --tag "$IMAGE" \
  "$CONTEXT_DIR"

IMAGE_BYTES="$(docker image inspect --format '{{.Size}}' "$IMAGE")"
printf 'Docker image virtual size: %s bytes\n' "$IMAGE_BYTES"
# The official notice says “50G” without defining the unit. Use the project's
# conservative 50,000,000,000-byte ceiling for this local gate.
test "$IMAGE_BYTES" -le 50000000000

mkdir -p "$RUN_ROOT" "$RESULT_DIR"
docker image save "$IMAGE" --output "$RUN_ROOT/image.tar"
du -h "$RUN_ROOT/image.tar"
sha256sum "$RUN_ROOT/image.tar" | tee "$RUN_ROOT/image.tar.sha256"

docker run --rm \
  --gpus all \
  --network none \
  --shm-size 1g \
  --volume "$TEST_FILE:/input/test_input.jsonl:ro" \
  --volume "$RESULT_DIR:/output:rw" \
  "$IMAGE" \
  /input/test_input.jsonl /output

python scripts/e02_docker_package.py verify-output \
  --test-file "$TEST_FILE" \
  --result-dir "$RESULT_DIR"
```

`--network none` demonstrates offline inference after the image is built. The
container still starts its own local vLLM server and sends requests to its
loopback interface. No host HF cache or host Python environment is mounted.
Before starting vLLM, the container entrypoint checks the input, result
directory writability, and existing Run artifacts; it clears inherited API
key variables because the internal loopback service does not need them.

The output verifier checks input/output ID alignment, every submission row
against the competition Schema, complete performance-report counts, and the
configured local Base/Adapter/vLLM identities. The report continues to mark
server-side weight identity as `unverified`; the local file hash and the
reported vLLM model alias do not independently prove which weights vLLM
loaded.

When reporting the run, include the verifier JSON, image virtual size, saved
image tar size and SHA-256. Keep `context/`, `image.tar`, model files, and
inference outputs under ignored `outputs/`; do not commit them.
