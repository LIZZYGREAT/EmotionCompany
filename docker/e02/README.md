# E02.2-C Docker package

## State

The CPU package preflight, pinned runtime files, image definition, official
entrypoint wrapper, and output verifier are prepared. The actual model package,
Docker image build, image-size/hash evidence, and Docker GPU Smoke have not been
run. `check` does not read or hash model files. `prepare` performs the full
Base/Adapter verification and copies roughly 4 GB of Base weights into the
ignored `outputs/` tree.

The package keeps the competition entrypoint
`bash /root/participant/start.sh TEST_FILE RESULT_DIR` and its
`submission.jsonl` / `performance_report.json` outputs. It contains only the
runtime files, the frozen Base/Adapter/Tokenizer, P0 and shared protocol. Raw
Train/Dev/Val data, targets, predictions, `.venv`, HF cache, and Git metadata do
not enter the image context.

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
test "$IMAGE_BYTES" -le 53687091200

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
