# E03 final dual-Adapter Docker delivery

This image contains the Qwen3-1.7B Base, A8-P4504 profile LoRA,
E03-B Replay-1152 emotion-response LoRA, the four required Prompts, and the
official two-argument `participant/start.sh` entrypoint. Inference runs offline
with one local vLLM service and serial requests to `profile` and
`emotion-response`.

The build pins the CUDA runtime and Miniforge images by digest, then creates the
fixed `conda_env` from the repository's Linux package locks. Model locks and
package manifests are packaging records; the official inference path does not
require their hashes, Git metadata, or a GPU diagnostic command.

From the repository root, create the package context and image:

```bash
python3 scripts/final_dual_docker_package.py check
python3 scripts/final_dual_docker_package.py prepare \
  --config configs/deployment/final_dual_docker.json \
  --output-dir outputs/final_delivery/E03-DUAL-20261109-04/docker/context

docker build --progress=plain \
  -f docker/final_dual/Dockerfile \
  -t emotionalcompany/final-dual:E03-DUAL-20261109-04 \
  outputs/final_delivery/E03-DUAL-20261109-04/docker/context
```

The official command inside the container is:

```bash
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

The image entrypoint also supports the following forms:

```bash
docker run --rm --gpus all --network none IMAGE TEST_FILE RESULT_DIR
docker run --rm --gpus all --network none IMAGE bash /root/participant/start.sh TEST_FILE RESULT_DIR
docker run --rm --gpus all --network none IMAGE /root/participant/start.sh TEST_FILE RESULT_DIR
docker run --rm --gpus all --network none IMAGE bash -lc "bash /root/participant/start.sh TEST_FILE RESULT_DIR"
docker run --rm --gpus all --network none IMAGE /bin/bash -c "bash /root/participant/start.sh TEST_FILE RESULT_DIR"
docker run --rm --gpus all --network none IMAGE sh -c "bash /root/participant/start.sh TEST_FILE RESULT_DIR"
docker run --rm --gpus all --network none --entrypoint /bin/bash IMAGE /root/participant/start.sh TEST_FILE RESULT_DIR
```

The entrypoint deliberately passes unknown command forms through instead of
rejecting them.  When no arguments are supplied it stays alive only so the
platform can attach with `docker exec`.

Runtime portability behavior:

- vLLM uses `--dtype auto`; BF16 is retained on Ampere+ GPUs and vLLM falls
  back to FP16 on pre-Ampere GPUs instead of exiting on a dtype check.
- Startup retries lower `gpu_memory_utilization`, tries alternate local ports,
  and retries with `--enforce-eager` before giving up.
- `fallback.degraded_output_on_runtime_failure` in the deployment config makes
  a non-startable runtime still emit a complete, Schema-valid submission with
  empty fields and a `degraded` report, rather than failing without files.
- On failure the image writes `vllm_server.log` and `runtime_failure.json`
  into `RESULT_DIR` so the reason survives even when the platform exposes no
  container logs.

To keep a container available for a later `docker exec`, mount the input and
result locations when creating it. With no command arguments the image idles
without starting vLLM or using the GPU:

```bash
docker run -d --gpus all --network none --name final-dual-idle \
  -v /host/input:/input:ro -v /host/result:/result:rw IMAGE
docker exec final-dual-idle bash /root/participant/start.sh /input/test.jsonl /result
docker rm -f final-dual-idle
```

Each completed run writes only `submission.jsonl` and
`performance_report.json` in `RESULT_DIR`. Existing files with those names are
replaced; other files are preserved. All inputs are processed, while timing is
limited to the first 100 samples. The output verifier checks row count, ID
order, completion, and the official submission Schema; it does not compare
generated text or quality metrics across runs.

Example output-contract check:

```bash
python3 scripts/final_dual_docker_package.py verify-output \
  --test-file outputs/final_delivery/E03-DUAL-20261109-04/input/smoke.jsonl \
  --result-dir outputs/final_delivery/E03-DUAL-20261109-04/docker/smoke \
  --expected-rows 3
```
