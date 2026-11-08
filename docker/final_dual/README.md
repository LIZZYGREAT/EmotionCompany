# E03 final dual-Adapter Docker delivery

This package contains the validated Qwen3-1.7B Base, A8-P4504 profile LoRA,
E03-B Replay-1152 emotion-response LoRA, locked Prompts, and the official
two-argument `participant/start.sh` entrypoint. Runtime inference is offline;
vLLM binds to container loopback and the container does not use a host model
cache or Python environment.

Build context and model files belong under ignored `outputs/`. The package
script checks the previous Transformers, vLLM, and official-entry gates before
copying any weights. It verifies Base, Tokenizer, Prompt, and both Adapter
identities, writes a container-relative model lock, and excludes datasets,
training checkpoints, caches, and Git metadata.

From the repository root:

```bash
python3 scripts/final_dual_docker_package.py check
python3 scripts/final_dual_docker_package.py prepare \
  --config configs/deployment/final_dual_docker.json \
  --output-dir outputs/final_delivery/E03-DUAL-20261108-02/docker/context

docker build --progress=plain \
  -f docker/final_dual/Dockerfile \
  -t emotionalcompany/final-dual:E03-DUAL-20261108-02 \
  outputs/final_delivery/E03-DUAL-20261108-02/docker/context

docker run --rm --gpus all --network none --shm-size 1g \
  -v "$PWD/outputs/final_delivery/E03-DUAL-20261108-02/transformers_smoke20/input/smoke20_input.jsonl:/input/smoke.jsonl:ro" \
  -v "$PWD/outputs/final_delivery/E03-DUAL-20261108-02/docker/smoke:/output:rw" \
  emotionalcompany/final-dual:E03-DUAL-20261108-02 \
  /input/smoke.jsonl /output

python3 scripts/final_dual_docker_package.py verify-output \
  --test-file outputs/final_delivery/E03-DUAL-20261108-02/transformers_smoke20/input/smoke20_input.jsonl \
  --result-dir outputs/final_delivery/E03-DUAL-20261108-02/docker/smoke \
  --reference-dir outputs/evaluation/final_dual/E03-DUAL-20261108-02-official-start-vllm20
```

The container receives no network (`--network none`). A successful run must
load one Base and both named LoRAs, return exact server Prompt Token IDs on
both routes, produce 20 schema-valid rows, and match the host vLLM submission
and strict-format counts. The verifier also checks the container log for OOM
and crash markers.

After the first verified smoke, inspect the image virtual size, save it under
the same ignored run directory, hash the tar, remove the just-built local tag,
load from that tar, and repeat the offline GPU smoke. The package script
refuses to overwrite an existing context, image archive, or result run.
