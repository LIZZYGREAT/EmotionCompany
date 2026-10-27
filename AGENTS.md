# Repository work rules

- Read [赛事文件边界与运行契约](docs/competition/赛事文件边界与运行契约.md) before changing competition files.
- Treat `data/train/train_public.jsonl`, `data/val/val_public.jsonl`, official schema files, and `configs/prompts/p0.txt` as protected inputs. Do not rewrite or delete them in routine work. Update them only with provenance, an explicit version record, and regression coverage.
- Preserve the official entry point `bash /root/participant/start.sh TEST_FILE RESULT_DIR`, its fixed `conda_env` activation, and the `submission.jsonl` / `performance_report.json` output contract.
- `participant/run_inference.py`, its config, and requirements are editable reference implementation files. Any changes must retain the competition contract and pass protocol tests; do not freeze the whole `participant/` directory.
- Keep P0/F2/T1 and D0.3 split semantics stable. A semantic change requires a new protocol/data version; a bug fix requires focused regression tests.
- Do not commit real data, model weights/cache, generated predictions, or `outputs/`. Run relevant tests and `git diff --check` before committing.

## Team Git and PR workflow

- Follow [双人 Git 协作与 PR 操作规范](data/knowledge/双人Git协作与PR操作规范.md) for task branches, explicit staging, commits, syncing, reviews, and merges.
- Integrate into `main` through a teammate-reviewed PR after the `unit-cpu` CI check passes. Do not push or force-push directly to `main`.
- Keep the GitHub PR template and CI workflow aligned with the Required Check configured for `main`.
