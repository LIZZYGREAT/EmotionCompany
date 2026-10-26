from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PROTECTED_SHA256 = {
    "participant/templates/submission_schema.json": "91c42d56ae433212aab7905eabdbb96cf8463c2f887dc46b3f31647ab3a5d591",
    "participant/templates/submission_template.json": "d6a630a114ffd91e83fab02deb22a1a7e2bd84266519f45bf775f48a47e31a3c",
    "data/train/train_public.schema.json": "a9467e520b06a96859acaa2ff022e395d08ff6ad5e8ae3532a22cc24c630cada",
    "data/val/val_public.schema.json": "4b995d80aba99883bda63e2fa18b161a2784e501f98316295f4618b0a0c4af10",
}


def test_official_schemas_and_submission_template_keep_the_reviewed_fingerprints():
    for relative_path, expected_sha256 in PROTECTED_SHA256.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_sha256, f"受保护赛事文件指纹变化: {relative_path}"


def test_start_script_preserves_entry_arguments_and_fixed_environment_contract():
    script = (ROOT / "participant/start.sh").read_text(encoding="utf-8")

    assert "source activate conda_env" in script
    assert '[[ $# -ne 2 ]]' in script
    assert 'TEST_FILE="$1"' in script
    assert 'RESULT_DIR="$2"' in script
    assert '"$SCRIPT_DIR/run_inference.py" "$TEST_FILE" "$RESULT_DIR"' in script

