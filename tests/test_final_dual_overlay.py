from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import final_dual_overlay as overlay


def test_overlay_dockerfile_only_inherits_image_and_copies_code_and_pdf() -> None:
    dockerfile = overlay.ROOT / "docker/final_dual/Dockerfile.overlay"

    overlay.validate_overlay_dockerfile(dockerfile)
    text = dockerfile.read_text(encoding="utf-8").lower()

    assert "from ${base_image}" in text
    assert "copy participant/model" not in text
    assert "copy participant/adapters" not in text
    assert "conda create" not in text
    assert "pip install" not in text
    assert "apt-get" not in text


def test_prepare_code_context_is_small_and_records_pdf_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    for relative in overlay.OVERLAY_FILES:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == overlay.PDF_PATH:
            destination.write_bytes(b"%PDF-1.7\nfixture\n")
        else:
            destination.write_text("# overlay fixture\n", encoding="utf-8")
    dockerfile = repo / "docker/final_dual/Dockerfile.overlay"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    shutil_source = overlay.ROOT / "docker/final_dual/Dockerfile.overlay"
    dockerfile.write_bytes(shutil_source.read_bytes())
    facts = {"image_id": "sha256:parent", "runtime": {"vllm": "0.11.1"}, "runtime_lock_sha256": {}}
    monkeypatch.setattr(overlay, "ROOT", repo)
    monkeypatch.setattr(overlay, "_check_base_facts", lambda *_args: (facts, {}))
    output = repo / "outputs/deployment/final_overlay/run/context"

    result = overlay.prepare_code_context(
        image="emotionalcompany/final-dual:parent",
        tag="emotionalcompany/final-dual:overlay",
        output_dir=output,
        facts_file=repo / "outputs/deployment/final_overlay/base_image.json",
    )

    files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected = {path.as_posix() for path in overlay.OVERLAY_FILES}
    assert expected.issubset(files)
    assert "participant/model" not in files
    assert "participant/adapters" not in files
    assert "Dockerfile" in files
    assert result["parent_image_id"] == "sha256:parent"
    manifest = json.loads((output / "overlay_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependency_installation"] is False
    assert manifest["model_copy"] is False
    assert manifest["report_pdf"]["sha256"] == overlay.sha256_file(output / overlay.PDF_PATH)


def test_smoke_fixture_uses_unrestricted_ids_and_extra_metadata() -> None:
    rows = overlay.make_smoke_rows(3)

    assert [row["id"] for row in rows] == [
        "test1_000001_t5", "test_000001_t0", "overlay-smoke:3"
    ]
    assert rows[0]["platform_metadata"] == {"source": "overlay-smoke"}
    assert rows[0]["history"][0]["smoke_metadata"] == "must be ignored"
