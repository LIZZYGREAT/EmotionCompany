from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARGUMENT_PARSER = ROOT / "docker/final_dual/entrypoint_args.sh"
ENTRYPOINT = ROOT / "docker/final_dual/entrypoint.sh"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["/input/test.jsonl", "/result"], ["/input/test.jsonl", "/result"]),
        (
            ["/root/participant/start.sh", "/input/test.jsonl", "/result"],
            ["/input/test.jsonl", "/result"],
        ),
        (
            ["bash", "/root/participant/start.sh", "/input/test.jsonl", "/result"],
            ["/input/test.jsonl", "/result"],
        ),
    ],
)
def test_docker_entrypoint_forwards_supported_argument_forms(arguments: list[str], expected: list[str]) -> None:
    result = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; shift; final_dual_parse_entrypoint_args "$@"; printf "%s\\n%s\\n" "$TEST_FILE" "$RESULT_DIR"',
            "bash", str(ARGUMENT_PARSER), *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


def test_no_arguments_selects_idle_container_without_running_inference() -> None:
    result = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; final_dual_parse_entrypoint_args; printf "%s\\n" "$ENTRYPOINT_IDLE"',
            "bash", str(ARGUMENT_PARSER),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "true"


def test_docker_entrypoint_rejects_unapproved_command() -> None:
    result = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; shift; final_dual_parse_entrypoint_args "$@"',
            "bash", str(ARGUMENT_PARSER),
            "bash", "/bin/echo", "bad.jsonl", "/result",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "only /root/participant/start.sh" in result.stderr


def test_container_entrypoint_has_no_gpu_diagnostic_gate() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "nvidia-smi" not in text
    assert "exec sleep infinity" in text
    assert 'exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"' in text
