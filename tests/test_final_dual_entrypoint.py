from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARGUMENT_PARSER = ROOT / "docker/final_dual/entrypoint_args.sh"
ENTRYPOINT = ROOT / "docker/final_dual/entrypoint.sh"


def _parse(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash", "-c",
            'source "$1"; shift; final_dual_parse_entrypoint_args "$@"; '
            'printf "%s\\n%s\\n%s\\n%s\\n" '
            '"$ENTRYPOINT_IDLE" "$ENTRYPOINT_START_SH" "$TEST_FILE" "$RESULT_DIR"',
            "bash", str(ARGUMENT_PARSER), *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_direct_two_path_form_is_mapped_to_the_official_entrypoint() -> None:
    result = _parse(["/input/test.jsonl", "/result"])

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["false", "true", "/input/test.jsonl", "/result"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["/root/participant/start.sh", "/input/test.jsonl", "/result"],
        ["bash", "/root/participant/start.sh", "/input/test.jsonl", "/result"],
        ["/bin/bash", "-c", "bash /root/participant/start.sh /input/test.jsonl /result"],
        ["bash", "-lc", "bash /root/participant/start.sh /input/test.jsonl /result"],
        ["sh", "-c", "bash /root/participant/start.sh /input/test.jsonl /result"],
        ["bash", "/bin/echo", "bad.jsonl", "/result"],
    ],
)
def test_command_forms_are_passed_through_instead_of_rejected(arguments: list[str]) -> None:
    result = _parse(arguments)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["false", "false", "", ""]


def test_single_string_command_is_marked_for_shell_execution() -> None:
    result = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; shift; final_dual_parse_entrypoint_args "$@"; '
            'printf "%s\\n%s\\n" "$ENTRYPOINT_START_SH" "$ENTRYPOINT_SHELL_COMMAND"',
            "bash", str(ARGUMENT_PARSER),
            "bash /root/participant/start.sh /input/test.jsonl /result",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "false", "bash /root/participant/start.sh /input/test.jsonl /result"
    ]


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


def test_container_entrypoint_has_no_gpu_diagnostic_gate_and_passes_commands_through() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "nvidia-smi" not in text
    assert "exec sleep infinity" in text
    assert 'exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"' in text
    assert 'exec "$@"' in text
