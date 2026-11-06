"""CPU-only helpers for the E03-A6 F4 full-field Pilot Fast evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.shared.f4_protocol import F4ParseError, parse_f4


@dataclass(frozen=True)
class F4FastResult:
    status: str
    reason: str
    raw_prefix: str
    parsed: Mapping[str, Any] | None
    generated_token_ids: tuple[int, ...]

    @property
    def valid(self) -> bool:
        return self.status == "valid"


class F4FastPrefixStopper:
    """Stop only after the complete generated ``\nResponse=`` field marker."""

    def __init__(
        self,
        decode: Callable[[Sequence[int]], str],
        enums: Mapping[str, list[str]],
        *,
        deadline: float | None = None,
    ) -> None:
        self.decode = decode
        self.enums = enums
        self.deadline = deadline
        self.stop_reason: str | None = None
        self.result: F4FastResult | None = None

    def inspect(self, generated_token_ids: Sequence[int]) -> bool:
        ids = tuple(int(value) for value in generated_token_ids)
        if self.deadline is not None:
            import time

            if time.monotonic() >= self.deadline:
                self.stop_reason = "budget_exhausted"
                self.result = F4FastResult("budget_exhausted", self.stop_reason, "", None, ids)
                return True
        decoded = self.decode(ids)
        marker = decoded.find("\nResponse=")
        if marker < 0:
            return False
        boundary = marker + len("\nResponse=")
        prefix = decoded[:boundary]
        try:
            parsed = parse_f4(prefix, dict(self.enums))
        except F4ParseError as exc:
            self.stop_reason = "invalid_at_response_marker"
            self.result = F4FastResult("invalid", exc.code, prefix, None, ids)
            return True
        self.stop_reason = "response_marker_closed"
        self.result = F4FastResult("valid", self.stop_reason, prefix, parsed, ids)
        return True

    def finish(self, stop_reason: str, generated_token_ids: Sequence[int]) -> F4FastResult:
        if self.result is not None:
            return self.result
        ids = tuple(int(value) for value in generated_token_ids)
        result = F4FastResult("incomplete", stop_reason, self.decode(ids), None, ids)
        self.stop_reason = stop_reason
        self.result = result
        return result
