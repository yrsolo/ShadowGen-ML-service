from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TritonBackendSettings:
    url: str | None
    protocol: str
    timeout_ms: int
    transport: str = "native"

    @property
    def enabled(self) -> bool:
        return bool(self.url)
