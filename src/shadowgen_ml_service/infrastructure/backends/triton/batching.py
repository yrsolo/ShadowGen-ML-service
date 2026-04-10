from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchingCapabilities:
    supported: bool
    preferred_batch_size: int | None = None
