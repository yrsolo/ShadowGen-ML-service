from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RasterAsset:
    mime_type: str
    mode: str
    width: int
    height: int
    data: bytes

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)
