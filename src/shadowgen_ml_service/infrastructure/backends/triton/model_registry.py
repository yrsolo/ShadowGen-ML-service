from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TritonModelBinding:
    stage_key: str
    model_variant: str
    model_name: str


class TritonModelRegistry:
    def __init__(self, bindings: list[TritonModelBinding]) -> None:
        self._bindings = {(item.stage_key, item.model_variant): item for item in bindings}

    def get(self, stage_key: str, model_variant: str) -> TritonModelBinding | None:
        return self._bindings.get((stage_key, model_variant))
