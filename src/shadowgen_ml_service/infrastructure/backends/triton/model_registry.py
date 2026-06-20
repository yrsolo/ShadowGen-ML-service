from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TritonTensorBinding:
    tensor_name: str
    datatype: str
    expected_ranks: tuple[int, ...] = field(default_factory=tuple)
    shape_policy: str | None = None
    channels: int | None = None


@dataclass(frozen=True)
class TritonModelBinding:
    stage_key: str
    model_variant: str
    model_name: str
    inputs: dict[str, TritonTensorBinding] = field(default_factory=dict)
    outputs: dict[str, TritonTensorBinding] = field(default_factory=dict)


class TritonModelRegistry:
    def __init__(self, bindings: list[TritonModelBinding]) -> None:
        self._bindings = {(item.stage_key, item.model_variant): item for item in bindings}

    def get(self, stage_key: str, model_variant: str) -> TritonModelBinding | None:
        return self._bindings.get((stage_key, model_variant))
