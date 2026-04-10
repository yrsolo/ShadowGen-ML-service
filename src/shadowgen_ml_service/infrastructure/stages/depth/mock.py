from __future__ import annotations

from shadowgen_ml_service.core.contracts import DepthEstimator
from shadowgen_ml_service.core.models import DepthResult
from shadowgen_ml_service.core.stage_io import DepthInput
from shadowgen_ml_service.utils.images import depth_from_mask, ensure_pil, pil_to_asset


class MockDepthEstimator(DepthEstimator):
    def __init__(self) -> None:
        self.device_label = "cpu"

    def estimate(self, stage_input: DepthInput) -> DepthResult:
        return DepthResult(depth_map=pil_to_asset(depth_from_mask(ensure_pil(stage_input.mask))))
