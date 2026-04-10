from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import DepthEstimator
from shadowgen_ml_service.core.models import DepthResult
from shadowgen_ml_service.core.stage_io import DepthInput
from shadowgen_ml_service.utils.images import depth_from_mask


class MockDepthEstimator(DepthEstimator):
    def __init__(self) -> None:
        self.device_label = "cpu"

    def estimate(self, stage_input: DepthInput) -> DepthResult:
        return DepthResult(depth_map=depth_from_mask(stage_input.mask))
