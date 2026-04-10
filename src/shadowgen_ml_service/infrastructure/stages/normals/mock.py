from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult
from shadowgen_ml_service.core.stage_io import NormalsInput


class MockNormalEstimator(NormalEstimator):
    def __init__(self) -> None:
        self.device_label = "cpu"
        self.backend_name = "mock"
        self.model_variant = "mock"

    def estimate(self, stage_input: NormalsInput) -> NormalResult:
        size = stage_input.depth_map.size if stage_input.depth_map is not None else stage_input.image.size
        normal_map = Image.new("RGB", size, (127, 127, 255))
        return NormalResult(normal_map=normal_map)
