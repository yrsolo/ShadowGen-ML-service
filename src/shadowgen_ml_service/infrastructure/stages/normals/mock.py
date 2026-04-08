from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult


class MockNormalEstimator(NormalEstimator):
    def __init__(self) -> None:
        self.device_label = "cpu"

    def estimate(self, depth_map: Image.Image) -> NormalResult:
        normal_map = Image.new("RGB", depth_map.size, (127, 127, 255))
        return NormalResult(normal_map=normal_map)
