from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult


class MockNormalEstimator(NormalEstimator):
    def __init__(self) -> None:
        self.device_label = "cpu"
        self.backend_name = "mock"
        self.model_variant = "mock"

    def estimate(self, image: Image.Image, depth_map: Image.Image | None = None) -> NormalResult:
        size = depth_map.size if depth_map is not None else image.size
        normal_map = Image.new("RGB", size, (127, 127, 255))
        return NormalResult(normal_map=normal_map)
