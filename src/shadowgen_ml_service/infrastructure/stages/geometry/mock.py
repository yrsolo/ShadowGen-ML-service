from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import GeometryEstimator
from shadowgen_ml_service.core.models import GeometryResult


class MockGeometryEstimator(GeometryEstimator):
    def estimate(self, image: Image.Image) -> GeometryResult:
        aspect = image.width / max(image.height, 1)
        fov = 42.0 + min(aspect, 2.0) * 8.0
        pitch = -4.0 if image.height >= image.width else -2.0
        return GeometryResult(camera_fov=fov, camera_pitch=pitch, camera_roll=0.0, confidence=0.55)
