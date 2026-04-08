from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import Detector
from shadowgen_ml_service.core.models import DetectionResult
from shadowgen_ml_service.utils.images import bbox_from_mask, estimate_foreground_mask


class MockDetector(Detector):
    def detect(self, image: Image.Image, padding_px: int) -> DetectionResult:
        mask = estimate_foreground_mask(image)
        bbox = bbox_from_mask(mask, padding_px)
        return DetectionResult(bbox=bbox, confidence=0.68)
