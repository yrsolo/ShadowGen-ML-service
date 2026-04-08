from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import Segmenter
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.utils.images import create_cutout, estimate_foreground_mask


class MockSegmenter(Segmenter):
    def segment(self, image: Image.Image) -> SegmentationResult:
        crop_mask = estimate_foreground_mask(image)
        cutout_rgba = create_cutout(image, crop_mask)
        return SegmentationResult(
            bbox=(0, 0, image.width, image.height),
            mask=crop_mask,
            cutout_rgba=cutout_rgba,
            crop_rgba=image,
        )
