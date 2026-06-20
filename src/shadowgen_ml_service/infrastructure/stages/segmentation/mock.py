from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import Segmenter
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.core.stage_io import SegmentationInput
from shadowgen_ml_service.utils.images import create_cutout, ensure_pil, estimate_foreground_mask, pil_to_asset


class MockSegmenter(Segmenter):
    def segment(self, stage_input: SegmentationInput) -> SegmentationResult:
        image = ensure_pil(stage_input.image)
        crop_mask = estimate_foreground_mask(image)
        cutout_rgba = create_cutout(image, crop_mask)
        return SegmentationResult(
            bbox=(0, 0, image.width, image.height),
            mask=pil_to_asset(crop_mask),
            cutout_rgba=pil_to_asset(cutout_rgba),
            crop_rgba=pil_to_asset(image),
        )
