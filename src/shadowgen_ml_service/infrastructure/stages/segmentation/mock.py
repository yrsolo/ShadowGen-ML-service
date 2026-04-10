from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import Segmenter
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.core.stage_io import SegmentationInput
from shadowgen_ml_service.utils.images import create_cutout, estimate_foreground_mask


class MockSegmenter(Segmenter):
    def segment(self, stage_input: SegmentationInput) -> SegmentationResult:
        crop_mask = estimate_foreground_mask(stage_input.image)
        cutout_rgba = create_cutout(stage_input.image, crop_mask)
        return SegmentationResult(
            bbox=(0, 0, stage_input.image.width, stage_input.image.height),
            mask=crop_mask,
            cutout_rgba=cutout_rgba,
            crop_rgba=stage_input.image,
        )
