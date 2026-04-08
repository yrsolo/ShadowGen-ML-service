from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import ForegroundColorEstimator
from shadowgen_ml_service.core.models import ForegroundRefinementResult


class PassthroughForegroundColorEstimator(ForegroundColorEstimator):
    def refine(self, image: Image.Image, alpha: Image.Image) -> ForegroundRefinementResult:
        cutout_rgba = image.convert("RGBA")
        cutout_rgba.putalpha(alpha.convert("L"))
        return ForegroundRefinementResult(cutout_rgba=cutout_rgba)
