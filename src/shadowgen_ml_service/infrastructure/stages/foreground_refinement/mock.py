from __future__ import annotations

from shadowgen_ml_service.core.contracts import ForegroundColorEstimator
from shadowgen_ml_service.core.models import ForegroundRefinementResult
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class PassthroughForegroundColorEstimator(ForegroundColorEstimator):
    def refine(self, image, alpha) -> ForegroundRefinementResult:
        cutout_rgba = ensure_pil(image).convert("RGBA")
        cutout_rgba.putalpha(ensure_pil(alpha).convert("L"))
        return ForegroundRefinementResult(cutout_rgba=pil_to_asset(cutout_rgba))
