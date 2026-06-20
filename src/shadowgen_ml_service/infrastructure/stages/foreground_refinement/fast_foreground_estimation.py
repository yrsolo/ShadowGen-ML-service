from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from shadowgen_ml_service.core.contracts import ForegroundColorEstimator
from shadowgen_ml_service.core.models import ForegroundRefinementResult
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, import_module, module_available
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


def _to_unit_float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_unit_float_alpha(alpha: Image.Image) -> np.ndarray:
    return (np.asarray(alpha.convert("L"), dtype=np.float32) / 255.0)[:, :, None]


def blur_fusion_foreground_estimator(
    image: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    radius: int,
    cv2_module: Any,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = (max(1, int(radius)), max(1, int(radius)))
    blurred_alpha = cv2_module.blur(alpha, kernel)
    if blurred_alpha.ndim == 2:
        blurred_alpha = blurred_alpha[:, :, None]
    blurred_foreground_alpha = cv2_module.blur(foreground * alpha, kernel)
    blurred_foreground = blurred_foreground_alpha / (blurred_alpha + 1e-5)
    blurred_background_alpha = cv2_module.blur(background * (1.0 - alpha), kernel)
    blurred_background = blurred_background_alpha / ((1.0 - blurred_alpha) + 1e-5)
    estimated_foreground = blurred_foreground + alpha * (
        image - alpha * blurred_foreground - (1.0 - alpha) * blurred_background
    )
    return np.clip(estimated_foreground, 0.0, 1.0), blurred_background


class FastForegroundColorEstimator(ForegroundColorEstimator):
    def __init__(self, cv2_module: Any | None = None, *, coarse_radius: int = 90, refine_radius: int = 6) -> None:
        self._cv2 = cv2_module or import_module("cv2")
        self.coarse_radius = coarse_radius
        self.refine_radius = refine_radius

    def refine(self, image, alpha) -> ForegroundRefinementResult:
        image_pil = ensure_pil(image)
        alpha_pil = ensure_pil(alpha)
        image_array = _to_unit_float_rgb(image_pil)
        alpha_array = _to_unit_float_alpha(alpha_pil)
        coarse_foreground, blurred_background = blur_fusion_foreground_estimator(
            image_array,
            image_array,
            image_array,
            alpha_array,
            self.coarse_radius,
            self._cv2,
        )
        refined_foreground, _ = blur_fusion_foreground_estimator(
            image_array,
            coarse_foreground,
            blurred_background,
            alpha_array,
            self.refine_radius,
            self._cv2,
        )
        refined_rgb = Image.fromarray((refined_foreground * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
        refined_rgba = refined_rgb.convert("RGBA")
        refined_rgba.putalpha(alpha_pil.convert("L"))
        return ForegroundRefinementResult(cutout_rgba=pil_to_asset(refined_rgba))


def probe_fast_foreground_estimation() -> RealAdapterProbe:
    available = module_available("cv2")
    detail = None if available else "requires opencv-python-headless (cv2)"
    return RealAdapterProbe(
        "Photoroom/fast-foreground-estimation",
        "approximate-fast-foreground-colour-estimation",
        available,
        detail,
    )
