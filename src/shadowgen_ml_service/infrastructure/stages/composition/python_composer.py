from __future__ import annotations

from shadowgen_ml_service.core.commands import BackgroundSpec, OutputSpec
from shadowgen_ml_service.core.contracts import Composer
from shadowgen_ml_service.core.models import CompositionResult
from shadowgen_ml_service.utils.images import compose_on_background, ensure_pil, pil_to_asset


class PythonComposer(Composer):
    def compose(self, cutout_rgba, shadow_rgba, background: BackgroundSpec, output: OutputSpec) -> CompositionResult:
        final_image = compose_on_background(
            cutout_rgba=ensure_pil(cutout_rgba),
            shadow_rgba=ensure_pil(shadow_rgba),
            color_hex=background.color_hex,
            width=output.width,
            height=output.height,
        )
        return CompositionResult(final_image=pil_to_asset(final_image))
