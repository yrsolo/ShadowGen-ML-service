from __future__ import annotations

from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import ShadowResult
from shadowgen_ml_service.core.stage_io import ShadowInput
from shadowgen_ml_service.utils.images import ensure_pil, generate_shadow_layer, pil_to_asset


class DeterministicShadowGenerator(ShadowGenerator):
    def __init__(self) -> None:
        self.device_label = "cpu"
        self.backend_name = "deterministic-stub"

    def generate(self, stage_input: ShadowInput) -> ShadowResult:
        shadow_image = generate_shadow_layer(
            mask=ensure_pil(stage_input.mask),
            angle_deg=stage_input.angle,
            elevation_deg=stage_input.elevation,
            softness=stage_input.softness,
            opacity=stage_input.opacity,
            reflection=stage_input.reflection,
            camera_pitch=0.0,
        )
        return ShadowResult(shadow_image=pil_to_asset(shadow_image))
