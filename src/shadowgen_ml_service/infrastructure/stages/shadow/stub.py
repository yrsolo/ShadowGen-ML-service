from __future__ import annotations

from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import ShadowResult
from shadowgen_ml_service.core.stage_io import ShadowInput
from shadowgen_ml_service.utils.images import generate_shadow_layer


class DeterministicShadowGenerator(ShadowGenerator):
    def __init__(self) -> None:
        self.device_label = "cpu"
        self.backend_name = "deterministic-stub"

    def generate(self, stage_input: ShadowInput) -> ShadowResult:
        shadow_rgba = generate_shadow_layer(
            mask=stage_input.mask,
            angle_deg=stage_input.angle,
            elevation_deg=stage_input.elevation,
            softness=stage_input.softness,
            opacity=stage_input.opacity,
            reflection=stage_input.reflection,
            camera_pitch=0.0,
        )
        return ShadowResult(shadow_rgba=shadow_rgba)
