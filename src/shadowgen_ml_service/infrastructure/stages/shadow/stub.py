from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.commands import ShadowSpec
from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import GeometryResult, ShadowResult
from shadowgen_ml_service.utils.images import generate_shadow_layer


class DeterministicShadowGenerator(ShadowGenerator):
    def generate(
        self,
        mask: Image.Image,
        depth_map: Image.Image,
        normal_map: Image.Image,
        geometry: GeometryResult,
        shadow: ShadowSpec,
    ) -> ShadowResult:
        shadow_rgba = generate_shadow_layer(
            mask=mask,
            angle_deg=shadow.angle_deg,
            elevation_deg=shadow.elevation_deg,
            softness=shadow.softness,
            opacity=shadow.opacity,
            reflection=shadow.reflection,
            camera_pitch=geometry.camera_pitch,
        )
        return ShadowResult(shadow_rgba=shadow_rgba)
