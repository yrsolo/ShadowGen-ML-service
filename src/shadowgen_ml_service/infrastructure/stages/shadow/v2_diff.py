from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from shadowgen_ml_service.core.commands import ShadowSpec
from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import GeometryResult, ShadowResult


@dataclass(frozen=True)
class DiffShadowInputs:
    img: Image.Image
    mask: Image.Image
    depth: Image.Image
    normal: Image.Image
    angle: float
    elevation: float
    softness: float
    reflection: float


class V2DiffShadowGenerator(ShadowGenerator):
    """
    Recommended skeleton for the future diffusion shadow model.

    Expected model contract:
    - img: refined RGBA cutout or RGB foreground crop
    - mask: foreground mask
    - depth: depth map
    - normal: normal map
    - angle: azimuth in degrees
    - elevation: light elevation in degrees
    - softness: conditioning control, not a post-blur
    - reflection: optional conditioning input, may be ignored by the first checkpoint

    Notes for the future implementation:
    - keep all preprocessing local to this class
    - do not post-blur the model result to emulate softness
    - return a standalone RGBA shadow layer
    - keep runtime metadata on `backend_name`, `model_variant`, and `device_label`
    """

    def __init__(self, *, target_device: str = "cuda") -> None:
        self.backend_name = "v2-diff"
        self.model_variant = "V2-DIFF"
        self.device_label = target_device

    def prepare_inputs(
        self,
        *,
        cutout_rgba: Image.Image,
        mask: Image.Image,
        depth_map: Image.Image,
        normal_map: Image.Image,
        shadow: ShadowSpec,
    ) -> DiffShadowInputs:
        return DiffShadowInputs(
            img=cutout_rgba,
            mask=mask,
            depth=depth_map,
            normal=normal_map,
            angle=float(shadow.angle_deg),
            elevation=float(shadow.elevation_deg),
            softness=float(shadow.softness),
            reflection=float(shadow.reflection),
        )

    def generate(
        self,
        cutout_rgba: Image.Image,
        mask: Image.Image,
        depth_map: Image.Image,
        normal_map: Image.Image,
        geometry: GeometryResult,
        shadow: ShadowSpec,
    ) -> ShadowResult:
        _ = self.prepare_inputs(
            cutout_rgba=cutout_rgba,
            mask=mask,
            depth_map=depth_map,
            normal_map=normal_map,
            shadow=shadow,
        )
        raise NotImplementedError("V2-DIFF shadow model is not connected yet. Implement the model loader and inference path in this class.")
