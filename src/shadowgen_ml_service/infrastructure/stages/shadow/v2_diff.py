from __future__ import annotations

from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import ShadowResult
from shadowgen_ml_service.core.stage_io import ShadowInput


class V2DiffShadowGenerator(ShadowGenerator):
    """
    Recommended skeleton for the diffusion shadow model.

    Current temporary model contract:
    - img: refined RGBA cutout or RGB foreground crop
    - mask: foreground mask
    - output: standalone RGBA shadow layer

    The pipeline still passes depth, normals, angle, elevation, softness, reflection,
    and opacity through `ShadowInput` for compatibility with `V1-GAN`, mock, and
    the future controllable diffusion model. This temporary V2-DIFF implementation
    intentionally ignores those controls because the current research target is
    "draw a plausible shadow" rather than "draw a fully controllable shadow".

    Notes for the future implementation:
    - keep all preprocessing local to this class
    - do not post-blur the model result to emulate softness when controls return
    - return a standalone RGBA shadow layer
    - keep runtime metadata on `backend_name`, `model_variant`, and `device_label`
    """

    def __init__(self, *, target_device: str = "cuda") -> None:
        self.backend_name = "v2-diff"
        self.model_variant = "V2-DIFF"
        self.device_label = target_device

    def prepare_inputs(self, stage_input: ShadowInput) -> ShadowInput:
        return stage_input

    def generate(self, stage_input: ShadowInput) -> ShadowResult:
        _ = self.prepare_inputs(stage_input)
        raise NotImplementedError("V2-DIFF shadow model is not connected yet. Implement the model loader and inference path in this class.")
