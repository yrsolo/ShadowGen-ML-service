from __future__ import annotations

from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import ShadowResult
from shadowgen_ml_service.core.stage_io import ShadowInput
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import (
    grayscale_to_nchw_float32_input,
    image_to_nchw_float32_input,
    rgb_to_nchw_float32_input,
    rgba_output_to_image,
    scalar_to_input,
)
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class TritonShadowGenerator(ShadowGenerator):
    backend_name = "triton-shadow"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding, *, model_variant: str) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = model_variant

    def generate(self, stage_input: ShadowInput) -> ShadowResult:
        image = ensure_pil(stage_input.img)
        mask = ensure_pil(stage_input.mask)
        depth = ensure_pil(stage_input.depth)
        normal = ensure_pil(stage_input.normal)
        response = self.client.infer(
            self.binding,
            inputs=[
                image_to_nchw_float32_input(self.binding.inputs["img"].tensor_name, image.convert("RGBA")),
                grayscale_to_nchw_float32_input(self.binding.inputs["mask"].tensor_name, mask.convert("L")),
                grayscale_to_nchw_float32_input(self.binding.inputs["depth"].tensor_name, depth.convert("L")),
                rgb_to_nchw_float32_input(self.binding.inputs["normal"].tensor_name, normal.convert("RGB")),
                scalar_to_input(self.binding.inputs["angle"].tensor_name, stage_input.angle),
                scalar_to_input(self.binding.inputs["elevation"].tensor_name, stage_input.elevation),
                scalar_to_input(self.binding.inputs["softness"].tensor_name, stage_input.softness),
                scalar_to_input(self.binding.inputs["reflection"].tensor_name, stage_input.reflection),
            ],
        )
        return ShadowResult(shadow_rgba=pil_to_asset(rgba_output_to_image(response[self.binding.outputs["shadow"].tensor_name])))
