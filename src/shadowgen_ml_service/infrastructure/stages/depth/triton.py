from __future__ import annotations

from shadowgen_ml_service.core.contracts import DepthEstimator
from shadowgen_ml_service.core.models import DepthResult
from shadowgen_ml_service.core.stage_io import DepthInput
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import grayscale_output_to_image, image_to_nchw_float32_input, mask_to_nchw_float32_input
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class TritonDepthEstimator(DepthEstimator):
    backend_name = "triton-depth"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def estimate(self, stage_input: DepthInput) -> DepthResult:
        image = ensure_pil(stage_input.image)
        mask = ensure_pil(stage_input.mask)
        response = self.client.infer(
            self.binding,
            inputs=[
                image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, image.convert("RGB")),
                mask_to_nchw_float32_input(self.binding.inputs["mask"].tensor_name, mask.convert("L")),
            ],
        )
        return DepthResult(depth_map=pil_to_asset(grayscale_output_to_image(response[self.binding.outputs["depth"].tensor_name])))
