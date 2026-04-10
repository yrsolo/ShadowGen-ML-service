from __future__ import annotations

from shadowgen_ml_service.core.contracts import DepthEstimator
from shadowgen_ml_service.core.models import DepthResult
from shadowgen_ml_service.core.stage_io import DepthInput
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import grayscale_output_to_image, image_to_nchw_float32_input, mask_to_nchw_float32_input


class TritonDepthEstimator(DepthEstimator):
    backend_name = "triton-depth"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def estimate(self, stage_input: DepthInput) -> DepthResult:
        response = self.client.infer(
            self.binding.model_name,
            inputs=[
                image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, stage_input.image.convert("RGB")),
                mask_to_nchw_float32_input(self.binding.inputs["mask"].tensor_name, stage_input.mask.convert("L")),
            ],
            outputs=[self.binding.outputs["depth"].tensor_name],
        )
        return DepthResult(depth_map=grayscale_output_to_image(response[self.binding.outputs["depth"].tensor_name]))
