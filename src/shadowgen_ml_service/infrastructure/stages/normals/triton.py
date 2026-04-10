from __future__ import annotations

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult
from shadowgen_ml_service.core.stage_io import NormalsInput
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import grayscale_to_nchw_float32_input, image_to_nchw_float32_input, rgb_output_to_image


class TritonNormalEstimator(NormalEstimator):
    backend_name = "triton-normals"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def estimate(self, stage_input: NormalsInput) -> NormalResult:
        inputs = [image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, stage_input.image.convert("RGB"))]
        if stage_input.depth_map is not None and "depth" in self.binding.inputs:
            inputs.append(grayscale_to_nchw_float32_input(self.binding.inputs["depth"].tensor_name, stage_input.depth_map.convert("L")))
        response = self.client.infer(
            self.binding.model_name,
            inputs=inputs,
            outputs=[self.binding.outputs["normal"].tensor_name],
        )
        return NormalResult(normal_map=rgb_output_to_image(response[self.binding.outputs["normal"].tensor_name]))
