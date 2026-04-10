from __future__ import annotations

from shadowgen_ml_service.core.contracts import DepthEstimator
from shadowgen_ml_service.core.models import DepthResult
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import base64_png_to_image, image_to_base64_png


class TritonDepthEstimator(DepthEstimator):
    backend_name = "triton-depth"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_name

    def estimate(self, image, mask) -> DepthResult:
        response = self.client.infer_json(
            self.binding.model_name,
            {
                "image_base64": image_to_base64_png(image.convert("RGBA")),
                "mask_base64": image_to_base64_png(mask.convert("L")),
            },
        )
        return DepthResult(depth_map=base64_png_to_image(response["depth_base64"], mode="L"))
