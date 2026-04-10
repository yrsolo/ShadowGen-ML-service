from __future__ import annotations

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import base64_png_to_image, image_to_base64_png


class TritonNormalEstimator(NormalEstimator):
    backend_name = "triton-normals"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_name

    def estimate(self, image, depth_map=None) -> NormalResult:
        payload = {"image_base64": image_to_base64_png(image.convert("RGBA"))}
        if depth_map is not None:
            payload["depth_base64"] = image_to_base64_png(depth_map.convert("L"))
        response = self.client.infer_json(self.binding.model_name, payload)
        return NormalResult(normal_map=base64_png_to_image(response["normal_base64"], mode="RGB"))
