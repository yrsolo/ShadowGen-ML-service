from __future__ import annotations

from shadowgen_ml_service.core.contracts import Detector
from shadowgen_ml_service.core.models import DetectionResult
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import image_to_base64_png


class TritonDetector(Detector):
    backend_name = "triton-detector"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_name

    def detect(self, image, padding_px: int) -> DetectionResult:
        response = self.client.infer_json(
            self.binding.model_name,
            {"image_base64": image_to_base64_png(image.convert("RGB")), "padding_px": padding_px},
        )
        bbox = tuple(int(value) for value in response["bbox"])
        return DetectionResult(bbox=bbox, confidence=float(response["confidence"]))
