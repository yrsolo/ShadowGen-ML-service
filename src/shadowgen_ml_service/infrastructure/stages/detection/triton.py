from __future__ import annotations

from shadowgen_ml_service.core.contracts import Detector
from shadowgen_ml_service.core.models import DetectionResult
from shadowgen_ml_service.core.stage_io import DetectionInput
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import image_to_nchw_float32_input, scalar_to_input, tensor_to_bbox, tensor_to_scalar
from shadowgen_ml_service.utils.images import ensure_pil


class TritonDetector(Detector):
    backend_name = "triton-detector"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def detect(self, stage_input: DetectionInput) -> DetectionResult:
        image = ensure_pil(stage_input.image)
        response = self.client.infer(
            self.binding,
            inputs=[
                image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, image.convert("RGB")),
                scalar_to_input(self.binding.inputs["padding_px"].tensor_name, int(stage_input.padding_px), datatype=self.binding.inputs["padding_px"].datatype),
            ],
        )
        bbox = tensor_to_bbox(response[self.binding.outputs["bbox"].tensor_name])
        return DetectionResult(
            bbox=bbox,
            confidence=tensor_to_scalar(response[self.binding.outputs["confidence"].tensor_name]),
        )
