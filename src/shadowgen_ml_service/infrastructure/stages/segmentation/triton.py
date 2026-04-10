from __future__ import annotations

from shadowgen_ml_service.core.contracts import Segmenter
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import base64_png_to_image, image_to_base64_png


class TritonSegmenter(Segmenter):
    backend_name = "triton-segmenter"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_name

    def segment(self, image) -> SegmentationResult:
        response = self.client.infer_json(
            self.binding.model_name,
            {"image_base64": image_to_base64_png(image.convert("RGBA"))},
        )
        bbox = tuple(int(value) for value in response["bbox"])
        mask = base64_png_to_image(response["mask_base64"], mode="L")
        cutout = base64_png_to_image(response["cutout_base64"], mode="RGBA")
        crop = base64_png_to_image(response.get("crop_base64", response["cutout_base64"]), mode="RGBA")
        return SegmentationResult(bbox=bbox, mask=mask, cutout_rgba=cutout, crop_rgba=crop)
