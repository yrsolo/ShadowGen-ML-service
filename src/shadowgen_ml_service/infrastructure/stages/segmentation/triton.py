from __future__ import annotations

from shadowgen_ml_service.core.contracts import Segmenter
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.core.stage_io import SegmentationInput
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import image_to_nchw_float32_input, mask_output_to_image, rgba_output_to_image, tensor_to_bbox
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class TritonSegmenter(Segmenter):
    backend_name = "triton-segmenter"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def segment(self, stage_input: SegmentationInput) -> SegmentationResult:
        image = ensure_pil(stage_input.image)
        response = self.client.infer(
            self.binding,
            inputs=[image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, image.convert("RGB"))],
        )
        bbox = tensor_to_bbox(response[self.binding.outputs["bbox"].tensor_name])
        mask = mask_output_to_image(response[self.binding.outputs["mask"].tensor_name])
        cutout = rgba_output_to_image(response[self.binding.outputs["cutout"].tensor_name])
        crop = rgba_output_to_image(response[self.binding.outputs["crop"].tensor_name])
        return SegmentationResult(
            bbox=bbox,
            mask=pil_to_asset(mask),
            cutout_rgba=pil_to_asset(cutout),
            crop_rgba=pil_to_asset(crop),
        )
