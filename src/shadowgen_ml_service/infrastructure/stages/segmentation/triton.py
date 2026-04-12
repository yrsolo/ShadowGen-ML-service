from __future__ import annotations

from shadowgen_ml_service.core.contracts import Segmenter
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.core.stage_io import SegmentationInput
from shadowgen_ml_service.infrastructure.backends.triton.batching import TritonStageBatchCoordinator
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import (
    batch_input_tensors,
    image_to_nchw_float32_input,
    mask_output_to_image,
    rgba_output_to_image,
    split_output_tensor,
    tensor_to_bbox,
)
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class TritonSegmenter(Segmenter):
    backend_name = "triton-segmenter"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding, batcher: TritonStageBatchCoordinator | None = None) -> None:
        self.client = client
        self.binding = binding
        self.batcher = batcher
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def segment(self, stage_input: SegmentationInput) -> SegmentationResult:
        if self.batcher is None:
            return self._segment_many([stage_input])[0]
        return self.batcher.execute(
            stage_key="segmenter",
            model_variant=self.model_variant,
            payload=stage_input,
            run_batch=self._segment_many,
        )

    def _segment_many(self, stage_inputs: list[SegmentationInput]) -> list[SegmentationResult]:
        batched_inputs = batch_input_tensors(
            [
                [image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, ensure_pil(stage_input.image).convert("RGB"))]
                for stage_input in stage_inputs
            ]
        )
        response = self.client.infer(self.binding, inputs=batched_inputs)
        batch_size = len(stage_inputs)
        bbox_tensors = split_output_tensor(response[self.binding.outputs["bbox"].tensor_name], batch_size)
        mask_tensors = split_output_tensor(response[self.binding.outputs["mask"].tensor_name], batch_size)
        cutout_tensors = split_output_tensor(response[self.binding.outputs["cutout"].tensor_name], batch_size)
        crop_tensors = split_output_tensor(response[self.binding.outputs["crop"].tensor_name], batch_size)
        return [
            SegmentationResult(
                bbox=tensor_to_bbox(bbox_tensor),
                mask=pil_to_asset(mask_output_to_image(mask_tensor)),
                cutout_rgba=pil_to_asset(rgba_output_to_image(cutout_tensor)),
                crop_rgba=pil_to_asset(rgba_output_to_image(crop_tensor)),
            )
            for bbox_tensor, mask_tensor, cutout_tensor, crop_tensor in zip(
                bbox_tensors,
                mask_tensors,
                cutout_tensors,
                crop_tensors,
                strict=True,
            )
        ]
