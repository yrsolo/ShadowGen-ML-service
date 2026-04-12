from __future__ import annotations

from PIL import Image

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
from shadowgen_ml_service.utils.images import bbox_from_mask, create_cutout, ensure_pil, pil_to_asset


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
        images = [ensure_pil(stage_input.image).convert("RGBA") for stage_input in stage_inputs]
        batched_inputs = batch_input_tensors(
            [
                [image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, image.convert("RGB"))]
                for image in images
            ]
        )
        response = self.client.infer(self.binding, inputs=batched_inputs)
        batch_size = len(stage_inputs)
        mask_tensors = split_output_tensor(response[self.binding.outputs["mask"].tensor_name], batch_size)
        bbox_tensors = self._optional_batched_output("bbox", response, batch_size)
        cutout_tensors = self._optional_batched_output("cutout", response, batch_size)
        crop_tensors = self._optional_batched_output("crop", response, batch_size)

        results: list[SegmentationResult] = []
        for index, (image, mask_tensor) in enumerate(zip(images, mask_tensors, strict=True)):
            mask_image = mask_output_to_image(mask_tensor)
            if mask_image.size != image.size:
                mask_image = mask_image.resize(image.size, Image.Resampling.BILINEAR)
            if bbox_tensors is not None:
                bbox = tensor_to_bbox(bbox_tensors[index])
            else:
                bbox = (0, 0, image.width, image.height)
                inferred_bbox = bbox_from_mask(mask_image, padding_px=0)
                if inferred_bbox != bbox:
                    bbox = inferred_bbox
            if cutout_tensors is not None:
                cutout_image = rgba_output_to_image(cutout_tensors[index])
                if cutout_image.size != image.size:
                    cutout_image = cutout_image.resize(image.size, Image.Resampling.BILINEAR)
            else:
                cutout_image = create_cutout(image, mask_image)
            if crop_tensors is not None:
                crop_image = rgba_output_to_image(crop_tensors[index])
                if crop_image.size != image.size:
                    crop_image = crop_image.resize(image.size, Image.Resampling.BILINEAR)
            else:
                crop_image = image
            results.append(
                SegmentationResult(
                    bbox=bbox,
                    mask=pil_to_asset(mask_image),
                    cutout_rgba=pil_to_asset(cutout_image),
                    crop_rgba=pil_to_asset(crop_image),
                )
            )
        return results

    def _optional_batched_output(self, alias: str, response: dict, batch_size: int):
        output_binding = self.binding.outputs.get(alias)
        if output_binding is None:
            return None
        tensor = response.get(output_binding.tensor_name)
        if tensor is None:
            return None
        return split_output_tensor(tensor, batch_size)
