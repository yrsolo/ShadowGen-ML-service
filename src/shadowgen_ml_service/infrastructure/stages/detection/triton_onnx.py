from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps

from shadowgen_ml_service.core.contracts import Detector
from shadowgen_ml_service.core.models import DetectionResult
from shadowgen_ml_service.core.stage_io import DetectionInput
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import tensor_input
from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import (
    clamp_bbox,
    load_grounding_dino_classes,
    post_process_kwargs,
    select_primary_detection,
)
from shadowgen_ml_service.utils.images import ensure_pil


class TritonOnnxGroundingDinoDetector(Detector):
    backend_name = "triton-onnx-grounding-dino"

    def __init__(
        self,
        client: TritonInferenceClient,
        binding: TritonModelBinding,
        *,
        model_id: str,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        self.client = client
        self.binding = binding
        self.model_id = model_id
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device_label = "triton"
        self.model_variant = binding.model_variant
        processor_cls, _ = load_grounding_dino_classes()
        self.processor = processor_cls.from_pretrained(model_id)

    def detect(self, stage_input: DetectionInput) -> DetectionResult:
        image_rgb = ensure_pil(stage_input.image).convert("RGB")
        infer_image, content_bbox = self._letterbox_square(image_rgb)
        encoded = self.processor(images=infer_image, text=self.prompt, return_tensors="pt")
        inputs = []
        for alias, tensor_binding in self.binding.inputs.items():
            value = encoded.get(alias)
            if value is None:
                raise RuntimeError(f"GroundingDINO ONNX processor did not produce input {alias}")
            array = value.detach().cpu().numpy()
            inputs.append(tensor_input(tensor_binding.tensor_name, array, datatype=tensor_binding.datatype))
        response = self.client.infer(self.binding, inputs=inputs)
        outputs = SimpleNamespace(
            logits=torch.from_numpy(np.asarray(response[self.binding.outputs["logits"].tensor_name])),
            pred_boxes=torch.from_numpy(np.asarray(response[self.binding.outputs["pred_boxes"].tensor_name])),
        )
        kwargs = post_process_kwargs(
            self.processor,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=encoded.get("input_ids"),
            target_sizes=[infer_image.size[::-1]],
            **kwargs,
        )
        candidates = []
        for box, score in zip(processed[0].get("boxes", []), processed[0].get("scores", []), strict=False):
            bbox = self._bbox_from_square(tuple(float(value) for value in box.detach().cpu().tolist()), content_bbox, image_rgb.size)
            candidates.append((bbox, float(score.detach().cpu().item())))
        bbox, confidence = select_primary_detection(candidates)
        return DetectionResult(bbox=bbox, confidence=confidence)

    def _letterbox_square(self, image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
        size = max(image.size)
        canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
        contained = ImageOps.contain(image, (size, size), method=Image.Resampling.BILINEAR)
        left = (size - contained.width) // 2
        top = (size - contained.height) // 2
        canvas.paste(contained, (left, top))
        return canvas, (left, top, left + contained.width, top + contained.height)

    def _bbox_from_square(
        self,
        bbox: tuple[float, float, float, float],
        content_bbox: tuple[int, int, int, int],
        target_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = content_bbox
        content_width = max(1, right - left)
        content_height = max(1, bottom - top)
        scale_x = target_size[0] / content_width
        scale_y = target_size[1] / content_height
        mapped = (
            (bbox[0] - left) * scale_x,
            (bbox[1] - top) * scale_y,
            (bbox[2] - left) * scale_x,
            (bbox[3] - top) * scale_y,
        )
        return clamp_bbox(mapped, target_size)
