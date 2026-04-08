from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, ClassVar

from PIL import Image

from shadowgen_ml_service.core.contracts import Detector
from shadowgen_ml_service.core.models import BBox, DetectionResult
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, import_module, module_available, tensor_to_float, to_flat_floats


def load_grounding_dino_classes() -> tuple[type[Any], type[Any]]:
    processor_module = importlib.import_module("transformers.models.grounding_dino.processing_grounding_dino")
    model_module = importlib.import_module("transformers.models.grounding_dino.modeling_grounding_dino")
    return processor_module.GroundingDinoProcessor, model_module.GroundingDinoForObjectDetection


def clamp_bbox(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> BBox:
    width, height = image_size
    left = max(0, min(int(round(box[0])), width - 1))
    top = max(0, min(int(round(box[1])), height - 1))
    right = max(left + 1, min(int(round(box[2])), width))
    bottom = max(top + 1, min(int(round(box[3])), height))
    return (left, top, right, bottom)


def bbox_area(bbox: BBox) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def select_primary_detection(candidates: list[tuple[BBox, float]], *, confidence_margin: float = 0.03) -> tuple[BBox, float]:
    if not candidates:
        raise ValueError("no detection candidates available")
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    best_bbox, best_score = ordered[0]
    close_candidates = [item for item in ordered if abs(item[1] - best_score) <= confidence_margin]
    if len(close_candidates) == 1:
        return best_bbox, best_score
    return max(close_candidates, key=lambda item: (bbox_area(item[0]), item[1]))


class RealDetector(Detector):
    _RESOURCE_CACHE: ClassVar[dict[tuple[str, bool], tuple[Any, Any]]] = {}

    def __init__(
        self,
        transformers_module: ModuleType | None = None,
        torch_module: Any | None = None,
        *,
        model_id: str = "IDEA-Research/grounding-dino-base",
        prompt: str = "object.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.local_files_only = local_files_only
        self._torch = torch_module
        if transformers_module is not None:
            processor_cls = transformers_module.GroundingDinoProcessor
            model_cls = transformers_module.GroundingDinoForObjectDetection
            self._processor = processor_cls.from_pretrained(model_id, local_files_only=local_files_only)
            self._model = model_cls.from_pretrained(model_id, local_files_only=local_files_only)
        else:
            processor_cls, model_cls = load_grounding_dino_classes()
            cache_key = (model_id, local_files_only)
            cached = self._RESOURCE_CACHE.get(cache_key)
            if cached is None:
                processor = processor_cls.from_pretrained(model_id, local_files_only=local_files_only)
                model = model_cls.from_pretrained(model_id, local_files_only=local_files_only)
                if hasattr(model, "eval"):
                    model.eval()
                cached = (processor, model)
                self._RESOURCE_CACHE[cache_key] = cached
            self._processor, self._model = cached
        if hasattr(self._model, "eval"):
            self._model.eval()

    def detect(self, image: Image.Image, padding_px: int) -> DetectionResult:
        image_rgb = image.convert("RGB")
        inputs = self._processor(images=image_rgb, text=self.prompt, return_tensors="pt")
        torch_module = self._torch or import_module("torch")
        with torch_module.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image_rgb.size[::-1]],
        )
        if not results:
            raise ValueError("GroundingDINO returned no detection result")
        result = results[0]
        boxes = result.get("boxes") if isinstance(result, dict) else getattr(result, "boxes", None)
        scores = result.get("scores") if isinstance(result, dict) else getattr(result, "scores", None)
        if boxes is None or scores is None:
            raise ValueError("GroundingDINO output does not contain boxes and scores")
        candidates: list[tuple[BBox, float]] = []
        for box, score in zip(boxes, scores):
            bbox = clamp_bbox(tuple(to_flat_floats(box, limit=4)), image_rgb.size)
            confidence = max(0.0, min(tensor_to_float(score), 1.0))
            candidates.append((bbox, confidence))
        bbox, confidence = select_primary_detection(candidates)
        return DetectionResult(bbox=bbox, confidence=confidence)


def probe_grounding_dino() -> RealAdapterProbe:
    available = module_available("torch") and module_available("transformers")
    detail = None if available else "requires torch and transformers"
    return RealAdapterProbe("IDEA-Research/grounding-dino-base", "bootstrap-probe", available, detail)
