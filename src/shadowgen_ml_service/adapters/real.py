from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import dataclass
import tempfile
from types import ModuleType
from typing import ClassVar
from typing import Any

import numpy as np
from PIL import Image

from shadowgen_ml_service.pipeline.contracts import Detector, GeometryEstimator, Segmenter
from shadowgen_ml_service.pipeline.types import BBox, DetectionResult, GeometryResult, SegmentationResult

@dataclass(frozen=True)
class RealAdapterProbe:
    model_name: str
    model_version: str
    available: bool
    detail: str | None = None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _import_module(name: str) -> ModuleType:
    return __import__(name, fromlist=["*"])


def _load_grounding_dino_classes() -> tuple[type[Any], type[Any]]:
    processor_module = importlib.import_module(
        "transformers.models.grounding_dino.processing_grounding_dino"
    )
    model_module = importlib.import_module(
        "transformers.models.grounding_dino.modeling_grounding_dino"
    )
    return (
        processor_module.GroundingDinoProcessor,
        model_module.GroundingDinoForObjectDetection,
    )


def _load_transformers_auto_classes() -> tuple[type[Any], type[Any]]:
    auto_processing_module = importlib.import_module(
        "transformers.models.auto.image_processing_auto"
    )
    auto_model_module = importlib.import_module(
        "transformers.models.auto.modeling_auto"
    )
    return (
        auto_processing_module.AutoImageProcessor,
        auto_model_module.AutoModelForImageSegmentation,
    )


def _clamp_bbox(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> BBox:
    width, height = image_size
    left = max(0, min(int(round(box[0])), width - 1))
    top = max(0, min(int(round(box[1])), height - 1))
    right = max(left + 1, min(int(round(box[2])), width))
    bottom = max(top + 1, min(int(round(box[3])), height))
    return (left, top, right, bottom)


def _bbox_area(bbox: BBox) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def select_primary_detection(candidates: list[tuple[BBox, float]], *, confidence_margin: float = 0.03) -> tuple[BBox, float]:
    if not candidates:
        raise ValueError("no detection candidates available")

    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    best_bbox, best_score = ordered[0]
    close_candidates = [
        item for item in ordered
        if abs(item[1] - best_score) <= confidence_margin
    ]
    if len(close_candidates) == 1:
        return best_bbox, best_score
    return max(close_candidates, key=lambda item: (_bbox_area(item[0]), item[1]))


def _to_flat_floats(value: Any, limit: int | None = None) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        raw = value.tolist()
    elif hasattr(value, "__array__"):
        raw = np.asarray(value).tolist()
    else:
        raw = value

    flattened: list[float] = []

    def _collect(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                _collect(child)
            return
        flattened.append(float(item))

    _collect(raw)
    if limit is not None:
        return flattened[:limit]
    return flattened


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
            self._processor = processor_cls.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
            self._model = model_cls.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
        else:
            processor_cls, model_cls = _load_grounding_dino_classes()
            cache_key = (model_id, local_files_only)
            cached = self._RESOURCE_CACHE.get(cache_key)
            if cached is None:
                processor = processor_cls.from_pretrained(
                    model_id,
                    local_files_only=local_files_only,
                )
                model = model_cls.from_pretrained(
                    model_id,
                    local_files_only=local_files_only,
                )
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
        torch_module = self._torch or _import_module("torch")
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
            bbox = _clamp_bbox(tuple(_to_flat_floats(box, limit=4)), image_rgb.size)
            confidence = max(0.0, min(_tensor_to_float(score), 1.0))
            candidates.append((bbox, confidence))
        if not candidates:
            raise ValueError("GroundingDINO produced zero valid boxes after post-processing")

        bbox, confidence = select_primary_detection(candidates)
        return DetectionResult(bbox=bbox, confidence=confidence)


def _largest_connected_component(mask_array: np.ndarray) -> np.ndarray:
    binary = mask_array.astype(np.uint8)
    height, width = binary.shape
    visited = np.zeros((height, width), dtype=bool)
    largest_component: list[tuple[int, int]] = []
    directions = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for start_y in range(height):
        for start_x in range(width):
            if binary[start_y, start_x] == 0 or visited[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            component: list[tuple[int, int]] = []

            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for dy, dx in directions:
                    next_y = y + dy
                    next_x = x + dx
                    if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                        continue
                    if visited[next_y, next_x] or binary[next_y, next_x] == 0:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))

            if len(component) > len(largest_component):
                largest_component = component

    if not largest_component:
        return binary

    result = np.zeros((height, width), dtype=np.uint8)
    for y, x in largest_component:
        result[y, x] = 1
    return result


class RealSegmenter(Segmenter):
    _RESOURCE_CACHE: ClassVar[dict[tuple[str, int, bool], tuple[Any, Any]]] = {}

    def __init__(
        self,
        transformers_module: ModuleType | None = None,
        torch_module: Any | None = None,
        transforms_module: Any | None = None,
        *,
        model_id: str = "ZhengPeng7/BiRefNet_lite-matting",
        resolution: int = 1024,
        mask_threshold: float = 0.5,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.local_files_only = local_files_only
        self._torch = torch_module or _import_module("torch")
        self._transforms = transforms_module or _import_module("torchvision.transforms")
        self._transform_image = self._transforms.Compose(
            [
                self._transforms.Resize((resolution, resolution)),
                self._transforms.ToTensor(),
                self._transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if transformers_module is not None:
            processor_cls = getattr(transformers_module, "AutoImageProcessor", None)
            model_cls = getattr(transformers_module, "AutoModelForImageSegmentation", None)
            self._processor = processor_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_files_only,
            ) if processor_cls is not None else None
            self._model = model_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        else:
            processor_cls, model_cls = _load_transformers_auto_classes()
            cache_key = (model_id, resolution, local_files_only)
            cached = self._RESOURCE_CACHE.get(cache_key)
            if cached is None:
                processor = None
                try:
                    processor = processor_cls.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                except Exception:
                    processor = None
                model = model_cls.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                if hasattr(model, "eval"):
                    model.eval()
                cached = (processor, model)
                self._RESOURCE_CACHE[cache_key] = cached
            self._processor, self._model = cached
        if hasattr(self._model, "eval"):
            self._model.eval()

    def segment(self, image: Image.Image) -> SegmentationResult:
        image_rgb = image.convert("RGB")
        image_tensor = self._transform_image(image_rgb).unsqueeze(0)
        if hasattr(image_tensor, "to"):
            parameter = None
            if hasattr(self._model, "parameters"):
                try:
                    parameter = next(self._model.parameters())
                except Exception:
                    parameter = None
            if parameter is not None:
                image_tensor = image_tensor.to(device=parameter.device, dtype=parameter.dtype)
        with self._torch.no_grad():
            prediction = self._model(image_tensor)[-1].sigmoid().cpu()
        alpha = prediction[0].squeeze()
        alpha_array = alpha.detach().numpy().astype(np.float32)
        alpha_image = Image.fromarray((alpha_array * 255.0).astype(np.uint8)).resize(
            image_rgb.size,
            Image.Resampling.BILINEAR,
        )
        alpha_resized = np.asarray(alpha_image, dtype=np.uint8)
        binary_mask = (alpha_resized >= int(self.mask_threshold * 255)).astype(np.uint8)
        binary_mask = _largest_connected_component(binary_mask)
        mask_image = Image.fromarray(binary_mask * 255)

        cutout_rgba = image_rgb.convert("RGBA")
        cutout_rgba.putalpha(alpha_image)
        return SegmentationResult(
            bbox=(0, 0, image.width, image.height),
            mask=mask_image,
            cutout_rgba=cutout_rgba,
            crop_rgba=image,
        )


def _tensor_to_float(value: Any) -> float:
    if value is None:
        raise ValueError("missing numeric value")
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty numeric sequence")
        return _tensor_to_float(value[0])
    if hasattr(value, "__array__"):
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0])
    return float(value)


def _as_degrees(value: Any) -> float:
    numeric = _tensor_to_float(value)
    if abs(numeric) <= math.pi * 2.5:
        return math.degrees(numeric)
    return numeric


def _extract_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"could not find any of {names}")


def _camera_to_fov_degrees(camera: Any) -> float:
    for candidate in ("vfov", "fov", "camera_fov"):
        try:
            return _as_degrees(_extract_attr(camera, candidate))
        except Exception:
            continue
    for candidate in ("fov_deg", "vfov_deg", "camera_fov_deg"):
        try:
            return _tensor_to_float(_extract_attr(camera, candidate))
        except Exception:
            continue
    raise ValueError("camera FOV field not found in GeoCalib output")


def _gravity_to_angle_degrees(gravity: Any, *candidates: str) -> float:
    for candidate in candidates:
        try:
            return _as_degrees(_extract_attr(gravity, candidate))
        except Exception:
            continue
    raise ValueError(f"gravity field not found for {candidates}")


def _confidence_from_result(result: Any) -> float:
    for path in ("confidence", "score", "gravity_confidence"):
        try:
            value = _extract_attr(result, path)
            confidence = _tensor_to_float(value)
            return max(0.0, min(confidence, 1.0))
        except Exception:
            continue

    camera = result.get("camera") if isinstance(result, dict) else getattr(result, "camera", None)
    gravity = result.get("gravity") if isinstance(result, dict) else getattr(result, "gravity", None)
    fov = _camera_to_fov_degrees(camera) if camera is not None else 60.0
    pitch = _gravity_to_angle_degrees(gravity, "pitch") if gravity is not None else 0.0
    roll = _gravity_to_angle_degrees(gravity, "roll") if gravity is not None else 0.0

    score = 0.9
    score -= min(abs(pitch) / 120.0, 0.25)
    score -= min(abs(roll) / 180.0, 0.2)
    if fov < 10 or fov > 150:
        score -= 0.2
    return max(0.05, min(score, 0.95))


class RealGeometryEstimator(GeometryEstimator):
    def __init__(
        self,
        geocalib_module: ModuleType | None = None,
        *,
        weights: str = "pinhole",
        camera_model: str = "pinhole",
        shared_intrinsics: bool = False,
    ) -> None:
        self._module = geocalib_module or _import_module("geocalib")
        self.weights = weights
        self.camera_model = camera_model
        self.shared_intrinsics = shared_intrinsics
        self._model = self._module.GeoCalib(weights=weights)

    def estimate(self, image: Image.Image) -> GeometryResult:
        prepared = self._prepare_input(image)
        result = self._model.calibrate(
            prepared,
            camera_model=self.camera_model,
            shared_intrinsics=self.shared_intrinsics,
        )
        camera = _extract_attr(result, "camera")
        gravity = _extract_attr(result, "gravity")
        return GeometryResult(
            camera_fov=_camera_to_fov_degrees(camera),
            camera_pitch=_gravity_to_angle_degrees(gravity, "pitch"),
            camera_roll=_gravity_to_angle_degrees(gravity, "roll"),
            confidence=_confidence_from_result(result),
        )

    def _prepare_input(self, image: Image.Image) -> Any:
        image_rgb = image.convert("RGB")
        if hasattr(self._model, "load_image"):
            for candidate in (image_rgb, np.asarray(image_rgb)):
                try:
                    return self._model.load_image(candidate)
                except Exception:
                    continue
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                image_rgb.save(temp_file.name, format="PNG")
                return self._model.load_image(temp_file.name)
        return np.asarray(image_rgb)


def probe_grounding_dino() -> RealAdapterProbe:
    available = _module_available("torch") and _module_available("transformers")
    detail = None if available else "requires torch and transformers"
    return RealAdapterProbe("IDEA-Research/grounding-dino-base", "bootstrap-probe", available, detail)


def probe_geocalib() -> RealAdapterProbe:
    available = _module_available("geocalib")
    detail = None if available else "requires the geocalib package in the current virtual environment"
    return RealAdapterProbe("GeoCalib", "bootstrap-probe", available, detail)


def probe_birefnet(*, allow_cpu: bool = False) -> RealAdapterProbe:
    dependencies_ready = (
        _module_available("torch")
        and _module_available("transformers")
        and _module_available("torchvision")
    )
    if not dependencies_ready:
        return RealAdapterProbe(
            "ZhengPeng7/BiRefNet_lite-matting",
            "bootstrap-probe",
            False,
            "requires torch, torchvision, transformers, and the BiRefNet model files",
        )

    torch_module = _import_module("torch")
    has_cuda = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
    available = has_cuda or allow_cpu
    if available:
        detail = "CUDA runtime detected" if has_cuda else "CPU mode explicitly enabled"
    else:
        detail = "BiRefNet real mode is disabled on CPU by default; set SHADOWGEN_BIREFNET_ALLOW_CPU=true to opt in"
    return RealAdapterProbe("ZhengPeng7/BiRefNet_lite-matting", "bootstrap-probe", available, detail)


def probe_depth_anything() -> RealAdapterProbe:
    available = _module_available("torch") and _module_available("transformers")
    detail = None if available else "requires torch, transformers, and a local Depth Anything integration"
    return RealAdapterProbe("depth-anything/Depth-Anything-V2-Small-hf", "bootstrap-probe", available, detail)
