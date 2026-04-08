from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from shadowgen_ml_service.core.contracts import Segmenter
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, import_module, module_available


def load_transformers_auto_classes() -> tuple[type[Any], type[Any]]:
    auto_processing_module = importlib.import_module("transformers.models.auto.image_processing_auto")
    auto_model_module = importlib.import_module("transformers.models.auto.modeling_auto")
    return auto_processing_module.AutoImageProcessor, auto_model_module.AutoModelForImageSegmentation


def largest_connected_component(mask_array: np.ndarray) -> np.ndarray:
    binary = mask_array.astype(np.uint8)
    height, width = binary.shape
    visited = np.zeros((height, width), dtype=bool)
    largest_component: list[tuple[int, int]] = []
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

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
        self._torch = torch_module or import_module("torch")
        self._transforms = transforms_module or import_module("torchvision.transforms")
        self.device_label = "cpu"
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
            self._processor = processor_cls.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only) if processor_cls is not None else None
            self._model = model_cls.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
        else:
            processor_cls, model_cls = load_transformers_auto_classes()
            cache_key = (model_id, resolution, local_files_only)
            cached = self._RESOURCE_CACHE.get(cache_key)
            if cached is None:
                processor = None
                try:
                    processor = processor_cls.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
                except Exception:
                    processor = None
                model = model_cls.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
                if hasattr(model, "eval"):
                    model.eval()
                cached = (processor, model)
                self._RESOURCE_CACHE[cache_key] = cached
            self._processor, self._model = cached
        if hasattr(self._model, "eval"):
            self._model.eval()
        self.device_label = self._infer_device_label()

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
        alpha_image = Image.fromarray((alpha_array * 255.0).astype(np.uint8)).resize(image_rgb.size, Image.Resampling.BILINEAR)
        alpha_resized = np.asarray(alpha_image, dtype=np.uint8)
        binary_mask = (alpha_resized >= int(self.mask_threshold * 255)).astype(np.uint8)
        binary_mask = largest_connected_component(binary_mask)
        mask_image = Image.fromarray(binary_mask * 255)
        cutout_rgba = image_rgb.convert("RGBA")
        cutout_rgba.putalpha(alpha_image)
        return SegmentationResult(
            bbox=(0, 0, image.width, image.height),
            mask=mask_image,
            cutout_rgba=cutout_rgba,
            crop_rgba=image,
        )

    def _infer_device_label(self) -> str:
        if hasattr(self._model, "device"):
            return str(self._model.device)
        if hasattr(self._model, "parameters"):
            try:
                return str(next(self._model.parameters()).device)
            except Exception:
                pass
        return "cpu"


def probe_birefnet(*, allow_cpu: bool = False) -> RealAdapterProbe:
    dependencies_ready = module_available("torch") and module_available("transformers") and module_available("torchvision")
    if not dependencies_ready:
        return RealAdapterProbe(
            "ZhengPeng7/BiRefNet_lite-matting",
            "bootstrap-probe",
            False,
            "requires torch, torchvision, transformers, and the BiRefNet model files",
        )
    torch_module = import_module("torch")
    has_cuda = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
    available = has_cuda or allow_cpu
    if available:
        detail = "CUDA runtime detected" if has_cuda else "CPU mode explicitly enabled"
    else:
        detail = "BiRefNet real mode is disabled on CPU by default; set SHADOWGEN_BIREFNET_ALLOW_CPU=true to opt in"
    return RealAdapterProbe("ZhengPeng7/BiRefNet_lite-matting", "bootstrap-probe", available, detail)
