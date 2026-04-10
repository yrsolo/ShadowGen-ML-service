from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from shadowgen_ml_service.core.contracts import DepthEstimator
from shadowgen_ml_service.core.models import DepthResult
from shadowgen_ml_service.core.stage_io import DepthInput
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, import_module, module_available
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


def load_transformers_depth_classes() -> tuple[type[Any], type[Any]]:
    auto_processing_module = importlib.import_module("transformers.models.auto.image_processing_auto")
    auto_model_module = importlib.import_module("transformers.models.auto.modeling_auto")
    return auto_processing_module.AutoImageProcessor, auto_model_module.AutoModelForDepthEstimation


def normalize_depth_map(predicted_depth: Any, output_size: tuple[int, int]) -> Image.Image:
    depth = predicted_depth
    if hasattr(depth, "detach"):
        depth = depth.detach()
    if hasattr(depth, "cpu"):
        depth = depth.cpu()
    if hasattr(depth, "numpy"):
        depth_array = depth.numpy()
    else:
        depth_array = np.asarray(depth)
    depth_array = np.asarray(depth_array, dtype=np.float32)
    if depth_array.ndim == 3:
        depth_array = depth_array[0]
    min_value = float(depth_array.min())
    max_value = float(depth_array.max())
    if max_value - min_value < 1e-6:
        normalized = np.zeros_like(depth_array, dtype=np.float32)
    else:
        normalized = (depth_array - min_value) / (max_value - min_value)
    image = Image.fromarray((normalized * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    return image.resize(output_size, Image.Resampling.BILINEAR)


class RealDepthEstimator(DepthEstimator):
    _RESOURCE_CACHE: ClassVar[dict[tuple[str, bool], tuple[Any, Any]]] = {}

    def __init__(
        self,
        transformers_module: ModuleType | None = None,
        torch_module: Any | None = None,
        *,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        target_device: str = "cuda",
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.target_device = target_device
        self.local_files_only = local_files_only
        self._torch = torch_module or import_module("torch")
        self.device_label = self._resolve_device_label()
        if transformers_module is not None:
            processor_cls = getattr(transformers_module, "AutoImageProcessor", None)
            model_cls = getattr(transformers_module, "AutoModelForDepthEstimation", None)
            self._processor = processor_cls.from_pretrained(model_id, local_files_only=local_files_only)
            self._model = model_cls.from_pretrained(model_id, local_files_only=local_files_only)
        else:
            processor_cls, model_cls = load_transformers_depth_classes()
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
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device_label)
        self.device_label = self._infer_device_label()

    def estimate(self, stage_input: DepthInput) -> DepthResult:
        image_rgb = ensure_pil(stage_input.image).convert("RGB")
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device_label)
        elif isinstance(inputs, dict):
            inputs = {key: value.to(self.device_label) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        predicted_depth = outputs.predicted_depth if hasattr(outputs, "predicted_depth") else outputs["predicted_depth"]
        depth_map = normalize_depth_map(predicted_depth, image_rgb.size)
        if stage_input.mask is not None:
            depth_map = Image.composite(depth_map, Image.new("L", depth_map.size, 0), ensure_pil(stage_input.mask).convert("L"))
        return DepthResult(depth_map=pil_to_asset(depth_map))

    def _infer_device_label(self) -> str:
        if hasattr(self._model, "device"):
            return str(self._model.device)
        if hasattr(self._model, "parameters"):
            try:
                return str(next(self._model.parameters()).device)
            except Exception:
                pass
        return "cpu"

    def _resolve_device_label(self) -> str:
        if str(self.target_device).startswith("cuda") and bool(getattr(getattr(self._torch, "cuda", None), "is_available", lambda: False)()):
            return str(self.target_device)
        return "cpu"


def probe_depth_anything() -> RealAdapterProbe:
    available = module_available("torch") and module_available("transformers")
    detail = None if available else "requires torch and transformers"
    return RealAdapterProbe("depth-anything/Depth-Anything-V2-Small-hf", "bootstrap-probe", available, detail)
