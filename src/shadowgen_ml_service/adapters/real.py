from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from io import BytesIO
import tempfile
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

from shadowgen_ml_service.pipeline.contracts import GeometryEstimator
from shadowgen_ml_service.pipeline.types import GeometryResult

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


def probe_birefnet() -> RealAdapterProbe:
    available = _module_available("torch")
    detail = None if available else "requires torch and a local BiRefNet integration"
    return RealAdapterProbe("ZhengPeng7/BiRefNet_lite-matting", "bootstrap-probe", available, detail)


def probe_depth_anything() -> RealAdapterProbe:
    available = _module_available("torch") and _module_available("transformers")
    detail = None if available else "requires torch, transformers, and a local Depth Anything integration"
    return RealAdapterProbe("depth-anything/Depth-Anything-V2-Small-hf", "bootstrap-probe", available, detail)
