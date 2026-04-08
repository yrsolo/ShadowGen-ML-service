from __future__ import annotations

import tempfile
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

from shadowgen_ml_service.core.contracts import GeometryEstimator
from shadowgen_ml_service.core.models import GeometryResult
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, as_degrees, extract_attr, import_module, module_available, tensor_to_float


def camera_to_fov_degrees(camera: Any) -> float:
    for candidate in ("vfov", "fov", "camera_fov"):
        try:
            return as_degrees(extract_attr(camera, candidate))
        except Exception:
            continue
    for candidate in ("fov_deg", "vfov_deg", "camera_fov_deg"):
        try:
            return tensor_to_float(extract_attr(camera, candidate))
        except Exception:
            continue
    raise ValueError("camera FOV field not found in GeoCalib output")


def gravity_to_angle_degrees(gravity: Any, *candidates: str) -> float:
    for candidate in candidates:
        try:
            return as_degrees(extract_attr(gravity, candidate))
        except Exception:
            continue
    raise ValueError(f"gravity field not found for {candidates}")


def confidence_from_result(result: Any) -> float:
    for path in ("confidence", "score", "gravity_confidence"):
        try:
            value = extract_attr(result, path)
            return max(0.0, min(tensor_to_float(value), 1.0))
        except Exception:
            continue

    camera = result.get("camera") if isinstance(result, dict) else getattr(result, "camera", None)
    gravity = result.get("gravity") if isinstance(result, dict) else getattr(result, "gravity", None)
    fov = camera_to_fov_degrees(camera) if camera is not None else 60.0
    pitch = gravity_to_angle_degrees(gravity, "pitch") if gravity is not None else 0.0
    roll = gravity_to_angle_degrees(gravity, "roll") if gravity is not None else 0.0

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
        self._module = geocalib_module or import_module("geocalib")
        self.weights = weights
        self.camera_model = camera_model
        self.shared_intrinsics = shared_intrinsics
        self._model = self._module.GeoCalib(weights=weights)
        self.device_label = self._infer_device_label()

    def estimate(self, image: Image.Image) -> GeometryResult:
        prepared = self._prepare_input(image)
        result = self._model.calibrate(prepared, camera_model=self.camera_model, shared_intrinsics=self.shared_intrinsics)
        camera = extract_attr(result, "camera")
        gravity = extract_attr(result, "gravity")
        return GeometryResult(
            camera_fov=camera_to_fov_degrees(camera),
            camera_pitch=gravity_to_angle_degrees(gravity, "pitch"),
            camera_roll=gravity_to_angle_degrees(gravity, "roll"),
            confidence=confidence_from_result(result),
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

    def _infer_device_label(self) -> str:
        if hasattr(self._model, "device"):
            return str(self._model.device)
        if hasattr(self._model, "parameters"):
            try:
                return str(next(self._model.parameters()).device)
            except Exception:
                pass
        return "cpu"


def probe_geocalib() -> RealAdapterProbe:
    available = module_available("geocalib")
    detail = None if available else "requires the geocalib package in the current virtual environment"
    return RealAdapterProbe("GeoCalib", "bootstrap-probe", available, detail)
