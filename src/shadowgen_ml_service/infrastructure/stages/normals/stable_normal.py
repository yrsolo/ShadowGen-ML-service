from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, import_module, module_available


class StableNormalEstimator(NormalEstimator):
    _RESOURCE_CACHE: ClassVar[dict[tuple[str, str, str, int, bool], Any]] = {}

    def __init__(
        self,
        torch_module: Any | None = None,
        *,
        model_variant: str = "StableNormal_turbo",
        target_device: str = "cuda",
        cache_dir: str | Path | None = None,
        resolution: int = 1024,
        allow_cpu: bool = False,
        match_input_resolution: bool = True,
    ) -> None:
        self.model_variant = model_variant
        self.target_device = target_device
        self.resolution = resolution
        self.allow_cpu = allow_cpu
        self.match_input_resolution = match_input_resolution
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._torch = torch_module or import_module("torch")
        self.device_label = self._resolve_device_label()
        self.backend_name = "stable-normal"

        cache_key = (
            model_variant,
            self.device_label,
            str(self.cache_dir or ""),
            resolution,
            match_input_resolution,
        )
        if torch_module is not None:
            predictor = self._load_predictor()
        else:
            predictor = self._RESOURCE_CACHE.get(cache_key)
            if predictor is None:
                predictor = self._load_predictor()
                self._RESOURCE_CACHE[cache_key] = predictor
        self._predictor = predictor

    def estimate(self, image: Image.Image, depth_map: Image.Image | None = None) -> NormalResult:
        source = image if image.mode == "RGBA" else image.convert("RGBA")
        normal_map = self._predictor(
            source,
            resolution=self.resolution,
            match_input_resolution=self.match_input_resolution,
            data_type="object",
        )
        if not isinstance(normal_map, Image.Image):
            normal_map = Image.fromarray(normal_map)
        if normal_map.mode != "RGB":
            normal_map = normal_map.convert("RGB")
        return NormalResult(normal_map=normal_map)

    def _load_predictor(self):
        kwargs = {
            "trust_repo": True,
            "device": self.device_label,
        }
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["local_cache_dir"] = str(self.cache_dir)
        return self._torch.hub.load("Stable-X/StableNormal", self.model_variant, **kwargs)

    def _resolve_device_label(self) -> str:
        if str(self.target_device).startswith("cuda") and bool(getattr(getattr(self._torch, "cuda", None), "is_available", lambda: False)()):
            return str(self.target_device)
        if self.allow_cpu:
            return "cpu"
        raise RuntimeError("StableNormal real mode requires CUDA by default; set SHADOWGEN_STABLE_NORMAL_ALLOW_CPU=true to opt in")


def probe_stable_normal(*, allow_cpu: bool = False, target_device: str = "cuda") -> RealAdapterProbe:
    dependencies_ready = all(
        module_available(name)
        for name in ("torch", "torchvision", "transformers", "diffusers")
    )
    if not dependencies_ready:
        return RealAdapterProbe(
            "Stable-X/StableNormal",
            "bootstrap-probe",
            False,
            "requires torch, torchvision, transformers, and diffusers",
        )

    torch_module = import_module("torch")
    has_cuda = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
    wants_cuda = str(target_device).startswith("cuda")
    available = has_cuda if wants_cuda else (has_cuda or allow_cpu)
    if available:
        detail = "CUDA runtime detected" if has_cuda else "CPU mode explicitly enabled"
    else:
        detail = "StableNormal real mode is disabled on CPU by default; set SHADOWGEN_STABLE_NORMAL_ALLOW_CPU=true to opt in"
    return RealAdapterProbe("Stable-X/StableNormal", "bootstrap-probe", available, detail)
