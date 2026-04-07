from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from os import getenv
from pathlib import Path


@dataclass(frozen=True)
class RealAdapterProbe:
    model_name: str
    model_version: str
    available: bool
    detail: str | None = None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _path_available(env_name: str) -> bool:
    raw = getenv(env_name)
    if not raw:
        return False
    return Path(raw).exists()


def probe_grounding_dino() -> RealAdapterProbe:
    available = _module_available("torch") and _module_available("transformers") and _path_available("SHADOWGEN_GROUNDING_DINO_PATH")
    detail = None if available else "requires torch, transformers, and SHADOWGEN_GROUNDING_DINO_PATH"
    return RealAdapterProbe("IDEA-Research/grounding-dino-base", "bootstrap-probe", available, detail)


def probe_geocalib() -> RealAdapterProbe:
    available = _module_available("torch") and _path_available("SHADOWGEN_GEOCALIB_PATH")
    detail = None if available else "requires torch and SHADOWGEN_GEOCALIB_PATH"
    return RealAdapterProbe("GeoCalib", "bootstrap-probe", available, detail)


def probe_birefnet() -> RealAdapterProbe:
    available = _module_available("torch") and _path_available("SHADOWGEN_BIREFNET_PATH")
    detail = None if available else "requires torch and SHADOWGEN_BIREFNET_PATH"
    return RealAdapterProbe("ZhengPeng7/BiRefNet_lite-matting", "bootstrap-probe", available, detail)


def probe_depth_anything() -> RealAdapterProbe:
    available = _module_available("torch") and _module_available("transformers") and _path_available("SHADOWGEN_DEPTH_ANYTHING_PATH")
    detail = None if available else "requires torch, transformers, and SHADOWGEN_DEPTH_ANYTHING_PATH"
    return RealAdapterProbe("depth-anything/Depth-Anything-V2-Small-hf", "bootstrap-probe", available, detail)
