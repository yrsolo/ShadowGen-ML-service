from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RealAdapterProbe:
    model_name: str
    model_version: str
    available: bool
    detail: str | None = None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def import_module(name: str) -> ModuleType:
    return __import__(name, fromlist=["*"])


def tensor_to_float(value: Any) -> float:
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
        return tensor_to_float(value[0])
    if hasattr(value, "__array__"):
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0])
    return float(value)


def as_degrees(value: Any) -> float:
    numeric = tensor_to_float(value)
    if abs(numeric) <= math.pi * 2.5:
        return math.degrees(numeric)
    return numeric


def extract_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"could not find any of {names}")


def to_flat_floats(value: Any, limit: int | None = None) -> list[float]:
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

    def collect(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
            return
        flattened.append(float(item))

    collect(raw)
    return flattened[:limit] if limit is not None else flattened
