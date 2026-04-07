from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class GeometryResult:
    camera_fov: float
    camera_pitch: float
    camera_roll: float
    confidence: float


@dataclass(frozen=True)
class SegmentationResult:
    bbox: BBox
    mask: Image.Image
    cutout_rgba: Image.Image
    crop_rgba: Image.Image


@dataclass(frozen=True)
class DepthResult:
    depth_map: Image.Image


@dataclass(frozen=True)
class NormalResult:
    normal_map: Image.Image


@dataclass(frozen=True)
class ShadowResult:
    shadow_rgba: Image.Image


@dataclass(frozen=True)
class CompositionResult:
    final_image: Image.Image


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    implementation: str
    model_name: str
    model_version: str
    available: bool
    using_mock: bool
    detail: str | None = None


@dataclass(frozen=True)
class RuntimeDescriptor:
    mode: str
    degraded: bool
    components: list[ComponentStatus]
    model_version: str


@dataclass
class CachedPreprocess:
    detection: DetectionResult
    geometry: GeometryResult
    segmentation: SegmentationResult
    depth: DepthResult
    normals: NormalResult
    cache_path: Path | None = None


@dataclass
class RenderResult:
    request_id: str | None
    artifacts: list[dict[str, str]]
    metrics: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    model_info: dict[str, str] = field(default_factory=dict)


@dataclass
class TimedValue:
    value: Any
    elapsed_ms: int
