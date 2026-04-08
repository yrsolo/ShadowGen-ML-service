from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
class EncodedArtifact:
    name: str
    kind: str
    mime_type: str
    image_base64: str


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
class PreprocessSnapshot:
    detection: DetectionResult
    geometry: GeometryResult
    segmentation: SegmentationResult
    depth: DepthResult
    normals: NormalResult
    cache_path: Path | None = None


@dataclass(frozen=True)
class HealthStatus:
    status: str
    service_version: str
    active_backend_mode: str


@dataclass
class CapabilitiesInfo:
    service_version: str
    model_version: str
    supported_input_mime_types: tuple[str, ...]
    supported_output_formats: tuple[str, ...]
    supports_debug_artifacts: bool
    max_image_bytes: int
    active_backend_mode: str
    degraded: bool
    components: list[ComponentStatus] = field(default_factory=list)
