from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image


BBox = tuple[int, int, int, int]
BackendKind = Literal["mock", "local", "triton", "internal"]


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
class ForegroundRefinementResult:
    cutout_rgba: Image.Image


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
class StageBackendId:
    stage_key: str
    backend_kind: BackendKind
    model_variant: str


@dataclass(frozen=True)
class StageBackendDescriptor:
    backend_id: StageBackendId
    model_name: str
    model_version: str
    available: bool
    detail: str | None = None
    device: str = "cpu"
    endpoint: str | None = None
    supports_batching: bool = False
    supports_async: bool = False
    is_default: bool = False

    @property
    def stage_key(self) -> str:
        return self.backend_id.stage_key

    @property
    def backend_kind(self) -> BackendKind:
        return self.backend_id.backend_kind

    @property
    def model_variant(self) -> str:
        return self.backend_id.model_variant


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    implementation: str
    model_name: str
    model_version: str
    available: bool
    using_mock: bool
    detail: str | None = None
    backend_kind: BackendKind = "mock"
    model_variant: str = "default"
    device: str = "cpu"
    endpoint: str | None = None
    supports_batching: bool = False
    supports_async: bool = False
    fallback_reason: str | None = None
    backends: list[StageBackendDescriptor] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeDescriptor:
    mode: str
    degraded: bool
    components: list[ComponentStatus]
    model_version: str
    execution_default_backend: BackendKind = "local"
    async_enabled: bool = False


@dataclass
class PreprocessSnapshot:
    detection: DetectionResult
    geometry: GeometryResult
    segmentation: SegmentationResult
    depth: DepthResult
    normals: NormalResult
    foreground_refinement: ForegroundRefinementResult | None = None
    cache_path: Path | None = None


@dataclass(frozen=True)
class HealthStatus:
    status: str
    service_version: str
    active_backend_mode: str
    async_enabled: bool = False


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
    execution_default_backend: BackendKind = "local"
    async_enabled: bool = False
    components: list[ComponentStatus] = field(default_factory=list)
