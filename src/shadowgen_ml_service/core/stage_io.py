from __future__ import annotations

from dataclasses import dataclass

from shadowgen_ml_service.core.assets import RasterAsset

@dataclass(frozen=True)
class DetectionInput:
    image: RasterAsset
    padding_px: int


@dataclass(frozen=True)
class SegmentationInput:
    image: RasterAsset


@dataclass(frozen=True)
class DepthInput:
    image: RasterAsset
    mask: RasterAsset


@dataclass(frozen=True)
class NormalsInput:
    image: RasterAsset
    depth_map: RasterAsset | None = None


@dataclass(frozen=True)
class ShadowInput:
    img: RasterAsset
    mask: RasterAsset
    depth: RasterAsset
    normal: RasterAsset
    angle: float
    elevation: float
    softness: float
    reflection: float
    opacity: float
