from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

@dataclass(frozen=True)
class DetectionInput:
    image: Image.Image
    padding_px: int


@dataclass(frozen=True)
class SegmentationInput:
    image: Image.Image


@dataclass(frozen=True)
class DepthInput:
    image: Image.Image
    mask: Image.Image


@dataclass(frozen=True)
class NormalsInput:
    image: Image.Image
    depth_map: Image.Image | None = None


@dataclass(frozen=True)
class ShadowInput:
    img: Image.Image
    mask: Image.Image
    depth: Image.Image
    normal: Image.Image
    angle: float
    elevation: float
    softness: float
    reflection: float
    opacity: float
