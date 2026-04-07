from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from shadowgen_ml_service.pipeline.types import (
    CompositionResult,
    DepthResult,
    DetectionResult,
    GeometryResult,
    NormalResult,
    SegmentationResult,
    ShadowResult,
)
from shadowgen_ml_service.schemas import BackgroundPayload, OutputPayload, ShadowPayload


class Detector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image, padding_px: int) -> DetectionResult:
        raise NotImplementedError


class GeometryEstimator(ABC):
    @abstractmethod
    def estimate(self, image: Image.Image) -> GeometryResult:
        raise NotImplementedError


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image: Image.Image) -> SegmentationResult:
        raise NotImplementedError


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, image: Image.Image, mask: Image.Image) -> DepthResult:
        raise NotImplementedError


class NormalEstimator(ABC):
    @abstractmethod
    def estimate(self, depth_map: Image.Image) -> NormalResult:
        raise NotImplementedError


class ShadowGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        mask: Image.Image,
        depth_map: Image.Image,
        normal_map: Image.Image,
        geometry: GeometryResult,
        shadow: ShadowPayload,
    ) -> ShadowResult:
        raise NotImplementedError


class Composer(ABC):
    @abstractmethod
    def compose(
        self,
        cutout_rgba: Image.Image,
        shadow_rgba: Image.Image,
        background: BackgroundPayload,
        output: OutputPayload,
    ) -> CompositionResult:
        raise NotImplementedError


class ArtifactEncoder(ABC):
    @abstractmethod
    def encode(
        self,
        final_image: Image.Image,
        output_format: str,
        debug_images: dict[str, Image.Image],
        return_debug: bool,
    ) -> list[dict[str, str]]:
        raise NotImplementedError
