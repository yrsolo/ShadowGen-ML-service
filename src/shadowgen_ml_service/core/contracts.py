from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from shadowgen_ml_service.core.commands import BackgroundSpec, OutputSpec, ShadowSpec
from shadowgen_ml_service.core.models import (
    CompositionResult,
    DepthResult,
    DetectionResult,
    EncodedArtifact,
    ForegroundRefinementResult,
    GeometryResult,
    NormalResult,
    PreprocessSnapshot,
    SegmentationResult,
    ShadowResult,
)


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


class ForegroundColorEstimator(ABC):
    @abstractmethod
    def refine(self, image: Image.Image, alpha: Image.Image) -> ForegroundRefinementResult:
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
        shadow: ShadowSpec,
    ) -> ShadowResult:
        raise NotImplementedError


class Composer(ABC):
    @abstractmethod
    def compose(
        self,
        cutout_rgba: Image.Image,
        shadow_rgba: Image.Image,
        background: BackgroundSpec,
        output: OutputSpec,
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
    ) -> list[EncodedArtifact]:
        raise NotImplementedError


class PreprocessCacheRepository(ABC):
    @abstractmethod
    def make_key(self, raw_bytes: bytes, runtime_signature: str, padding_px: int, working_size: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self, key: str) -> PreprocessSnapshot | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, key: str, snapshot: PreprocessSnapshot) -> None:
        raise NotImplementedError


class PreviewBuilderRegistry(ABC):
    @abstractmethod
    def build(self, stage_key: str, stage_value: object, context: object) -> dict[str, Image.Image]:
        raise NotImplementedError
