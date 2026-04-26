from __future__ import annotations

from abc import ABC, abstractmethod

from shadowgen_ml_service.core.assets import RasterAsset
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
from shadowgen_ml_service.core.stage_io import DetectionInput, DepthInput, NormalsInput, SegmentationInput, ShadowInput


class Detector(ABC):
    @abstractmethod
    def detect(self, stage_input: DetectionInput) -> DetectionResult:
        raise NotImplementedError


class GeometryEstimator(ABC):
    @abstractmethod
    def estimate(self, image: RasterAsset) -> GeometryResult:
        raise NotImplementedError


class Segmenter(ABC):
    @abstractmethod
    def segment(self, stage_input: SegmentationInput) -> SegmentationResult:
        raise NotImplementedError


class ForegroundColorEstimator(ABC):
    @abstractmethod
    def refine(self, image: RasterAsset, alpha: RasterAsset) -> ForegroundRefinementResult:
        raise NotImplementedError


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, stage_input: DepthInput) -> DepthResult:
        raise NotImplementedError


class NormalEstimator(ABC):
    @abstractmethod
    def estimate(self, stage_input: NormalsInput) -> NormalResult:
        raise NotImplementedError


class ShadowGenerator(ABC):
    @abstractmethod
    def generate(self, stage_input: ShadowInput) -> ShadowResult:
        raise NotImplementedError


class Composer(ABC):
    @abstractmethod
    def compose(
        self,
        cutout_rgba: RasterAsset,
        shadow_image: RasterAsset,
        background: BackgroundSpec,
        output: OutputSpec,
    ) -> CompositionResult:
        raise NotImplementedError


class ArtifactEncoder(ABC):
    @abstractmethod
    def encode(
        self,
        final_image: RasterAsset,
        output_format: str,
        debug_images: dict[str, RasterAsset],
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
    def build(self, stage_key: str, stage_value: object, context: object) -> dict[str, RasterAsset]:
        raise NotImplementedError
