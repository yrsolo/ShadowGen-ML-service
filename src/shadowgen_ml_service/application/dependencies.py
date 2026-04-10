from __future__ import annotations

from dataclasses import dataclass, field

from shadowgen_ml_service.core.contracts import (
    ArtifactEncoder,
    Composer,
    DepthEstimator,
    Detector,
    ForegroundColorEstimator,
    GeometryEstimator,
    NormalEstimator,
    PreprocessCacheRepository,
    PreviewBuilderRegistry,
    Segmenter,
    ShadowGenerator,
)
from shadowgen_ml_service.core.models import RuntimeDescriptor, StageBackendDescriptor, StageBackendId


@dataclass
class RegisteredStageBackend:
    descriptor: StageBackendDescriptor
    handler: object | None


@dataclass
class PipelineBackendRegistry:
    backends: dict[StageBackendId, RegisteredStageBackend] = field(default_factory=dict)
    defaults: dict[str, StageBackendId] = field(default_factory=dict)

    def register(self, descriptor: StageBackendDescriptor, handler: object | None) -> None:
        self.backends[descriptor.backend_id] = RegisteredStageBackend(descriptor=descriptor, handler=handler)

    def set_default(self, stage_key: str, backend_id: StageBackendId) -> None:
        self.defaults[stage_key] = backend_id

    def get(self, backend_id: StageBackendId) -> RegisteredStageBackend | None:
        return self.backends.get(backend_id)

    def list_stage(self, stage_key: str) -> list[RegisteredStageBackend]:
        return [item for item in self.backends.values() if item.descriptor.stage_key == stage_key]

    def default_for_stage(self, stage_key: str) -> RegisteredStageBackend | None:
        backend_id = self.defaults.get(stage_key)
        if backend_id is None:
            return None
        return self.get(backend_id)


@dataclass
class PipelineRuntime:
    registry: PipelineBackendRegistry
    encoder: ArtifactEncoder
    cache: PreprocessCacheRepository
    previews: PreviewBuilderRegistry
    descriptor: RuntimeDescriptor

    @property
    def signature(self) -> str:
        return "|".join(
            f"{component.name}:{component.backend_kind}:{component.model_variant}:{component.model_version}"
            for component in self.descriptor.components
        )

    def backend(self, stage_key: str, backend_kind: str, model_variant: str = "default") -> RegisteredStageBackend | None:
        return self.registry.get(StageBackendId(stage_key=stage_key, backend_kind=backend_kind, model_variant=model_variant))

    def default_backend(self, stage_key: str) -> RegisteredStageBackend | None:
        return self.registry.default_for_stage(stage_key)

    def _compat_get(self, stage_key: str, backend_kind: str, model_variant: str) -> object | None:
        backend = self.backend(stage_key, backend_kind, model_variant)
        return None if backend is None else backend.handler

    def _compat_set(self, stage_key: str, backend_kind: str, model_variant: str, handler: object | None) -> None:
        backend = self.backend(stage_key, backend_kind, model_variant)
        if backend is not None:
            backend.handler = handler

    @property
    def detector(self):
        active = self.default_backend("detector")
        return None if active is None else active.handler

    @detector.setter
    def detector(self, value) -> None:
        active = self.default_backend("detector")
        if active is not None:
            active.handler = value

    @property
    def mock_detector(self):
        return self._compat_get("detector", "mock", "mock-v1")

    @mock_detector.setter
    def mock_detector(self, value) -> None:
        self._compat_set("detector", "mock", "mock-v1", value)

    @property
    def real_detector(self):
        return self._compat_get("detector", "local", "grounding-dino")

    @real_detector.setter
    def real_detector(self, value) -> None:
        self._compat_set("detector", "local", "grounding-dino", value)

    @property
    def geometry(self):
        active = self.default_backend("geometry_estimator")
        return None if active is None else active.handler

    @geometry.setter
    def geometry(self, value) -> None:
        active = self.default_backend("geometry_estimator")
        if active is not None:
            active.handler = value

    @property
    def mock_geometry(self):
        return self._compat_get("geometry_estimator", "mock", "mock-v1")

    @mock_geometry.setter
    def mock_geometry(self, value) -> None:
        self._compat_set("geometry_estimator", "mock", "mock-v1", value)

    @property
    def real_geometry(self):
        return self._compat_get("geometry_estimator", "local", "geocalib")

    @real_geometry.setter
    def real_geometry(self, value) -> None:
        self._compat_set("geometry_estimator", "local", "geocalib", value)

    @property
    def segmenter(self):
        active = self.default_backend("segmenter")
        return None if active is None else active.handler

    @segmenter.setter
    def segmenter(self, value) -> None:
        active = self.default_backend("segmenter")
        if active is not None:
            active.handler = value

    @property
    def mock_segmenter(self):
        return self._compat_get("segmenter", "mock", "mock-v1")

    @mock_segmenter.setter
    def mock_segmenter(self, value) -> None:
        self._compat_set("segmenter", "mock", "mock-v1", value)

    @property
    def real_segmenter(self):
        return self._compat_get("segmenter", "local", "birefnet")

    @real_segmenter.setter
    def real_segmenter(self, value) -> None:
        self._compat_set("segmenter", "local", "birefnet", value)

    @property
    def foreground_refiner(self):
        active = self.default_backend("foreground_refiner")
        return None if active is None else active.handler

    @foreground_refiner.setter
    def foreground_refiner(self, value) -> None:
        active = self.default_backend("foreground_refiner")
        if active is not None:
            active.handler = value

    @property
    def mock_foreground_refiner(self):
        return self._compat_get("foreground_refiner", "mock", "passthrough-v1")

    @mock_foreground_refiner.setter
    def mock_foreground_refiner(self, value) -> None:
        self._compat_set("foreground_refiner", "mock", "passthrough-v1", value)

    @property
    def real_foreground_refiner(self):
        return self._compat_get("foreground_refiner", "local", "fast-foreground-estimation")

    @real_foreground_refiner.setter
    def real_foreground_refiner(self, value) -> None:
        self._compat_set("foreground_refiner", "local", "fast-foreground-estimation", value)

    @property
    def depth(self):
        active = self.default_backend("depth_estimator")
        return None if active is None else active.handler

    @depth.setter
    def depth(self, value) -> None:
        active = self.default_backend("depth_estimator")
        if active is not None:
            active.handler = value

    @property
    def mock_depth(self):
        return self._compat_get("depth_estimator", "mock", "mock-v1")

    @mock_depth.setter
    def mock_depth(self, value) -> None:
        self._compat_set("depth_estimator", "mock", "mock-v1", value)

    @property
    def real_depth(self):
        return self._compat_get("depth_estimator", "local", "depth-anything-v2-small")

    @real_depth.setter
    def real_depth(self, value) -> None:
        self._compat_set("depth_estimator", "local", "depth-anything-v2-small", value)

    @property
    def normals(self):
        active = self.default_backend("normal_estimator")
        return None if active is None else active.handler

    @normals.setter
    def normals(self, value) -> None:
        active = self.default_backend("normal_estimator")
        if active is not None:
            active.handler = value

    @property
    def mock_normals(self):
        return self._compat_get("normal_estimator", "mock", "mock-v1")

    @mock_normals.setter
    def mock_normals(self, value) -> None:
        self._compat_set("normal_estimator", "mock", "mock-v1", value)

    @property
    def real_normals(self):
        return self._compat_get("normal_estimator", "local", "stable-normal") or self._compat_get("normal_estimator", "local", "from-depth-v2")

    @real_normals.setter
    def real_normals(self, value) -> None:
        if self.backend("normal_estimator", "local", "stable-normal") is not None:
            self._compat_set("normal_estimator", "local", "stable-normal", value)
        if self.backend("normal_estimator", "local", "from-depth-v2") is not None:
            self._compat_set("normal_estimator", "local", "from-depth-v2", value)

    @property
    def shadow(self):
        active = self.default_backend("shadow_generator")
        return None if active is None else active.handler

    @shadow.setter
    def shadow(self, value) -> None:
        active = self.default_backend("shadow_generator")
        if active is not None:
            active.handler = value

    @property
    def mock_shadow(self):
        return self._compat_get("shadow_generator", "mock", "mock")

    @mock_shadow.setter
    def mock_shadow(self, value) -> None:
        self._compat_set("shadow_generator", "mock", "mock", value)

    @property
    def real_shadow(self):
        return self._compat_get("shadow_generator", "local", "v1-gan")

    @real_shadow.setter
    def real_shadow(self, value) -> None:
        self._compat_set("shadow_generator", "local", "v1-gan", value)

    @property
    def shadow_v1_gan(self):
        return self._compat_get("shadow_generator", "local", "v1-gan")

    @shadow_v1_gan.setter
    def shadow_v1_gan(self, value) -> None:
        self._compat_set("shadow_generator", "local", "v1-gan", value)

    @property
    def shadow_v2_diff(self):
        return self._compat_get("shadow_generator", "triton", "v2-diff")

    @shadow_v2_diff.setter
    def shadow_v2_diff(self, value) -> None:
        self._compat_set("shadow_generator", "triton", "v2-diff", value)

    @property
    def composer(self):
        return self._compat_get("composer", "local", "python-composer")
