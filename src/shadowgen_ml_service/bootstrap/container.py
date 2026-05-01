from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineBackendRegistry, PipelineRuntime
from shadowgen_ml_service.bootstrap.probes import (
    probe_birefnet,
    probe_depth_anything,
    probe_fast_foreground_estimation,
    probe_geocalib,
    probe_grounding_dino,
    probe_stable_normal,
    probe_shadow_pix2pix,
    probe_shadow_v2_diff,
)
from shadowgen_ml_service.bootstrap.runtime_descriptor import backend_descriptor, build_runtime_descriptor, component_status
from shadowgen_ml_service.bootstrap.triton_bindings import build_triton_model_registry
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.models import ComponentStatus, StageBackendId
from shadowgen_ml_service.infrastructure.backends.triton.batching import TritonStageBatchCoordinator
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelRegistry
from shadowgen_ml_service.infrastructure.cache.preprocess_cache_repository import FilesystemPreprocessCacheRepository
from shadowgen_ml_service.infrastructure.encoding.default import DefaultArtifactEncoder
from shadowgen_ml_service.infrastructure.presentation.preview_registry import DefaultPreviewBuilderRegistry
from shadowgen_ml_service.infrastructure.stages.composition.python_composer import PythonComposer
from shadowgen_ml_service.infrastructure.stages.depth.depth_anything import RealDepthEstimator
from shadowgen_ml_service.infrastructure.stages.depth.mock import MockDepthEstimator
from shadowgen_ml_service.infrastructure.stages.depth.triton import TritonDepthEstimator
from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import RealDetector
from shadowgen_ml_service.infrastructure.stages.detection.mock import MockDetector
from shadowgen_ml_service.infrastructure.stages.detection.triton import TritonDetector
from shadowgen_ml_service.infrastructure.stages.foreground_refinement.fast_foreground_estimation import FastForegroundColorEstimator
from shadowgen_ml_service.infrastructure.stages.foreground_refinement.mock import PassthroughForegroundColorEstimator
from shadowgen_ml_service.infrastructure.stages.geometry.geocalib import RealGeometryEstimator
from shadowgen_ml_service.infrastructure.stages.geometry.mock import MockGeometryEstimator
from shadowgen_ml_service.infrastructure.stages.normals.from_depth import NormalFromDepthEstimator
from shadowgen_ml_service.infrastructure.stages.normals.mock import MockNormalEstimator
from shadowgen_ml_service.infrastructure.stages.normals.stable_normal import StableNormalEstimator
from shadowgen_ml_service.infrastructure.stages.normals.triton import TritonNormalEstimator
from shadowgen_ml_service.infrastructure.stages.lazy import LazyStageAdapter
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import RealSegmenter
from shadowgen_ml_service.infrastructure.stages.segmentation.mock import MockSegmenter
from shadowgen_ml_service.infrastructure.stages.segmentation.triton import TritonSegmenter
from shadowgen_ml_service.infrastructure.stages.shadow.pix2pix import Pix2PixShadowGenerator
from shadowgen_ml_service.infrastructure.stages.shadow.stub import DeterministicShadowGenerator
from shadowgen_ml_service.infrastructure.stages.shadow.triton import TritonShadowGenerator
from shadowgen_ml_service.infrastructure.stages.shadow.v2_diff import V2DiffShadowGenerator


HEAVY_STAGES = {"detector", "segmenter", "depth_estimator", "normal_estimator", "shadow_generator"}


def build_runtime(settings: Settings) -> PipelineRuntime:
    registry = PipelineBackendRegistry()
    components: list[ComponentStatus] = []

    execution_default_backend = _effective_default_backend(settings)
    triton_settings = TritonBackendSettings(
        url=settings.triton_url,
        protocol=settings.triton_protocol,
        timeout_ms=settings.triton_timeout_ms,
    )
    triton_client = TritonInferenceClient(triton_settings)
    triton_ready = triton_client.ping()
    triton_models = build_triton_model_registry(settings)
    triton_batcher = TritonStageBatchCoordinator(
        enabled=settings.batching_enabled,
        window_ms=settings.batch_window_ms,
        max_size=settings.batch_max_size,
        stage_enabled={
            "segmenter": settings.batch_segmenter_enabled,
            "depth_estimator": settings.batch_depth_enabled,
            "normal_estimator": settings.batch_normals_enabled,
            "shadow_generator": settings.batch_shadow_enabled,
        },
    )

    _register_detector(settings, registry, triton_client, triton_models, triton_ready)
    _register_geometry(settings, registry)
    _register_segmenter(settings, registry, triton_client, triton_models, triton_ready, triton_batcher)
    _register_foreground_refiner(registry)
    _register_depth(settings, registry, triton_client, triton_models, triton_ready, triton_batcher)
    _register_normals(settings, registry, triton_client, triton_models, triton_ready, triton_batcher)
    _register_shadow(settings, registry, triton_client, triton_models, triton_ready, triton_batcher)
    _register_composer(registry)
    _register_encoder(registry)

    default_stage_backend = {
        "detector": "mock" if execution_default_backend == "mock" else _stage_backend_preference(settings.detector_backend_kind, execution_default_backend),
        "geometry_estimator": "mock" if execution_default_backend == "mock" else ("internal" if not settings.geometry_enabled else settings.geometry_backend_kind),
        "segmenter": "mock" if execution_default_backend == "mock" else _stage_backend_preference(settings.segmenter_backend_kind, execution_default_backend),
        "foreground_refiner": "mock" if execution_default_backend == "mock" else settings.foreground_refiner_backend_kind,
        "depth_estimator": "mock" if execution_default_backend == "mock" else _stage_backend_preference(settings.depth_backend_kind, execution_default_backend),
        "normal_estimator": "mock" if execution_default_backend == "mock" else _stage_backend_preference(settings.normals_backend_kind, execution_default_backend),
        "shadow_generator": "mock" if execution_default_backend == "mock" else _stage_backend_preference(settings.shadow_backend_kind, execution_default_backend),
        "composer": "local",
        "artifact_encoder": "internal",
    }

    for stage_key in (
        "detector",
        "geometry_estimator",
        "segmenter",
        "foreground_refiner",
        "depth_estimator",
        "normal_estimator",
        "shadow_generator",
        "composer",
        "artifact_encoder",
    ):
        preferred_backend_kind = default_stage_backend[stage_key]
        preferred_variant = _preferred_variant(stage_key, settings)
        active_backend, fallback_reason = _resolve_active_backend(
            registry=registry,
            stage_key=stage_key,
            preferred_backend_kind=preferred_backend_kind,
            preferred_variant=preferred_variant,
        )
        if active_backend is None:
            raise RuntimeError(f"no backend registered for stage {stage_key}")
        registry.set_default(stage_key, active_backend.descriptor.backend_id)
        components.append(
            _component_from_active_backend(
                stage_key=stage_key,
                preferred_backend_kind=preferred_backend_kind,
                preferred_variant=preferred_variant,
                active_backend=active_backend,
                fallback_reason=fallback_reason,
                stage_backends=registry.list_stage(stage_key),
            )
        )

    return PipelineRuntime(
        registry=registry,
        encoder=DefaultArtifactEncoder(),
        cache=FilesystemPreprocessCacheRepository(settings.preprocess_cache_dir),
        previews=DefaultPreviewBuilderRegistry(),
        descriptor=build_runtime_descriptor(
            execution_default_backend=execution_default_backend,
            async_enabled=settings.async_enabled,
            components=components,
        ),
    )


def _register_detector(settings: Settings, registry: PipelineBackendRegistry, triton_client: TritonInferenceClient, triton_models: TritonModelRegistry, triton_ready: bool) -> None:
    probe = probe_grounding_dino()
    mock = MockDetector()
    registry.register(
        backend_descriptor(
            stage_key="detector",
            backend_kind="mock",
            model_variant="mock-v1",
            model_name="mock-detector",
            model_version="mock-v1",
            available=True,
            detail="deterministic fallback detector",
            device="cpu",
        ),
        mock,
    )
    handler = None
    detail = probe.detail
    device = "cpu"
    available = False
    if probe.available:
        def make_local_detector() -> RealDetector:
            return RealDetector(
                model_id=settings.grounding_dino_model_id,
                prompt=settings.grounding_dino_prompt,
                box_threshold=settings.grounding_dino_box_threshold,
                text_threshold=settings.grounding_dino_text_threshold,
                target_device=settings.target_device,
            )

        preferred_detector_backend = _stage_backend_preference(
            settings.detector_backend_kind,
            _effective_default_backend(settings),
        )
        if preferred_detector_backend == "local":
            try:
                handler = make_local_detector()
                available = True
                detail = (
                    "GroundingDINO local backend active "
                    f"(model_id={settings.grounding_dino_model_id}, prompt={settings.grounding_dino_prompt!r})"
                )
                device = getattr(handler, "device_label", "cpu")
            except Exception as exc:
                detail = f"GroundingDINO local init failed: {exc}"
        else:
            handler = LazyStageAdapter(make_local_detector, backend_name="lazy-grounding-dino", device_label=settings.target_device)
            available = True
            detail = f"GroundingDINO local backend available for lazy fallback (model_id={settings.grounding_dino_model_id})"
            device = settings.target_device
    registry.register(
        backend_descriptor(
            stage_key="detector",
            backend_kind="local",
            model_variant="grounding-dino",
            model_name=probe.model_name,
            model_version=probe.model_version,
            available=available,
            detail=detail,
            device=device,
            supports_batching=False,
            supports_async=True,
        ),
        handler,
    )
    binding = triton_models.get("detector", "grounding-dino")
    triton_available, triton_detail = _probe_triton_backend(
        triton_client=triton_client,
        binding=binding,
        triton_ready=triton_ready,
        unavailable_detail="Triton detector endpoint or model is unavailable",
    )
    registry.register(
        backend_descriptor(
            stage_key="detector",
            backend_kind="triton",
            model_variant="grounding-dino",
            model_name=binding.model_name if binding is not None else settings.triton_detector_model,
            model_version="triton-managed",
            available=triton_available,
            detail=triton_detail,
            device="triton",
            endpoint=triton_client.endpoint,
            supports_batching=False,
            supports_async=True,
        ),
        TritonDetector(triton_client, binding) if triton_available and binding is not None else None,
    )


def _register_geometry(settings: Settings, registry: PipelineBackendRegistry) -> None:
    probe = probe_geocalib()
    mock = MockGeometryEstimator()
    registry.register(
        backend_descriptor(
            stage_key="geometry_estimator",
            backend_kind="mock",
            model_variant="mock-v1",
            model_name="mock-geometry",
            model_version="mock-v1",
            available=True,
            detail="deterministic fallback geometry estimator",
        ),
        mock,
    )
    if not settings.geometry_enabled:
        registry.register(
            backend_descriptor(
                stage_key="geometry_estimator",
                backend_kind="internal",
                model_variant="disabled",
                model_name="geometry-disabled",
                model_version="disabled",
                available=True,
                detail="GeoCalib stage disabled by SHADOWGEN_GEOMETRY_ENABLED=false",
                device="none",
                supports_async=True,
            ),
            None,
        )
        return
    handler = None
    detail = probe.detail
    available = False
    if probe.available:
        try:
            handler = RealGeometryEstimator(
                weights=settings.geocalib_weights,
                camera_model=settings.geocalib_camera_model,
                shared_intrinsics=settings.geocalib_shared_intrinsics,
            )
            available = True
            detail = (
                "GeoCalib local backend active "
                f"(weights={settings.geocalib_weights}, camera_model={settings.geocalib_camera_model})"
            )
        except Exception as exc:
            detail = f"GeoCalib local init failed: {exc}"
    registry.register(
        backend_descriptor(
            stage_key="geometry_estimator",
            backend_kind="local",
            model_variant="geocalib",
            model_name=probe.model_name,
            model_version=probe.model_version,
            available=available,
            detail=detail,
            device="cpu",
            supports_async=False,
        ),
        handler,
    )


def _register_segmenter(
    settings: Settings,
    registry: PipelineBackendRegistry,
    triton_client: TritonInferenceClient,
    triton_models: TritonModelRegistry,
    triton_ready: bool,
    triton_batcher: TritonStageBatchCoordinator,
) -> None:
    probe = probe_birefnet(model_id=settings.birefnet_model_id, allow_cpu=settings.birefnet_allow_cpu)
    mock = MockSegmenter()
    registry.register(
        backend_descriptor(
            stage_key="segmenter",
            backend_kind="mock",
            model_variant="mock-v1",
            model_name="mock-segmenter",
            model_version="mock-v1",
            available=True,
            detail="deterministic fallback matting stage",
        ),
        mock,
    )
    handler = None
    detail = probe.detail
    device = "cpu"
    available = False
    if probe.available:
        def make_local_segmenter() -> RealSegmenter:
            return RealSegmenter(
                model_id=settings.birefnet_model_id,
                resolution=settings.birefnet_resolution,
                mask_threshold=settings.birefnet_mask_threshold,
                target_device=settings.target_device,
                compile_enabled=settings.birefnet_compile_enabled,
                compile_mode=settings.birefnet_compile_mode,
                compile_backend=settings.birefnet_compile_backend,
                matmul_precision=settings.birefnet_matmul_precision,
            )

        preferred_segmenter_backend = _stage_backend_preference(
            settings.segmenter_backend_kind,
            _effective_default_backend(settings),
        )
        if preferred_segmenter_backend == "triton":
            handler = LazyStageAdapter(make_local_segmenter, backend_name="lazy-birefnet", device_label=settings.target_device)
            available = True
            detail = f"BiRefNet local backend available for lazy fallback (model_id={settings.birefnet_model_id})"
            device = settings.target_device
        else:
            try:
                handler = make_local_segmenter()
                available = True
                detail = (
                    f"BiRefNet local backend active (model_id={settings.birefnet_model_id}, "
                    f"compile={getattr(handler, 'compile_status', 'disabled')})"
                )
                device = getattr(handler, "device_label", "cpu")
            except Exception as exc:
                detail = f"BiRefNet local init failed: {exc}"
    registry.register(
        backend_descriptor(
            stage_key="segmenter",
            backend_kind="local",
            model_variant="birefnet",
            model_name=probe.model_name,
            model_version=probe.model_version,
            available=available,
            detail=detail,
            device=device,
            supports_batching=False,
            supports_async=True,
        ),
        handler,
    )
    binding = triton_models.get("segmenter", "birefnet")
    triton_available, triton_detail = _probe_triton_backend(
        triton_client=triton_client,
        binding=binding,
        triton_ready=triton_ready,
        unavailable_detail="Triton segmenter endpoint or model is unavailable",
    )
    registry.register(
        backend_descriptor(
            stage_key="segmenter",
            backend_kind="triton",
            model_variant="birefnet",
            model_name=binding.model_name if binding is not None else settings.triton_segmenter_model,
            model_version="triton-managed",
            available=triton_available,
            detail=triton_detail,
            device="triton",
            endpoint=triton_client.endpoint,
            supports_batching=True,
            supports_async=True,
        ),
        TritonSegmenter(triton_client, binding, triton_batcher) if triton_available and binding is not None else None,
    )


def _register_foreground_refiner(registry: PipelineBackendRegistry) -> None:
    probe = probe_fast_foreground_estimation()
    mock = PassthroughForegroundColorEstimator()
    registry.register(
        backend_descriptor(
            stage_key="foreground_refiner",
            backend_kind="mock",
            model_variant="passthrough-v1",
            model_name="passthrough-foreground",
            model_version="passthrough-v1",
            available=True,
            detail="passthrough foreground refinement",
        ),
        mock,
    )
    handler = None
    detail = probe.detail
    available = False
    if probe.available:
        try:
            handler = FastForegroundColorEstimator()
            available = True
            detail = "Fast Foreground Colour Estimation local backend active"
        except Exception as exc:
            detail = f"foreground refinement local init failed: {exc}"
    registry.register(
        backend_descriptor(
            stage_key="foreground_refiner",
            backend_kind="local",
            model_variant="fast-foreground-estimation",
            model_name=probe.model_name,
            model_version=probe.model_version,
            available=available,
            detail=detail,
        ),
        handler,
    )


def _register_depth(
    settings: Settings,
    registry: PipelineBackendRegistry,
    triton_client: TritonInferenceClient,
    triton_models: TritonModelRegistry,
    triton_ready: bool,
    triton_batcher: TritonStageBatchCoordinator,
) -> None:
    probe = probe_depth_anything()
    mock = MockDepthEstimator()
    registry.register(
        backend_descriptor(
            stage_key="depth_estimator",
            backend_kind="mock",
            model_variant="mock-v1",
            model_name="mock-depth",
            model_version="mock-v1",
            available=True,
            detail="deterministic fallback depth estimator",
        ),
        mock,
    )
    handler = None
    detail = probe.detail
    device = "cpu"
    available = False
    if probe.available:
        def make_local_depth() -> RealDepthEstimator:
            return RealDepthEstimator(model_id=settings.depth_anything_model_id, target_device=settings.target_device)

        preferred_depth_backend = _stage_backend_preference(
            settings.depth_backend_kind,
            _effective_default_backend(settings),
        )
        if preferred_depth_backend == "local":
            try:
                handler = make_local_depth()
                available = True
                detail = f"Depth Anything local backend active (model_id={settings.depth_anything_model_id})"
                device = getattr(handler, "device_label", "cpu")
            except Exception as exc:
                detail = f"Depth Anything local init failed: {exc}"
        else:
            handler = LazyStageAdapter(make_local_depth, backend_name="lazy-depth-anything", device_label=settings.target_device)
            available = True
            detail = f"Depth Anything local backend available for lazy fallback (model_id={settings.depth_anything_model_id})"
            device = settings.target_device
    registry.register(
        backend_descriptor(
            stage_key="depth_estimator",
            backend_kind="local",
            model_variant="depth-anything-v2-small",
            model_name=probe.model_name,
            model_version=probe.model_version,
            available=available,
            detail=detail,
            device=device,
            supports_batching=False,
            supports_async=True,
        ),
        handler,
    )
    binding = triton_models.get("depth_estimator", "depth-anything-v2-small")
    triton_available, triton_detail = _probe_triton_backend(
        triton_client=triton_client,
        binding=binding,
        triton_ready=triton_ready,
        unavailable_detail="Triton depth endpoint or model is unavailable",
    )
    registry.register(
        backend_descriptor(
            stage_key="depth_estimator",
            backend_kind="triton",
            model_variant="depth-anything-v2-small",
            model_name=binding.model_name if binding is not None else settings.triton_depth_model,
            model_version="triton-managed",
            available=triton_available,
            detail=triton_detail,
            device="triton",
            endpoint=triton_client.endpoint,
            supports_batching=True,
            supports_async=True,
        ),
        TritonDepthEstimator(triton_client, binding, triton_batcher) if triton_available and binding is not None else None,
    )


def _register_normals(
    settings: Settings,
    registry: PipelineBackendRegistry,
    triton_client: TritonInferenceClient,
    triton_models: TritonModelRegistry,
    triton_ready: bool,
    triton_batcher: TritonStageBatchCoordinator,
) -> None:
    mock = MockNormalEstimator()
    registry.register(
        backend_descriptor(
            stage_key="normal_estimator",
            backend_kind="mock",
            model_variant="mock-v1",
            model_name="flat-normal-map",
            model_version="mock-v1",
            available=True,
            detail="flat fallback normals",
        ),
        mock,
    )
    depth_fallback = NormalFromDepthEstimator()
    registry.register(
        backend_descriptor(
            stage_key="normal_estimator",
            backend_kind="local",
            model_variant="from-depth-v2",
            model_name="normal-map-from-depth",
            model_version=depth_fallback.model_variant,
            available=True,
            detail="depth-derived normal map fallback",
            device=getattr(depth_fallback, "device_label", "cpu"),
            supports_async=True,
        ),
        depth_fallback,
    )
    probe = probe_stable_normal(allow_cpu=settings.stable_normal_allow_cpu, target_device=settings.target_device)
    handler = None
    detail = probe.detail
    available = False
    device = "cpu"
    stable_requested = settings.normals_model_variant == "stable-normal"
    if not stable_requested:
        detail = "StableNormal local backend is disabled by default; set SHADOWGEN_NORMALS_MODEL_VARIANT=stable-normal to enable it"
    if stable_requested and probe.available:
        try:
            handler = StableNormalEstimator(
                model_variant=settings.stable_normal_variant,
                resolution=settings.stable_normal_resolution,
                target_device=settings.target_device,
                allow_cpu=settings.stable_normal_allow_cpu,
                cache_dir=settings.model_cache_dir / "stable-normal",
            )
            available = True
            detail = f"StableNormal local backend active (variant={settings.stable_normal_variant})"
            device = getattr(handler, "device_label", "cpu")
        except Exception as exc:
            detail = f"StableNormal local init failed: {exc}"
    registry.register(
        backend_descriptor(
            stage_key="normal_estimator",
            backend_kind="local",
            model_variant="stable-normal",
            model_name=probe.model_name,
            model_version=settings.stable_normal_variant,
            available=available,
            detail=detail,
            device=device,
            supports_batching=False,
            supports_async=True,
        ),
        handler,
    )
    binding = triton_models.get("normal_estimator", "stable-normal")
    triton_available, triton_detail = _probe_triton_backend(
        triton_client=triton_client,
        binding=binding,
        triton_ready=triton_ready,
        unavailable_detail="Triton normals endpoint or model is unavailable",
    )
    registry.register(
        backend_descriptor(
            stage_key="normal_estimator",
            backend_kind="triton",
            model_variant="stable-normal",
            model_name=binding.model_name if binding is not None else settings.triton_normals_model,
            model_version="triton-managed",
            available=triton_available,
            detail=triton_detail,
            device="triton",
            endpoint=triton_client.endpoint,
            supports_batching=True,
            supports_async=True,
        ),
        TritonNormalEstimator(triton_client, binding, triton_batcher) if triton_available and binding is not None else None,
    )


def _register_shadow(
    settings: Settings,
    registry: PipelineBackendRegistry,
    triton_client: TritonInferenceClient,
    triton_models: TritonModelRegistry,
    triton_ready: bool,
    triton_batcher: TritonStageBatchCoordinator,
) -> None:
    mock = DeterministicShadowGenerator()
    registry.register(
        backend_descriptor(
            stage_key="shadow_generator",
            backend_kind="mock",
            model_variant="mock",
            model_name="deterministic-shadow",
            model_version="stub-v1",
            available=True,
            detail="deterministic analytical shadow generator",
            supports_async=True,
        ),
        mock,
    )
    probe = probe_shadow_pix2pix(settings.shadow_pix2pix_weights_path, target_device=settings.target_device)
    handler = None
    detail = probe.detail
    available = False
    device = "cpu"
    if probe.available:
        def make_v1_shadow() -> Pix2PixShadowGenerator:
            return Pix2PixShadowGenerator(
                weights_path=settings.shadow_pix2pix_weights_path,
                target_device=settings.target_device,
            )

        preferred_shadow_backend = _stage_backend_preference(
            settings.shadow_backend_kind,
            _effective_default_backend(settings),
        )
        preferred_shadow_variant = settings.shadow_model_variant.lower()
        if preferred_shadow_backend == "local" and preferred_shadow_variant == "v1-gan":
            try:
                handler = make_v1_shadow()
                available = True
                detail = f"V1-GAN local backend active (weights={settings.shadow_pix2pix_weights_path})"
                device = getattr(handler, "device_label", "cpu")
            except Exception as exc:
                detail = f"V1-GAN local init failed: {exc}"
        else:
            handler = LazyStageAdapter(make_v1_shadow, backend_name="lazy-v1-gan", device_label=settings.target_device)
            available = True
            detail = f"V1-GAN local backend available for lazy fallback (weights={settings.shadow_pix2pix_weights_path})"
            device = settings.target_device
    registry.register(
        backend_descriptor(
            stage_key="shadow_generator",
            backend_kind="local",
            model_variant="v1-gan",
            model_name="V1-GAN",
            model_version=probe.model_version,
            available=available,
            detail=detail,
            device=device,
            supports_batching=False,
            supports_async=True,
        ),
        handler,
    )
    v2_probe = probe_shadow_v2_diff(settings.shadow_v2_diff_bundle_path, settings.shadow_v2_diff_background_path)
    v2_handler = None
    v2_detail = v2_probe.detail
    v2_available = False
    v2_device = "cpu"
    if v2_probe.available:
        def make_v2_shadow() -> V2DiffShadowGenerator:
            return V2DiffShadowGenerator(
                bundle_path=settings.shadow_v2_diff_bundle_path,
                background_path=settings.shadow_v2_diff_background_path,
                target_device=settings.target_device,
                seed=settings.shadow_v2_diff_seed,
                steps=settings.shadow_v2_diff_steps,
                guidance_scale=settings.shadow_v2_diff_guidance_scale,
                fast_lcm=settings.shadow_v2_diff_fast_lcm,
                compile_enabled=settings.shadow_v2_diff_compile_enabled,
                compile_mode=settings.shadow_v2_diff_compile_mode,
                compile_backend=settings.shadow_v2_diff_compile_backend or None,
            )

        preferred_shadow_backend = _stage_backend_preference(
            settings.shadow_backend_kind,
            _effective_default_backend(settings),
        )
        preferred_shadow_variant = settings.shadow_model_variant.lower()
        if preferred_shadow_backend == "local" and preferred_shadow_variant == "v2-diff":
            try:
                v2_handler = make_v2_shadow()
                v2_available = True
                v2_detail = f"V2-DIFF local backend active (bundle={settings.shadow_v2_diff_bundle_path})"
                v2_device = getattr(v2_handler, "device_label", "cpu")
            except Exception as exc:
                v2_detail = f"V2-DIFF local init failed: {exc}"
        else:
            v2_handler = LazyStageAdapter(make_v2_shadow, backend_name="lazy-v2-diff", device_label=settings.target_device)
            v2_available = True
            v2_detail = f"V2-DIFF local backend available for lazy fallback (bundle={settings.shadow_v2_diff_bundle_path})"
            v2_device = settings.target_device
    registry.register(
        backend_descriptor(
            stage_key="shadow_generator",
            backend_kind="local",
            model_variant="v2-diff",
            model_name=v2_probe.model_name,
            model_version=v2_probe.model_version,
            available=v2_available,
            detail=v2_detail,
            device=v2_device,
            supports_batching=False,
            supports_async=True,
        ),
        v2_handler,
    )
    binding = triton_models.get("shadow_generator", "v2-diff")
    triton_available, triton_detail = _probe_triton_backend(
        triton_client=triton_client,
        binding=binding,
        triton_ready=triton_ready,
        unavailable_detail="V2-DIFF Triton backend scaffold only",
    )
    registry.register(
        backend_descriptor(
            stage_key="shadow_generator",
            backend_kind="triton",
            model_variant="v2-diff",
            model_name=binding.model_name if binding is not None else settings.triton_shadow_v2_model,
            model_version="triton-managed",
            available=triton_available,
            detail=triton_detail,
            device="triton",
            endpoint=triton_client.endpoint,
            supports_batching=True,
            supports_async=True,
        ),
        TritonShadowGenerator(triton_client, binding, model_variant="v2-diff", batcher=triton_batcher) if triton_available and binding is not None else None,
    )


def _register_composer(registry: PipelineBackendRegistry) -> None:
    handler = PythonComposer()
    registry.register(
        backend_descriptor(
            stage_key="composer",
            backend_kind="local",
            model_variant="python-composer",
            model_name="solid-background-composer",
            model_version="v1",
            available=True,
            detail="Python compositor",
        ),
        handler,
    )


def _register_encoder(registry: PipelineBackendRegistry) -> None:
    registry.register(
        backend_descriptor(
            stage_key="artifact_encoder",
            backend_kind="internal",
            model_variant="default-encoder",
            model_name="artifact-encoder",
            model_version="v1",
            available=True,
            detail="Python artifact encoder",
        ),
        None,
    )


def _effective_default_backend(settings: Settings) -> str:
    if settings.execution_default_backend:
        return settings.execution_default_backend
    return "mock" if settings.runtime_mode.lower() == "mock" else "local"


def _stage_backend_preference(stage_override: str, global_default: str) -> str:
    return stage_override or global_default


def _preferred_variant(stage_key: str, settings: Settings) -> str:
    if stage_key == "shadow_generator":
        return settings.shadow_model_variant.lower()
    if stage_key == "normal_estimator":
        return settings.normals_model_variant
    if stage_key == "detector":
        return "grounding-dino"
    if stage_key == "segmenter":
        return "birefnet"
    if stage_key == "depth_estimator":
        return "depth-anything-v2-small"
    if stage_key == "geometry_estimator":
        return "geocalib" if settings.geometry_enabled else "disabled"
    if stage_key == "foreground_refiner":
        return "fast-foreground-estimation"
    if stage_key == "composer":
        return "python-composer"
    return "default"


def _fallback_order(stage_key: str, preferred_backend_kind: str) -> list[str]:
    if preferred_backend_kind == "mock":
        return ["mock"]
    if preferred_backend_kind == "internal":
        return ["internal"]
    if stage_key == "normal_estimator":
        if preferred_backend_kind == "triton":
            return ["triton", "local", "local", "mock"]
        return ["local", "local", "mock"]
    if stage_key in {"geometry_estimator", "foreground_refiner", "composer"}:
        return [preferred_backend_kind, "local", "mock"]
    if preferred_backend_kind == "triton":
        return ["triton", "local", "mock"]
    return [preferred_backend_kind, "mock"] if preferred_backend_kind == "local" else ["local", "mock"]


def _resolve_active_backend(*, registry: PipelineBackendRegistry, stage_key: str, preferred_backend_kind: str, preferred_variant: str):
    fallback_reason = None
    if stage_key == "normal_estimator":
        candidate_ids = []
        if preferred_backend_kind == "mock":
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock-v1"))
        elif preferred_variant == "from-depth-v2":
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="from-depth-v2"))
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock-v1"))
        else:
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind=preferred_backend_kind, model_variant="stable-normal"))
            if preferred_backend_kind == "triton":
                candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="stable-normal"))
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="from-depth-v2"))
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock-v1"))
        for index, backend_id in enumerate(candidate_ids):
            registered = registry.get(backend_id)
            if registered is not None and registered.descriptor.available and registered.handler is not None:
                if index > 0:
                    fallback_reason = f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}"
                return registered, fallback_reason

    if stage_key == "shadow_generator":
        candidate_ids = []
        if preferred_backend_kind == "mock":
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock"))
        elif preferred_backend_kind == "triton":
            candidate_ids.extend(
                [
                    StageBackendId(stage_key=stage_key, backend_kind="triton", model_variant=preferred_variant),
                    StageBackendId(stage_key=stage_key, backend_kind="local", model_variant=preferred_variant),
                    StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="v1-gan"),
                    StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock"),
                ]
            )
        else:
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind=preferred_backend_kind, model_variant=preferred_variant))
            if preferred_variant == "v2-diff":
                candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="v1-gan"))
            candidate_ids.append(StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock"))
        for index, backend_id in enumerate(candidate_ids):
            registered = registry.get(backend_id)
            if registered is not None and registered.descriptor.available and registered.handler is not None:
                if index > 0:
                    fallback_reason = f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}"
                return registered, fallback_reason

    candidate_ids = [
        StageBackendId(
            stage_key=stage_key,
            backend_kind=backend_kind,
            model_variant=_default_variant_for_backend(stage_key, backend_kind, preferred_variant),
        )
        for backend_kind in _fallback_order(stage_key, preferred_backend_kind)
    ]
    for index, backend_id in enumerate(candidate_ids):
        registered = registry.get(backend_id)
        if registered is not None and registered.descriptor.available and registered.handler is not None or (
            registered is not None and registered.descriptor.available and stage_key in {"artifact_encoder", "geometry_estimator"} and registered.descriptor.backend_kind == "internal"
        ):
            if index > 0:
                fallback_reason = f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}"
            return registered, fallback_reason

    if stage_key == "normal_estimator":
        for variant in ("stable-normal", "from-depth-v2"):
            registered = registry.get(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant=variant))
            if registered is not None and registered.descriptor.available:
                fallback_reason = f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}"
                return registered, fallback_reason

    for registered in registry.list_stage(stage_key):
        if registered.descriptor.available:
            fallback_reason = f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}"
            return registered, fallback_reason
    return None, fallback_reason


def _default_variant_for_backend(stage_key: str, backend_kind: str, preferred_variant: str) -> str:
    if backend_kind == "internal" and stage_key == "geometry_estimator":
        return preferred_variant
    if backend_kind == "mock":
        if stage_key == "shadow_generator":
            return "mock"
        if stage_key == "foreground_refiner":
            return "passthrough-v1"
        return "mock-v1"
    if stage_key == "normal_estimator" and backend_kind == "local":
        return preferred_variant if preferred_variant in {"stable-normal", "from-depth-v2"} else "stable-normal"
    if stage_key == "shadow_generator":
        return preferred_variant
    if stage_key == "detector":
        return "grounding-dino"
    if stage_key == "segmenter":
        return "birefnet"
    if stage_key == "depth_estimator":
        return "depth-anything-v2-small"
    if stage_key == "geometry_estimator":
        return "geocalib"
    if stage_key == "foreground_refiner":
        return "fast-foreground-estimation"
    if stage_key == "composer":
        return "python-composer"
    if stage_key == "artifact_encoder":
        return "default-encoder"
    return preferred_variant


def _component_from_active_backend(
    *,
    stage_key: str,
    preferred_backend_kind: str,
    preferred_variant: str,
    active_backend,
    fallback_reason: str | None,
    stage_backends,
) -> ComponentStatus:
    descriptor = active_backend.descriptor
    using_mock = descriptor.backend_kind == "mock"
    implementation = _derive_implementation(
        actual_backend_kind=descriptor.backend_kind,
        preferred_backend_kind=preferred_backend_kind,
        preferred_variant=preferred_variant,
        actual_variant=descriptor.model_variant,
        fallback_reason=fallback_reason,
    )
    return component_status(
        name=stage_key,
        implementation=implementation,
        model_name=descriptor.model_name,
        model_version=descriptor.model_version,
        available=descriptor.available,
        using_mock=using_mock,
        detail=descriptor.detail,
        backend_kind=descriptor.backend_kind,
        model_variant=descriptor.model_variant,
        device=descriptor.device,
        endpoint=descriptor.endpoint,
        supports_batching=descriptor.supports_batching,
        supports_async=descriptor.supports_async,
        fallback_reason=fallback_reason,
        backends=[item.descriptor for item in stage_backends],
    )


def _derive_implementation(
    *,
    actual_backend_kind: str,
    preferred_backend_kind: str,
    preferred_variant: str,
    actual_variant: str,
    fallback_reason: str | None,
) -> str:
    if not fallback_reason and actual_backend_kind == preferred_backend_kind and actual_variant == preferred_variant:
        return actual_backend_kind
    if actual_backend_kind == "mock":
        return "mock-fallback" if preferred_backend_kind != "mock" or fallback_reason else "mock"
    return f"{actual_backend_kind}-fallback" if fallback_reason or actual_variant != preferred_variant else actual_backend_kind


def _probe_triton_backend(
    *,
    triton_client: TritonInferenceClient,
    binding,
    triton_ready: bool,
    unavailable_detail: str,
) -> tuple[bool, str]:
    if binding is None:
        return False, unavailable_detail
    if not triton_ready:
        return False, "Triton endpoint is unavailable"
    available, detail = triton_client.probe_binding(binding)
    if available:
        return True, detail
    return False, detail or unavailable_detail
