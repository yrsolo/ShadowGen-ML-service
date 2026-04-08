from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.bootstrap.probes import (
    probe_birefnet,
    probe_depth_anything,
    probe_fast_foreground_estimation,
    probe_geocalib,
    probe_grounding_dino,
    probe_stable_normal,
)
from shadowgen_ml_service.bootstrap.runtime_descriptor import build_runtime_descriptor, component_status
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.infrastructure.cache.preprocess_cache_repository import FilesystemPreprocessCacheRepository
from shadowgen_ml_service.infrastructure.encoding.default import DefaultArtifactEncoder
from shadowgen_ml_service.infrastructure.presentation.preview_registry import DefaultPreviewBuilderRegistry
from shadowgen_ml_service.infrastructure.stages.composition.python_composer import PythonComposer
from shadowgen_ml_service.infrastructure.stages.depth.depth_anything import RealDepthEstimator
from shadowgen_ml_service.infrastructure.stages.depth.mock import MockDepthEstimator
from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import RealDetector
from shadowgen_ml_service.infrastructure.stages.detection.mock import MockDetector
from shadowgen_ml_service.infrastructure.stages.geometry.geocalib import RealGeometryEstimator
from shadowgen_ml_service.infrastructure.stages.geometry.mock import MockGeometryEstimator
from shadowgen_ml_service.infrastructure.stages.foreground_refinement.fast_foreground_estimation import FastForegroundColorEstimator
from shadowgen_ml_service.infrastructure.stages.foreground_refinement.mock import PassthroughForegroundColorEstimator
from shadowgen_ml_service.infrastructure.stages.normals.stable_normal import StableNormalEstimator
from shadowgen_ml_service.infrastructure.stages.normals.mock import MockNormalEstimator
from shadowgen_ml_service.infrastructure.stages.normals.from_depth import NormalFromDepthEstimator
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import RealSegmenter
from shadowgen_ml_service.infrastructure.stages.segmentation.mock import MockSegmenter
from shadowgen_ml_service.infrastructure.stages.shadow.stub import DeterministicShadowGenerator


def build_runtime(settings: Settings) -> PipelineRuntime:
    mode = settings.runtime_mode.lower()
    grounding = probe_grounding_dino()
    geocalib = probe_geocalib()
    birefnet = probe_birefnet(allow_cpu=settings.birefnet_allow_cpu)
    fast_foreground = probe_fast_foreground_estimation()
    depth_anything = probe_depth_anything()
    stable_normal = probe_stable_normal(
        allow_cpu=settings.stable_normal_allow_cpu,
        target_device=settings.target_device,
    )

    mock_detector = MockDetector()
    detector = mock_detector
    real_detector = None
    detector_component = component_status(
        "detector",
        "mock",
        grounding.model_name,
        "mock-v1",
        True,
        True,
        "deterministic fallback detector" if not grounding.available else "real wrapper scaffold exists",
    )
    if mode != "mock" and grounding.available:
        try:
            detector = RealDetector(
                model_id=settings.grounding_dino_model_id,
                prompt=settings.grounding_dino_prompt,
                box_threshold=settings.grounding_dino_box_threshold,
                text_threshold=settings.grounding_dino_text_threshold,
                target_device=settings.target_device,
            )
            real_detector = detector
            detector_component = component_status(
                "detector",
                "real",
                grounding.model_name,
                grounding.model_version,
                True,
                False,
                (
                    "GroundingDINO backend active "
                    f"(model_id={settings.grounding_dino_model_id}, prompt={settings.grounding_dino_prompt!r}, "
                    f"box_threshold={settings.grounding_dino_box_threshold}, text_threshold={settings.grounding_dino_text_threshold})"
                ),
            )
        except Exception as exc:
            detector_component = component_status("detector", "mock-fallback", grounding.model_name, "mock-v1", True, True, f"GroundingDINO init failed: {exc}")

    mock_geometry = MockGeometryEstimator()
    geometry = mock_geometry
    real_geometry = None
    geometry_component = component_status(
        "geometry_estimator",
        "mock",
        geocalib.model_name,
        "mock-v1",
        True,
        True,
        "deterministic fallback geometry estimator" if not geocalib.available else "real wrapper scaffold exists",
    )
    if mode != "mock" and geocalib.available:
        try:
            geometry = RealGeometryEstimator(
                weights=settings.geocalib_weights,
                camera_model=settings.geocalib_camera_model,
                shared_intrinsics=settings.geocalib_shared_intrinsics,
            )
            real_geometry = geometry
            geometry_component = component_status(
                "geometry_estimator",
                "real",
                geocalib.model_name,
                geocalib.model_version,
                True,
                False,
                (
                    "GeoCalib backend active "
                    f"(weights={settings.geocalib_weights}, camera_model={settings.geocalib_camera_model}, "
                    f"shared_intrinsics={settings.geocalib_shared_intrinsics})"
                ),
            )
        except Exception as exc:
            geometry_component = component_status("geometry_estimator", "mock-fallback", geocalib.model_name, "mock-v1", True, True, f"GeoCalib init failed: {exc}")

    mock_segmenter = MockSegmenter()
    segmenter = mock_segmenter
    real_segmenter = None
    segmenter_component = component_status(
        "segmenter",
        "mock",
        birefnet.model_name,
        "mock-v1",
        True,
        True,
        birefnet.detail or ("deterministic fallback matting stage" if not birefnet.available else "real wrapper scaffold exists"),
    )
    if mode != "mock" and birefnet.available:
        try:
            segmenter = RealSegmenter(
                model_id=settings.birefnet_model_id,
                resolution=settings.birefnet_resolution,
                mask_threshold=settings.birefnet_mask_threshold,
                target_device=settings.target_device,
            )
            real_segmenter = segmenter
            segmenter_component = component_status(
                "segmenter",
                "real",
                birefnet.model_name,
                birefnet.model_version,
                True,
                False,
                (
                    "BiRefNet backend active "
                    f"(model_id={settings.birefnet_model_id}, resolution={settings.birefnet_resolution}, "
                    f"mask_threshold={settings.birefnet_mask_threshold}, allow_cpu={settings.birefnet_allow_cpu})"
                ),
            )
        except Exception as exc:
            segmenter_component = component_status("segmenter", "mock-fallback", birefnet.model_name, "mock-v1", True, True, f"BiRefNet init failed: {exc}")

    mock_foreground_refiner = PassthroughForegroundColorEstimator()
    foreground_refiner = mock_foreground_refiner
    real_foreground_refiner = None
    foreground_refiner_component = component_status(
        "foreground_refiner",
        "mock",
        fast_foreground.model_name,
        "passthrough-v1",
        True,
        True,
        fast_foreground.detail or "passthrough fallback foreground refinement",
    )
    if mode != "mock" and fast_foreground.available:
        try:
            foreground_refiner = FastForegroundColorEstimator()
            real_foreground_refiner = foreground_refiner
            foreground_refiner_component = component_status(
                "foreground_refiner",
                "real",
                fast_foreground.model_name,
                fast_foreground.model_version,
                True,
                False,
                "Fast Foreground Colour Estimation backend active",
            )
        except Exception as exc:
            foreground_refiner_component = component_status(
                "foreground_refiner",
                "mock-fallback",
                fast_foreground.model_name,
                "passthrough-v1",
                True,
                True,
                f"foreground refinement init failed: {exc}",
            )

    mock_depth = MockDepthEstimator()
    depth = mock_depth
    real_depth = None
    depth_component = component_status(
        "depth_estimator",
        "mock",
        depth_anything.model_name,
        "mock-v1",
        True,
        True,
        "deterministic fallback depth estimator" if not depth_anything.available else "real wrapper scaffold exists",
    )
    if mode != "mock" and depth_anything.available:
        try:
            depth = RealDepthEstimator(
                model_id=settings.depth_anything_model_id,
                target_device=settings.target_device,
            )
            real_depth = depth
            depth_component = component_status(
                "depth_estimator",
                "real",
                depth_anything.model_name,
                depth_anything.model_version,
                True,
                False,
                f"Depth Anything backend active (model_id={settings.depth_anything_model_id})",
            )
        except Exception as exc:
            depth_component = component_status(
                "depth_estimator",
                "mock-fallback",
                depth_anything.model_name,
                "mock-v1",
                True,
                True,
                f"Depth Anything init failed: {exc}",
            )

    mock_normals = MockNormalEstimator()
    normals = mock_normals
    real_normals = None
    if mode == "mock":
        normals_component = component_status(
            "normal_estimator",
            "mock",
            "flat-normal-map",
            "mock-v1",
            True,
            True,
            "flat fallback normals",
        )
    else:
        fallback_normals = NormalFromDepthEstimator()
        normals = fallback_normals
        real_normals = fallback_normals
        normals_component = component_status(
            "normal_estimator",
            "real",
            "normal-map-from-depth",
            fallback_normals.model_variant,
            True,
            False,
            stable_normal.detail or "StableNormal unavailable; using depth-derived normals fallback",
        )
        if stable_normal.available:
            try:
                normals = StableNormalEstimator(
                    model_variant=settings.stable_normal_variant,
                    resolution=settings.stable_normal_resolution,
                    target_device=settings.target_device,
                    allow_cpu=settings.stable_normal_allow_cpu,
                    cache_dir=settings.model_cache_dir / "stable-normal",
                )
                real_normals = normals
                normals_component = component_status(
                    "normal_estimator",
                    "real",
                    stable_normal.model_name,
                    settings.stable_normal_variant,
                    True,
                    False,
                    (
                        "StableNormal backend active "
                        f"(variant={settings.stable_normal_variant}, resolution={settings.stable_normal_resolution}, "
                        f"allow_cpu={settings.stable_normal_allow_cpu})"
                    ),
                )
            except Exception as exc:
                normals_component = component_status(
                    "normal_estimator",
                    "real",
                    "normal-map-from-depth",
                    fallback_normals.model_variant,
                    True,
                    False,
                    f"StableNormal init failed; using depth-derived normals fallback: {exc}",
                )

    components = [
        detector_component,
        geometry_component,
        segmenter_component,
        foreground_refiner_component,
        depth_component,
        normals_component,
        component_status("shadow_generator", "deterministic-stub", "shadow-stub", "v1", True, False),
        component_status("composer", "python", "solid-background-composer", "v1", True, False),
        component_status("artifact_encoder", "python", "artifact-encoder", "v1", True, False),
    ]
    return PipelineRuntime(
        detector=detector,
        mock_detector=mock_detector,
        real_detector=real_detector,
        geometry=geometry,
        mock_geometry=mock_geometry,
        real_geometry=real_geometry,
        segmenter=segmenter,
        mock_segmenter=mock_segmenter,
        real_segmenter=real_segmenter,
        foreground_refiner=foreground_refiner,
        mock_foreground_refiner=mock_foreground_refiner,
        real_foreground_refiner=real_foreground_refiner,
        depth=depth,
        mock_depth=mock_depth,
        real_depth=real_depth,
        normals=normals,
        mock_normals=mock_normals,
        real_normals=real_normals,
        shadow=DeterministicShadowGenerator(),
        composer=PythonComposer(),
        encoder=DefaultArtifactEncoder(),
        cache=FilesystemPreprocessCacheRepository(settings.preprocess_cache_dir),
        previews=DefaultPreviewBuilderRegistry(),
        descriptor=build_runtime_descriptor(mode, components),
    )
