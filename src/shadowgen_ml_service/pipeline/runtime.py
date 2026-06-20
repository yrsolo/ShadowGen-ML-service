from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.bootstrap.container import build_runtime
from shadowgen_ml_service.bootstrap.probes import (
    probe_birefnet,
    probe_depth_anything,
    probe_fast_foreground_estimation,
    probe_geocalib,
    probe_grounding_dino,
    probe_shadow_pix2pix,
    probe_stable_normal,
)
from shadowgen_ml_service.infrastructure.stages.depth.depth_anything import RealDepthEstimator
from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import RealDetector
from shadowgen_ml_service.infrastructure.stages.foreground_refinement.fast_foreground_estimation import FastForegroundColorEstimator
from shadowgen_ml_service.infrastructure.stages.geometry.geocalib import RealGeometryEstimator
from shadowgen_ml_service.infrastructure.stages.normals.stable_normal import StableNormalEstimator
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import RealSegmenter
from shadowgen_ml_service.infrastructure.stages.shadow.pix2pix import Pix2PixShadowGenerator

__all__ = [
    "PipelineRuntime",
    "RealDetector",
    "RealGeometryEstimator",
    "RealSegmenter",
    "FastForegroundColorEstimator",
    "RealDepthEstimator",
    "StableNormalEstimator",
    "Pix2PixShadowGenerator",
    "build_runtime",
    "probe_birefnet",
    "probe_depth_anything",
    "probe_fast_foreground_estimation",
    "probe_stable_normal",
    "probe_shadow_pix2pix",
    "probe_geocalib",
    "probe_grounding_dino",
]
