from shadowgen_ml_service.infrastructure.stages.depth.depth_anything import RealDepthEstimator, probe_depth_anything
from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import RealDetector, probe_grounding_dino, select_primary_detection
from shadowgen_ml_service.infrastructure.stages.foreground_refinement.fast_foreground_estimation import FastForegroundColorEstimator, probe_fast_foreground_estimation
from shadowgen_ml_service.infrastructure.stages.geometry.geocalib import RealGeometryEstimator, probe_geocalib
from shadowgen_ml_service.infrastructure.stages.normals.from_depth import NormalFromDepthEstimator
from shadowgen_ml_service.infrastructure.stages.normals.stable_normal import StableNormalEstimator, probe_stable_normal
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import RealSegmenter, probe_birefnet
from shadowgen_ml_service.infrastructure.stages.shadow.pix2pix import Pix2PixShadowGenerator, probe_shadow_pix2pix
from shadowgen_ml_service.infrastructure.stages.shadow.v2_diff import V2DiffShadowGenerator
from shadowgen_ml_service.infrastructure.stages.shared.model_support import import_module as _import_module

__all__ = [
    "RealDetector",
    "FastForegroundColorEstimator",
    "RealDepthEstimator",
    "RealGeometryEstimator",
    "NormalFromDepthEstimator",
    "StableNormalEstimator",
    "RealSegmenter",
    "Pix2PixShadowGenerator",
    "V2DiffShadowGenerator",
    "_import_module",
    "probe_birefnet",
    "probe_depth_anything",
    "probe_fast_foreground_estimation",
    "probe_geocalib",
    "probe_grounding_dino",
    "probe_stable_normal",
    "probe_shadow_pix2pix",
    "select_primary_detection",
]
