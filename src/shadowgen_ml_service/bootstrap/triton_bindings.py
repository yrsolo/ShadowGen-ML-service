from __future__ import annotations

from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding, TritonModelRegistry, TritonTensorBinding


def build_triton_model_registry(settings: Settings) -> TritonModelRegistry:
    return TritonModelRegistry(
        [
            TritonModelBinding(
                "detector",
                "grounding-dino",
                settings.triton_detector_model,
                inputs={
                    "image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3),
                    "padding_px": TritonTensorBinding("padding_px", "INT32", expected_ranks=(1,), shape_policy="scalar"),
                },
                outputs={
                    "bbox": TritonTensorBinding("bbox", "FP32", expected_ranks=(1, 2), shape_policy="bbox4"),
                    "confidence": TritonTensorBinding("confidence", "FP32", expected_ranks=(1,), shape_policy="scalar"),
                },
            ),
            TritonModelBinding(
                "segmenter",
                "birefnet",
                settings.triton_segmenter_model,
                inputs={"image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3)},
                outputs={
                    "bbox": TritonTensorBinding("bbox", "FP32", expected_ranks=(1, 2), shape_policy="bbox4"),
                    "mask": TritonTensorBinding("mask", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=1),
                    "cutout": TritonTensorBinding("cutout", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=4),
                    "crop": TritonTensorBinding("crop", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=4),
                },
            ),
            TritonModelBinding(
                "depth_estimator",
                "depth-anything-v2-small",
                settings.triton_depth_model,
                inputs={
                    "image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3),
                    "mask": TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1),
                },
                outputs={"depth": TritonTensorBinding("depth", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=1)},
            ),
            TritonModelBinding(
                "normal_estimator",
                "stable-normal",
                settings.triton_normals_model,
                inputs={
                    "image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3),
                    "depth": TritonTensorBinding("depth", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1),
                },
                outputs={"normal": TritonTensorBinding("normal", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=3)},
            ),
            TritonModelBinding(
                "shadow_generator",
                "v2-diff",
                settings.triton_shadow_v2_model,
                inputs={
                    "img": TritonTensorBinding("img", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=4),
                    "mask": TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1),
                    "depth": TritonTensorBinding("depth", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1),
                    "normal": TritonTensorBinding("normal", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3),
                    "angle": TritonTensorBinding("angle", "FP32", expected_ranks=(1,), shape_policy="scalar"),
                    "elevation": TritonTensorBinding("elevation", "FP32", expected_ranks=(1,), shape_policy="scalar"),
                    "softness": TritonTensorBinding("softness", "FP32", expected_ranks=(1,), shape_policy="scalar"),
                    "reflection": TritonTensorBinding("reflection", "FP32", expected_ranks=(1,), shape_policy="scalar"),
                },
                outputs={"shadow": TritonTensorBinding("shadow", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=4)},
            ),
        ]
    )
