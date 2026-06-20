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
                    "padding_px": TritonTensorBinding("padding_px", "INT32", expected_ranks=(1, 2), shape_policy="scalar"),
                },
                outputs={
                    "bbox": TritonTensorBinding("bbox", "FP32", expected_ranks=(1, 2), shape_policy="bbox4"),
                    "confidence": TritonTensorBinding("confidence", "FP32", expected_ranks=(1, 2), shape_policy="scalar"),
                },
            ),
            TritonModelBinding(
                "detector",
                "grounding-dino-onnx",
                settings.triton_detector_onnx_model,
                inputs={
                    "pixel_values": TritonTensorBinding("pixel_values", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3),
                    "pixel_mask": TritonTensorBinding("pixel_mask", "INT64", expected_ranks=(3,), shape_policy=None),
                    "input_ids": TritonTensorBinding("input_ids", "INT64", expected_ranks=(2,), shape_policy=None),
                    "token_type_ids": TritonTensorBinding("token_type_ids", "INT64", expected_ranks=(2,), shape_policy=None),
                    "attention_mask": TritonTensorBinding("attention_mask", "INT64", expected_ranks=(2,), shape_policy=None),
                },
                outputs={
                    "logits": TritonTensorBinding("logits", "FP32", expected_ranks=(3,), shape_policy=None),
                    "pred_boxes": TritonTensorBinding("pred_boxes", "FP32", expected_ranks=(3,), shape_policy=None),
                },
            ),
            TritonModelBinding(
                "segmenter",
                "birefnet",
                settings.triton_segmenter_model,
                inputs={"image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3)},
                outputs={"mask": TritonTensorBinding("mask", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=1)},
            ),
            TritonModelBinding(
                "segmenter",
                "rmbg-2.0",
                settings.triton_segmenter_rmbg2_model,
                inputs={
                    "image": TritonTensorBinding(
                        settings.triton_segmenter_rmbg2_input,
                        "FP32",
                        expected_ranks=(4,),
                        shape_policy="channel-first",
                        channels=3,
                    )
                },
                outputs={
                    "mask": TritonTensorBinding(
                        settings.triton_segmenter_rmbg2_output,
                        "FP32",
                        expected_ranks=(3, 4),
                        shape_policy="channel-first",
                        channels=1,
                    )
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
                },
                outputs={"shadow_image": TritonTensorBinding("shadow_image", "FP32", expected_ranks=(3, 4), shape_policy="channel-first", channels=4)},
            ),
        ]
    )
