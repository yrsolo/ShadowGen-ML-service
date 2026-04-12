from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from shadowgen_ml_service.core.stage_io import SegmentationInput
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding, TritonTensorBinding
from shadowgen_ml_service.infrastructure.stages.segmentation.onnx_export import (
    default_segmenter_onnx_contract,
    validate_segmenter_export_names,
)
from shadowgen_ml_service.infrastructure.stages.segmentation.triton import TritonSegmenter
from shadowgen_ml_service.utils.images import asset_to_pil, pil_to_asset


class _FakeClient:
    def __init__(self, payload: dict[str, np.ndarray]) -> None:
        self.payload = payload
        self.calls: list[list[dict]] = []

    def infer(self, binding, *, inputs: list[dict]) -> dict[str, np.ndarray]:
        self.calls.append(inputs)
        return self.payload


class TritonSegmenterTests(unittest.TestCase):
    def test_segmenter_onnx_export_contract_requires_mask_only_names(self) -> None:
        contract = default_segmenter_onnx_contract()
        validate_segmenter_export_names(input_names=[contract.input_name], output_names=[contract.output_name])
        with self.assertRaises(ValueError):
            validate_segmenter_export_names(input_names=["input"], output_names=[contract.output_name])

    def test_triton_segmenter_builds_cutout_from_mask_only_output(self) -> None:
        binding = TritonModelBinding(
            stage_key="segmenter",
            model_variant="birefnet",
            model_name="shadowgen_segmenter",
            inputs={"image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3)},
            outputs={"mask": TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1)},
        )
        mask = np.ones((1, 1, 6, 8), dtype=np.float32)
        mask[:, :, :2, :] = 0.0
        mask[:, :, :, :2] = 0.0
        client = _FakeClient({"mask": mask})
        segmenter = TritonSegmenter(client, binding)

        image = Image.new("RGBA", (8, 6), (255, 100, 0, 255))
        result = segmenter.segment(SegmentationInput(image=pil_to_asset(image)))

        self.assertEqual(result.mask.size, (8, 6))
        self.assertEqual(result.crop_rgba.size, (8, 6))
        self.assertEqual(result.bbox, (2, 2, 8, 6))
        cutout = asset_to_pil(result.cutout_rgba)
        self.assertEqual(cutout.getchannel("A").getpixel((0, 0)), 0)
        self.assertGreater(cutout.getchannel("A").getpixel((4, 4)), 0)

    def test_triton_segmenter_uses_full_output_payload_when_available(self) -> None:
        binding = TritonModelBinding(
            stage_key="segmenter",
            model_variant="birefnet",
            model_name="shadowgen_segmenter",
            inputs={"image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3)},
            outputs={
                "bbox": TritonTensorBinding("bbox", "FP32", expected_ranks=(2,), shape_policy="bbox4"),
                "mask": TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1),
                "cutout": TritonTensorBinding("cutout", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=4),
                "crop": TritonTensorBinding("crop", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=4),
            },
        )
        client = _FakeClient(
            {
                "bbox": np.asarray([[1.0, 2.0, 7.0, 5.0]], dtype=np.float32),
                "mask": np.ones((1, 1, 6, 8), dtype=np.float32),
                "cutout": np.ones((1, 4, 6, 8), dtype=np.float32),
                "crop": np.zeros((1, 4, 6, 8), dtype=np.float32),
            }
        )
        segmenter = TritonSegmenter(client, binding)

        image = Image.new("RGBA", (8, 6), (10, 20, 30, 255))
        result = segmenter.segment(SegmentationInput(image=pil_to_asset(image)))

        self.assertEqual(result.bbox, (1, 2, 7, 5))
        self.assertEqual(asset_to_pil(result.cutout_rgba).getpixel((0, 0)), (255, 255, 255, 255))
        self.assertEqual(asset_to_pil(result.crop_rgba).getpixel((0, 0)), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
