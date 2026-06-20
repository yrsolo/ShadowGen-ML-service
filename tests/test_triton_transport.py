from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.batching import TritonStageBatchCoordinator
from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings
from shadowgen_ml_service.infrastructure.backends.triton.errors import TritonSchemaMismatchError
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding, TritonTensorBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import (
    batch_input_tensors,
    image_to_nchw_float32_input,
    scalar_to_input,
    split_output_tensor,
    tensor_map_from_response,
    validate_tensor_against_binding,
)
from shadowgen_ml_service.infrastructure.stages.shadow.triton import TritonShadowGenerator
from shadowgen_ml_service.core.stage_io import ShadowInput
from shadowgen_ml_service.utils.images import pil_to_asset


class _FakeHttpResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTritonClient:
    def __init__(self) -> None:
        self.last_inputs: list[dict] | None = None

    def infer(self, binding: TritonModelBinding, *, inputs: list[dict]):
        self.last_inputs = inputs
        return {"shadow_image": np.zeros((1, 4, 4, 4), dtype=np.float32)}


class TritonTransportTests(unittest.TestCase):
    def test_image_input_uses_standard_nchw_fp32_shape(self) -> None:
        tensor = image_to_nchw_float32_input("image", Image.new("RGB", (8, 6), "white"))
        self.assertEqual(tensor["name"], "image")
        self.assertEqual(tensor["datatype"], "FP32")
        self.assertEqual(tensor["shape"], [1, 3, 6, 8])

    def test_scalar_input_uses_requested_datatype(self) -> None:
        tensor = scalar_to_input("padding_px", 128, datatype="INT32")
        self.assertEqual(tensor["datatype"], "INT32")
        self.assertEqual(tensor["shape"], [1])
        self.assertEqual(tensor["data"], [128])

    def test_tensor_map_from_response_parses_named_outputs(self) -> None:
        response = {
            "outputs": [
                {"name": "bbox", "datatype": "FP32", "shape": [1, 4], "data": [1.0, 2.0, 3.0, 4.0]},
                {"name": "confidence", "datatype": "FP32", "shape": [1], "data": [0.75]},
            ]
        }
        tensors = tensor_map_from_response(response)
        self.assertEqual(sorted(tensors.keys()), ["bbox", "confidence"])
        self.assertEqual(tensors["bbox"].shape, (1, 4))
        self.assertAlmostEqual(float(tensors["confidence"][0]), 0.75)

    def test_client_posts_standard_triton_infer_payload(self) -> None:
        settings = TritonBackendSettings(url="http://triton.local", protocol="http", timeout_ms=1000, transport="json")
        client = TritonInferenceClient(settings)
        captured: dict[str, object] = {}
        binding = TritonModelBinding(
            stage_key="detector",
            model_variant="grounding-dino",
            model_name="detector",
            inputs={"image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3)},
            outputs={
                "bbox": TritonTensorBinding("bbox", "FP32", expected_ranks=(2,), shape_policy="bbox4"),
                "confidence": TritonTensorBinding("confidence", "FP32", expected_ranks=(1,), shape_policy="scalar"),
            },
        )

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeHttpResponse(
                {
                    "outputs": [
                        {"name": "bbox", "datatype": "FP32", "shape": [1, 4], "data": [10.0, 20.0, 30.0, 40.0]},
                        {"name": "confidence", "datatype": "FP32", "shape": [1], "data": [0.91]},
                    ]
                }
            )

        with patch("shadowgen_ml_service.infrastructure.backends.triton.client.request.urlopen", side_effect=fake_urlopen):
            tensors = client.infer(
                binding,
                inputs=[{"name": "image", "datatype": "FP32", "shape": [1, 3, 6, 8], "data": [0.0] * (1 * 3 * 6 * 8)}],
            )

        self.assertEqual(captured["url"], "http://triton.local/v2/models/detector/infer")
        self.assertEqual(captured["payload"]["outputs"], [{"name": "bbox"}, {"name": "confidence"}])
        self.assertEqual(captured["payload"]["inputs"][0]["name"], "image")
        self.assertEqual(tensors["bbox"].shape, (1, 4))

    def test_probe_binding_checks_model_ready_and_metadata(self) -> None:
        settings = TritonBackendSettings(url="http://triton.local", protocol="http", timeout_ms=1000)
        client = TritonInferenceClient(settings)
        binding = TritonModelBinding(
            stage_key="segmenter",
            model_variant="birefnet",
            model_name="shadowgen_segmenter",
            inputs={"image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3)},
            outputs={"mask": TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1)},
        )

        def fake_urlopen(req, timeout):
            if req.full_url.endswith("/v2/models/shadowgen_segmenter/ready"):
                return _FakeHttpResponse({}, status=200)
            if req.full_url.endswith("/v2/models/shadowgen_segmenter"):
                return _FakeHttpResponse(
                    {
                        "name": "shadowgen_segmenter",
                        "inputs": [{"name": "image", "datatype": "FP32", "shape": ["3", "-1", "-1"]}],
                        "outputs": [{"name": "mask", "datatype": "FP32", "shape": ["1", "-1", "-1"]}],
                    }
                )
            raise AssertionError(f"unexpected url {req.full_url}")

        with patch("shadowgen_ml_service.infrastructure.backends.triton.client.request.urlopen", side_effect=fake_urlopen):
            available, detail = client.probe_binding(binding)

        self.assertTrue(available)
        self.assertIn("shadowgen_segmenter", detail)

    def test_probe_binding_rejects_schema_mismatch(self) -> None:
        settings = TritonBackendSettings(url="http://triton.local", protocol="http", timeout_ms=1000)
        client = TritonInferenceClient(settings)
        binding = TritonModelBinding(
            stage_key="segmenter",
            model_variant="birefnet",
            model_name="shadowgen_segmenter",
            inputs={"image": TritonTensorBinding("image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=3)},
            outputs={"mask": TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1)},
        )

        def fake_urlopen(req, timeout):
            if req.full_url.endswith("/v2/models/shadowgen_segmenter/ready"):
                return _FakeHttpResponse({}, status=200)
            if req.full_url.endswith("/v2/models/shadowgen_segmenter"):
                return _FakeHttpResponse(
                    {
                        "name": "shadowgen_segmenter",
                        "inputs": [{"name": "image", "datatype": "FP32", "shape": ["4", "-1", "-1"]}],
                        "outputs": [{"name": "mask", "datatype": "FP32", "shape": ["1", "-1", "-1"]}],
                    }
                )
            raise AssertionError(f"unexpected url {req.full_url}")

        with patch("shadowgen_ml_service.infrastructure.backends.triton.client.request.urlopen", side_effect=fake_urlopen):
            available, detail = client.probe_binding(binding)

        self.assertFalse(available)
        self.assertIn("channels", detail)

    def test_model_binding_carries_tensor_schema(self) -> None:
        binding = TritonModelBinding(
            stage_key="shadow_generator",
            model_variant="v2-diff",
            model_name="shadow-v2",
            inputs={"img": TritonTensorBinding("img", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=4)},
            outputs={"shadow_image": TritonTensorBinding("shadow_image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=4)},
        )
        self.assertEqual(binding.inputs["img"].tensor_name, "img")
        self.assertEqual(binding.outputs["shadow_image"].datatype, "FP32")
        self.assertEqual(binding.outputs["shadow_image"].channels, 4)

    def test_shadow_v2_triton_adapter_sends_only_declared_inputs(self) -> None:
        binding = TritonModelBinding(
            stage_key="shadow_generator",
            model_variant="v2-diff",
            model_name="shadowgen_shadow_v2",
            inputs={
                "img": TritonTensorBinding("img", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=4),
                "mask": TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1),
            },
            outputs={"shadow_image": TritonTensorBinding("shadow_image", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=4)},
        )
        fake_client = _FakeTritonClient()
        generator = TritonShadowGenerator(fake_client, binding, model_variant="v2-diff")
        image = Image.new("RGBA", (4, 4), (255, 128, 64, 255))
        mask = Image.new("L", (4, 4), 255)
        stage_input = ShadowInput(
            img=pil_to_asset(image),
            mask=pil_to_asset(mask),
            depth=pil_to_asset(mask),
            normal=pil_to_asset(Image.new("RGB", (4, 4), (127, 127, 255))),
            angle=45.0,
            elevation=35.0,
            softness=0.5,
            reflection=0.1,
            opacity=0.65,
        )

        result = generator.generate(stage_input)

        self.assertEqual(result.shadow_image.width, 4)
        self.assertEqual([item["name"] for item in fake_client.last_inputs or []], ["img", "mask"])

    def test_schema_validation_rejects_wrong_channel_count(self) -> None:
        binding = TritonTensorBinding("mask", "FP32", expected_ranks=(4,), shape_policy="channel-first", channels=1)
        tensor = np.zeros((1, 3, 16, 16), dtype=np.float32)
        with self.assertRaises(TritonSchemaMismatchError):
            validate_tensor_against_binding("mask", tensor, binding)

    def test_batch_input_tensors_concatenates_batch_dimension(self) -> None:
        first = [image_to_nchw_float32_input("image", Image.new("RGB", (4, 4), "white"))]
        second = [image_to_nchw_float32_input("image", Image.new("RGB", (4, 4), "black"))]
        merged = batch_input_tensors([first, second])
        self.assertEqual(merged[0]["shape"], [2, 3, 4, 4])

    def test_split_output_tensor_returns_per_item_views(self) -> None:
        tensor = np.zeros((2, 1, 8, 8), dtype=np.float32)
        parts = split_output_tensor(tensor, 2)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (1, 1, 8, 8))

    def test_stage_batch_coordinator_groups_compatible_calls(self) -> None:
        coordinator = TritonStageBatchCoordinator(
            enabled=True,
            window_ms=25,
            max_size=4,
            stage_enabled={"segmenter": True},
        )
        seen = []
        result_holder = []

        def worker(value: int) -> None:
            result_holder.append(
                coordinator.execute(
                    stage_key="segmenter",
                    model_variant="birefnet",
                    payload=value,
                    run_batch=lambda payloads: seen.append(list(payloads)) or [item * 10 for item in payloads],
                )
            )

        import threading

        first = threading.Thread(target=worker, args=(1,))
        second = threading.Thread(target=worker, args=(2,))
        first.start()
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertEqual(len(seen), 1)
        self.assertEqual(sorted(seen[0]), [1, 2])
        self.assertEqual(sorted(result_holder), [10, 20])


if __name__ == "__main__":
    unittest.main()
