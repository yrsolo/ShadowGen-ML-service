from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding, TritonTensorBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import (
    image_to_nchw_float32_input,
    scalar_to_input,
    tensor_map_from_response,
)


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
        settings = TritonBackendSettings(url="http://triton.local", protocol="http", timeout_ms=1000)
        client = TritonInferenceClient(settings)
        captured: dict[str, object] = {}

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
                "detector",
                inputs=[{"name": "image", "datatype": "FP32", "shape": [1, 3, 6, 8], "data": [0.0] * (1 * 3 * 6 * 8)}],
                outputs=["bbox", "confidence"],
            )

        self.assertEqual(captured["url"], "http://triton.local/v2/models/detector/infer")
        self.assertEqual(captured["payload"]["outputs"], [{"name": "bbox"}, {"name": "confidence"}])
        self.assertEqual(captured["payload"]["inputs"][0]["name"], "image")
        self.assertEqual(tensors["bbox"].shape, (1, 4))

    def test_model_binding_carries_tensor_schema(self) -> None:
        binding = TritonModelBinding(
            stage_key="shadow_generator",
            model_variant="v2-diff",
            model_name="shadow-v2",
            inputs={"img": TritonTensorBinding("img", "FP32")},
            outputs={"shadow": TritonTensorBinding("shadow", "FP32")},
        )
        self.assertEqual(binding.inputs["img"].tensor_name, "img")
        self.assertEqual(binding.outputs["shadow"].datatype, "FP32")


if __name__ == "__main__":
    unittest.main()
