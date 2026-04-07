from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from shadowgen_ml_service.app import create_app
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.types import SegmentationResult
from shadowgen_ml_service.pipeline.service import TimeoutServiceError


def make_image_base64(image_format: str = "PNG") -> tuple[str, str]:
    image = Image.new("RGBA", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 40, 170, 200), radius=22, fill=(220, 55, 55, 255))
    buffer = BytesIO()
    save_image = image.convert("RGB") if image_format == "JPEG" else image
    save_image.save(buffer, format=image_format)
    mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
    return mime_type, base64.b64encode(buffer.getvalue()).decode("ascii")


def make_request(image_format: str = "PNG") -> dict:
    mime_type, image_base64 = make_image_base64(image_format=image_format)
    return {
        "request_id": "test-request",
        "pipeline_version": "ml-shadowgen-v1",
        "source": {"mime_type": mime_type, "image_base64": image_base64},
        "shadow": {
            "angle_deg": 45,
            "elevation_deg": 35,
            "softness": 0.5,
            "opacity": 0.65,
            "reflection": 0.1,
        },
        "background": {"mode": "solid", "color_hex": "#FFFFFF"},
        "output": {"format": "png", "width": None, "height": None, "return_debug": True},
    }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app(Settings())
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_capabilities(self) -> None:
        response = self.client.get("/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("components", payload)
        self.assertIn("active_backend_mode", payload)

    def test_playground_page_exists(self) -> None:
        response = self.client.get("/playground")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ShadowGen Pipeline Playground", response.text)

    def test_render_success_with_debug_artifacts(self) -> None:
        response = self.client.post("/v1/render", json=make_request())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        artifact_names = {artifact["name"] for artifact in payload["artifacts"]}
        self.assertIn("final", artifact_names)
        self.assertIn("shadow", artifact_names)
        self.assertEqual(payload["metrics"]["total_ms"] >= 0, True)
        self.assertIsInstance(payload["warnings"], list)

    def test_padding_default_and_override(self) -> None:
        default_response = self.client.post("/v1/render", json=make_request())
        custom_request = make_request()
        custom_request["preprocess"] = {"padding_px": 160}
        custom_response = self.client.post("/v1/render", json=custom_request)
        self.assertEqual(default_response.status_code, 200)
        self.assertEqual(custom_response.status_code, 200)

    def test_angle_and_elevation_change_output(self) -> None:
        base_request = make_request()
        base_response = self.client.post("/v1/render", json=base_request)

        changed_request = make_request()
        changed_request["shadow"]["angle_deg"] = 220
        changed_request["shadow"]["elevation_deg"] = 12
        changed_response = self.client.post("/v1/render", json=changed_request)

        base_final = next(item for item in base_response.json()["artifacts"] if item["name"] == "final")["image_base64"]
        changed_final = next(item for item in changed_response.json()["artifacts"] if item["name"] == "final")["image_base64"]
        self.assertNotEqual(base_final, changed_final)

    def test_invalid_mime_returns_unsupported_input(self) -> None:
        payload = make_request()
        payload["source"]["mime_type"] = "image/webp"
        response = self.client.post("/v1/render", json=payload)
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["error"]["code"], "unsupported_input")

    def test_validation_error_shape(self) -> None:
        payload = make_request()
        payload["output"]["format"] = "gif"
        response = self.client.post("/v1/render", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")

    def test_debug_pipeline_run_all(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-all",
            json={"render_request": make_request(), "stage_modes": {"shadow_generator": "real", "composer": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["stages"][0]["stage_key"], "decode")
        self.assertEqual(payload["stages"][-1]["stage_key"], "composer")
        geometry_stage = next(item for item in payload["stages"] if item["stage_key"] == "geometry_estimator")
        self.assertIn("details", geometry_stage)
        preview_names = {preview["name"] for preview in geometry_stage["previews"]}
        self.assertIn("geometry_overlay", preview_names)

    def test_debug_pipeline_stage_real_failure(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/detector",
            json={"render_request": make_request(), "stage_modes": {"detector": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        detector = payload["stages"][-1]
        self.assertEqual(detector["stage_key"], "detector")
        self.assertEqual(detector["status"], "failed")
        self.assertIn("Real detector", detector["error"])

    def test_debug_pipeline_geometry_real_uses_mock_fallback(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/geometry_estimator",
            json={"render_request": make_request(), "stage_modes": {"geometry_estimator": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        geometry = response.json()["stages"][-1]
        self.assertEqual(geometry["stage_key"], "geometry_estimator")
        self.assertEqual(geometry["status"], "completed")
        self.assertIn("camera_fov", geometry["details"])
        self.assertIn(geometry["actual_mode"], {"real", "mock-fallback"})
        if geometry["actual_mode"] == "real":
            self.assertEqual(geometry["details"]["backend"], "real")
        else:
            self.assertEqual(geometry["details"]["backend"], "mock")

    def test_debug_pipeline_geometry_real_smoke(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/geometry_estimator",
            json={"render_request": make_request(), "stage_modes": {"geometry_estimator": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        geometry = response.json()["stages"][-1]
        self.assertEqual(geometry["status"], "completed")
        self.assertGreaterEqual(geometry["elapsed_ms"], 0)
        self.assertIn("camera_pitch", geometry["details"])
        preview_names = {preview["name"] for preview in geometry["previews"]}
        self.assertIn("geometry_input", preview_names)
        self.assertIn("geometry_overlay", preview_names)

    def test_segmentation_runs_after_crop_and_resize(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(Settings(preprocess_cache_dir=Path(temp_dir)))
            service = app.state.render_service

            class RecordingSegmenter:
                def __init__(self) -> None:
                    self.seen_size = None

                def segment(self, image):
                    self.seen_size = image.size
                    mask = Image.new("L", image.size, 255)
                    cutout = image.copy()
                    cutout.putalpha(mask)
                    return SegmentationResult(
                        bbox=(0, 0, image.width, image.height),
                        mask=mask,
                        cutout_rgba=cutout,
                        crop_rgba=image,
                    )

            recorder = RecordingSegmenter()
            service.runtime.segmenter = recorder
            client = TestClient(app)
            response = client.post("/v1/render", json=make_request())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(recorder.seen_size, (service.settings.working_size, service.settings.working_size))

    def test_timeout_error_mapping(self) -> None:
        class FakeService:
            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {"service_version": "0.1.0"}

            def render(self, payload):
                raise TimeoutServiceError("forced timeout", request_id=payload.request_id)

        app = create_app(Settings())
        app.state.render_service = FakeService()
        client = TestClient(app)
        response = client.post("/v1/render", json=make_request())
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "timeout")


if __name__ == "__main__":
    unittest.main()
