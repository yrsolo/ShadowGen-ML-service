from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from shadowgen_ml_service.app import create_app
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.types import DetectionResult, ForegroundRefinementResult, GeometryResult
from shadowgen_ml_service.pipeline.types import SegmentationResult
from shadowgen_ml_service.pipeline.service import TimeoutServiceError
from shadowgen_ml_service.utils.images import asset_to_pil, pil_to_asset


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
        self.assertIn(response.json()["status"], {"ok", "degraded"})
        self.assertIn("accepting_jobs", response.json())
        self.assertIn("preferred_submit_mode", response.json())

    def test_capabilities(self) -> None:
        response = self.client.get("/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("components", payload)
        self.assertIn("active_backend_mode", payload)
        self.assertIn("execution_default_backend", payload)
        self.assertIn("async_enabled", payload)
        self.assertIn("supported_submit_modes", payload)
        self.assertIn("preferred_submit_mode", payload)
        self.assertIn("job_execution", payload)
        self.assertIn("batching_strategy", payload)

    def test_playground_page_exists(self) -> None:
        response = self.client.get("/playground")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ShadowGen Pipeline Playground", response.text)

    def test_async_render_job_lifecycle(self) -> None:
        submit = self.client.post("/v1/render/jobs", json=make_request())
        self.assertEqual(submit.status_code, 202)
        job_id = submit.json()["job_id"]
        self.assertTrue(job_id)
        self.assertEqual(submit.json()["submit_mode"], "async")
        poll = None
        for _ in range(30):
            poll = self.client.get(f"/v1/render/jobs/{job_id}")
            self.assertEqual(poll.status_code, 200)
            if poll.json()["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        self.assertIsNotNone(poll)
        self.assertIn(poll.json()["status"], {"completed", "failed"})
        self.assertIn("capacity_snapshot", poll.json())

    def test_async_submit_is_idempotent_by_request_id(self) -> None:
        first = self.client.post("/v1/render/jobs", json=make_request())
        second = self.client.post("/v1/render/jobs", json=make_request())
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])

    def test_async_queue_full_returns_429(self) -> None:
        app = create_app(Settings(job_max_running=1, job_max_pending=1))
        entered = threading.Event()
        release = threading.Event()
        original_execute = app.state.render_use_case.execute

        class SlowRenderUseCase:
            def execute(self, command):
                entered.set()
                release.wait(timeout=2)
                return original_execute(command)

        app.state.submit_job_use_case.render_use_case = SlowRenderUseCase()
        app.state.render_service._submit_job_use_case.render_use_case = app.state.submit_job_use_case.render_use_case
        client = TestClient(app)

        first_request = make_request()
        first_request["request_id"] = "queue-full-1"
        second_request = make_request()
        second_request["request_id"] = "queue-full-2"
        third_request = make_request()
        third_request["request_id"] = "queue-full-3"

        first = client.post("/v1/render/jobs", json=first_request)
        entered.wait(timeout=1)
        second = client.post("/v1/render/jobs", json=second_request)
        third = client.post("/v1/render/jobs", json=third_request)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(third.status_code, 429)
        self.assertEqual(third.json()["error"]["code"], "queue_full")
        release.set()

    def test_sync_render_respects_capacity_limit(self) -> None:
        app = create_app(Settings(job_max_running=1))
        client = TestClient(app)

        entered = threading.Event()
        release = threading.Event()
        original_execute = app.state.render_use_case.execute

        class SlowRenderUseCase:
            def execute(self, command):
                entered.set()
                release.wait(timeout=2)
                return original_execute(command)

        app.state.render_use_case = SlowRenderUseCase()
        app.state.render_service._render_use_case = app.state.render_use_case

        holder: dict[str, object] = {}

        def run_first_render():
            holder["response"] = client.post("/v1/render", json=make_request())

        thread = threading.Thread(target=run_first_render)
        thread.start()
        entered.wait(timeout=1)

        blocked = client.post("/v1/render", json=make_request())
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.json()["error"]["code"], "queue_full")

        release.set()
        thread.join(timeout=3)
        self.assertEqual(holder["response"].status_code, 200)

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
            json={"render_request": make_request(), "stage_modes": {"shadow_generator": "v1-gan", "composer": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["stages"][0]["stage_key"], "decode")
        self.assertEqual(payload["stages"][-1]["stage_key"], "composer")
        geometry_stage = next(item for item in payload["stages"] if item["stage_key"] == "geometry_estimator")
        self.assertIn("details", geometry_stage)
        preview_names = {preview["name"] for preview in geometry_stage["previews"]}
        self.assertIn("geometry_overlay", preview_names)
        detection_stage = next(item for item in payload["stages"] if item["stage_key"] == "detector")
        detection_preview_names = {preview["name"] for preview in detection_stage["previews"]}
        self.assertIn("crop_for_resize", detection_preview_names)

    def test_debug_pipeline_segmenter_real_smoke(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/segmenter",
            json={"render_request": make_request(), "stage_modes": {"segmenter": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        segmenter = payload["stages"][-1]
        self.assertEqual(segmenter["stage_key"], "segmenter")
        self.assertEqual(segmenter["status"], "completed")
        self.assertIn(segmenter["actual_mode"], {"local", "local-fallback", "mock-fallback"})
        self.assertIn("mask_width", segmenter["details"])
        preview_names = {preview["name"] for preview in segmenter["previews"]}
        self.assertIn("working_crop", preview_names)
        self.assertIn("cutout", preview_names)
        self.assertIn("mask", preview_names)

    def test_debug_pipeline_foreground_refiner_real_smoke(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/foreground_refiner",
            json={"render_request": make_request(), "stage_modes": {"foreground_refiner": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        foreground = payload["stages"][-1]
        self.assertEqual(foreground["stage_key"], "foreground_refiner")
        self.assertEqual(foreground["status"], "completed")
        self.assertIn(foreground["actual_mode"], {"local", "local-fallback", "mock-fallback"})
        preview_names = {preview["name"] for preview in foreground["previews"]}
        self.assertIn("segmenter_cutout", preview_names)
        self.assertIn("foreground_cutout", preview_names)

    def test_debug_pipeline_detector_real_smoke(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/detector",
            json={"render_request": make_request(), "stage_modes": {"detector": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        detector = payload["stages"][-1]
        self.assertEqual(detector["stage_key"], "detector")
        self.assertEqual(detector["status"], "completed")
        self.assertIn(detector["actual_mode"], {"local", "local-fallback", "mock-fallback"})
        self.assertIn("confidence", detector["details"])
        self.assertIn("bbox_left", detector["details"])
        preview_names = {preview["name"] for preview in detector["previews"]}
        self.assertIn("detection_overlay", preview_names)
        self.assertIn("crop_for_resize", preview_names)

    def test_debug_pipeline_depth_real_smoke(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/depth_estimator",
            json={"render_request": make_request(), "stage_modes": {"depth_estimator": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        depth = payload["stages"][-1]
        self.assertEqual(depth["stage_key"], "depth_estimator")
        self.assertEqual(depth["status"], "completed")
        self.assertIn(depth["actual_mode"], {"local", "triton", "local-fallback", "mock-fallback"})
        self.assertIn("depth_width", depth["details"])
        preview_names = {preview["name"] for preview in depth["previews"]}
        self.assertIn("depth", preview_names)

    def test_debug_pipeline_normals_real_smoke(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/normal_estimator",
            json={"render_request": make_request(), "stage_modes": {"depth_estimator": "real", "normal_estimator": "real"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        normals = payload["stages"][-1]
        self.assertEqual(normals["stage_key"], "normal_estimator")
        self.assertEqual(normals["status"], "completed")
        self.assertIn(normals["actual_mode"], {"local", "triton", "local-fallback", "mock-fallback"})
        self.assertIn("normals_width", normals["details"])
        self.assertIn(normals["details"]["backend"], {"local", "triton", "mock"})
        self.assertIn(normals["details"]["variant"], {"stable-normal", "from-depth-v2", "mock-v1"})
        preview_names = {preview["name"] for preview in normals["previews"]}
        self.assertIn("normals", preview_names)

    def test_debug_pipeline_shadow_real_smoke(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/shadow_generator",
            json={"render_request": make_request(), "stage_modes": {"shadow_generator": "v1-gan"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        shadow = payload["stages"][-1]
        self.assertEqual(shadow["stage_key"], "shadow_generator")
        self.assertEqual(shadow["status"], "completed")
        self.assertIn(shadow["actual_mode"], {"local", "triton", "local-fallback", "mock-fallback"})
        self.assertIn("backend", shadow["details"])
        self.assertIn("variant", shadow["details"])
        preview_names = {preview["name"] for preview in shadow["previews"]}
        self.assertIn("shadow", preview_names)

    def test_debug_pipeline_shadow_v2_diff_reports_fallback_or_unavailable(self) -> None:
        response = self.client.post(
            "/v1/dev/pipeline/run-stage/shadow_generator",
            json={"render_request": make_request(), "stage_modes": {"shadow_generator": "v2-diff"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        shadow = payload["stages"][-1]
        self.assertEqual(shadow["stage_key"], "shadow_generator")
        self.assertIn(shadow["status"], {"completed", "failed"})
        self.assertIn(shadow["actual_mode"], {"unavailable", "local-fallback", "mock-fallback"})

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
        self.assertIn(geometry["actual_mode"], {"local", "mock-fallback"})
        if geometry["actual_mode"] == "local":
            self.assertEqual(geometry["details"]["backend"], "local")
        else:
            self.assertIn(geometry["details"]["backend"], {"mock", "mock-fallback"})

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

    def test_debug_pipeline_detector_mock_uses_mock_adapter_even_when_real_is_available(self) -> None:
        app = create_app(Settings())

        class BombDetector:
            def detect(self, stage_input):
                raise AssertionError("real detector should not run in mock mode")

        class StubMockDetector:
            def detect(self, stage_input):
                return DetectionResult(bbox=(1, 2, 30, 40), confidence=0.123)

        app.state.render_service.runtime.real_detector = BombDetector()
        app.state.render_service.runtime.detector = BombDetector()
        app.state.render_service.runtime.mock_detector = StubMockDetector()
        client = TestClient(app)

        response = client.post(
            "/v1/dev/pipeline/run-stage/detector",
            json={"render_request": make_request(), "stage_modes": {"detector": "mock"}},
        )
        self.assertEqual(response.status_code, 200)
        detector = response.json()["stages"][-1]
        self.assertEqual(detector["actual_mode"], "mock")
        self.assertEqual(detector["details"]["confidence"], 0.123)
        self.assertEqual(detector["details"]["prompt"], "mock")

    def test_debug_pipeline_geometry_mock_uses_mock_adapter_even_when_real_is_available(self) -> None:
        app = create_app(Settings())

        class BombGeometry:
            def estimate(self, image):
                raise AssertionError("real geometry should not run in mock mode")

        class StubMockGeometry:
            def estimate(self, image):
                return GeometryResult(camera_fov=11.0, camera_pitch=22.0, camera_roll=33.0, confidence=0.44)

        app.state.render_service.runtime.real_geometry = BombGeometry()
        app.state.render_service.runtime.geometry = BombGeometry()
        app.state.render_service.runtime.mock_geometry = StubMockGeometry()
        client = TestClient(app)

        response = client.post(
            "/v1/dev/pipeline/run-stage/geometry_estimator",
            json={"render_request": make_request(), "stage_modes": {"geometry_estimator": "mock"}},
        )
        self.assertEqual(response.status_code, 200)
        geometry = response.json()["stages"][-1]
        self.assertEqual(geometry["actual_mode"], "mock")
        self.assertEqual(geometry["details"]["camera_fov"], 11.0)
        self.assertEqual(geometry["details"]["backend"], "mock")

    def test_debug_pipeline_segmenter_mock_uses_mock_adapter_even_when_real_is_available(self) -> None:
        app = create_app(Settings())

        class BombSegmenter:
            def segment(self, stage_input):
                raise AssertionError("real segmenter should not run in mock mode")

        class StubMockSegmenter:
            def segment(self, stage_input):
                image = asset_to_pil(stage_input.image)
                mask = Image.new("L", image.size, 255)
                cutout = image.copy()
                cutout.putalpha(mask)
                return SegmentationResult(
                    bbox=(0, 0, image.width, image.height),
                    mask=pil_to_asset(mask),
                    cutout_rgba=pil_to_asset(cutout),
                    crop_rgba=pil_to_asset(image),
                )

        app.state.render_service.runtime.real_segmenter = BombSegmenter()
        app.state.render_service.runtime.segmenter = BombSegmenter()
        app.state.render_service.runtime.mock_segmenter = StubMockSegmenter()
        client = TestClient(app)

        response = client.post(
            "/v1/dev/pipeline/run-stage/segmenter",
            json={"render_request": make_request(), "stage_modes": {"segmenter": "mock"}},
        )
        self.assertEqual(response.status_code, 200)
        segmenter = response.json()["stages"][-1]
        self.assertEqual(segmenter["actual_mode"], "mock")
        self.assertEqual(segmenter["details"]["backend"], "mock")

    def test_debug_pipeline_depth_mock_uses_mock_adapter_even_when_real_is_available(self) -> None:
        app = create_app(Settings())

        class BombDepthEstimator:
            def estimate(self, stage_input):
                raise AssertionError("real depth estimator should not run in mock mode")

        app.state.render_service.runtime.real_depth = BombDepthEstimator()
        app.state.render_service.runtime.depth = BombDepthEstimator()
        client = TestClient(app)

        response = client.post(
            "/v1/dev/pipeline/run-stage/depth_estimator",
            json={"render_request": make_request(), "stage_modes": {"depth_estimator": "mock"}},
        )
        self.assertEqual(response.status_code, 200)
        depth = response.json()["stages"][-1]
        self.assertEqual(depth["actual_mode"], "mock")
        self.assertEqual(depth["details"]["backend"], "mock")

    def test_debug_pipeline_normals_mock_uses_mock_adapter_even_when_real_is_available(self) -> None:
        app = create_app(Settings())

        class BombNormalsEstimator:
            def estimate(self, stage_input):
                raise AssertionError("real normals estimator should not run in mock mode")

        app.state.render_service.runtime.real_normals = BombNormalsEstimator()
        app.state.render_service.runtime.normals = BombNormalsEstimator()
        client = TestClient(app)

        response = client.post(
            "/v1/dev/pipeline/run-stage/normal_estimator",
            json={"render_request": make_request(), "stage_modes": {"normal_estimator": "mock"}},
        )
        self.assertEqual(response.status_code, 200)
        normals = response.json()["stages"][-1]
        self.assertEqual(normals["actual_mode"], "mock")
        self.assertEqual(normals["details"]["backend"], "mock")

    def test_debug_pipeline_shadow_mock_uses_mock_adapter_even_when_real_is_available(self) -> None:
        app = create_app(Settings())

        class BombShadowGenerator:
            def generate(self, stage_input):
                raise AssertionError("real shadow generator should not run in mock mode")

        app.state.render_service.runtime.real_shadow = BombShadowGenerator()
        app.state.render_service.runtime.shadow = BombShadowGenerator()
        client = TestClient(app)

        response = client.post(
            "/v1/dev/pipeline/run-stage/shadow_generator",
            json={"render_request": make_request(), "stage_modes": {"shadow_generator": "mock"}},
        )
        self.assertEqual(response.status_code, 200)
        shadow = response.json()["stages"][-1]
        self.assertEqual(shadow["actual_mode"], "mock")
        self.assertEqual(shadow["details"]["backend"], "mock")

    def test_debug_pipeline_foreground_refiner_mock_uses_mock_adapter_even_when_real_is_available(self) -> None:
        app = create_app(Settings())

        class BombForegroundRefiner:
            def refine(self, image, alpha):
                raise AssertionError("real foreground refiner should not run in mock mode")

        class StubMockForegroundRefiner:
            def refine(self, image, alpha):
                cutout = asset_to_pil(image).convert("RGBA")
                cutout.putalpha(asset_to_pil(alpha))
                return ForegroundRefinementResult(cutout_rgba=pil_to_asset(cutout))

        app.state.render_service.runtime.real_foreground_refiner = BombForegroundRefiner()
        app.state.render_service.runtime.foreground_refiner = BombForegroundRefiner()
        app.state.render_service.runtime.mock_foreground_refiner = StubMockForegroundRefiner()
        client = TestClient(app)

        response = client.post(
            "/v1/dev/pipeline/run-stage/foreground_refiner",
            json={"render_request": make_request(), "stage_modes": {"foreground_refiner": "mock"}},
        )
        self.assertEqual(response.status_code, 200)
        foreground = response.json()["stages"][-1]
        self.assertEqual(foreground["actual_mode"], "mock")
        self.assertEqual(foreground["details"]["backend"], "mock")

    def test_segmentation_runs_after_crop_and_resize(self) -> None:
        temp_dir = Path("var/cache/test-preprocess") / uuid4().hex
        try:
            app = create_app(Settings(preprocess_cache_dir=temp_dir))
            service = app.state.render_service

            class RecordingSegmenter:
                def __init__(self) -> None:
                    self.seen_size = None

                def segment(self, stage_input):
                    image = asset_to_pil(stage_input.image)
                    self.seen_size = image.size
                    mask = Image.new("L", image.size, 255)
                    cutout = image.copy()
                    cutout.putalpha(mask)
                    return SegmentationResult(
                        bbox=(0, 0, image.width, image.height),
                        mask=pil_to_asset(mask),
                        cutout_rgba=pil_to_asset(cutout),
                        crop_rgba=pil_to_asset(image),
                    )

            recorder = RecordingSegmenter()
            service.runtime.segmenter = recorder
            client = TestClient(app)
            response = client.post("/v1/render", json=make_request())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(recorder.seen_size, (service.settings.working_size, service.settings.working_size))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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
