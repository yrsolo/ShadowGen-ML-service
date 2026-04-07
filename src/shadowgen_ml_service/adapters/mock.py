from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.pipeline.contracts import (
    ArtifactEncoder,
    Composer,
    DepthEstimator,
    Detector,
    GeometryEstimator,
    NormalEstimator,
    Segmenter,
    ShadowGenerator,
)
from shadowgen_ml_service.pipeline.types import (
    CompositionResult,
    DepthResult,
    DetectionResult,
    GeometryResult,
    NormalResult,
    SegmentationResult,
    ShadowResult,
)
from shadowgen_ml_service.schemas import BackgroundPayload, OutputPayload, ShadowPayload
from shadowgen_ml_service.utils.images import (
    bbox_from_mask,
    compose_on_background,
    create_cutout,
    depth_from_mask,
    encode_image,
    estimate_foreground_mask,
    generate_shadow_layer,
    normals_from_depth,
)


class MockDetector(Detector):
    def detect(self, image: Image.Image, padding_px: int) -> DetectionResult:
        mask = estimate_foreground_mask(image)
        bbox = bbox_from_mask(mask, padding_px)
        return DetectionResult(bbox=bbox, confidence=0.68)


class MockGeometryEstimator(GeometryEstimator):
    def estimate(self, image: Image.Image) -> GeometryResult:
        aspect = image.width / max(image.height, 1)
        fov = 42.0 + min(aspect, 2.0) * 8.0
        pitch = -4.0 if image.height >= image.width else -2.0
        return GeometryResult(camera_fov=fov, camera_pitch=pitch, camera_roll=0.0, confidence=0.55)


class MockSegmenter(Segmenter):
    def segment(self, image: Image.Image) -> SegmentationResult:
        crop_mask = estimate_foreground_mask(image)
        cutout_rgba = create_cutout(image, crop_mask)
        return SegmentationResult(bbox=(0, 0, image.width, image.height), mask=crop_mask, cutout_rgba=cutout_rgba, crop_rgba=image)


class MockDepthEstimator(DepthEstimator):
    def estimate(self, image: Image.Image, mask: Image.Image) -> DepthResult:
        return DepthResult(depth_map=depth_from_mask(mask))


class MockNormalEstimator(NormalEstimator):
    def estimate(self, depth_map: Image.Image) -> NormalResult:
        return NormalResult(normal_map=normals_from_depth(depth_map))


class MockShadowGenerator(ShadowGenerator):
    def generate(
        self,
        mask: Image.Image,
        depth_map: Image.Image,
        normal_map: Image.Image,
        geometry: GeometryResult,
        shadow: ShadowPayload,
    ) -> ShadowResult:
        shadow_rgba = generate_shadow_layer(
            mask=mask,
            angle_deg=shadow.angle_deg,
            elevation_deg=shadow.elevation_deg,
            softness=shadow.softness,
            opacity=shadow.opacity,
            reflection=shadow.reflection,
            camera_pitch=geometry.camera_pitch,
        )
        return ShadowResult(shadow_rgba=shadow_rgba)


class MockComposer(Composer):
    def compose(
        self,
        cutout_rgba: Image.Image,
        shadow_rgba: Image.Image,
        background: BackgroundPayload,
        output: OutputPayload,
    ) -> CompositionResult:
        final_image = compose_on_background(
            cutout_rgba=cutout_rgba,
            shadow_rgba=shadow_rgba,
            color_hex=background.color_hex,
            width=output.width,
            height=output.height,
        )
        return CompositionResult(final_image=final_image)


class DefaultArtifactEncoder(ArtifactEncoder):
    def encode(
        self,
        final_image: Image.Image,
        output_format: str,
        debug_images: dict[str, Image.Image],
        return_debug: bool,
    ) -> list[dict[str, str]]:
        artifacts: list[dict[str, str]] = []
        mime_type, image_base64 = encode_image(final_image, output_format)
        artifacts.append(
            {
                "name": "final",
                "kind": "final",
                "mime_type": mime_type,
                "image_base64": image_base64,
            }
        )
        if return_debug:
            for name, image in debug_images.items():
                debug_mime_type, debug_base64 = encode_image(image, "png")
                artifacts.append(
                    {
                        "name": name,
                        "kind": "debug",
                        "mime_type": debug_mime_type,
                        "image_base64": debug_base64,
                    }
                )
        return artifacts
