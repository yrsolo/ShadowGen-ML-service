from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import StageBackendSelection


UNAVAILABLE_MESSAGES = {
    "detector": "Real detector is unavailable. Configure the GroundingDINO integration and model weights first.",
    "geometry_estimator": "Real geometry estimator is unavailable. Configure GeoCalib integration first.",
    "segmenter": "Real segmenter is unavailable. Configure the BiRefNet adapter first.",
    "foreground_refiner": "Real foreground colour estimator is unavailable. Install the Fast Foreground Colour Estimation dependencies first.",
    "depth_estimator": "Real depth estimator is unavailable. Configure the Depth Anything adapter first.",
    "normal_estimator": "Real normal estimator is unavailable. Configure the StableNormal adapter or use the depth-derived fallback.",
    "shadow_generator": "Real shadow generator is unavailable. Configure the pix2pix shadow backend and local weights first.",
}


class BackendSelector:
    def __init__(self, runtime: PipelineRuntime) -> None:
        self.runtime = runtime

    def select_for_debug(self, stage_key: str, requested_mode: str) -> StageBackendSelection:
        component = next((item for item in self.runtime.descriptor.components if item.name == stage_key), None)
        if component is None:
            return StageBackendSelection(requested_mode=requested_mode, actual_mode="internal")
        if requested_mode == "real":
            if stage_key == "composer":
                return StageBackendSelection(requested_mode=requested_mode, actual_mode="real")
            if component.implementation == "real" and component.available:
                return StageBackendSelection(requested_mode=requested_mode, actual_mode="real")
            if component.available and component.using_mock:
                return StageBackendSelection(requested_mode=requested_mode, actual_mode="mock-fallback")
            return StageBackendSelection(
                requested_mode=requested_mode,
                actual_mode="unavailable",
                unavailable_message=UNAVAILABLE_MESSAGES.get(stage_key, "Requested real mode is unavailable for this stage."),
            )
        return StageBackendSelection(requested_mode=requested_mode, actual_mode="mock")

    def actual_mode_for_public(self, stage_key: str) -> str:
        component = next((item for item in self.runtime.descriptor.components if item.name == stage_key), None)
        if component is None:
            return "internal"
        if component.implementation == "real":
            return "real"
        if component.using_mock:
            return "mock-fallback" if component.implementation == "mock-fallback" else "mock"
        return component.implementation

    def select_shadow_variant_for_debug(self, requested_mode: str) -> StageBackendSelection:
        if requested_mode == "mock":
            return StageBackendSelection(requested_mode=requested_mode, actual_mode="mock")
        if requested_mode == "v1-gan":
            if self.runtime.shadow_v1_gan is not None:
                return StageBackendSelection(requested_mode=requested_mode, actual_mode="real")
            return StageBackendSelection(
                requested_mode=requested_mode,
                actual_mode="mock-fallback",
                unavailable_message="V1-GAN is unavailable. Falling back to the mock shadow generator.",
            )
        if requested_mode == "v2-diff":
            if self.runtime.shadow_v2_diff is not None:
                return StageBackendSelection(requested_mode=requested_mode, actual_mode="real")
            return StageBackendSelection(
                requested_mode=requested_mode,
                actual_mode="unavailable",
                unavailable_message="V2-DIFF is not connected yet. The model scaffold exists, but the inference backend is not implemented.",
            )
        return StageBackendSelection(requested_mode=requested_mode, actual_mode="unavailable", unavailable_message="Unknown shadow backend requested.")
