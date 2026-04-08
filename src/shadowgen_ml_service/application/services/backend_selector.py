from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import StageBackendSelection


UNAVAILABLE_MESSAGES = {
    "detector": "Real detector is unavailable. Configure the GroundingDINO integration and model weights first.",
    "geometry_estimator": "Real geometry estimator is unavailable. Configure GeoCalib integration first.",
    "segmenter": "Real segmenter is unavailable. Configure the BiRefNet adapter first.",
    "foreground_refiner": "Real foreground colour estimator is unavailable. Install the Fast Foreground Colour Estimation dependencies first.",
    "depth_estimator": "Real depth estimator is unavailable. Configure the Depth Anything adapter first.",
}


class BackendSelector:
    def __init__(self, runtime: PipelineRuntime) -> None:
        self.runtime = runtime

    def select_for_debug(self, stage_key: str, requested_mode: str) -> StageBackendSelection:
        component = next((item for item in self.runtime.descriptor.components if item.name == stage_key), None)
        if component is None:
            return StageBackendSelection(requested_mode=requested_mode, actual_mode="internal")
        if requested_mode == "real":
            if stage_key in {"normal_estimator", "shadow_generator", "composer"}:
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
