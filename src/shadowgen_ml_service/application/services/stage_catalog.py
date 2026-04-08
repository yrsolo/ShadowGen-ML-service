from __future__ import annotations

from shadowgen_ml_service.application.models import StageDefinition


STAGE_CATALOG = [
    StageDefinition("decode", "Decode", "Decode input image and validate the payload."),
    StageDefinition("geometry_estimator", "Geometry", "Estimate camera geometry from the original image."),
    StageDefinition("detector", "Detection", "Locate the main foreground object and compute the crop area."),
    StageDefinition("segmenter", "Segmentation", "Build the foreground mask and cut out the object."),
    StageDefinition("depth_estimator", "Depth", "Estimate the relative depth map for the foreground crop."),
    StageDefinition("normal_estimator", "Normals", "Compute surface normals from the depth map."),
    StageDefinition("shadow_generator", "Shadow", "Generate the shadow layer from geometry and user lighting controls."),
    StageDefinition("composer", "Composition", "Composite the object and shadow on the target background."),
]


def get_stage_definition(stage_key: str) -> StageDefinition:
    for stage in STAGE_CATALOG:
        if stage.key == stage_key:
            return stage
    return StageDefinition(stage_key, stage_key, stage_key)
