# API Summary

## Endpoints

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/render`

## Request highlights

`POST /v1/render` accepts:

- `source`
- `preprocess`
- `shadow`
- `background`
- `output`

### New preprocess block

```json
{
  "preprocess": {
    "padding_px": 100
  }
}
```

Rules:

- `padding_px` is optional in the payload
- default value is `100`
- minimum is `0`
- it controls crop and padding around the detected object before later inference stages

## Success rules

- exactly one `final` artifact
- debug artifacts only when `output.return_debug=true`
- `metrics.total_ms` always present
- `metrics.foreground_refinement_ms` may be present when the dedicated foreground colour refinement stage runs before depth/composition
- `warnings` always present
- `model_info` always present

## Error rules

Errors always use:

```json
{
  "error": {
    "code": "validation_error",
    "message": "..."
  }
}
```

## Playground debug metadata

`/v1/dev/pipeline/run-all` and `/v1/dev/pipeline/run-stage/{stage_key}` return per-stage debug entries.

For `geometry_estimator`, each stage entry may include:

- `details.camera_fov`
- `details.camera_pitch`
- `details.camera_roll`
- `details.confidence`
- `details.backend`

Geometry previews include:

- `geometry_input`
- `geometry_overlay`

For `detector`, each stage entry may include:

- `details.bbox_left`
- `details.bbox_top`
- `details.bbox_right`
- `details.bbox_bottom`
- `details.confidence`
- `details.backend`
- `details.prompt`

Detection previews include:

- `detection_overlay`
- `crop_for_resize`

For `segmenter`, each stage entry may include:

- `details.bbox_left`
- `details.bbox_top`
- `details.bbox_right`
- `details.bbox_bottom`
- `details.mask_width`
- `details.mask_height`
- `details.backend`

Segmentation previews include:

- `working_crop`
- `mask`
- `cutout`

For `foreground_refiner`, each stage entry may include:

- `details.cutout_width`
- `details.cutout_height`
- `details.backend`

Foreground refinement previews include:

- `segmenter_cutout`
- `foreground_cutout`

For `depth_estimator`, each stage entry may include:

- `details.depth_width`
- `details.depth_height`
- `details.backend`

Depth previews include:

- `depth`
- `working_cutout`

For `normal_estimator`, each stage entry may include:

- `details.normals_width`
- `details.normals_height`
- `details.backend`
- `details.variant`

Normals previews include:

- `normals`

For `shadow_generator`, each stage entry may include:

- `details.backend`

Shadow previews include:

- `shadow`
