# API Summary

## Public Endpoints

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/render`

## Dev Endpoints

- `GET /playground`
- `POST /v1/dev/pipeline/run-all`
- `POST /v1/dev/pipeline/run-stage/{stage_key}`

## `POST /v1/render`

Main request blocks:

- `source`
- `preprocess`
- `shadow`
- `background`
- `output`

### `preprocess`

Current active field:

```json
{
  "preprocess": {
    "padding_px": 100
  }
}
```

Rules:

- `padding_px` is optional in payload
- default is `100`
- minimum is `0`

## Success Contract

A successful render returns:

- exactly one `final` artifact
- optional debug artifacts when `output.return_debug=true`
- `metrics.total_ms`
- `warnings`
- `model_info`

## Error Contract

Errors use the shape:

```json
{
  "error": {
    "code": "validation_error",
    "message": "..."
  }
}
```

## `GET /v1/capabilities`

Capabilities expose:

- service version
- model version summary
- supported input mime types
- supported output formats
- active backend mode
- degraded flag
- component list

Each component includes:

- `name`
- `implementation`
- `model_name`
- `model_version`
- `available`
- `using_mock`
- `detail`

This is the main machine-readable place to understand which backend is actually active.

## Debug Pipeline Responses

The dev pipeline endpoints return an ordered list of stage executions.

Each stage includes:

- `stage_key`
- `title`
- `description`
- `status`
- `requested_mode`
- `actual_mode`
- `elapsed_ms`
- `error`
- `details`
- `previews`

## Stage Detail Summary

### `geometry_estimator`

May include:

- `camera_fov`
- `camera_pitch`
- `camera_roll`
- `confidence`
- `backend`
- `camera_model`
- `weights`
- `shared_intrinsics`
- `device`

Previews:

- `geometry_input`
- `geometry_overlay`

### `detector`

May include:

- `bbox_left`
- `bbox_top`
- `bbox_right`
- `bbox_bottom`
- `confidence`
- `backend`
- `prompt`
- `device`

Previews:

- `detection_overlay`
- `crop_for_resize`

### `segmenter`

May include:

- `bbox_left`
- `bbox_top`
- `bbox_right`
- `bbox_bottom`
- `mask_width`
- `mask_height`
- `backend`
- `device`

Previews:

- `working_crop`
- `mask`
- `cutout`

### `foreground_refiner`

May include:

- `cutout_width`
- `cutout_height`
- `backend`
- `device`

Previews:

- `segmenter_cutout`
- `foreground_cutout`

### `depth_estimator`

May include:

- `depth_width`
- `depth_height`
- `backend`
- `device`

Previews:

- `depth`
- `working_cutout`

### `normal_estimator`

May include:

- `normals_width`
- `normals_height`
- `backend`
- `variant`
- `device`

Previews:

- `normals`

### `shadow_generator`

May include:

- `backend`
- `variant`
- `device`

Current requested variants in the dev interface:

- `mock`
- `v1-gan`
- `v2-diff`

Previews:

- `shadow`

### `composer`

Previews:

- `final`

## Notes

- `requested_mode` is what the caller asked for
- `actual_mode` is what the runtime really used
- `mock-fallback` means the requested model backend was not usable and the stage fell back
- `unavailable` means the requested backend variant exists in the interface but is not wired to inference yet
