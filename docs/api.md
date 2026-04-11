# API Summary

## Public Endpoints

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/render`
- `POST /v1/render/jobs`
- `GET /v1/render/jobs/{job_id}`
- `DELETE /v1/render/jobs/{job_id}`

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

## Sync Success Contract

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

## Async Render API

### `POST /v1/render/jobs`

Submits the same logical render request as the sync endpoint, but returns job metadata instead of the render result.

Response shape:

- `job_id`
- `request_id`
- `status`
- `created_at`
- `updated_at`

### `GET /v1/render/jobs/{job_id}`

Returns:

- job metadata
- current status
- optional error
- optional final `RenderResponse` result

### `DELETE /v1/render/jobs/{job_id}`

Cancels a pending or running job.

## `GET /v1/capabilities`

Capabilities expose:

- service version
- model version summary
- supported input mime types
- supported output formats
- active backend mode
- degraded flag
- execution default backend
- async availability
- component list

Each component includes:

- `name`
- `implementation`
- `model_name`
- `model_version`
- `available`
- `using_mock`
- `detail`
- `backend_kind`
- `model_variant`
- `device`
- `endpoint`
- `supports_batching`
- `supports_async`
- `fallback_reason`
- `backends`

`backends` is the detailed descriptor list for all registered executors of that stage.

Each backend descriptor includes:

- `backend_kind`
- `model_variant`
- `model_name`
- `model_version`
- `available`
- `detail`
- `device`
- `endpoint`
- `supports_batching`
- `supports_async`

This is the primary machine-readable place to understand whether a stage currently runs via:

- `mock`
- `local`
- `triton`

This is also the primary worker-facing handshake endpoint for deciding:

- whether the ML core should be used in sync or async mode
- whether batching is supported by the active heavy-stage backends

## `GET /health`

Current fields:

- `status`
- `service_version`
- `active_backend_mode`
- `async_enabled`

## Debug Pipeline Request

The dev endpoints accept:

- `render_request`
- `stage_modes`
- `stage_backend_kinds`
- `stage_variants`

Notes:

- `stage_modes` is a compatibility input and should be treated as transitional
- `stage_backend_kinds` is the execution-aware selector
- `stage_variants` chooses the logical model variant

## Debug Pipeline Responses

The dev pipeline endpoints return an ordered list of stage executions.

Each stage includes:

- `stage_key`
- `title`
- `description`
- `status`
- `requested_mode`
- `actual_mode`
- `requested_backend_kind`
- `actual_backend_kind`
- `model_variant`
- `model_name`
- `model_version`
- `device`
- `endpoint`
- `cache_status`
- `fallback_reason`
- `elapsed_ms`
- `error`
- `details`
- `previews`

Compatibility notes:

- `requested_mode` and `actual_mode` still exist
- execution-aware fields are now the source of truth

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

Current stage variants:

- `mock`
- `v1-gan`
- `v2-diff`

Current execution backend kinds:

- `mock`
- `local`
- `triton`

Previews:

- `shadow`

### `composer`

Previews:

- `final`

## Execution Semantics

- `requested_backend_kind` is what the caller asked for
- `actual_backend_kind` is what the runtime actually used
- `model_variant` identifies the logical stage backend family
- `mock-fallback` means a non-mock backend was requested but the stage fell back to mock
- `local-fallback` means a Triton backend was requested but the stage fell back to local execution
- `unavailable` means the interface knows the backend slot but no executable backend is currently wired

## Worker Integration Notes

The recommended worker contract is documented separately in:

- [worker-core-contract.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/worker-core-contract.md)

Worker authors should treat:

- `GET /v1/capabilities`
  - as the feature-discovery handshake
- `POST /v1/render`
  - as the sync compatibility path
- `POST /v1/render/jobs`
  - as the preferred async path when `async_enabled=true`
