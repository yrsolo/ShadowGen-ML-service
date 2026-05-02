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
- `POST /v1/dev/service/shutdown`

## `POST /v1/render`

Main request blocks:

- `source`
- `preprocess`
- `shadow`
- `background`
- `output`

### `shadow`

Current fields:

```json
{
  "shadow": {
    "model": "v1-gan",
    "angle_deg": 45,
    "elevation_deg": 35,
    "softness": 0.5,
    "opacity": 0.65,
    "reflection": 0.0
  }
}
```

Rules:

- `model` is optional for backward compatibility
- allowed `model` values are `v1-gan` and `v2-diff`
- if `model` is omitted, the service uses its configured runtime default
- `v1-gan` uses `angle_deg` as the active rot/top-view shadow direction control
- current `v2-diff` is control-free and ignores `angle_deg`, `elevation_deg`, `softness`, and `reflection`
- `opacity` remains accepted for API stability and legacy/mock paths

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
- after detection crop, the service keeps extra canonical canvas margin with `SHADOWGEN_WORKING_CONTENT_SCALE`
- default `SHADOWGEN_WORKING_CONTENT_SCALE` is `0.68`, so crop content targets about 68% of the `512x512` working canvas and leaves room for generated shadows

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
- `submit_mode`

Behavior:

- returns `202 Accepted` on successful async submission
- may return the same `job_id` for repeated live or completed `request_id`
- returns `429` with `queue_full` when the bounded pending queue is full
- returns `503` with `not_accepting_jobs` when the service is draining or not accepting new jobs

### `GET /v1/render/jobs/{job_id}`

Returns:

- job metadata
- current status
- `submit_mode`
- optional `capacity_snapshot`
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
- supported submit modes
- preferred submit mode
- job execution capacity snapshot
- batching strategy
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
- whether the core is currently able to accept more jobs

For the live Triton rollout, the most important capability checks are:

- `detector.backends[]`
  - `backend_kind = triton`
  - `available = true`
  - `model_name = shadowgen_detector`
- `segmenter.backends[]`
  - `backend_kind = triton`
  - `available = true`
  - `model_name = shadowgen_segmenter`

Additional worker-facing fields:

- `supported_submit_modes`
- `preferred_submit_mode`
- `job_execution`
- `batching_strategy`

`job_execution` includes:

- `queue_backend`
- `accepting_jobs`
- `max_running_jobs`
- `max_pending_jobs`
- `running_jobs`
- `pending_jobs`
- `cancel_mode`
- `idempotency_supported`

## `GET /health`

Current fields:

- `status`
- `service_version`
- `active_backend_mode`
- `async_enabled`
- `accepting_jobs`
- `preferred_submit_mode`

Current status values:

- `ok`
- `degraded`
- `draining`
- `overloaded`

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

## Dev Service Shutdown

`POST /v1/dev/service/shutdown` asks the currently running ML-service process to terminate.

This endpoint exists only as a local playground convenience. It is useful when the service was started through `start-service.cmd`, which opens a visible Windows console and runs FastAPI without reload by default.

Response shape:

- `status`
- `pid`
- `message`

Operational notes:

- when the service is started without reload, the process exits and the visible console remains available for logs
- when the service is started with `uvicorn --reload`, the reloader may spawn a new worker process after shutdown
- this endpoint does not stop the Triton Docker container; stop Triton separately with `docker rm -f shadowgen-triton-segmenter`

## Stage Detail Summary

### `geometry_estimator`

Default state:

- disabled by default with `SHADOWGEN_GEOMETRY_ENABLED=false`
- omitted from normal `run-all` playground execution while disabled
- direct `run-stage/geometry_estimator` calls return a skipped stage while disabled

When enabled, details may include:

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

Operational note:

- when `actual_backend_kind = triton`, the current live contract is a GroundingDINO Triton Python backend returning `bbox` and `confidence`
- detection overlay and crop preview are still reconstructed inside ML-core postprocess

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

Operational note:

- when `actual_backend_kind = triton`, the current live contract is a mask-first Triton Python backend
- `cutout` and compatibility `bbox` are reconstructed in ML-core postprocess after Triton returns `mask`

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

Default variant:

- `from-depth-v2`

Notes:

- `from-depth-v2` is the fast depth-derived local path
- `stable-normal` is an opt-in neural local variant and may fallback when its model/runtime is unavailable

### `shadow_generator`

May include:

- `backend`
- `variant`
- `device`

Current stage variants:

- `mock`
- `v1-gan`: controllable local GAN path for rot/top-view shadow generation
- `v2-diff`: control-free diffusion path; current model contract is `img + mask -> shadow_image`

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

`request_id` remains optional for public compatibility, but when present it now acts as the ML-core idempotency key for async job submission.
