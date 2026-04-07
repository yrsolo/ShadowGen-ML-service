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
