# New ML Service Specification

## Purpose

This document defines the target contract for the new ShadowGen ML service.

It is intentionally **not** based on the old `ShadowGEN` transport contract.
Legacy compatibility remains an adapter concern in the current worker runtime.

The goal of the new service is to provide a clean, stable, stateless ML interface that fits the ShadowGen v2 architecture:

```text
Web -> API -> Queue / Store -> Worker -> Render Pipeline Adapter -> New ML Service
```

## Scope

The new ML service is responsible for:

- receiving a single source image
- extracting the foreground object
- generating a shadow from explicit light and shadow parameters
- compositing the object on the requested background
- returning final and optional debug artifacts
- returning processing metrics and warnings

The new ML service is **not** responsible for:

- job queueing
- job lifecycle state
- user management
- cloud storage orchestration
- public product API design
- legacy protocol compatibility

## Design Principles

- stateless
- synchronous per request
- versioned API
- explicit request and response schema
- machine-readable errors
- no hidden fallback behavior
- no dependency on project-internal asset IDs
- no dependency on Yandex infrastructure details

## Base URL And Versioning

The service should expose a versioned API surface:

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/render`

Future breaking changes should use a new major version path such as `/v2/...`.

## Required Endpoints

### `GET /health`

Purpose:

- liveness probe
- readiness probe for the worker adapter

Example response:

```json
{
  "status": "ok"
}
```

### `GET /v1/capabilities`

Purpose:

- advertise supported input types
- advertise supported output types
- expose service and model version metadata
- support worker-side diagnostics and compatibility checks

Example response:

```json
{
  "service_version": "1.0.0",
  "model_version": "shadow-model-2026-04-07",
  "supported_input_mime_types": ["image/jpeg", "image/png", "image/webp"],
  "supported_output_formats": ["png", "webp"],
  "supports_debug_artifacts": true,
  "max_image_bytes": 10485760
}
```

### `POST /v1/render`

Purpose:

- process one image synchronously
- return final and optional debug artifacts

Transport:

- `Content-Type: application/json`
- binary image bytes are passed as base64 in the request body

## Request Contract

### Top-Level Request

```json
{
  "request_id": "optional-trace-id",
  "pipeline_version": "ml-shadowgen-v1",
  "source": {
    "mime_type": "image/jpeg",
    "image_base64": "..."
  },
  "preprocess": {
    "padding_px": 100
  },
  "shadow": {
    "angle_deg": 45,
    "elevation_deg": 35,
    "softness": 0.5,
    "opacity": 0.6,
    "reflection": 0.0
  },
  "background": {
    "mode": "solid",
    "color_hex": "#FFFFFF"
  },
  "output": {
    "format": "png",
    "width": null,
    "height": null,
    "return_debug": true
  }
}
```

### Field Definitions

#### `request_id`

- optional string
- used for tracing and logs
- echoed back in the response when provided

#### `pipeline_version`

- required string
- identifies the expected ML behavior family
- allows the worker to request a specific generation pipeline

Initial recommendation:

- `ml-shadowgen-v1`

#### `source`

Required object:

```json
{
  "mime_type": "image/jpeg",
  "image_base64": "..."
}
```

Rules:

- exactly one image per request
- supported formats must be declared in `/v1/capabilities`
- the ML service must decode the image from base64
- the service must validate the MIME type and image payload consistency

#### `preprocess`

Optional object with defaults:

```json
{
  "padding_px": 100
}
```

Definitions:

- `padding_px`
  - optional integer
  - default: `100`
  - minimum: `0`
  - controls the padding around the detected object before crop / resize

#### `shadow`

Required object:

```json
{
  "angle_deg": 45,
  "elevation_deg": 35,
  "softness": 0.5,
  "opacity": 0.6,
  "reflection": 0.0
}
```

Definitions:

- `angle_deg`
  - required number
  - range: `0..360`
  - direction of the shadow on the image plane

- `elevation_deg`
  - required number
  - range: `0..90`
  - light source elevation above the horizon
  - lower values produce longer shadows
  - higher values produce shorter, more compact shadows

- `softness`
  - required number
  - range: `0..1`
  - controls edge blur / softness of the shadow

- `opacity`
  - required number
  - range: `0..1`
  - controls shadow density

- `reflection`
  - required number
  - range: `0..1`
  - optional reflection or glossy-floor effect strength
  - `0` means disabled

#### `background`

Required object:

```json
{
  "mode": "solid",
  "color_hex": "#FFFFFF"
}
```

Initial supported values:

- `mode`
  - required string
  - initial allowed values:
    - `solid`

- `color_hex`
  - required string when `mode=solid`
  - expected format: `#RRGGBB`

Future extension may allow transparent or generated backgrounds, but that is out of scope for v1.

#### `output`

Required object:

```json
{
  "format": "png",
  "width": null,
  "height": null,
  "return_debug": true
}
```

Definitions:

- `format`
  - required string
  - initial allowed values:
    - `png`
    - `webp`

- `width`
  - optional integer
  - minimum: `1`

- `height`
  - optional integer
  - minimum: `1`

- `return_debug`
  - required boolean
  - when `true`, the service should return intermediate debug artifacts when available

Output sizing rules:

- if both `width` and `height` are omitted, keep the natural pipeline output size
- if one side is omitted, preserve aspect ratio
- if both are provided, the service may resize into the requested dimensions

## Response Contract

### Success Response

```json
{
  "request_id": "optional-trace-id",
  "artifacts": [
    {
      "name": "final",
      "kind": "final",
      "mime_type": "image/png",
      "image_base64": "..."
    },
    {
      "name": "cutout",
      "kind": "debug",
      "mime_type": "image/png",
      "image_base64": "..."
    },
    {
      "name": "shadow",
      "kind": "debug",
      "mime_type": "image/png",
      "image_base64": "..."
    }
  ],
  "metrics": {
    "total_ms": 842,
    "segmentation_ms": 130,
    "shadow_ms": 402,
    "composition_ms": 56
  },
  "warnings": [],
  "model_info": {
    "service_version": "1.0.0",
    "model_version": "shadow-model-2026-04-07"
  }
}
```

### Artifact Rules

Each artifact object:

```json
{
  "name": "final",
  "kind": "final",
  "mime_type": "image/png",
  "image_base64": "..."
}
```

Fields:

- `name`
  - required string
  - stable semantic name such as `final`, `cutout`, `shadow`, `mask`

- `kind`
  - required string
  - allowed initial values:
    - `final`
    - `debug`

- `mime_type`
  - required string

- `image_base64`
  - required string

Rules:

- the response must contain exactly one `final` artifact
- `debug` artifacts are optional
- `debug` artifacts should only be returned when `output.return_debug=true`

### Metrics

`metrics` must always be present.

Minimum required field:

- `total_ms`

Recommended additional fields:

- `decode_ms`
- `segmentation_ms`
- `shadow_ms`
- `composition_ms`
- `encode_ms`

### Warnings

`warnings` must always be present.

Rules:

- warnings describe degraded but successful processing
- warnings must not be mixed into `errors`
- warnings must be safe to display in diagnostics

Examples:

- foreground extraction had low confidence
- reflection was clamped to the supported range
- requested debug artifact was unavailable

### Model Info

`model_info` should always be present in successful responses.

Fields:

- `service_version`
- `model_version`

Optional future fields:

- `build_id`
- `weights_hash`

## Error Contract

Errors must use a consistent machine-readable shape.

Example:

```json
{
  "error": {
    "code": "validation_error",
    "message": "output.format must be one of: png, webp"
  }
}
```

Required error fields:

- `code`
- `message`

Recommended optional fields:

- `details`
- `request_id`

## Error Categories

Initial required categories:

- `validation_error`
- `unsupported_input`
- `processing_failed`
- `timeout`
- `internal_error`

Recommended HTTP mapping:

- `400` for `validation_error`
- `415` for `unsupported_input`
- `422` for semantically invalid but well-formed requests
- `408` or `504` for `timeout`
- `500` for `processing_failed` and `internal_error`

## Processing Semantics

The new service should be:

- synchronous
- deterministic enough for repeated use with the same input and settings
- free from project-internal filesystem assumptions
- free from cloud-specific coupling

The service should:

- process exactly one request independently
- not require persistent session state
- not require shared disk between requests

## Logging And Traceability

The service should produce structured logs containing:

- timestamp
- request ID when present
- service version
- model version
- total processing time
- error code when failed

The service should not log full base64 image payloads.

## Worker Integration Expectations

From the ShadowGen worker perspective, the service must map cleanly to this internal pipeline shape:

- input:
  - `source_image: bytes`
  - `source_mime_type: str`
  - `RenderRequest`
- output:
  - final artifact bytes
  - optional debug artifact bytes
  - metrics
  - warnings

This means the worker adapter should be able to:

1. build the request body from `RenderRequest` and the source image bytes
2. call the ML service
3. map returned artifacts into `PipelineOutput`

The ML service must not require:

- ShadowGen asset IDs
- direct bucket access
- queue message fields
- web or API-specific DTOs

## Non-Goals For V1

These are intentionally out of scope for the first clean version:

- batch inference
- asynchronous callback protocol
- job orchestration inside the ML service
- direct storage upload from the ML service
- user-specific authorization model
- backward compatibility with old `ShadowGEN` HTTP payloads

## Acceptance Criteria

The new ML service is considered contract-complete for ShadowGen v2 when:

- `POST /v1/render` accepts one image plus explicit render settings
- `angle_deg` and `elevation_deg` are both supported
- one final artifact is always returned on success
- optional debug artifacts are returned when requested
- `metrics.total_ms` is always present
- errors use a stable machine-readable shape
- the worker adapter can integrate without legacy-specific field names

## Recommended First Implementation Checklist

- implement `GET /health`
- implement `GET /v1/capabilities`
- implement `POST /v1/render`
- validate all request fields explicitly
- add unit tests for request validation
- add integration tests for at least:
  - JPEG input
  - PNG input
  - `angle_deg` changes
  - `elevation_deg` changes
  - `return_debug=true`
  - invalid MIME type
  - timeout behavior
