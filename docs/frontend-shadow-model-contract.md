# Frontend Shadow Model Contract

This document describes how the product frontend should expose shadow model selection and call the ML core.

## Current User-Facing Capability

The product currently has two shadow model choices:

- `V1-GAN`
- `V2-DIFF`

They are intentionally different.

### `V1-GAN`

`V1-GAN` is the current controllable local GAN model.

Use it when the user wants to rotate the shadow direction manually.

Behavior:

- generates a top-view/rot-style shadow
- uses `angle_deg` as the active direction control
- uses `opacity` as final shadow density
- does not currently use `elevation_deg`, `softness`, or `reflection`

Recommended UI copy:

```text
V1-GAN
Classic controllable shadow model.
Use it when you want to rotate the shadow direction manually.
The Angle control changes the shadow direction.
```

### `V2-DIFF`

`V2-DIFF` is the new diffusion model slot.

Current limitation:

- it is temporarily control-free
- it receives object image + mask
- it draws a plausible shadow automatically
- it ignores `angle_deg`, `elevation_deg`, `softness`, and `reflection`

Recommended UI copy:

```text
V2-DIFF
Automatic diffusion shadow model.
It draws a shadow without manual controls.
Angle, Elevation, Softness, and Reflection are not used yet.
```

### Short UI Hint

```text
V1-GAN: controllable shadow, Angle rotates direction.
V2-DIFF: automatic diffusion shadow, currently without controls.
```

### Disabled Controls Hint For `V2-DIFF`

```text
These controls do not affect V2-DIFF yet.
V2-DIFF currently generates the shadow automatically from the object and mask.
Switch to V1-GAN if you need manual direction control.
```

## Frontend State Model

Recommended frontend model enum:

```ts
type ShadowModel = "v1-gan" | "v2-diff";
```

Recommended control availability map:

```ts
const shadowModelControls = {
  "v1-gan": {
    angle: true,
    opacity: true,
    elevation: false,
    softness: false,
    reflection: false,
  },
  "v2-diff": {
    angle: false,
    opacity: true,
    elevation: false,
    softness: false,
    reflection: false,
  },
} satisfies Record<ShadowModel, Record<string, boolean>>;
```

`opacity` remains useful for legacy/mock shadow generation and final presentation policy. Current diffusion output is treated as a full shadow image, not as a separate shadow layer.

## Public API

Use:

```http
POST /v1/render
Content-Type: application/json
```

The selected model is sent as:

```json
{
  "shadow": {
    "model": "v1-gan"
  }
}
```

Allowed values:

- `v1-gan`
- `v2-diff`

Compatibility:

- `shadow.model` is optional
- if omitted, ML core uses its configured runtime default
- new frontend clients should send it explicitly

## Sync Request Example: `V1-GAN`

```json
{
  "request_id": "front-2026-04-26-001",
  "pipeline_version": "ml-shadowgen-v1",
  "source": {
    "mime_type": "image/png",
    "image_base64": "<BASE64_IMAGE_WITHOUT_DATA_URL_PREFIX>"
  },
  "preprocess": {
    "padding_px": 100
  },
  "shadow": {
    "model": "v1-gan",
    "angle_deg": 45,
    "elevation_deg": 35,
    "softness": 0.5,
    "opacity": 0.65,
    "reflection": 0.1
  },
  "background": {
    "mode": "solid",
    "color_hex": "#FFFFFF"
  },
  "output": {
    "format": "png",
    "width": null,
    "height": null,
    "return_debug": false
  }
}
```

Notes:

- `angle_deg` affects `V1-GAN`
- `opacity` affects final shadow density
- `elevation_deg`, `softness`, and `reflection` are accepted for API stability, but do not currently affect `V1-GAN`

## Sync Request Example: `V2-DIFF`

```json
{
  "request_id": "front-2026-04-26-002",
  "pipeline_version": "ml-shadowgen-v1",
  "source": {
    "mime_type": "image/png",
    "image_base64": "<BASE64_IMAGE_WITHOUT_DATA_URL_PREFIX>"
  },
  "preprocess": {
    "padding_px": 100
  },
  "shadow": {
    "model": "v2-diff",
    "angle_deg": 45,
    "elevation_deg": 35,
    "softness": 0.5,
    "opacity": 0.65,
    "reflection": 0.1
  },
  "background": {
    "mode": "solid",
    "color_hex": "#FFFFFF"
  },
  "output": {
    "format": "png",
    "width": null,
    "height": null,
    "return_debug": false
  }
}
```

Notes:

- `V2-DIFF` currently ignores `angle_deg`, `elevation_deg`, `softness`, and `reflection`
- these fields are still required by the current request schema
- keep sending safe defaults until the public schema is simplified further

Recommended defaults for `V2-DIFF`:

```json
{
  "angle_deg": 45,
  "elevation_deg": 35,
  "softness": 0.5,
  "opacity": 0.65,
  "reflection": 0.0
}
```

## TypeScript Request Helper

```ts
type ShadowModel = "v1-gan" | "v2-diff";

type RenderRequestInput = {
  imageBase64: string;
  mimeType: "image/png" | "image/jpeg" | "image/webp";
  model: ShadowModel;
  angleDeg?: number;
  opacity?: number;
};

function buildRenderRequest(input: RenderRequestInput) {
  return {
    request_id: crypto.randomUUID(),
    pipeline_version: "ml-shadowgen-v1",
    source: {
      mime_type: input.mimeType,
      image_base64: input.imageBase64,
    },
    preprocess: {
      padding_px: 100,
    },
    shadow: {
      model: input.model,
      angle_deg: input.angleDeg ?? 45,
      elevation_deg: 35,
      softness: 0.5,
      opacity: input.opacity ?? 0.65,
      reflection: 0.0,
    },
    background: {
      mode: "solid",
      color_hex: "#FFFFFF",
    },
    output: {
      format: "png",
      width: null,
      height: null,
      return_debug: false,
    },
  };
}
```

## TypeScript Fetch Example

```ts
async function renderShadow(input: RenderRequestInput) {
  const response = await fetch("http://localhost:8000/v1/render", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(buildRenderRequest(input)),
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload?.error?.message ?? "Shadow render failed");
  }

  const finalArtifact = payload.artifacts.find((item: { kind: string }) => item.kind === "final");
  if (!finalArtifact) {
    throw new Error("ML core response does not contain a final artifact");
  }

  return {
    payload,
    finalImageSrc: `data:${finalArtifact.mime_type};base64,${finalArtifact.image_base64}`,
  };
}
```

## Async API

For worker-oriented or long-running flows, submit the same request body to:

```http
POST /v1/render/jobs
Content-Type: application/json
```

Example:

```ts
async function submitRenderJob(input: RenderRequestInput) {
  const response = await fetch("http://localhost:8000/v1/render/jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(buildRenderRequest(input)),
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload?.error?.message ?? "Shadow job submit failed");
  }

  return payload.job_id as string;
}
```

Poll:

```http
GET /v1/render/jobs/{job_id}
```

Completed job response contains `result`, which has the same shape as the sync `/v1/render` response.

## Fallback Behavior

The ML core may fallback when a requested backend is unavailable.

Current important case:

- requested `v2-diff` uses the local diffusion backend today
- future deployments may route `v2-diff` through Triton without changing the public request shape
- if the local/Triton model is unavailable, ML core may fallback to `v1-gan` or mock depending on runtime availability
- response `warnings` may include `shadow_generator_fallback_active`

Frontend recommendation:

- show the final image if render succeeds
- optionally show a small non-blocking warning when `warnings` contains `shadow_generator_fallback_active`
- do not assume `v2-diff` was used unless backend metadata is exposed by a debug or future product response

## Validation Rules

`shadow.model`:

- optional
- allowed: `v1-gan`, `v2-diff`
- unknown values return `422 validation_error`

`shadow.angle_deg`:

- range: `0..360`
- active for `V1-GAN`
- ignored by current `V2-DIFF`

`shadow.elevation_deg`:

- range: `0..90`
- ignored by current `V1-GAN` and `V2-DIFF`

`shadow.softness`:

- range: `0..1`
- ignored by current `V1-GAN` and `V2-DIFF`
- coarse softness behavior exists only in mock/debug paths

`shadow.opacity`:

- range: `0..1`
- accepted for API stability and legacy/mock paths

`shadow.reflection`:

- range: `0..1`
- reserved for future models

## Recommended Frontend UX

When model is `v1-gan`:

- enable `Angle`
- enable `Opacity`
- hide or disable `Elevation`, `Softness`, `Reflection`
- label it as controllable/classic

When model is `v2-diff`:

- disable `Angle`, `Elevation`, `Softness`, `Reflection`
- keep `Opacity` if the product wants density control
- show a hint that the model works automatically
