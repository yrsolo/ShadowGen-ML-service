# Module Map

This document is the practical map of the repository: where things live, what they are responsible for, and where to add new code.

## Top-Level Folders

```text
agent/         runtime contract for AI agents
docs/          active docs + historical bootstrap material
src/           service code
tests/         automated tests
work/          local tracking and roadmap notes
```

## `src/shadowgen_ml_service/`

### Root files

- [app.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/app.py)
  Compatibility export for app creation.

- [main.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/main.py)
  Module entrypoint used by Uvicorn.

- [config.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/config.py)
  Central environment-backed settings object for local, Triton, and async runtime behavior.

- [schemas.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/schemas.py)
- [web_ui.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/web_ui.py)
  Legacy compatibility files. Do not put new logic here.

## `core/`

Purpose:

- define the service language
- keep stable internal contracts
- stay free of HTTP, UI, and transport details

Main files:

- [commands.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/commands.py)
  Request-side command objects such as `RenderCommand`, `ShadowSpec`, and debug command metadata.

- [contracts.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/contracts.py)
  Stage interfaces like `Detector`, `Segmenter`, `ShadowGenerator`.

- [models.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/models.py)
  Internal result models and execution-aware runtime metadata.

- [assets.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/assets.py)
  Transport-neutral raster asset contract used by heavy-stage inputs and outputs.

- [stage_io.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/stage_io.py)
  Canonical heavy-stage input objects. This is the real boundary between orchestration and executors.

- [job_contracts.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/job_contracts.py)
  Async render job state and repository-facing contracts.

- [errors.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/errors.py)
  Service-level error hierarchy.

Add code here when:

- you need a new internal contract
- you need a new typed result object
- you need a canonical stage input/output shape
- you need a new service error

Do not add:

- model loading
- FastAPI routes
- Triton transport code
- HTML

## `application/`

Purpose:

- orchestrate pipeline execution
- manage stage order and backend selection
- keep sync and async use cases stable

### `application/use_cases/`

- [render_pipeline.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/render_pipeline.py)
  Main synchronous public render path.

- [debug_pipeline.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/debug_pipeline.py)
  Stage-by-stage debug execution path for the playground.

- [submit_render_job.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/submit_render_job.py)
- [get_render_job.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/get_render_job.py)
- [cancel_render_job.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/cancel_render_job.py)
  Async render job orchestration.

- [get_health.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/get_health.py)
- [get_capabilities.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/get_capabilities.py)

### `application/services/`

- [backend_selector.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/services/backend_selector.py)
  Selects `mock`, `local`, or `triton` plus the effective model variant.

- [stage_runner.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/services/stage_runner.py)
  Shared stage execution wrapper with metadata capture and normalized stage-level error handling.

- [stage_catalog.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/services/stage_catalog.py)
  Source of truth for stage order and human-facing names.

### Other files

- [dependencies.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/dependencies.py)
  Registry-based runtime container and backend registration primitives.

- [models.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/models.py)
  Pipeline context, metrics collector, stage execution metadata, async job records.

Add code here when:

- logic is orchestration-oriented
- behavior is stage-agnostic
- it belongs to execution policy, not model internals

## `bootstrap/`

Purpose:

- assemble the runtime
- probe dependency availability
- register all backends
- create capability metadata

Main files:

- [container.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/bootstrap/container.py)
  Composition root. Registers backends per stage and chooses defaults.

- [triton_bindings.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/bootstrap/triton_bindings.py)
  Centralized Triton model binding schemas with tensor-name, dtype, rank, and layout expectations.

- [probes.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/bootstrap/probes.py)
  Lightweight backend availability probes.

- [runtime_descriptor.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/bootstrap/runtime_descriptor.py)
  Converts registered backend descriptors into health/capability payloads.

Add code here when:

- a new stage backend needs runtime wiring
- a stage becomes Triton-aware
- capability metadata rules change

## `infrastructure/`

Purpose:

- concrete implementations
- technical persistence and encoding
- preview building
- transport integrations

### `infrastructure/backends/triton/`

This is the shared Triton subsystem.

Main files:

- [client.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/backends/triton/client.py)
  Shared transport wrapper for standard Triton infer calls with explicit tensor inputs and outputs.

- [config.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/backends/triton/config.py)
  Triton endpoint settings.

- [model_registry.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/backends/triton/model_registry.py)
  Mapping from stage and model variant to Triton model identifiers plus tensor binding schemas.

- [serializers.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/backends/triton/serializers.py)
  Tensor serialization, response decoding, and schema validation helpers for the standard Triton tensor protocol.

- [errors.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/backends/triton/errors.py)
  Triton-specific error types.

- [batching.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/backends/triton/batching.py)
  Batching capability helpers.

Rule:

- transport-level concerns live here, not in `application/`

### `infrastructure/stages/`

This is the main place for stage code.

Current stage packages:

- `geometry/`
- `detection/`
- `segmentation/`
- `foreground_refinement/`
- `depth/`
- `normals/`
- `shadow/`
- `composition/`
- `shared/`

Current backend pattern for heavy stages:

```text
stage_name/
  mock.py
  local_backend.py
  triton.py
```

Examples:

- [geometry/geocalib.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/geometry/geocalib.py)
- [detection/grounding_dino.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/detection/grounding_dino.py)
- [detection/triton.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/detection/triton.py)
- [segmentation/birefnet.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/segmentation/birefnet.py)
- [segmentation/triton.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/segmentation/triton.py)
- [depth/depth_anything.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/depth/depth_anything.py)
- [depth/triton.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/depth/triton.py)
- [normals/stable_normal.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/normals/stable_normal.py)
- [normals/from_depth.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/normals/from_depth.py)
- [normals/triton.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/normals/triton.py)
- [shadow/pix2pix.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/shadow/pix2pix.py)
- [shadow/v2_diff.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/shadow/v2_diff.py)
- [shadow/triton.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/shadow/triton.py)

Shadow stage specifics:

- `mock`: analytical fallback
- `V1-GAN`: current local backend
- `V2-DIFF`: preferred Triton-ready slot

### `infrastructure/cache/`

- [preprocess_cache_repository.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/cache/preprocess_cache_repository.py)

Purpose:

- store and restore preprocess snapshots
- keep cache logic outside use cases

### `infrastructure/encoding/`

- [default.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/encoding/default.py)

Purpose:

- encode final and debug artifacts

### `infrastructure/presentation/`

- [preview_registry.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/presentation/preview_registry.py)

Purpose:

- build per-stage debug previews for the playground

### `infrastructure/jobs/`

- [in_memory.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/jobs/in_memory.py)

Purpose:

- current in-process async render job backend
- replaceable job repository/queue implementation

Add code here when:

- it is model-specific
- it is file/cache/preview implementation code
- it depends on transport, persistence, or runtime libraries

## `interfaces/http/`

Purpose:

- define the external HTTP surface

Main files:

- [app.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/interfaces/http/app.py)
- [public_routes.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/interfaces/http/public_routes.py)
- [dev_routes.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/interfaces/http/dev_routes.py)
- [public_schemas.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/interfaces/http/public_schemas.py)
- [dev_schemas.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/interfaces/http/dev_schemas.py)
- [mappers.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/interfaces/http/mappers.py)

Add code here when:

- API payload shape changes
- async API changes
- debug request/response schema changes
- route behavior changes

## `interfaces/dev/`

Purpose:

- browser playground for manual testing

Main file:

- [playground.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/interfaces/dev/playground.py)

This file contains the inline HTML/CSS/JS page used for:

- file upload
- contract parameter control
- stage reruns
- backend-kind switching
- shadow model variant switching
- preview rendering
- execution metadata inspection

## `pipeline/`

Purpose:

- compatibility shims for older imports and earlier structure

Files such as:

- [runtime.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/pipeline/runtime.py)
- [service.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/pipeline/service.py)
- [types.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/pipeline/types.py)

Rule:

- do not add new core logic here
- use them only to preserve compatibility

## `adapters/`

Purpose:

- compatibility export layer for old import sites

Rule:

- useful for transition
- not the place for new behavior

## `utils/`

Main file:

- [images.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/utils/images.py)

Purpose:

- shared image decoding
- `RasterAsset <-> PIL` conversion helpers
- crop preparation
- overlays
- encoding helpers
- non-model image processing utilities

Use with care:

- if utility logic becomes stage-specific, move it into the relevant stage package

## Where To Edit Common Changes

If you want to add a new stage:

1. add contracts/models if needed in `core/`
2. add implementation under `infrastructure/stages/<stage>/`
3. wire it in `bootstrap/container.py`
4. expose it in `application/services/stage_catalog.py`
5. update selector/runtime metadata if execution options changed
6. add tests
7. update docs

If you want to add a new backend variant for an existing stage:

1. create a backend file inside that stage package
2. if it is Triton-based, use the shared Triton subsystem
3. update runtime probes and registry wiring
4. update backend selection behavior
5. update debug details and playground labels
6. document the new variant

If you want to change API payloads:

1. update `interfaces/http/*schemas.py`
2. update `interfaces/http/mappers.py`
3. update use cases if command shape changes
4. update tests
5. update `docs/api.md`

If you want to change async execution:

1. update `core/job_contracts.py`
2. update `application/use_cases/*render_job.py`
3. update `infrastructure/jobs/`
4. update public schemas/routes
5. update docs

## Non-Code Folders

### `tests/`

Includes:

- API tests
- runtime wiring tests
- architecture boundary tests
- stage-specific tests

### `work/`

Includes:

- current task tracking
- roadmap
- campaign evidence

### `docs/first/`

Historical early-phase planning material.

Keep it as reference, but do not treat it as the primary source of truth for the running service.
