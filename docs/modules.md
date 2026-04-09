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
  Central environment-backed settings object.

- [schemas.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/schemas.py)
- [web_ui.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/web_ui.py)
  Legacy compatibility files. Do not put new logic here.

## `core/`

Purpose:

- define the service language
- keep type-safe contracts
- stay free of HTTP and UI details

Main files:

- [commands.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/commands.py)
  Request-side command objects such as `RenderCommand` and `ShadowSpec`.

- [contracts.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/contracts.py)
  Stage interfaces like `Detector`, `Segmenter`, `ShadowGenerator`.

- [models.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/models.py)
  Internal result and runtime metadata objects.

- [errors.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/core/errors.py)
  Service-level error hierarchy.

Add code here when:

- you need a new internal contract
- you need a new typed result object
- you need a new service error

Do not add:

- model loading
- FastAPI routes
- HTML

## `application/`

Purpose:

- orchestrate pipeline execution
- manage stage order and backend selection
- collect metrics and warnings

### `application/use_cases/`

- [render_pipeline.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/render_pipeline.py)
  Main public render path.

- [debug_pipeline.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/debug_pipeline.py)
  Stage-by-stage debug execution path for the playground.

- [get_health.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/get_health.py)
- [get_capabilities.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/use_cases/get_capabilities.py)

### `application/services/`

- [backend_selector.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/services/backend_selector.py)
  Decides which backend mode is actually used.

- [stage_runner.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/services/stage_runner.py)
  Shared stage execution wrapper.

- [stage_catalog.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/services/stage_catalog.py)
  Source of truth for stage order and human-facing names.

### Other files

- [dependencies.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/dependencies.py)
  `PipelineRuntime` container.

- [models.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/application/models.py)
  Pipeline context, metrics collector, debug execution objects.

Add code here when:

- logic is orchestration-oriented
- behavior is stage-agnostic
- it belongs to execution policy, not model internals

## `bootstrap/`

Purpose:

- assemble the runtime
- probe dependency availability
- create capability metadata

Main files:

- [container.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/bootstrap/container.py)
  Composition root. This is where active implementations are selected.

- [probes.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/bootstrap/probes.py)
  Lightweight backend availability probes.

- [runtime_descriptor.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/bootstrap/runtime_descriptor.py)
  Runtime status assembly for capabilities/health.

Add code here when:

- a new stage backend needs runtime wiring
- a new model family needs probe metadata

## `infrastructure/`

Purpose:

- concrete implementations
- technical persistence and encoding
- preview builders

### `infrastructure/stages/`

This is the main place for model code.

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

Pattern:

```text
stage_name/
  mock.py
  real_backend.py
  __init__.py
```

Examples:

- [geometry/geocalib.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/geometry/geocalib.py)
- [detection/grounding_dino.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/detection/grounding_dino.py)
- [segmentation/birefnet.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/segmentation/birefnet.py)
- [normals/stable_normal.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/normals/stable_normal.py)
- [normals/from_depth.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/normals/from_depth.py)
- [shadow/pix2pix.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/shadow/pix2pix.py)
- [shadow/v2_diff.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/stages/shadow/v2_diff.py)

Shadow stage specifics:

- `mock`: analytical fallback
- `V1-GAN`: current migrated pix2pix backend
- `V2-DIFF`: scaffold class, not implemented yet

### `infrastructure/cache/`

- [preprocess_cache_repository.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/cache/preprocess_cache_repository.py)

Purpose:

- store and restore preprocess snapshots

### `infrastructure/encoding/`

- [default.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/encoding/default.py)

Purpose:

- encode final and debug artifacts

### `infrastructure/presentation/`

- [preview_registry.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/src/shadowgen_ml_service/infrastructure/presentation/preview_registry.py)

Purpose:

- build per-stage debug previews for the playground

Add code here when:

- it is model-specific
- it is file/cache/preview implementation code
- it depends on technical libraries

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
- backend switching
- preview rendering

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
- crop preparation
- overlays
- encoding helpers
- non-model image processing utilities

Use with care:

- if utility logic becomes stage-specific, move it back into the relevant stage package

## Where To Edit Common Changes

If you want to add a new stage:

1. add contracts/models if needed in `core/`
2. add implementation under `infrastructure/stages/<stage>/`
3. wire it in `bootstrap/container.py`
4. expose it in `application/services/stage_catalog.py`
5. add debug/UI schema handling if needed
6. add tests
7. update docs

If you want to add a new backend variant for an existing stage:

1. create a new backend file inside that stage package
2. update runtime probes/wiring
3. update backend selection behavior
4. update debug details and playground labels
5. document the new variant

If you want to change API payloads:

1. update `interfaces/http/*schemas.py`
2. update `interfaces/http/mappers.py`
3. update use cases if command shape changes
4. update tests
5. update `docs/api.md`

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
