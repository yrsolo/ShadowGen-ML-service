# Architecture

## Purpose

`ShadowGen ML Service` is a hybrid orchestrator for foreground extraction, scene analysis, shadow generation, and final composition.

The architectural goal is:

- stable control plane
- replaceable execution plane
- unchanged debug UX while heavy stages move between `mock`, `local`, and `triton`

## Architectural Split

### Control Plane

Lives in this service and owns:

- FastAPI public API
- FastAPI dev API
- web playground
- synchronous render orchestration
- asynchronous job orchestration
- crop / pad / resize preprocessing
- preview generation
- preprocess cache
- backend selection and fallback policy
- capabilities and health reporting

### Execution Plane

Provides stage executors:

- `mock`
- `local`
- `triton`

Application code does not depend on a specific transport or executor type. It only works with stage ports and execution metadata.

## Pipeline Flow

Current stage order:

1. decode and validate request payload
2. estimate geometry on the full image
3. detect the primary object on the full image
4. crop, pad, and fit into the canonical working canvas
5. segment on the prepared working crop
6. refine semi-transparent foreground colours
7. estimate depth
8. estimate normals
9. generate shadow
10. compose cutout and shadow
11. encode output artifacts

Processing rules:

- geometry and detection run on the original image
- segmentation runs after crop / pad / resize
- foreground refinement is a standalone stage
- heavy downstream stages consume canonical working-canvas inputs
- shadow consumes canonical `img`, `mask`, `depth`, `normal`, `angle`, `elevation`, `softness`, `reflection`

## Layered Structure

### `core/`

Contains stage-independent contracts and typed models:

- command models
- runtime metadata
- stage result models
- canonical stage I/O contracts
- async job contracts
- error types

Rules:

- no FastAPI
- no Pydantic
- no Triton transport code
- no HTML/UI logic

### `application/`

Contains orchestration and execution policy:

- sync render use case
- sync debug use case
- async job use cases
- backend selector
- stage runner
- pipeline context
- metrics collection
- registry-aware runtime access

Rules:

- depends on `core`
- may depend on generic stage ports
- must not depend on concrete `local` or `triton` backend classes

### `bootstrap/`

Contains runtime assembly:

- probes
- backend registration
- default backend policy
- runtime descriptors for capabilities and health

Responsibilities:

- register all known backends per stage
- decide active defaults per stage
- expose machine-readable execution metadata

### `infrastructure/`

Contains technical implementations:

- stage backends
- Triton client subsystem
- cache repository
- artifact encoder
- preview builders
- async job backend

This is the only layer that should know transport details, model-loading details, and persistence details.

### `interfaces/http/`

Contains:

- route handlers
- request/response schemas
- HTTP mappers
- app factory

### `interfaces/dev/`

Contains:

- browser playground UI

## Execution Model

The old `mock|real` model is no longer sufficient.

Each stage execution is described by:

- `backend_kind`: `mock`, `local`, `triton`, or `internal`
- `model_variant`
- `model_name`
- `model_version`
- `device`
- `endpoint`
- `supports_batching`
- `supports_async`
- `fallback_reason`

Compatibility fields such as `requested_mode` and `actual_mode` still exist in dev responses, but they are now derived from the execution-aware model.

## Registry-Based Runtime

The runtime is now registry-based rather than field-based.

Instead of storing fixed slots like:

- `real_detector`
- `mock_detector`
- `shadow_v1_gan`

the runtime keeps a backend registry keyed by:

- `stage_key`
- `backend_kind`
- `model_variant`

This enables:

- explicit per-stage backend selection
- clean fallback rules
- local and Triton coexistence
- async job reuse of the same runtime

## Stage Ports and Canonical I/O

Heavy stage execution is normalized through canonical inputs:

- `DetectionInput`
- `SegmentationInput`
- `DepthInput`
- `NormalsInput`
- `ShadowInput`

This matters because Triton adapters should not leak transport details into orchestration code.

The orchestrator works in terms of semantic stage inputs and outputs. Serialization to tensors or bytes happens inside adapters or Triton serializers.

## Triton Subsystem

Triton integration is a first-class subsystem under `infrastructure/backends/triton/`.

Main responsibilities:

- endpoint configuration
- health probing
- model registry
- request serialization
- response deserialization
- transport-level timeout and mismatch handling
- batching capability metadata

Stage-specific Triton adapters live under the relevant stage package and depend on this subsystem.

Current Triton-ready stage families:

- `detector`
- `segmenter`
- `depth_estimator`
- `normal_estimator`
- `shadow_generator` for `V2-DIFF`

## Sync and Async Flows

### Sync

Used by:

- `POST /v1/render`
- dev pipeline endpoints
- browser playground

Characteristics:

- immediate response
- detailed previews
- easier debugging

### Async

Used by:

- `POST /v1/render/jobs`
- `GET /v1/render/jobs/{job_id}`
- `DELETE /v1/render/jobs/{job_id}`

Characteristics:

- production-oriented path
- job state externalized through API
- reuses the same backend registry and stage runner

Current async backend is in-process and replaceable by design.

## Stage Overview

### Local-Only Phase-1 Stages

#### Geometry

- Stage key: `geometry_estimator`
- Backends: `mock`, `local`
- Local backend: GeoCalib
- Runs on the full image

#### Foreground Refinement

- Stage key: `foreground_refiner`
- Backends: `mock`, `local`
- Local backend: Fast Foreground Colour Estimation

#### Composition

- Stage key: `composer`
- Backend kind: `local`
- Local backend: Python compositor

#### Artifact Encoding

- Stage key: `artifact_encoder`
- Backend kind: `internal`

### Heavy Triton-Ready Stages

#### Detection

- Stage key: `detector`
- Backends: `mock`, `local`, `triton`
- Local backend: GroundingDINO
- Triton variant: `grounding-dino`

#### Segmentation

- Stage key: `segmenter`
- Backends: `mock`, `local`, `triton`
- Local backend: BiRefNet
- Triton variant: `birefnet`

#### Depth

- Stage key: `depth_estimator`
- Backends: `mock`, `local`, `triton`
- Local backend: Depth Anything V2 Small
- Triton variant: `depth-anything-v2-small`

#### Normals

- Stage key: `normal_estimator`
- Backends:
  - `mock`
  - `local stable-normal`
  - `local from-depth-v2` fallback
  - `triton stable-normal`

Important:

- normals are a first-class stage
- `from-depth-v2` is still an explicit local fallback

#### Shadow

- Stage key: `shadow_generator`
- Backends:
  - `mock`
  - `local v1-gan`
  - `triton v2-diff`

Important:

- `V1-GAN` remains the current working local model
- `V2-DIFF` now exists as the preferred Triton-ready slot
- `softness` is a model input for real backends
- coarse post-blur remains only in the mock backend

## Cache and Metadata

The preprocess cache stores expensive intermediate artifacts.

Cache signature includes runtime identity so cached data can be invalidated when:

- backend kind changes
- model variant changes
- model version changes
- runtime defaults change

Dev execution metadata now surfaces:

- requested backend kind
- actual backend kind
- model variant
- model name and version
- device
- Triton endpoint
- cache status
- fallback reason

## Web UI Role

The web playground remains in this service.

That is intentional:

- Triton should execute heavy inference
- the control plane should keep rich debug UX

The playground now selects:

- execution backend kind per stage
- model variant where relevant, especially for shadow

It does not know or care whether a stage is backed by a Python model instance or a remote Triton model.

## Compatibility Layer

The repository still includes compatibility shims such as:

- `pipeline/`
- `adapters/`
- root exports like `app.py`

These preserve older import paths but are not the architectural center.

New logic belongs in:

- `core/`
- `application/`
- `bootstrap/`
- `infrastructure/`
- `interfaces/`

## Design Intent

This architecture is optimized for:

- replacing local backends with Triton without rewriting orchestration
- preserving the debug web UI
- supporting sync and async execution side by side
- keeping stage ownership explicit and testable
