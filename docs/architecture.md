# Architecture

## Purpose

`ShadowGen ML Service` is a local-first synchronous ML service that takes a source image, analyzes the foreground object and scene cues, generates a shadow layer, and returns a final composited artifact.

The service is designed to be:

- stateless at the HTTP boundary
- explicit about runtime backend selection
- easy to test with deterministic mock stages
- ready for iterative model replacement without rewriting the whole service

## High-Level Flow

Pipeline order:

1. Decode and validate request payload
2. Estimate scene geometry on the full source image
3. Detect the primary foreground object
4. Crop, pad, and fit the object into the working canvas
5. Segment the object on the prepared working crop
6. Refine semi-transparent foreground colours
7. Estimate depth
8. Estimate normals
9. Generate shadow
10. Compose object and shadow on the background
11. Encode output artifacts and metrics

Important processing rules:

- geometry and detection run on the original image
- segmentation runs after crop / pad / resize
- foreground refinement is a standalone stage, not hidden inside segmentation
- depth and normals run on the refined foreground result
- shadow is its own stage with named model variants

## Layered Structure

The service follows a clean layered split.

### `core/`

Contains:

- commands
- typed models
- pipeline contracts
- service errors

Rules:

- no FastAPI
- no Pydantic
- no UI logic
- no model-specific imports

### `application/`

Contains:

- use cases
- pipeline context
- metrics collection
- backend selection
- stage catalog
- stage runner

Rules:

- orchestrates the pipeline
- depends on `core`
- does not know about concrete HTTP schema classes

### `bootstrap/`

Contains:

- runtime probes
- dependency composition
- runtime descriptor assembly

Purpose:

- decide which backends are active
- wire mock, fallback, and real implementations
- expose machine-readable runtime status to API/UI

### `infrastructure/`

Contains:

- model backends
- preview builders
- cache persistence
- artifact encoding

This is where stage-specific implementation code lives.

### `interfaces/http/`

Contains:

- FastAPI app factory
- public routes
- dev routes
- request and response schemas
- HTTP mappers

### `interfaces/dev/`

Contains:

- browser playground UI

## Runtime Philosophy

Each stage should ideally support:

- deterministic mock backend
- real backend
- predictable fallback behavior

This is already implemented for multiple stages and is especially important for:

- development without GPU
- debugging broken model initialization
- preserving a stable UI and API while model backends change

## Stage-by-Stage Overview

### Geometry

- Stage key: `geometry_estimator`
- Runs on: full source image
- Primary backend: GeoCalib
- Outputs:
  - `camera_fov`
  - `camera_pitch`
  - `camera_roll`
  - `confidence`

### Detection

- Stage key: `detector`
- Runs on: full source image
- Primary backend: GroundingDINO
- Purpose:
  - locate main object
  - produce crop bbox
  - prepare downstream working crop

### Segmentation

- Stage key: `segmenter`
- Runs on: prepared working crop
- Primary backend: BiRefNet
- Outputs:
  - binary mask
  - RGBA cutout
  - crop metadata

### Foreground Refinement

- Stage key: `foreground_refiner`
- Runs on: segmented crop plus alpha
- Primary backend: Fast Foreground Colour Estimation
- Purpose:
  - fix semi-transparent RGB contamination
  - improve downstream depth/shadow/composition quality

### Depth

- Stage key: `depth_estimator`
- Runs on: refined cutout
- Primary backend: Depth Anything V2 Small
- Output:
  - normalized single-channel depth map

### Normals

- Stage key: `normal_estimator`
- Primary backend: StableNormal
- Fallback backend: `from-depth`
- Mock backend: flat neutral normal map

Important:

- normals are no longer treated as “just depth postprocessing”
- the stage now has its own model identity and explicit fallback path

### Shadow

- Stage key: `shadow_generator`
- Current variants:
  - `mock`
  - `V1-GAN`
  - `V2-DIFF`

#### `mock`

- deterministic analytical shadow stub
- keeps the coarse softness blur behavior
- used for fast fallback and predictable tests

#### `V1-GAN`

- current real shadow model
- migrated from the legacy pix2pix ShadowGEN code
- only minimal inference code and one generator checkpoint were imported

Inputs:

- `img`
- `mask`
- `depth`
- `normal`
- `angle`
- `elevation`
- `softness`
- `reflection`

Important:

- `softness` is treated as model input
- no post-blur is applied after the real model output

#### `V2-DIFF`

- reserved runtime slot for the next shadow model family
- recommended class scaffold already exists
- inference backend intentionally not implemented yet

This means the service is already prepared for a controlled transition:

- `V1-GAN` can remain active
- `V2-DIFF` can be integrated without changing the external debug UX

### Composition

- Stage key: `composer`
- Current backend: Python compositor
- Purpose:
  - place cutout and shadow on target background
  - return final render image

## Cache and Artifacts

The service keeps a preprocess cache for expensive intermediate results.

Cached data can include:

- detection
- geometry
- segmentation
- foreground refinement
- depth
- normals

Cache key includes:

- input bytes hash
- runtime signature
- padding
- working size

This lets the service invalidate stale cached results when model variants or implementations change.

## Public vs Dev Interfaces

Public endpoints:

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/render`

Dev interface:

- `GET /playground`
- `POST /v1/dev/pipeline/run-all`
- `POST /v1/dev/pipeline/run-stage/{stage_key}`

The dev interface is intentionally richer than the public one and exposes:

- per-stage timing
- requested vs actual backend mode
- device info
- debug previews
- stage-local details

## Compatibility Shims

The repository still contains compatibility modules such as:

- `pipeline/`
- `adapters/`
- some root-level exports like `app.py`

These are not the architectural center anymore.
New logic should go into the layered structure, not into the shims.

## Design Intent

This repo is optimized for long-lived model iteration.

That means:

- model upgrades should mostly touch `infrastructure/stages/...`
- runtime selection should mostly touch `bootstrap/`
- orchestration should stay stable in `application/`
- public API should remain stable in `interfaces/http/`

That separation is the main architectural goal of the project.
