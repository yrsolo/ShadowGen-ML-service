# ShadowGen ML Service

Stateless synchronous ML service for foreground extraction, scene analysis, shadow generation, and final composition.

## Quick View

- API: `GET /health`, `GET /v1/capabilities`, `POST /v1/render`
- Dev UI: `GET /playground`
- Main runtime target: local NVIDIA GPU workstation
- Core pipeline:
  1. decode
  2. geometry
  3. detection
  4. crop / pad / resize
  5. segmentation
  6. foreground refinement
  7. depth
  8. normals
  9. shadow
  10. composition
  11. artifact encoding

## Current Model Stack

- `Geometry`: GeoCalib
- `Detection`: GroundingDINO
- `Segmentation`: BiRefNet
- `Foreground`: Fast Foreground Colour Estimation
- `Depth`: Depth Anything V2 Small
- `Normals`: StableNormal with `from-depth` fallback
- `Shadow`:
  - `mock`
  - `V1-GAN` for the migrated legacy pix2pix model
  - `V2-DIFF` scaffold only, backend not implemented yet

## Project Layout

Top-level structure:

```text
src/shadowgen_ml_service/
  core/             domain contracts, typed models, commands, errors
  application/      use cases, stage runner, backend selection, pipeline context
  bootstrap/        composition root and runtime probes
  infrastructure/   model adapters, cache, encoding, preview builders
  interfaces/http/  FastAPI routes, schemas, mappers
  interfaces/dev/   playground UI
  pipeline/         compatibility shims for older imports
  adapters/         compatibility exports for older imports
  utils/            image utilities shared across stages

docs/
  README.md         docs index
  architecture.md   system design and stage responsibilities
  modules.md        codebase map and folder-by-folder explanation
  api.md            public and debug API summary
  runbook-local.md  local environment, startup, GPU bring-up, model notes
  workflow.md       working rules and repository conventions
  first/            bootstrap-era source materials and drafts
```

## Quick Start

Base environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -e .[dev]
```

GPU-oriented ML setup:

```powershell
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
.venv\Scripts\python.exe -m pip install -e .[dev,ml]
```

Run the service:

```powershell
.venv\Scripts\python.exe -m uvicorn shadowgen_ml_service.main:app --reload
```

Or on Windows:

```cmd
run-service.cmd
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/playground`

## Where To Read Next

- [Docs Index](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/README.md)
- [Architecture](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/architecture.md)
- [Module Map](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/modules.md)
- [Local Runbook](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/runbook-local.md)
- [API Summary](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
- [Workflow](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/workflow.md)

## Current State

The repository already contains:

- a working FastAPI service
- a stage-by-stage playground UI
- real and mock backends for multiple ML stages
- local model and cache handling
- tests for API, runtime wiring, stage behavior, and architectural boundaries

The repository intentionally still includes:

- compatibility shim modules under `pipeline/`, `adapters/`, and a few root exports
- historical source material under `docs/first/`

These remain to support gradual cleanup without breaking the active workflow.
