# ShadowGen ML Service

`ShadowGen ML Service` is a hybrid orchestration service for foreground extraction, scene analysis, shadow generation, and final composition.

It is intentionally split into:

- a control plane: FastAPI API, web playground, pipeline orchestration, cache, previews, sync and async execution flows
- an execution plane: interchangeable `mock`, `local`, and `triton` backends for heavy ML stages

## Quick View

- Public API:
  - `GET /health`
  - `GET /v1/capabilities`
  - `POST /v1/render`
  - `POST /v1/render/jobs`
  - `GET /v1/render/jobs/{job_id}`
  - `DELETE /v1/render/jobs/{job_id}`
- Dev API:
  - `GET /playground`
  - `POST /v1/dev/pipeline/run-all`
  - `POST /v1/dev/pipeline/run-stage/{stage_key}`
- Main runtime target: local NVIDIA GPU workstation with optional Triton execution for heavy stages

## Pipeline

Current stage order:

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

- `Geometry`: GeoCalib, local-only in phase 1
- `Detection`: GroundingDINO, `mock|local|triton`
- `Segmentation`: BiRefNet, `mock|local|triton`
- `Foreground`: Fast Foreground Colour Estimation, local-only in phase 1
- `Depth`: Depth Anything V2 Small, `mock|local|triton`
- `Normals`:
  - `mock`
  - `StableNormal local`
  - `StableNormal triton scaffold`
  - `from-depth` local fallback
- `Shadow`:
  - `mock`
  - `V1-GAN local`
  - `V2-DIFF local`
  - `V2-DIFF triton scaffold`
- `Composition`: Python compositor, local-only

## Backend Model

The service no longer treats execution as just `mock|real`.

Each heavy stage is described by:

- `backend_kind`: `mock`, `local`, or `triton`
- `model_variant`: stage-specific variant such as `grounding-dino`, `birefnet`, `stable-normal`, `v1-gan`, `v2-diff`

The runtime resolves:

- requested backend kind
- actual backend kind
- active model variant
- device
- endpoint
- fallback reason

This metadata is exposed through both capabilities and dev-stage responses.

Heavy stages now enter execution through canonical `stage_io` input objects, so orchestration no longer depends on backend-specific argument lists.

The Triton subsystem now targets the standard Triton tensor infer schema with explicit `inputs` and `outputs` instead of stage-specific JSON bodies.

## Project Layout

```text
src/shadowgen_ml_service/
  core/                    domain contracts, stage I/O, typed models, job contracts
  application/             use cases, stage runner, backend selection, pipeline context
  bootstrap/               runtime assembly, probes, registry defaults, descriptors
  infrastructure/
    backends/triton/       Triton client, model registry, serializers, transport errors
    cache/                 preprocess cache repository
    encoding/              artifact encoding
    jobs/                  in-memory async job backend
    presentation/          preview registry
    stages/                stage implementations and per-stage local/mock/triton backends
  interfaces/http/         FastAPI routes, schemas, mappers
  interfaces/dev/          playground UI
  pipeline/                compatibility shims for older imports
  adapters/                compatibility exports for older imports
  utils/                   shared image utilities

docs/
  README.md
  architecture.md
  modules.md
  worker-core-contract.md
  shadow-v2-model-contract.md
  api.md
  runbook-local.md
  workflow.md
  first/
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

Rebuild the Triton image after changing code under `ops/triton/model_repository`:

```cmd
rebuild-triton.cmd
```

Start Triton when you need Triton-backed stages:

```cmd
start-triton.cmd
```

Start the FastAPI service:

```cmd
start-service.cmd
```

`start-triton.cmd` starts only the Triton Docker container and waits for model readiness. `start-service.cmd` starts only FastAPI in the current visible console. This keeps the Triton container lifecycle explicit.

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/playground`

## Triton Readiness

The repository is now structured so that the orchestrator stays stable while stage executors can move between `mock`, `local`, and `triton`.

Current Triton-ready stage boundaries:

- `detector`
- `segmenter`
- `depth_estimator`
- `normal_estimator`
- `shadow_generator`

Current local-only phase-1 stages:

- `geometry_estimator`
- `foreground_refiner`
- `composer`
- `artifact_encoder`

This means web UI, sync render, async jobs, cache, and previews remain in this service even when heavy inference is moved to Triton.

Current live Triton bridge:

- `detector` uses a temporary Triton `python` backend around GroundingDINO
- `detector` also has an experimental `grounding-dino-onnx` variant served as `shadowgen_detector_onnx` when exported locally
- `segmenter` uses a temporary Triton `python` backend around BiRefNet
- `segmenter` also has a live `rmbg-2.0` ONNX slot served as `shadowgen_segmenter_rmbg2` when gated weights are prepared locally
- ML-core now uses official Triton HTTP binary tensor transport by default; set `SHADOWGEN_TRITON_TRANSPORT=json` only for debugging
- `RMBG-2.0` is gated on Hugging Face; the preparation tool reads `HF_TOKEN` from process env or the repository `.env`
- the current BRIA RMBG-2.0 ONNX export has fixed batch `1`, so this variant is served without Triton dynamic batching
- `GroundingDINO` ONNX export returns model tensors (`logits`, `pred_boxes`); ML-core keeps bbox postprocess in the Triton adapter
- `torch.compile` remains an opt-in acceleration lever while ONNX export is blocked by `torchvision::deform_conv2d`
- `ONNX` stays the planned first long-term production model format

Normal local lifecycle:

```cmd
rebuild-triton.cmd
start-triton.cmd
start-service.cmd
```

Run `rebuild-triton.cmd` only after Triton Python backend code changes. Run `start-triton.cmd` to start or restart the Triton container. Run `start-service.cmd` for the FastAPI service.

The rebuild script builds [ops/triton/Dockerfile.segmenter-python](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/Dockerfile.segmenter-python). Large generated ONNX files are intentionally excluded from Docker build context and mounted by [start-triton.cmd](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/start-triton.cmd). The Triton start script exposes HTTP on host `8010`, gRPC on host `8011`, and metrics on host `8012`.

The default launcher mode uses Docker GPU (`TRITON_GPU=1`), `TRITON_DEVICE=cuda:0`, and a 512px segmenter resolution. Detector and segmenter default to local execution because the current Triton Python backends are still slower than local in-process models. Set `USE_TRITON_BACKENDS=1` or choose `triton` in `/playground` to test Triton execution.

Live smoke check:

```cmd
.venv\Scripts\python.exe tools\smoke_triton_detector.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
.venv\Scripts\python.exe tools\smoke_triton_detector.py --variant grounding-dino-onnx --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --variant rmbg-2.0 --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
.venv\Scripts\python.exe tools\benchmark_stage_backends.py --stage all --base-url http://127.0.0.1:8010 --transport local --transport triton-native
```

## Where To Read Next

- [Docs Index](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/README.md)
- [Architecture](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/architecture.md)
- [Module Map](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/modules.md)
- [Worker/Core Contract](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/worker-core-contract.md)
- [Shadow V2 Model Contract](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/shadow-v2-model-contract.md)
- [Local Runbook](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/runbook-local.md)
- [API Summary](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
- [Workflow](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/workflow.md)
