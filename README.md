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
- Dev API is disabled by default in application settings. Local launchers enable it for the playground; the process shutdown endpoint is enabled only for the visible-console `start-service.cmd` flow, not for the Docker service container.
- Main runtime target: local NVIDIA GPU workstation with either a two-container Docker stack or split local/Triton debugging.

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

In `/playground`, detector and segmenter can be switched by `model_variant` as well as backend:

- Detection: `grounding-dino` or `grounding-dino-onnx`
- Segmentation: `birefnet` or `rmbg-2.0`

Choosing an ONNX variant automatically selects the `triton` backend for that stage. Switching back to `local` resets the stage to its local variant.

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
  docker-local.md
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

Recommended production replacement Docker workflow:

```cmd
rebuild-service-container.cmd
start-service-container.cmd
```

Select the host GPU in `.env`:

```dotenv
SERVICE_GPU_DEVICE=1
SHADOWGEN_TARGET_DEVICE=cuda:0
SERVICE_HTTP_PORT=9001
```

Open:

- `http://127.0.0.1:9001/`

Stop the service container:

```cmd
stop-service-container.cmd
```

This starts one container:

- `shadowgen-ml-service`: FastAPI control plane and local ML backends on `SERVICE_HTTP_PORT` (default `9001`)

The selected host GPU is controlled by `SERVICE_GPU_DEVICE`. Inside the container the selected card is addressed as `cuda:0`, so `SHADOWGEN_TARGET_DEVICE=cuda:0` is the normal setting.

Runtime models, caches, outputs, and `.env` remain mounts and are not baked into the service image.

Triton/debug Docker workflow:

```cmd
rebuild-triton.cmd
rebuild-service-container.cmd
start-docker-stack.cmd
```

This starts two containers:

- `shadowgen-triton-segmenter`: Triton execution plane on host ports `8010`/`8011`/`8012`
- `shadowgen-ml-service`: FastAPI control plane and playground on `SERVICE_HTTP_PORT` (default `9001`)

The service container talks to Triton through the Docker network at `http://triton:8000`. Models, caches, outputs, and `.env` remain runtime mounts and are not baked into the service image.

Advanced split debugging workflow:

```cmd
rebuild-triton.cmd
start-triton.cmd
start-service.cmd
```

Use this when you want FastAPI in a visible Windows console or need the playground shutdown button to stop the local process. `start-triton.cmd` starts only the Triton container. `start-service.cmd` starts only FastAPI on the host Python environment.

For non-local deployments, leave `SHADOWGEN_DEV_API_ENABLED=0` and `SHADOWGEN_DEV_SHUTDOWN_ENABLED=0` unless the playground and shutdown endpoint are intentionally exposed.

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
- `RMBG-2.0` is fed at `1024x1024` by default through `SHADOWGEN_TRITON_SEGMENTER_RMBG2_RESOLUTION`; the current ONNX graph fails at `512x512` despite dynamic-shape metadata
- `GroundingDINO` ONNX export returns model tensors (`logits`, `pred_boxes`); ML-core keeps bbox postprocess in the Triton adapter
- `torch.compile` remains an opt-in acceleration lever while ONNX export is blocked by `torchvision::deform_conv2d`
- `ONNX` stays the planned first long-term production model format

Recommended production container lifecycle:

```cmd
rebuild-service-container.cmd
start-service-container.cmd
```

Run `rebuild-service-container.cmd` after service code or Python dependency changes. Run `start-service-container.cmd` to start the service-only production replacement container without Triton. Run `stop-service-container.cmd` to stop it.

The service rebuild script builds [Dockerfile.service](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/Dockerfile.service). Large generated model files, `.env`, `.models`, `.cache`, `artifacts`, and `var` are excluded from the service Docker build context and mounted at runtime.

The production service container reads `SERVICE_GPU_DEVICE` from `.env`, reserves that host GPU through Docker Compose, and sets `NVIDIA_VISIBLE_DEVICES` consistently. Inside the container the selected GPU is addressed as `cuda:0`. On the current workstation host GPU `1` is the RTX 4090. Use [.env.example](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/.env.example) as a safe configuration reference.

For Triton bring-up, use `rebuild-triton.cmd`, `start-docker-stack.cmd`, or the split debug flow `start-triton.cmd` + `start-service.cmd`.

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
- [Service Integration Contract](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/service-contract.md)
- [Worker/Core Contract](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/worker-core-contract.md)
- [Shadow V2 Model Contract](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/shadow-v2-model-contract.md)
- [Docker Local Runbook](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/docker-local.md)
- [Local Runbook](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/runbook-local.md)
- [API Summary](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
- [Workflow](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/workflow.md)
