# Local Runbook

## Goal

This runbook describes how to prepare a local environment, start the service, use the playground, and understand the current Triton-ready execution model.

## Recommended Environment

- OS: Windows workstation
- Python: `3.11`
- Runtime target: NVIDIA GPU
- Package manager: `pip` inside `.venv`

## Local Setup

### 1. Create `.venv`

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

### 2. Install base project

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
```

### 3. Install CUDA PyTorch

Install CUDA-enabled torch explicitly before ML extras:

```powershell
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

If your machine requires a different CUDA channel, replace `cu126` with the matching one.

### 4. Install ML stack

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev,ml]
```

## Verify GPU

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Healthy result:

- non-CPU torch build
- `True` for CUDA availability
- real NVIDIA device name

## Start the Service

### Uvicorn

```powershell
.venv\Scripts\python.exe -m uvicorn shadowgen_ml_service.main:app --reload
```

### Windows shortcut

```cmd
run-service.cmd
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/playground`

## Execution Model

The runtime is now execution-aware.

For heavy stages, separate two dimensions:

- `backend_kind`
  - `mock`
  - `local`
  - `triton`
- `model_variant`
  - stage-specific, for example `grounding-dino`, `birefnet`, `stable-normal`, `v1-gan`, `v2-diff`

The playground and capability responses expose:

- requested backend kind
- actual backend kind
- model variant
- device
- endpoint
- fallback reason

## Playground Use

The playground supports:

- image upload
- contract parameter control
- full pipeline run
- per-stage rerun
- preview scaling
- explicit backend switching
- explicit shadow model variant switching
- horizontal stage navigation with the mouse wheel
- vertical scrolling inside a stage card with `Shift` + mouse wheel

Current shadow stage variants in the UI:

- `mock`
- `V1-GAN`
- `V2-DIFF`

Current execution backend choices for heavy stages:

- `mock`
- `local`
- `triton`

## Model Bring-Up Summary

### Geometry

- Backends: `mock`, `local`
- Local backend: GeoCalib
- Env vars:
  - `SHADOWGEN_GEOCALIB_WEIGHTS`
  - `SHADOWGEN_GEOCALIB_CAMERA_MODEL`
  - `SHADOWGEN_GEOCALIB_SHARED_INTRINSICS`
  - `SHADOWGEN_GEOMETRY_BACKEND_KIND`

### Detection

- Backends: `mock`, `local`, `triton`
- Local backend: GroundingDINO
- Env vars:
  - `SHADOWGEN_GROUNDING_DINO_MODEL_ID`
  - `SHADOWGEN_GROUNDING_DINO_PROMPT`
  - `SHADOWGEN_GROUNDING_DINO_BOX_THRESHOLD`
  - `SHADOWGEN_GROUNDING_DINO_TEXT_THRESHOLD`
  - `SHADOWGEN_DETECTOR_BACKEND_KIND`
  - `SHADOWGEN_TRITON_DETECTOR_MODEL`

### Segmentation

- Backends: `mock`, `local`, `triton`
- Local backend: BiRefNet
- First live Triton target: `shadowgen_segmenter`
- Current live Triton packaging: temporary `python` backend
- Long-term production Triton format: `ONNX`
- Env vars:
  - `SHADOWGEN_BIREFNET_MODEL_ID`
  - `SHADOWGEN_BIREFNET_RESOLUTION`
  - `SHADOWGEN_BIREFNET_MASK_THRESHOLD`
  - `SHADOWGEN_BIREFNET_ALLOW_CPU`
  - `SHADOWGEN_BIREFNET_COMPILE_ENABLED`
  - `SHADOWGEN_BIREFNET_COMPILE_MODE`
  - `SHADOWGEN_BIREFNET_COMPILE_BACKEND`
  - `SHADOWGEN_BIREFNET_MATMUL_PRECISION`
  - `SHADOWGEN_SEGMENTER_BACKEND_KIND`
  - `SHADOWGEN_TRITON_SEGMENTER_MODEL`

#### Triton Python backend and model repository

Tracked repository scaffold:

- [ops/triton/model_repository/README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/model_repository/README.md)
- [ops/triton/model_repository/shadowgen_segmenter/config.pbtxt](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/model_repository/shadowgen_segmenter/config.pbtxt)
- [ops/triton/model_repository/shadowgen_segmenter/1/model.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/model_repository/shadowgen_segmenter/1/model.py)
- [ops/triton/Dockerfile.segmenter-python](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/Dockerfile.segmenter-python)
- [tools/run_triton_segmenter_python.cmd](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/tools/run_triton_segmenter_python.cmd)
- [tools/run_triton_segmenter_python.ps1](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/tools/run_triton_segmenter_python.ps1)

Current live contract:

- input `image`: `FP32`, batched `NCHW`, normalized `0..1`
- output `mask`: `FP32`, batched `NCHW`

Postprocess kept inside ML-core:

- `cutout`
- `crop`
- compatibility `bbox`

Current blocker note:

- in the current environment BiRefNet ONNX export is blocked by `torchvision::deform_conv2d`
- the export tool now tries both the modern and legacy ONNX exporters and reports this blocker explicitly
- `shadowgen_segmenter` now uses a temporary Triton Python backend so we can run a live Triton stage without replacing the model
- `torch.compile` is exposed as an opt-in acceleration path for both the local backend and the Triton Python backend

### Foreground Refinement

- Backends: `mock`, `local`
- Local backend: Fast Foreground Colour Estimation
- Dependency note:
  - OpenCV-based stage
- Env vars:
  - `SHADOWGEN_FOREGROUND_REFINER_BACKEND_KIND`

### Depth

- Backends: `mock`, `local`, `triton`
- Local backend: Depth Anything V2 Small
- Env vars:
  - `SHADOWGEN_DEPTH_ANYTHING_MODEL_ID`
  - `SHADOWGEN_DEPTH_BACKEND_KIND`
  - `SHADOWGEN_TRITON_DEPTH_MODEL`
  - `SHADOWGEN_TARGET_DEVICE`

### Normals

- Backends:
  - `mock`
  - `local stable-normal`
  - `local from-depth-v2` fallback
  - `triton stable-normal`
- Env vars:
  - `SHADOWGEN_STABLE_NORMAL_VARIANT`
  - `SHADOWGEN_STABLE_NORMAL_RESOLUTION`
  - `SHADOWGEN_STABLE_NORMAL_ALLOW_CPU`
  - `SHADOWGEN_NORMALS_BACKEND_KIND`
  - `SHADOWGEN_TRITON_NORMALS_MODEL`
  - `SHADOWGEN_TARGET_DEVICE`

### Shadow

- Variants:
  - `mock`
  - `V1-GAN`
  - `V2-DIFF`

#### `mock`

- deterministic analytical fallback
- keeps the coarse blur softness behavior

#### `V1-GAN`

- current working local shadow backend
- local weights path:
  - `.models/shadow/AveragedModel.pth`
- env vars:
  - `SHADOWGEN_SHADOW_BACKEND_KIND`
  - `SHADOWGEN_SHADOW_MODEL_VARIANT`
  - `SHADOWGEN_SHADOW_PIX2PIX_WEIGHTS_PATH`
  - `SHADOWGEN_TARGET_DEVICE`

Important:

- real shadow backends receive `softness` as model input
- real shadow backends do not use post-blur for softness

#### `V2-DIFF`

- preferred Triton-ready slot
- scaffolded intentionally
- not implemented as a working model yet
- env vars:
  - `SHADOWGEN_SHADOW_BACKEND_KIND`
  - `SHADOWGEN_SHADOW_MODEL_VARIANT`
  - `SHADOWGEN_TRITON_SHADOW_V2_MODEL`

Expected model inputs:

- `img`
- `mask`
- `depth`
- `normal`
- `angle`
- `elevation`
- `softness`
- `reflection`

## Triton Settings

Global Triton-related settings:

- `SHADOWGEN_EXECUTION_DEFAULT_BACKEND`
- `SHADOWGEN_TRITON_URL`
- `SHADOWGEN_TRITON_PROTOCOL`
- `SHADOWGEN_TRITON_TIMEOUT_MS`
- `SHADOWGEN_TRITON_MODEL_REPOSITORY`

Async settings:

- `SHADOWGEN_ASYNC_ENABLED`
- `SHADOWGEN_ASYNC_BACKEND`
- `SHADOWGEN_JOB_MAX_RUNNING`
- `SHADOWGEN_JOB_MAX_PENDING`
- `SHADOWGEN_JOB_ACCEPTING_ENABLED`
- `SHADOWGEN_JOB_CANCEL_MODE`
- `SHADOWGEN_BATCHING_ENABLED`
- `SHADOWGEN_BATCH_WINDOW_MS`
- `SHADOWGEN_BATCH_MAX_SIZE`
- `SHADOWGEN_BATCH_SEGMENTER_ENABLED`
- `SHADOWGEN_BATCH_DEPTH_ENABLED`
- `SHADOWGEN_BATCH_NORMALS_ENABLED`
- `SHADOWGEN_BATCH_SHADOW_ENABLED`

Per-stage execution defaults:

- `SHADOWGEN_DETECTOR_BACKEND_KIND`
- `SHADOWGEN_SEGMENTER_BACKEND_KIND`
- `SHADOWGEN_DEPTH_BACKEND_KIND`
- `SHADOWGEN_NORMALS_BACKEND_KIND`
- `SHADOWGEN_SHADOW_BACKEND_KIND`

Global behavior:

- if `SHADOWGEN_EXECUTION_DEFAULT_BACKEND=mock`, heavy stages default to mock
- otherwise the service defaults to `local` unless a stage override is set
- requesting `triton` without a reachable endpoint or registered model results in explicit fallback metadata

## Async API Use

The service now provides async render endpoints:

- `POST /v1/render/jobs`
- `GET /v1/render/jobs/{job_id}`
- `DELETE /v1/render/jobs/{job_id}`

Current async backend:

- in-process, in-memory

This is meant for architectural separation and API stability. It is replaceable later with Redis/RQ, Celery, or another worker backend.

## Local Storage Rules

Ignored local-only paths include:

- `.models/`
- `.cache/`
- `var/cache/`
- `var/tmp/`
- `artifacts/`

Use them for:

- model checkpoints
- downloaded weights
- preprocess cache
- temporary files
- generated outputs

## Health Checks

### Service health

```powershell
curl http://127.0.0.1:8000/health
```

### Runtime capabilities

```powershell
curl http://127.0.0.1:8000/v1/capabilities
```

Check:

- `active_backend_mode`
- `execution_default_backend`
- `async_enabled`
- `supported_submit_modes`
- `preferred_submit_mode`
- `job_execution`
- `batching_strategy`
- `components[]`

For model bring-up, the most useful fields are:

- `implementation`
- `backend_kind`
- `model_variant`
- `model_name`
- `model_version`
- `device`
- `endpoint`
- `fallback_reason`
- `backends`

## Troubleshooting

### Service uses CPU unexpectedly

Check:

- CUDA torch is installed in the same `.venv`
- service was started from that `.venv`
- stage `device` in the playground
- capabilities for that stage show `backend_kind=local` and a CUDA-capable device

### Requested Triton but got fallback

That means:

- the stage is Triton-aware
- the runtime could not use the requested Triton backend
- the selector fell back to `local` or `mock`

Inspect:

- `fallback_reason`
- `endpoint`
- `/v1/capabilities`
- startup logs

### How to verify the live Triton segmenter

The container is built locally from:

- [ops/triton/Dockerfile.segmenter-python](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/Dockerfile.segmenter-python)

The model repository copied into the Triton image is:

- [ops/triton/model_repository](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/model_repository)

The helper bakes the model repository into `/models` by default. This avoids Windows bind-mount problems for workspaces on drives such as `N:\`. Use `-BindModelRepository` only when Docker Desktop can reliably mount the workspace path.

Triton uses standard ports inside the container, but the helper maps them to offset host ports so FastAPI can keep using `8000` locally.

Default host ports:

- `8010`: Triton HTTP API, used by ML-core
- `8011`: Triton gRPC API
- `8012`: Triton metrics

Container ports:

- `8000`: Triton HTTP API
- `8001`: Triton gRPC API
- `8002`: Triton metrics

1. Build and run the Triton image:

```powershell
tools\run_triton_segmenter_python.cmd
```

Run in the background:

```powershell
tools\run_triton_segmenter_python.cmd -Detach
```

Detached containers are intentionally not started with `--rm`, so startup failures keep their logs:

```powershell
docker logs shadowgen-triton-segmenter
docker rm -f shadowgen-triton-segmenter
```

If PowerShell execution policy is already configured, the direct PowerShell script is also valid:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\run_triton_segmenter_python.ps1
```

This helper script:

- checks that Docker CLI is available
- checks that the Docker daemon is reachable
- builds the custom Triton image
- starts Triton without Docker GPU flags by default so the container and HTTP wiring can be verified first
- starts Triton with `--gpus all` when `-Gpu` is provided
- publishes HTTP/gRPC/metrics ports
- serves the image-baked model repository from `/models`

GPU mode:

```powershell
tools\run_triton_segmenter_python.cmd -Gpu -Detach
```

If `-Gpu` fails with an NVIDIA runtime error, Docker Desktop is running but Docker cannot expose the NVIDIA runtime to containers yet. Check:

- Docker Desktop is using the WSL2 Linux backend
- NVIDIA driver supports WSL CUDA
- `nvidia-smi` works on the Windows host
- `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi` works

The temporary Python backend model config uses `KIND_CPU` intentionally. The Python model still moves tensors to CUDA when the container has GPU access, but Triton can also load the model for local bring-up without GPU container runtime.

2. Check the Triton model readiness:

```powershell
.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010
```

3. Point ML-core at Triton:

```powershell
$env:SHADOWGEN_TRITON_URL="http://127.0.0.1:8010"
$env:SHADOWGEN_SEGMENTER_BACKEND_KIND="triton"
$env:SHADOWGEN_BIREFNET_COMPILE_ENABLED="true"
```

4. Start the service and check:

```powershell
curl http://127.0.0.1:8000/v1/capabilities
```

Expected segmenter signals:

- `backend_kind = triton`
- `model_name = shadowgen_segmenter`
- `available = true`
- `endpoint` is filled

5. In `/playground`, rerun `Segmentation` with `backend_kind = triton`.

Expected stage metadata:

- `actual_backend_kind = triton`
- `device = triton`
- `endpoint` is filled

Operational note:

- on the current workstation, a missing Docker Desktop WSL distro `docker-desktop` blocks live Triton container startup until Docker Desktop is repaired

### `V2-DIFF` is unavailable

That is expected right now.
The runtime slot and Triton adapter scaffold exist by design, but the actual model backend is not yet connected.

## Related Docs

- [README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/README.md)
- [architecture.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/architecture.md)
- [modules.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/modules.md)
- [api.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
