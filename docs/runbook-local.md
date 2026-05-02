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

The normal local workflow now has two root `.cmd` entrypoints.

Rebuild Triton after changing files under `ops/triton/model_repository`:

```cmd
rebuild-triton.cmd
```

Start the full service:

```cmd
start-service.cmd
```

`start-service.cmd` opens a visible Windows console, starts the prebuilt Triton container when needed, waits until `shadowgen_detector` and `shadowgen_segmenter` are ready, sets the ML-core Triton environment, and starts FastAPI with `RELOAD=0` by default. That makes the `/playground` shutdown button stop the visible FastAPI process instead of having `uvicorn --reload` immediately spawn it again.

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/playground`

## Execution Model

The runtime is now execution-aware.

## Working Canvas Margins

After detection, the selected crop is placed into the canonical working canvas. The default margin policy is intentionally airy for shadow generation:

- `SHADOWGEN_WORKING_SIZE`
  - canonical downstream canvas size
  - default `512`
- `SHADOWGEN_WORKING_CONTENT_SCALE`
  - fraction of the canonical canvas occupied by the detected crop after resize
  - default `0.68`
  - lower values leave more margin for generated shadows
- `preprocess.padding_px`
  - public-request crop context before canonical resize
  - default `100`

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
- dev-only shutdown button for the current ML-service process

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
- Current default: disabled, because the pipeline does not consume geometry prediction yet
- Env vars:
  - `SHADOWGEN_GEOMETRY_ENABLED` (default `false`)
  - `SHADOWGEN_GEOCALIB_WEIGHTS`
  - `SHADOWGEN_GEOCALIB_CAMERA_MODEL`
  - `SHADOWGEN_GEOCALIB_SHARED_INTRINSICS`
  - `SHADOWGEN_GEOMETRY_BACKEND_KIND`

When `SHADOWGEN_GEOMETRY_ENABLED=false`, the public pipeline does not run GeoCalib and records `geometry_ms=0`. The playground hides the Geometry card by default; the old debug endpoint still returns a skipped stage if called directly.

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
- Default local model: `ZhengPeng7/BiRefNet-matting`
- Previous fast default was `ZhengPeng7/BiRefNet_lite-matting`; use it only when speed matters more than edge quality
- Higher-quality experimental override: `ZhengPeng7/BiRefNet_HR-matting`
- First live Triton targets: `shadowgen_detector`, `shadowgen_segmenter`
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
- [ops/triton/model_repository/shadowgen_detector/config.pbtxt](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/model_repository/shadowgen_detector/config.pbtxt)
- [ops/triton/model_repository/shadowgen_detector/1/model.py](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/model_repository/shadowgen_detector/1/model.py)
- [ops/triton/Dockerfile.segmenter-python](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/ops/triton/Dockerfile.segmenter-python)
- [rebuild-triton.cmd](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/rebuild-triton.cmd)
- [start-service.cmd](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/start-service.cmd)

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
- `shadowgen_detector` now uses a temporary Triton Python backend around GroundingDINO
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
  - `local from-depth-v2`
  - `triton stable-normal`
- Current default: `local from-depth-v2`
- Env vars:
  - `SHADOWGEN_STABLE_NORMAL_VARIANT`
  - `SHADOWGEN_STABLE_NORMAL_RESOLUTION`
  - `SHADOWGEN_STABLE_NORMAL_ALLOW_CPU`
  - `SHADOWGEN_NORMALS_BACKEND_KIND`
  - `SHADOWGEN_NORMALS_MODEL_VARIANT` (default `from-depth-v2`; set `stable-normal` to opt into the neural local backend)
  - `SHADOWGEN_TRITON_NORMALS_MODEL`
  - `SHADOWGEN_TARGET_DEVICE`

Performance note:

- `from-depth-v2` is the default because it is deterministic and cheap enough for the current shadow inputs
- the previous depth-derived normal implementation used OpenCV inpainting and could take hundreds of milliseconds at `512x512`
- current `from-depth-v2` uses mask-normalized smoothing before gradients and is typically in the tens of milliseconds on the same input

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

- `V1-GAN` is the current controllable local model for top-view/rotated shadows
- `V2-DIFF` is currently control-free and ignores `angle`, `elevation`, `softness`, and `reflection`
- real controlled shadow backends must not use post-blur for softness; coarse blur remains mock-only

#### `V2-DIFF`

- working local diffusion backend, plus preferred future Triton slot
- currently simplified to a control-free diffusion model
- expected to draw a plausible full shadow image from the object image and mask
- local bundle default:
  - `.models/shadow/v2-diff/shadowgen_inpaint_lora_prod_current`
- neutral background default:
  - `.models/shadow/v2-diff/mean_background.png`
- env vars:
  - `SHADOWGEN_SHADOW_BACKEND_KIND`
  - `SHADOWGEN_SHADOW_MODEL_VARIANT`
  - `SHADOWGEN_SHADOW_V2_DIFF_BUNDLE_PATH`
  - `SHADOWGEN_SHADOW_V2_DIFF_BACKGROUND_PATH`
  - `SHADOWGEN_SHADOW_V2_DIFF_SEED`
  - `SHADOWGEN_SHADOW_V2_DIFF_FAST_LCM` (default `true`; fuses the Shadow LoRA with the official SD1.5 LCM-LoRA)
  - `SHADOWGEN_SHADOW_V2_DIFF_STEPS` (default comes from the active bundle mode; current fast LCM bundle default is `5`)
  - `SHADOWGEN_SHADOW_V2_DIFF_GUIDANCE_SCALE` (default comes from the active bundle mode; current fast LCM bundle default is `1.0`)
  - `SHADOWGEN_SHADOW_V2_DIFF_COMPILE_ENABLED` (default `false`; opt-in `torch.compile` for long-running local services)
  - `SHADOWGEN_SHADOW_V2_DIFF_COMPILE_MODE` (default `reduce-overhead`)
  - `SHADOWGEN_SHADOW_V2_DIFF_COMPILE_BACKEND` (optional)
  - `SHADOWGEN_TRITON_SHADOW_V2_MODEL`

Performance note:

- the local diffusion backend spends most runtime in UNet denoising
- on the local probe image, cached `V2-DIFF` inference measured about `2.65s` at `24` steps and about `1.43s` at `12` steps
- the current accelerated handoff bundle recommends fast LCM mode with `5` steps and `guidance_scale=1.0`
- local smoke with the resident pipeline measured about `13s` for the first call including pipeline load/fuse, then about `0.6s` for the next cached call
- normal quality mode is still available with `SHADOWGEN_SHADOW_V2_DIFF_FAST_LCM=false`
- `torch.compile` is intentionally disabled by default because the first compile pass can be much slower than a normal request

Current expected model inputs:

- `img`
- `mask`

Future controlled inputs may add `depth`, `normal`, `angle`, `elevation`, `softness`, and `reflection`. The ML core adapter sends only tensors declared by the active Triton binding.

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

1. Rebuild the Triton image after changing Triton model/backend code:

```cmd
rebuild-triton.cmd
```

This script:

- checks that Docker is reachable
- removes the old `shadowgen-triton-segmenter` container if it exists
- builds `shadowgen-triton-segmenter:py`
- bakes `ops/triton/model_repository` into `/models`

2. Start the service:

```cmd
start-service.cmd
```

This script:

- opens a visible Windows console window
- starts the prebuilt Triton container if it is not already running
- waits until `shadowgen_detector` and `shadowgen_segmenter` are ready
- defaults to `TRITON_GPU=1`, `TRITON_DEVICE=cuda:0`, `TRITON_RESOLUTION=512`
- sets `SHADOWGEN_TRITON_SEGMENTER_COMPILE_ENABLED=false` to avoid a long first-request `torch.compile` pause in the playground
- sets `SHADOWGEN_TRITON_URL=http://127.0.0.1:8010`
- sets `SHADOWGEN_DETECTOR_BACKEND_KIND=triton`
- sets `SHADOWGEN_SEGMENTER_BACKEND_KIND=triton`
- starts FastAPI on `http://127.0.0.1:8000/playground`
- starts FastAPI with `RELOAD=0` by default so the Playground shutdown button stops the actual process

Useful environment overrides:

```cmd
set PORT=8003
set TRITON_DEVICE=cuda:0
set TRITON_RESOLUTION=512
start-service.cmd
```

If `TRITON_GPU=1` fails with an NVIDIA runtime error, Docker Desktop is running but Docker cannot expose the NVIDIA runtime to containers yet. Check:

- Docker Desktop is using the WSL2 Linux backend
- NVIDIA driver supports WSL CUDA
- `nvidia-smi` works on the Windows host
- `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi` works

The temporary Python backend model config uses `KIND_CPU` intentionally. The Python model still moves tensors to CUDA when the container has GPU access, but Triton can also load the model for local bring-up without GPU container runtime.

3. Check the Triton model readiness manually if needed:

```powershell
.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010 --wait-seconds 240
.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010 shadowgen_detector --wait-seconds 240
```

4. Run a live smoke check. This performs:

- direct `TritonDetector` inference
- direct `TritonSegmenter` inference
- full `/v1/render` with one Triton stage active and other heavy stages set to `mock`
- artifact export to `artifacts/triton-smoke/`

```powershell
.venv\Scripts\python.exe tools\smoke_triton_detector.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
```

5. Check ML-core capabilities:

```powershell
curl http://127.0.0.1:8000/v1/capabilities
```

Expected detector signals:

- `backend_kind = triton`
- `model_name = shadowgen_detector`
- `available = true`
- `endpoint` is filled

Expected segmenter signals:

- `backend_kind = triton`
- `model_name = shadowgen_segmenter`
- `available = true`
- `endpoint` is filled

6. In `/playground`, rerun `Detection` and `Segmentation` with `backend_kind = triton`.

Expected stage metadata:

- `actual_backend_kind = triton`
- `device = triton`
- `endpoint` is filled

Operational note:

- the current Python backend is a bridge while ONNX export is blocked; the target production path remains ONNX first, then TensorRT
- if `TRITON_GPU=1` fails with `Auto-detected mode as 'legacy'`, the service can still run CPU smoke checks with `TRITON_GPU=0`, but Docker Desktop/NVIDIA runtime must be fixed before GPU Triton inference
- CPU Triton smoke is intentionally slow; on this workstation a 512px segmenter call was roughly tens of seconds
- GPU Triton smoke with the temporary Python backend measured about `1.2s` for segmentation after warm-up; it is much faster than CPU but still slower than the local in-process backend until we move to ONNX/TensorRT

### `V2-DIFF` is unavailable

That is expected right now.
The runtime slot and Triton adapter scaffold exist by design, but the actual model backend is not yet connected.

## Prepare Shadow V2 Sample Pack

To create a small handoff pack for model developers:

```powershell
.venv\Scripts\python.exe tools\prepare_shadow_v2_sample_pack.py --count 10 --backend-kind local --normal-variant from-depth-v2 --output-dir artifacts\shadow-v2-sample-pack
```

The helper downloads curated external object photos, runs the pipeline through detection, segmentation, foreground refinement, depth, and normals, then writes contract-ready files under `artifacts\shadow-v2-sample-pack`.

Generated outputs are ignored by git. Share the folder separately when handing samples to the model team.

## Related Docs

- [README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/README.md)
- [architecture.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/architecture.md)
- [modules.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/modules.md)
- [api.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
