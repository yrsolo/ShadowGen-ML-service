# Local Runbook

## Goal

This runbook describes how to prepare a local environment, start the service, use the playground, and understand the current model bring-up expectations.

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

## Playground Use

The playground supports:

- image upload
- contract parameter control
- full pipeline run
- per-stage rerun
- preview scaling
- explicit backend switching

Current shadow stage variants in the UI:

- `mock`
- `V1-GAN`
- `V2-DIFF`

Current behavior:

- `V1-GAN` is the working migrated legacy shadow model
- `V2-DIFF` is present as a scaffold and intentionally reports itself as not connected yet

## Model Bring-Up Summary

### Geometry

- Backend: GeoCalib
- Env vars:
  - `SHADOWGEN_GEOCALIB_WEIGHTS`
  - `SHADOWGEN_GEOCALIB_CAMERA_MODEL`
  - `SHADOWGEN_GEOCALIB_SHARED_INTRINSICS`

### Detection

- Backend: GroundingDINO
- Env vars:
  - `SHADOWGEN_GROUNDING_DINO_MODEL_ID`
  - `SHADOWGEN_GROUNDING_DINO_PROMPT`
  - `SHADOWGEN_GROUNDING_DINO_BOX_THRESHOLD`
  - `SHADOWGEN_GROUNDING_DINO_TEXT_THRESHOLD`

### Segmentation

- Backend: BiRefNet
- Env vars:
  - `SHADOWGEN_BIREFNET_MODEL_ID`
  - `SHADOWGEN_BIREFNET_RESOLUTION`
  - `SHADOWGEN_BIREFNET_MASK_THRESHOLD`
  - `SHADOWGEN_BIREFNET_ALLOW_CPU`

### Foreground Refinement

- Backend: Fast Foreground Colour Estimation
- Dependency note:
  - OpenCV-based stage

### Depth

- Backend: Depth Anything V2 Small
- Env vars:
  - `SHADOWGEN_DEPTH_ANYTHING_MODEL_ID`
  - `SHADOWGEN_TARGET_DEVICE`

### Normals

- Primary backend: StableNormal
- Fallback backend: `from-depth`
- Env vars:
  - `SHADOWGEN_STABLE_NORMAL_VARIANT`
  - `SHADOWGEN_STABLE_NORMAL_RESOLUTION`
  - `SHADOWGEN_STABLE_NORMAL_ALLOW_CPU`
  - `SHADOWGEN_TARGET_DEVICE`

### Shadow

- Variants:
  - `mock`
  - `V1-GAN`
  - `V2-DIFF`

#### `mock`

- deterministic analytical fallback
- still uses coarse blur for softness emulation

#### `V1-GAN`

- migrated legacy pix2pix generator
- local weights path:
  - `.models/shadow/AveragedModel.pth`
- env vars:
  - `SHADOWGEN_SHADOW_MODEL_VARIANT`
  - `SHADOWGEN_SHADOW_PIX2PIX_WEIGHTS_PATH`
  - `SHADOWGEN_TARGET_DEVICE`

Important:

- real shadow backends receive `softness` as model input
- real shadow backends do not use post-blur for softness anymore

#### `V2-DIFF`

- scaffold exists in code
- runtime slot exists
- backend intentionally not implemented yet

Expected future model inputs:

- `img`
- `mask`
- `depth`
- `normal`
- `angle`
- `elevation`
- `softness`
- `reflection`

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
- `degraded`
- `components[]`

For model bring-up, the most useful fields are:

- `name`
- `implementation`
- `model_name`
- `model_version`
- `using_mock`
- `detail`

## Troubleshooting

### Service uses CPU unexpectedly

Check:

- CUDA torch is installed in the same `.venv`
- service was started from that `.venv`
- stage `details.device` in the playground

### Stage shows `mock-fallback`

That means:

- the stage exists
- requested backend was not usable at runtime
- the service fell back to the mock or deterministic path

Inspect:

- `/v1/capabilities`
- stage details in playground
- startup logs

### `V2-DIFF` is unavailable

That is expected right now.
The scaffold is present by design, but the model backend is not implemented yet.

## Related Docs

- [README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/README.md)
- [architecture.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/architecture.md)
- [modules.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/modules.md)
- [api.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
