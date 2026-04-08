# Local Runbook

## Target shape

- Local-first development
- Synchronous HTTP API
- NVIDIA GPU target for future real-model bring-up

## Basic setup

1. Create a compatible Python environment.
2. Install base dependencies and dev dependencies.
3. Start the API server with Uvicorn.
4. Run tests before commit.

### Commands

```powershell
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m uvicorn shadowgen_ml_service.main:app --reload
.venv/Scripts/python.exe -m pytest
```

## Recreate `.venv` with CUDA

`pyproject.toml` now describes the project itself plus the non-torch ML stack.
CUDA-enabled `torch` is installed separately on purpose, because pip cannot infer the correct NVIDIA wheel from `pyproject.toml` alone.

Recommended Windows / PowerShell flow from the repository root:

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
.venv\Scripts\python.exe -m pip install -e .[dev,ml]
```

If your NVIDIA driver / CUDA runtime needs a different PyTorch channel, replace `cu126` with the matching one from the official PyTorch install matrix.

Quick verification:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Expected result on a healthy GPU environment:

- `torch` version without the `+cpu` suffix
- `True` for `torch.cuda.is_available()`
- a real NVIDIA device name

## Browser playground

After the server starts, open:

- `http://127.0.0.1:8000/playground`

The playground supports:

- uploading one source image
- running the full pipeline or a single stage
- switching each stage between `mock` and `real`
- showing stage-local errors and previews
- adjusting the global preview size
- comparing raw segmentation cutouts against the post-matting foreground colour refinement stage

### Optional ML bootstrap

```powershell
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
.venv/Scripts/python.exe -m pip install -e .[ml]
```

If direct GitHub install is blocked, a local fallback also works:

```powershell
Invoke-WebRequest https://github.com/cvg/GeoCalib/archive/refs/heads/main.zip -OutFile var/tmp/geocalib-main.zip
Expand-Archive var/tmp/geocalib-main.zip -DestinationPath var/tmp/geocalib-src -Force
.venv/Scripts/python.exe -m pip install -e var/tmp/geocalib-src/GeoCalib-main
```

## GeoCalib bring-up

The real `Geometry` step uses GeoCalib from the active `.venv`.

Expected workflow:

1. Install CUDA `torch` / `torchvision` into the current virtual environment.
2. Install `-e .[ml]` so GeoCalib and the supporting ML packages land in the same `.venv`.
3. Keep heavy weights and artifacts out of git in ignored folders such as `.models/`.
4. Start the service and inspect `GET /v1/capabilities`.

Useful env vars:

- `SHADOWGEN_MODEL_CACHE_DIR` for local model storage
- `TORCH_HOME` if you want GeoCalib weights to download into a custom cache directory instead of the default torch cache
- `SHADOWGEN_GEOCALIB_WEIGHTS` to switch GeoCalib weights (`pinhole`, `distorted`, or a checkpoint path)
- `SHADOWGEN_GEOCALIB_CAMERA_MODEL` to choose the camera model passed into calibration (`pinhole`, `simple_radial`, `simple_divisional`)
- `SHADOWGEN_GEOCALIB_SHARED_INTRINSICS` to toggle GeoCalib shared-intrinsics optimization

GeoCalib downloads its released weights on first initialization. That is expected and should happen only once per cache location.

What to verify:

- `components[].name == "geometry_estimator"`
- `implementation == "real"`
- `using_mock == false`

In the playground `Geometry` card:

- `geometry_overlay` now includes a synthetic floor grid to make the estimated horizon and perspective easier to judge visually
- stage details also show `camera_model`, `weights`, and `shared_intrinsics` for the active GeoCalib run

If GeoCalib is unavailable or fails to initialize, the service falls back to the mock geometry estimator and the playground shows `mock-fallback`.

## GroundingDINO bring-up

The real `Detection` step uses `IDEA-Research/grounding-dino-base` through `transformers`.

Expected workflow:

1. Install CUDA `torch` / `torchvision` into the active virtual environment.
2. Install `-e .[ml]` so `transformers` and its supporting packages land in the same `.venv`.
3. Start the service and let the model download into the local Hugging Face / torch cache on first initialization.
4. Inspect `GET /v1/capabilities`.

Useful env vars:

- `SHADOWGEN_GROUNDING_DINO_MODEL_ID` to switch the detector checkpoint
- `SHADOWGEN_GROUNDING_DINO_PROMPT` to override the zero-shot prompt, default `object.`
- `SHADOWGEN_GROUNDING_DINO_BOX_THRESHOLD` to change the box confidence threshold
- `SHADOWGEN_GROUNDING_DINO_TEXT_THRESHOLD` to change the text threshold used by post-processing
- `SHADOWGEN_WORKING_CONTENT_SCALE` to control how much of the square working canvas the detected crop may occupy; lower values leave more outer margin for shadow projection

What to verify:

- `components[].name == "detector"`
- `implementation == "real"`
- `using_mock == false`

In the playground `Detection` card:

- `detection_overlay` shows the selected bbox on the source image
- `crop_for_resize` shows the working crop that will go to segmentation, including the new outer margin reserved for shadows
- stage details expose bbox coordinates, confidence, backend mode, and prompt

If GroundingDINO is unavailable or fails to initialize, the service falls back to the mock detector and the playground shows `mock-fallback`.

## BiRefNet bring-up

The real `Segmentation` step uses `ZhengPeng7/BiRefNet_lite-matting`.

Useful env vars:

- `SHADOWGEN_BIREFNET_MODEL_ID` to switch the segmenter checkpoint
- `SHADOWGEN_BIREFNET_RESOLUTION` to control the internal BiRefNet inference size
- `SHADOWGEN_BIREFNET_MASK_THRESHOLD` to control matte-to-mask binarization
- `SHADOWGEN_BIREFNET_ALLOW_CPU=true` to opt into CPU execution

Default runtime behavior:

- if CUDA is available, `auto` mode can activate the real segmenter
- if CUDA is not available, the service stays on mock segmentation by default
- CPU execution is possible only with explicit opt-in because BiRefNet is too slow for the normal interactive path on this machine

What to verify:

- `components[].name == "segmenter"`
- `implementation == "real"` only when CUDA is available or CPU mode was explicitly enabled
- `using_mock == false` when the real backend is active

In the playground `Segmentation` card:

- `working_crop` shows the padded square crop passed into segmentation
- `mask` shows the binary foreground mask
- `cutout` shows the RGBA cutout after matting
- stage details expose mask size, bbox, and backend mode

## Foreground colour refinement bring-up

The dedicated `Foreground` step runs after segmentation and before depth/composition.

It uses the Fast Foreground Colour Estimation method from:

- `https://github.com/Photoroom/fast-foreground-estimation`

Why this exists:

- segmentation and matting are responsible for alpha
- foreground colour estimation is responsible for fixing the RGB values of semi-transparent edge pixels
- that separation keeps the architecture cleaner and lets us swap or benchmark this refinement stage independently from the segmenter

Runtime behavior:

- the real backend is active when `cv2` is installed
- the fallback backend is a passthrough refiner that keeps the alpha channel but skips colour correction
- the corrected cutout is what downstream depth/shadow/composition stages receive and what the preprocess cache stores

What to verify:

- `components[].name == "foreground_refiner"`
- `implementation == "real"` when OpenCV is available
- `using_mock == false` when the Fast Foreground Colour Estimation backend is active

In the playground `Foreground` card:

- `segmenter_cutout` shows the raw cutout coming from the segmentation stage
- `foreground_cutout` shows the corrected cutout after foreground colour estimation
- stage details expose the active backend mode

## Depth Anything bring-up

The real `Depth` step uses `depth-anything/Depth-Anything-V2-Small-hf`.

Useful env vars:

- `SHADOWGEN_DEPTH_ANYTHING_MODEL_ID` to switch the depth checkpoint
- `SHADOWGEN_TARGET_DEVICE` to control whether the service targets `cuda`, `cuda:0`, or `cpu`

What to verify:

- `components[].name == "depth_estimator"`
- `implementation == "real"` when Depth Anything initializes successfully
- `using_mock == false` when the real backend is active

In the playground `Depth` card:

- `depth` shows the normalized monocular depth map
- `working_cutout` shows the refined foreground cutout that was sent into the model
- stage details expose backend mode, output size, and execution device

If Depth Anything is unavailable or fails to initialize, the service falls back to the deterministic mock depth estimator and the playground shows `mock-fallback`.

## Normals bring-up

The `Normals` stage is now treated as its own runtime module.

Runtime behavior:

- `real` mode first tries the neural `Stable-X/StableNormal` backend
- if the neural backend is unavailable or fails to initialize, the stage falls back to the deterministic `from-depth` backend instead of disappearing
- `mock` mode returns a flat neutral normal map

Useful env vars:

- `SHADOWGEN_STABLE_NORMAL_VARIANT` to switch the StableNormal hub entry, default `StableNormal_turbo`
- `SHADOWGEN_STABLE_NORMAL_RESOLUTION` to control StableNormal inference resolution
- `SHADOWGEN_STABLE_NORMAL_ALLOW_CPU=true` to opt into CPU execution when CUDA is unavailable
- `SHADOWGEN_TARGET_DEVICE` to control the preferred device such as `cuda` or `cuda:0`

What to verify:

- `components[].name == "normal_estimator"`
- `implementation == "real"` in normal runtime mode
- `model_name == "Stable-X/StableNormal"` when the neural backend is active
- `model_name == "normal-map-from-depth"` only when the stage is running on the explicit depth-derived fallback
- switching the playground card to `mock` changes the stage backend and output

In the playground `Normals` card:

- `normals` shows the RGB normal map
- stage details expose backend kind, variant, output size, and execution device

## Legacy pix2pix shadow bring-up

The `Shadow` stage can now use the legacy pix2pix generator migrated from:

- `N:\PROJECTS\python\STUDY\ShadowGEN\SHADOW\pix2pix.py`

What was intentionally imported:

- generator inference architecture
- azimuth / `rot` conditioning logic
- one generator checkpoint

What was intentionally not imported:

- discriminator
- training loop
- dataset code
- old service helpers and unrelated utilities

Local weights location:

- `.models/shadow/AveragedModel.pth`

Useful env vars:

- `SHADOWGEN_SHADOW_PIX2PIX_WEIGHTS_PATH` to point the service at a different local checkpoint
- `SHADOWGEN_TARGET_DEVICE` to control whether inference targets `cuda`, `cuda:0`, or `cpu`

Runtime behavior:

- `real` mode uses the migrated pix2pix generator when the checkpoint is present
- `mock` mode uses the deterministic analytical shadow generator
- if the pix2pix backend fails to initialize, the service falls back to the deterministic stub and reports that in capabilities/debug metadata

What to verify:

- `components[].name == "shadow_generator"`
- `implementation == "real"` when the checkpoint is present and the backend initializes successfully
- `using_mock == false` when pix2pix is active

In the playground `Shadow` card:

- `shadow` shows the generated standalone RGBA shadow layer
- stage details expose backend mode and execution device

## Note on Python

This repository keeps the code compatible with Python `3.11+`.
Current local `.venv` may not be the final production baseline for heavy ML dependencies.

Important packaging note:

- `pyproject.toml` intentionally does not pin `torch` there anymore
- install the CUDA build of `torch` explicitly first
- then install `-e .[dev,ml]`
- if you run only `pip install -e .[ml]`, pip may silently choose a CPU-only torch in some environments
- `opencv-python-headless` is part of the base service dependencies because the foreground colour refinement stage uses it in both mock-heavy and real-model workflows
