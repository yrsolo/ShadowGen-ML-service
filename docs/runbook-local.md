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

## Browser playground

After the server starts, open:

- `http://127.0.0.1:8000/playground`

The playground supports:

- uploading one source image
- running the full pipeline or a single stage
- switching each stage between `mock` and `real`
- showing stage-local errors and previews
- adjusting the global preview size

### Optional ML bootstrap

```powershell
.venv/Scripts/python.exe -m pip install -e .[ml]
.venv/Scripts/python.exe -m pip install transformers
.venv/Scripts/python.exe -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
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

1. Install GeoCalib into the current virtual environment.
2. Keep heavy weights and artifacts out of git in ignored folders such as `.models/`.
3. Start the service and inspect `GET /v1/capabilities`.

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

1. Install `transformers` into the active virtual environment.
2. Start the service and let the model download into the local Hugging Face / torch cache on first initialization.
3. Inspect `GET /v1/capabilities`.

Useful env vars:

- `SHADOWGEN_GROUNDING_DINO_MODEL_ID` to switch the detector checkpoint
- `SHADOWGEN_GROUNDING_DINO_PROMPT` to override the zero-shot prompt, default `object.`
- `SHADOWGEN_GROUNDING_DINO_BOX_THRESHOLD` to change the box confidence threshold
- `SHADOWGEN_GROUNDING_DINO_TEXT_THRESHOLD` to change the text threshold used by post-processing

What to verify:

- `components[].name == "detector"`
- `implementation == "real"`
- `using_mock == false`

In the playground `Detection` card:

- `detection_overlay` shows the selected bbox on the source image
- `crop_for_resize` shows the working crop that will go to segmentation
- stage details expose bbox coordinates, confidence, backend mode, and prompt

If GroundingDINO is unavailable or fails to initialize, the service falls back to the mock detector and the playground shows `mock-fallback`.

## Note on Python

This repository keeps the code compatible with Python `3.11+`.
Current local `.venv` may not be the final production baseline for heavy ML dependencies.
