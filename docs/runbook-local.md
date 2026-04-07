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
.venv/Scripts/python.exe -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
```

For real-model probing, configure local model paths through env vars:

- `SHADOWGEN_GROUNDING_DINO_PATH`
- `SHADOWGEN_GEOCALIB_PATH`
- `SHADOWGEN_BIREFNET_PATH`
- `SHADOWGEN_DEPTH_ANYTHING_PATH`

## GeoCalib bring-up

The real `Geometry` step uses GeoCalib from the active `.venv`.

Expected workflow:

1. Install GeoCalib into the current virtual environment.
2. Keep heavy weights and artifacts out of git in ignored folders such as `.models/`.
3. Start the service and inspect `GET /v1/capabilities`.

What to verify:

- `components[].name == "geometry_estimator"`
- `implementation == "real"`
- `using_mock == false`

If GeoCalib is unavailable or fails to initialize, the service falls back to the mock geometry estimator and the playground shows `mock-fallback`.

## Note on Python

This repository keeps the code compatible with Python `3.11+`.
Current local `.venv` may not be the final production baseline for heavy ML dependencies.
