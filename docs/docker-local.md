# Docker Local Runbook

This repository supports two Docker modes:

- production replacement mode: one ML-service container with local ML backends
- Triton/debug mode: ML-service container plus Triton container

Use production replacement mode while Triton is not the main runtime path.

Secrets and model weights are not baked into images. Runtime state is mounted or injected at container start:

- `.env` through optional Compose `env_file`.
- `.models/` for local model bundles.
- `.cache/` for Hugging Face and runtime caches.
- `artifacts/` and `var/` for generated outputs and cache files.

## Production Replacement Mode

This mode starts only `shadowgen-ml-service`. It does not require Triton and is intended to replace the old ML core behind the existing product/worker.

### Build

Rebuild the ML service image after changing service code or Python dependencies:

```cmd
rebuild-service-container.cmd
```

### Start

Start the service container:

```cmd
start-service-container.cmd
```

Configure the GPU in `.env` before startup:

```dotenv
SERVICE_GPU_DEVICE=1
SHADOWGEN_TARGET_DEVICE=cuda:0
```

Default URL:

- ML service API: `http://127.0.0.1:8000`

The container runs with:

- `SHADOWGEN_DEV_API_ENABLED=0`
- `SHADOWGEN_DEV_SHUTDOWN_ENABLED=0`
- local detector/segmenter/depth/normals/shadow backends
- `SHADOWGEN_SHADOW_MODEL_VARIANT=v2-diff` unless overridden

### Stop

```cmd
stop-service-container.cmd
```

### GPU Selection

Choose the host GPU with `SERVICE_GPU_DEVICE` in `.env`.

Example for host GPU `1`:

```dotenv
SERVICE_GPU_DEVICE=1
SHADOWGEN_TARGET_DEVICE=cuda:0
```

Inside the container the selected GPU is exposed as `cuda:0`, so `SHADOWGEN_TARGET_DEVICE` should normally remain `cuda:0` regardless of the host GPU index.

The tracked [.env.example](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/.env.example) contains safe defaults. Do not replace an existing `.env` if it already contains credentials; add or update only the GPU keys.

### Runtime Mounts

The service-only compose file mounts:

- `.models/` to `/app/.models`
- `.cache/` to `/app/.cache`
- `artifacts/` to `/app/artifacts`
- `var/` to `/app/var`

Override them when deploying outside the repository directory:

```cmd
set SERVICE_MODELS_DIR=D:\shadowgen\.models
set SERVICE_CACHE_DIR=D:\shadowgen\.cache
set SERVICE_ARTIFACTS_DIR=D:\shadowgen\artifacts
set SERVICE_VAR_DIR=D:\shadowgen\var
start-service-container.cmd
```

## Triton/Debug Two-Container Mode

This mode starts:

- `shadowgen-triton-segmenter`: Triton execution plane for Triton-backed stages.
- `shadowgen-ml-service`: FastAPI orchestration/debug service with the playground.

Use it for Triton bring-up, not as the default production replacement path yet.

## Build Triton/Debug Images

Rebuild the Triton image after changing Triton model code:

```cmd
rebuild-triton.cmd
```

## Start Triton/Debug Stack

Start both containers:

```cmd
start-docker-stack.cmd
```

`start-docker-stack.cmd` does not force a rebuild. It starts the already-built images and removes stale stopped containers with the same names before `docker compose up -d`.

Startup ordering:

- Triton starts first.
- Docker healthcheck waits for `http://127.0.0.1:8000/v2/health/ready` inside the Triton container.
- The ML service starts only after Triton is healthy, so capability probes can see ready Triton models at startup.

Default URLs:

- Playground: `http://127.0.0.1:8000/playground`
- ML service API: `http://127.0.0.1:8000`
- Triton HTTP: `http://127.0.0.1:8010`
- Triton gRPC: `127.0.0.1:8011`
- Triton metrics: `http://127.0.0.1:8012`

## Stop Triton/Debug Stack

```cmd
stop-docker-stack.cmd
```

## GPU Selection For Triton/Debug Stack

By default both containers expose GPU `1`, which is expected to be the RTX 4090 on the current workstation. Override if needed:

```cmd
set TRITON_GPU_DEVICE=1
set SERVICE_GPU_DEVICE=1
start-docker-stack.cmd
```

Inside each container `cuda:0` means the first visible GPU after `NVIDIA_VISIBLE_DEVICES` filtering.

The production service-only compose file reserves the selected GPU through Docker Compose device reservations. The Triton/debug compose still uses `gpus: all` plus `NVIDIA_VISIBLE_DEVICES` filtering.

## Backend Selection

The service container defaults to local ML backends and can call Triton through the Docker network at `http://triton:8000`.

To request Triton-backed detector/segmenter by default:

```cmd
set SHADOWGEN_DETECTOR_BACKEND_KIND=triton
set SHADOWGEN_SEGMENTER_BACKEND_KIND=triton
start-docker-stack.cmd
```

The playground can still switch stage backend kinds per stage.

## Split Debug Mode

The older split workflow remains useful when FastAPI should run directly in a visible Windows console:

```cmd
rebuild-triton.cmd
start-triton.cmd
start-service.cmd
```

Differences from the Docker stack:

- `start-service.cmd` uses the host `.venv`.
- `SHADOWGEN_TRITON_URL` points to `http://127.0.0.1:8010`.
- `SHADOWGEN_DEV_SHUTDOWN_ENABLED=1`, so the playground shutdown button can stop the visible FastAPI process.
- The service is not isolated from host Python/package state.

Use the Docker stack when validating deployment shape. Use split mode when iterating on Python code with a debugger or visible console logs.
