# SGML-BOOTSTRAP Evidence

## Scope Achieved

- repository initialized on `dev`
- layered architecture established
- service API implemented
- playground UI implemented
- playground stage browser reworked into a horizontal scroller with wheel-driven horizontal navigation
- multiple ML stages wired with mock and real paths
- runtime refactored to a Triton-ready registry model
- sync and async render paths now coexist
- project docs rewritten into a usable current-state reference set
- worker-facing integration contract documented for `ShadowGen-v2`
- ML core upgraded to expose worker-facing readiness, capacity, and async idempotency semantics

## Captured Evidence

- `GET /health`, `GET /v1/capabilities`, `POST /v1/render` are implemented
- `POST /v1/render/jobs`, `GET /v1/render/jobs/{job_id}`, and `DELETE /v1/render/jobs/{job_id}` are implemented
- `GET /playground`, `POST /v1/dev/pipeline/run-all`, and `POST /v1/dev/pipeline/run-stage/{stage_key}` are implemented
- request contract supports `preprocess.padding_px`
- repository architecture is split into:
  - `core`
  - `application`
  - `bootstrap`
  - `infrastructure`
  - `interfaces`

### Execution Architecture Evidence

- runtime uses a registry of stage backends keyed by:
  - `stage_key`
  - `backend_kind`
  - `model_variant`
- heavy stages now advertise:
  - `mock`
  - `local`
  - `triton`
- dev and public metadata expose:
  - backend kind
  - model variant
  - model name and version
  - device
  - endpoint
  - fallback reason
  - batching and async support flags

### Async Flow Evidence

- in-memory async job backend exists under `infrastructure/jobs/`
- async use cases exist under `application/use_cases/`
- async endpoints are routed through the same orchestration model as sync execution
- async backend is now bounded and concurrent instead of single-thread serial
- async job metadata now includes submit mode and capacity snapshot
- `request_id` now acts as the async idempotency key when present

### Triton Readiness Evidence

- shared Triton subsystem exists under `infrastructure/backends/triton/`
- Triton transport now uses the standard tensor infer payload with explicit `inputs` and `outputs`
- stage/model bindings now include tensor schema metadata, not only model names
- heavy-stage canonical inputs now carry neutral `RasterAsset` payloads instead of implicit `PIL.Image` objects
- stage runtime faults are normalized into structured failed-stage executions for dev/debug and service errors for public sync execution
- heavy-stage Triton execution now supports internal micro-batching for:
  - `segmenter`
  - `depth_estimator`
  - `normal_estimator`
  - `shadow_generator`
- stage-specific Triton adapters exist for:
  - `detector`
  - `segmenter`
  - `depth_estimator`
  - `normal_estimator`
  - `shadow_generator`
- heavy-stage orchestration now enters executors through canonical `stage_io` contracts
- local-only phase-1 stages remain:
  - `geometry_estimator`
  - `foreground_refiner`
  - `composer`
  - `artifact_encoder`

### Stage Wiring Evidence

- `geometry_estimator`
  - GeoCalib local backend
  - mock backend
  - playground overlay and numeric details

- `detector`
  - GroundingDINO local backend
  - mock backend
  - Triton scaffold backend
  - bbox previews and working-crop preview

- `segmenter`
  - BiRefNet local backend
  - mock backend
  - live-first Triton mask-first backend contract
  - working crop, mask, and cutout previews
  - tracked Triton model repository scaffold under `ops/triton/model_repository/shadowgen_segmenter/`
  - tracked Triton Python backend implementation under `ops/triton/model_repository/shadowgen_segmenter/1/model.py`
  - tracked Triton custom image scaffold under `ops/triton/Dockerfile.segmenter-python`
  - tracked bring-up helper under `tools/run_triton_segmenter_python.ps1`
  - tracked readiness checker under `tools/check_triton_segmenter_ready.py`
  - reproducible ONNX export tool under `tools/export_segmenter_onnx.py`
  - export tool now tries both modern and legacy ONNX exporters and reports the current `torchvision::deform_conv2d` blocker explicitly
  - local BiRefNet runtime now exposes opt-in Torch acceleration controls (`torch.compile`, matmul precision)

- `foreground_refiner`
  - Fast Foreground Colour Estimation stage is standalone
  - no longer hidden inside segmentation

- `depth_estimator`
  - Depth Anything V2 local backend
  - mock backend
  - Triton scaffold backend

- `normal_estimator`
  - StableNormal local backend
  - `from-depth-v2` local fallback backend
  - mock backend
  - Triton scaffold backend

- `shadow_generator`
  - deterministic mock backend
  - `V1-GAN` local backend migrated from legacy pix2pix inference
  - `V2-DIFF` Triton-ready scaffold backend

### Shadow-Specific Evidence

- local `V1-GAN` checkpoint stored under ignored `.models/shadow/AveragedModel.pth`
- real shadow backend successfully loaded on local GPU
- live smoke previously verified:
  - backend `Pix2PixShadowGenerator`
  - device `cuda:0`
  - debug stage completed successfully
- real shadow outputs no longer use coarse post-blur softness
- coarse softness blur remains only in the mock shadow backend

### Documentation Evidence

The active docs now provide:

- quick entry overview in `README.md`
- docs index in `docs/README.md`
- system architecture in `docs/architecture.md`
- codebase map in `docs/modules.md`
- local runtime and model bring-up notes in `docs/runbook-local.md`
- public, debug, and async API summary in `docs/api.md`
- worker-to-ML-core integration contract in `docs/worker-core-contract.md`
- repository workflow rules in `docs/workflow.md`

### Playground Evidence

- dev playground now uses a horizontal stage scroller on desktop
- each stage card uses top execution controls/details and bottom preview area
- mouse wheel over the pipeline maps vertical wheel motion to horizontal scrolling
- `Shift` + mouse wheel scrolls the active stage card vertically
- stage cards use internal vertical overflow so previews are scrollable instead of being clipped
- top controls were compacted so stage cards and previews remain visible without vertical page scroll

### Triton Bring-Up Evidence

- local segmenter Triton container is built from `ops/triton/Dockerfile.segmenter-python`
- tracked Triton model repository lives under `ops/triton/model_repository`
- Triton image bakes the model repository into `/models` by default to avoid Windows bind-mount issues
- optional `-BindModelRepository` remains available for live model repository mounts
- Triton launcher supports `-Detach` for background local bring-up
- Triton Python image includes BiRefNet dynamic-module dependencies (`einops`, `kornia`, `timm`)
- local Triton `shadowgen_segmenter` readiness passed on `http://127.0.0.1:8010`
- ML-core capabilities reported `segmenter` as `backend_kind=triton`, `available=true`, `model_name=shadowgen_segmenter`, with no fallback reason
- Windows launcher exists at `tools/run_triton_segmenter_python.cmd`
- PowerShell launcher exists at `tools/run_triton_segmenter_python.ps1`
- service launcher exists at `run-service-triton-segmenter.cmd` to set Triton segmenter env defaults
- local helper maps standard Triton container ports to offset host ports:
  - HTTP `8010`
  - gRPC `8011`
  - metrics `8012`
- ML-core should use `SHADOWGEN_TRITON_URL=http://127.0.0.1:8010` when FastAPI uses local port `8000`
- Triton launcher defaults to no Docker GPU flag for bring-up and supports explicit `-Gpu`
- temporary Python backend uses `KIND_CPU` so Triton can load without NVIDIA container runtime, while model code still chooses CUDA when available
- debug fallback reason now includes the unavailable backend descriptor detail, for example `Triton endpoint is unavailable`

### Validation Evidence

- `.venv\Scripts\python.exe -m pytest` passed: `81 passed`
- `python -m compileall src tests tools` passed
- browser smoke passed for the generated playground HTML:
  - 9 stage cards rendered
  - horizontal overflow detected
  - wheel moved pipeline `scrollLeft` from `4` to `780`
- PowerShell syntax parse passed for `tools/run_triton_segmenter_python.ps1`
- `tools\run_triton_segmenter_python.cmd -NoBuild -Detach` started Triton successfully after baking the model repository into the image
- `.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010` passed
- `python -m py_compile ops/triton/model_repository/shadowgen_segmenter/1/model.py` passed
- `tools/run_triton_segmenter_python.ps1` fails fast with a clear Docker / WSL diagnostic when the local Triton container backend is unavailable

## Remaining Bootstrap Gaps

- no heavy stage has been smoke-tested yet against a real external Triton server
- current BiRefNet ONNX export is blocked in this environment by `torchvision::deform_conv2d`, so the live `segmenter` bridge currently depends on the temporary Triton Python backend
- `V2-DIFF` shadow backend is scaffolded but not implemented
- compatibility shims still remain in the repository
- the current workstation has a Docker Desktop / WSL blocker (`docker-desktop` distro missing), so the temporary Triton segmenter image has not yet been validated end-to-end against a real running container
