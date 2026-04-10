# SGML-BOOTSTRAP Evidence

## Scope Achieved

- repository initialized on `dev`
- layered architecture established
- service API implemented
- playground UI implemented
- multiple ML stages wired with mock and real paths
- runtime refactored to a Triton-ready registry model
- sync and async render paths now coexist
- project docs rewritten into a usable current-state reference set

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

### Triton Readiness Evidence

- shared Triton subsystem exists under `infrastructure/backends/triton/`
- Triton transport now uses the standard tensor infer payload with explicit `inputs` and `outputs`
- stage/model bindings now include tensor schema metadata, not only model names
- heavy-stage canonical inputs now carry neutral `RasterAsset` payloads instead of implicit `PIL.Image` objects
- stage runtime faults are normalized into structured failed-stage executions for dev/debug and service errors for public sync execution
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
  - Triton scaffold backend
  - working crop, mask, and cutout previews

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
- repository workflow rules in `docs/workflow.md`

### Validation Evidence

- `.venv\Scripts\python.exe -m pytest` passed: `61 passed`
- `python -m compileall src tests` passed

## Remaining Bootstrap Gaps

- no heavy stage is connected to a live Triton server yet
- `V2-DIFF` shadow backend is scaffolded but not implemented
- compatibility shims still remain in the repository
- Docker/deployment documentation is not yet part of bootstrap
