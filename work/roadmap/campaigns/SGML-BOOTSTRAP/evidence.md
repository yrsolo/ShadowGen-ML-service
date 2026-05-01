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
  - tracked live smoke helper under `tools/smoke_triton_segmenter.py`
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
- tablet/desktop playground now keeps a minimum pipeline height instead of shrinking stage previews to fit the viewport; the page scrolls vertically when needed

### Triton Bring-Up Evidence

- local segmenter Triton container is built from `ops/triton/Dockerfile.segmenter-python`
- tracked Triton model repository lives under `ops/triton/model_repository`
- Triton image bakes the model repository into `/models` by default to avoid Windows bind-mount issues
- optional `-BindModelRepository` remains available for live model repository mounts
- Triton launcher supports `-Detach` for background local bring-up
- Triton launcher supports `-Wait` to block until `shadowgen_segmenter` is ready
- Triton launcher mounts host HuggingFace cache into `/root/.cache/huggingface` to avoid filling Docker overlay storage
- Triton launcher defaults to CPU-safe `SHADOWGEN_TRITON_SEGMENTER_RESOLUTION=512` without `-Gpu` and quality `1024` with `-Gpu`
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
- temporary Python backend accepts runtime env overrides for model id, device, and resolution
- local heavy fallback adapters are now lazy when their stage is not selected as the active backend, avoiding unnecessary local model loads during Triton smoke and Triton-first service startup
- direct live Triton segmenter smoke on `C:\Users\solofarm\Pictures\Screenshots\1.jpg` succeeded through `TritonSegmenter`: input `(512, 508)`, bbox `(239, 172, 384, 362)`, mask extrema `(0, 254)`
- full live `/v1/render` smoke with `segmenter=triton` and other heavy stages set to `mock` succeeded against the running Triton container; no-cache metrics reported `segmentation_ms=22982`, `total_ms=24041` in the tracked smoke script run
- debug fallback reason now includes the unavailable backend descriptor detail, for example `Triton endpoint is unavailable`
- V2-DIFF model training/export/serving requirements are captured in `docs/shadow-v2-model-contract.md`
- V2-DIFF integration is temporarily simplified to a control-free `img + mask -> shadow_image` binding
- Local V2-DIFF backend is connected through the SD1.5 inpaint LoRA bundle under `.models/shadow/v2-diff/shadowgen_inpaint_lora_prod_current`
- V2-DIFF uses `.models/shadow/v2-diff/mean_background.png` as the neutral conditioning background before inpaint inference
- Manual `artifacts/manual-inputs/1.jpg` V2-DIFF check stored under `artifacts/manual-runs/screenshot-1-v2-diff-local`; debug metadata reported `shadow_generator completed local v2-diff`
- V2-DIFF inpaint mask now uses inverted foreground semantics: object is preserved black, editable background/shadow region is white
- Manual inverted-mask V2-DIFF check stored under `artifacts/manual-runs/screenshot-1-v2-diff-local-inverted-mask`; generated shadow image changed materially from the previous object-mask run
- shadow stage output contract now represents a full model image instead of a separate `shadow_rgba` layer
- `V1-GAN` remains the controllable local rot/top-view shadow model
- the Triton shadow adapter now serializes only tensors declared by the active binding, so future controlled V2-DIFF tensors can be added without changing pipeline orchestration
- public render requests now accept optional `shadow.model` with `v1-gan` or `v2-diff`
- frontend-facing model selection guidance is documented in `docs/frontend-shadow-model-contract.md`
- Shadow V2 sample pack generated under `artifacts/shadow-v2-sample-pack` with 10 local-backend samples and contract-ready `shadow_input.npz` files
- Shadow V2 sample pack now uses curated product-case sources and includes an in-folder README for model developers
- Shadow V2 sample pack replacements: sample `06` now uses `Hot Coffee on a rainy day`; sample `09` now uses `Wallet on a table`
- BiRefNet cutout alpha is now clamped by the cleaned largest-component foreground mask, preventing low-confidence matting leaks from reaching V1-GAN/V2-DIFF shadow inputs
- Default local and Triton-Python segmenter model id changed from the lite matting checkpoint to `ZhengPeng7/BiRefNet-matting` for better cutout quality; the lite checkpoint remains available through `SHADOWGEN_BIREFNET_MODEL_ID`
- Manual pipeline check stored under `artifacts/manual-runs/v1-after-alpha-cleanup`; sample `01` cutout alpha bbox changed from `(44, 167, 468, 512)` to `(46, 169, 466, 361)`
- Default working-canvas content scale is now `0.68`, restoring larger margins around foreground objects for shadow generation
- Preprocess cache keys now include `working_content_scale`, so old tight-crop cache entries do not mask margin changes
- Manual `artifacts/manual-inputs/1.jpg` check stored under `artifacts/manual-runs/screenshot-1-v1-scale068`; cutout alpha bbox is `(115, 84, 397, 430)` after the margin change
- Local `V2-DIFF` profiling showed cached runtime is dominated by UNet denoising: about `3178 ms` of `3559 ms` at `24` steps; VAE decode was about `82 ms`, text encoding about `40 ms`, and remaining Python/PIL/postprocess about `259 ms`
- Local `V2-DIFF` defaults now use `12` denoising steps instead of bundle-driven `24`, while keeping `guidance_scale=2.0`; manual benchmark on `artifacts/manual-inputs/1.jpg` reduced cached shadow inference from about `2652 ms` to about `1433 ms`
- Manual V2-DIFF optimization comparison images are stored under `artifacts/manual-runs/v2-diff-optimization`
- The accelerated production `V2-DIFF` bundle from `ShadowGen-ML-training/outputs/prod_bundles/shadowgen_inpaint_lora_prod_current` has been copied into the ignored `.models/shadow/v2-diff/shadowgen_inpaint_lora_prod_current` service slot
- Local `V2-DIFF` now defaults to fast LCM mode: Shadow LoRA is fused with `latent-consistency/lcm-lora-sdv1-5`, scheduler switches to `LCMScheduler`, bundle defaults use `5` steps and `guidance_scale=1.0`
- Real local fast LCM smoke on the manual cat artifact succeeded under `artifacts/manual-runs/v2-diff-fast-lcm-smoke`; first call including pipeline load/fuse took about `12958 ms`, second resident call took about `582 ms`
- Geometry is disabled by default through `SHADOWGEN_GEOMETRY_ENABLED=false`; public render records `geometry_ms=0`, and the playground run-all flow skips the unused stage
- `normal_estimator` now defaults to `local/from-depth-v2`, avoiding the current broken/slow StableNormal path unless explicitly requested
- Depth-derived normals no longer use OpenCV inpainting; local timing on the 512px manual depth artifact changed from about `198..231 ms` to about `26..37 ms`

### Validation Evidence

- `.venv\Scripts\python.exe -m pytest -q` passed after the `shadow_image` contract switch: `88 passed, 3 warnings`
- `.venv\Scripts\python.exe -m compileall src tests` passed after the `shadow_image` contract switch
- `.venv\Scripts\python.exe -m pytest -q` passed after BiRefNet alpha cleanup: `89 passed, 3 warnings`
- `.venv\Scripts\python.exe -m compileall src tests` passed after BiRefNet alpha cleanup
- `.venv\Scripts\python.exe -m pytest -q` passed after working-canvas margin change: `91 passed, 3 warnings`
- `.venv\Scripts\python.exe -m compileall src tests` passed after working-canvas margin change
- `.venv\Scripts\python.exe -m pytest -q` passed after local V2-DIFF integration: `93 passed, 4 warnings`
- `.venv\Scripts\python.exe -m compileall src tests` passed after local V2-DIFF integration
- `.venv\Scripts\python.exe -m pytest -q` passed after V2-DIFF mask inversion: `93 passed, 4 warnings`
- `.venv\Scripts\python.exe -m compileall src tests` passed after V2-DIFF mask inversion
- `.venv\Scripts\python.exe -m pytest tests\test_shadow_v2_diff.py -q` passed after V2-DIFF latency optimization: `3 passed, 1 warning`
- `.venv\Scripts\python.exe -m compileall src tests` passed after V2-DIFF latency optimization
- `.venv\Scripts\python.exe -m pytest -q` passed after V2-DIFF latency optimization: `94 passed, 4 warnings`
- `.venv\Scripts\python.exe -m pytest tests\test_shadow_v2_diff.py -q` passed after fast LCM integration: `4 passed, 1 warning`
- `.venv\Scripts\python.exe -m compileall src tests` passed after fast LCM integration
- `.venv\Scripts\python.exe -m pytest -q` passed after fast LCM integration: `95 passed, 4 warnings`
- `.venv\Scripts\python.exe -m pytest tests\test_segmenter.py tests\test_runtime.py -q` passed after BiRefNet quality default switch: `14 passed`
- `.venv\Scripts\python.exe -m compileall src tests ops\triton\model_repository\shadowgen_segmenter\1\model.py` passed after BiRefNet quality default switch
- `.venv\Scripts\python.exe -m pytest -q` passed after BiRefNet quality default switch: `95 passed, 4 warnings`
- `.venv\Scripts\python.exe -m pytest tests\test_api.py tests\test_runtime.py tests\test_normals.py -q` passed after geometry/normal changes: `50 passed, 3 warnings`
- `.venv\Scripts\python.exe -m compileall src tests` passed after geometry/normal changes
- `.venv\Scripts\python.exe -m pytest -q` passed after geometry/normal changes: `94 passed, 4 warnings`
- `.venv\Scripts\python.exe -m compileall src/shadowgen_ml_service/interfaces/dev/playground.py` passed after playground min-height fix
- `.venv\Scripts\python.exe -m pytest` passed: `88 passed`
- `.venv\Scripts\python.exe -m compileall src tests` passed
- `.venv\Scripts\python.exe -m pytest` passed: `85 passed`
- `.venv\Scripts\python.exe -m compileall src tests` passed
- `.venv\Scripts\python.exe -m pytest` passed: `81 passed`
- `python -m compileall src tests tools` passed
- browser smoke passed for the generated playground HTML:
  - 9 stage cards rendered
  - horizontal overflow detected
  - wheel moved pipeline `scrollLeft` from `4` to `780`
- PowerShell syntax parse passed for `tools/run_triton_segmenter_python.ps1`
- `tools\run_triton_segmenter_python.cmd -NoBuild -Detach` started Triton successfully after baking the model repository into the image
- `.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010` passed
- `tools\run_triton_segmenter_python.cmd -NoBuild -Detach -Wait -Resolution 512` started Triton successfully with host-mounted HuggingFace cache
- direct `TritonSegmenter` smoke against `http://127.0.0.1:8010` passed and saved artifacts under `artifacts/triton-smoke-real`
- full `/v1/render` smoke with `segmenter=triton` passed and saved artifacts under `artifacts/triton-smoke-full`
- `.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --output-dir artifacts\triton-smoke-script` passed and saved direct/render artifacts
- `.venv\Scripts\python.exe -m pytest tests\test_runtime.py tests\test_api.py tests\test_triton_transport.py tests\test_segmenter_triton.py -q` passed after live Triton hardening: `65 passed, 3 warnings`
- `.venv\Scripts\python.exe -m pytest -q` passed after live Triton hardening: `95 passed, 4 warnings`
- `.venv\Scripts\python.exe -m compileall src tests tools ops\triton\model_repository\shadowgen_segmenter\1\model.py` passed after live Triton hardening
- `python -m py_compile ops/triton/model_repository/shadowgen_segmenter/1/model.py` passed
- `tools/run_triton_segmenter_python.ps1` fails fast with a clear Docker / WSL diagnostic when the local Triton container backend is unavailable

## Remaining Bootstrap Gaps

- current BiRefNet ONNX export is blocked in this environment by `torchvision::deform_conv2d`, so the live `segmenter` bridge currently depends on the temporary Triton Python backend
- `V2-DIFF` Triton backend is scaffolded but not implemented
- compatibility shims still remain in the repository
- Docker Desktop GPU mode still fails on this workstation with NVIDIA runtime `legacy` mode, so current Triton validation is CPU-only until Docker Desktop / NVIDIA runtime is fixed
