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
  - live Triton Python backend
  - bbox previews and working-crop preview
  - tracked Triton model repository scaffold under `ops/triton/model_repository/shadowgen_detector/`
  - tracked Triton Python backend implementation under `ops/triton/model_repository/shadowgen_detector/1/model.py`
  - tracked live smoke helper under `tools/smoke_triton_detector.py`

- `segmenter`
  - BiRefNet local backend
  - mock backend
  - live-first Triton mask-first backend contract
  - working crop, mask, and cutout previews
  - tracked Triton model repository scaffold under `ops/triton/model_repository/shadowgen_segmenter/`
  - tracked Triton Python backend implementation under `ops/triton/model_repository/shadowgen_segmenter/1/model.py`
  - tracked Triton custom image scaffold under `ops/triton/Dockerfile.segmenter-python`
  - tracked bring-up helpers consolidated into root `rebuild-triton.cmd` and `start-service.cmd`
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
- root launch flow is now reduced to two user-facing CMD scripts:
  - `rebuild-triton.cmd` rebuilds the Triton image with the baked model repository and removes the stale Triton container
  - `start-service.cmd` opens a visible FastAPI console, starts the prebuilt Triton container, waits for readiness, and starts ML-core with Triton segmenter defaults
- `start-service.cmd` now keeps detector and segmenter on local defaults unless `USE_TRITON_BACKENDS=1`, because the current Triton Python backends are debug bridges rather than final performance targets
- ML-core Triton inference now uses the official Triton HTTP client with binary tensor payloads by default through `SHADOWGEN_TRITON_TRANSPORT=native`; `SHADOWGEN_TRITON_TRANSPORT=json` remains available as a debug fallback
- direct benchmark on `C:\Users\solofarm\Pictures\Screenshots\1.jpg` showed JSON transport overhead clearly:
  - detector Triton JSON mean `1510.53 ms`
  - detector Triton native mean `337.90 ms`
  - segmenter Triton JSON mean `2324.50 ms`
  - segmenter Triton native mean `954.46 ms`
- local-vs-native benchmark on the same image showed:
  - detector local mean `200.77 ms`
  - detector Triton native mean `328.95 ms`
  - segmenter local mean `830.94 ms`
  - segmenter Triton native mean `956.00 ms`
- RMBG-2.0 is registered as optional segmenter variant `rmbg-2.0` with Triton model name `shadowgen_segmenter_rmbg2`
- `tools/prepare_rmbg2_onnx_triton.py` prepares the gated BRIA RMBG-2.0 ONNX model repository entry after local Hugging Face authentication; generated `model.onnx` files stay ignored by git
- `tools/export_detector_onnx.py` adds an experimental model-only GroundingDINO export path that emits `logits` and `pred_boxes`; runtime replacement still requires a dedicated postprocess adapter or Triton ensemble step
- Playground includes a dev-only shutdown button backed by `POST /v1/dev/service/shutdown`, so the current ML-service process can be stopped from the UI
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
- direct live Triton detector smoke on `C:\Users\solofarm\Pictures\Screenshots\1.jpg` succeeded through `TritonDetector`: input `(768, 762)`, bbox `(354, 257, 586, 544)`, confidence `0.419`
- full `/v1/render` smoke with `detector=triton` and other heavy stages set to `mock` succeeded; no-cache metrics reported `detection_ms=923`, `total_ms=1905`
- `docker exec shadowgen-triton-segmenter python3 -c "import torch; ..."` confirmed CUDA is available and container `cuda:0` is `NVIDIA GeForce RTX 4090` after adding the detector model
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
- legacy Triton run helpers were removed after consolidation into `rebuild-triton.cmd` and `start-service.cmd`
- `.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010` passed
- direct `TritonSegmenter` smoke against `http://127.0.0.1:8010` passed and saved artifacts under `artifacts/triton-smoke-real`
- full `/v1/render` smoke with `segmenter=triton` passed and saved artifacts under `artifacts/triton-smoke-full`
- `.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --output-dir artifacts\triton-smoke-script` passed and saved direct/render artifacts
- `http://127.0.0.1:8003/v1/capabilities` reported `segmenter.backend_kind=triton`, `endpoint=http://127.0.0.1:8010` after starting the service with `RELOAD=0`
- `http://127.0.0.1:8003/v1/dev/pipeline/run-stage/segmenter` returned `segmenter completed triton triton` with `elapsed_ms=19871`
- `.venv\Scripts\python.exe -m pytest tests\test_runtime.py tests\test_api.py tests\test_triton_transport.py tests\test_segmenter_triton.py -q` passed after live Triton hardening: `65 passed, 3 warnings`
- `.venv\Scripts\python.exe -m pytest -q` passed after live Triton hardening: `95 passed, 4 warnings`
- `.venv\Scripts\python.exe -m compileall src tests tools ops\triton\model_repository\shadowgen_segmenter\1\model.py` passed after live Triton hardening
- `python -m py_compile ops/triton/model_repository/shadowgen_segmenter/1/model.py` passed
- `start-service.cmd` now owns the local Triton container start path and fails fast when Docker or the prebuilt Triton image is unavailable
- `.venv\Scripts\python.exe -m pytest tests\test_api.py -q` passed after adding the visible service launcher and Playground shutdown control: `38 passed, 3 warnings`
- `.venv\Scripts\python.exe -m compileall src tests` passed after adding the visible service launcher and Playground shutdown control
- `cmd /c rebuild-triton.cmd /?` passed after consolidating root launch scripts
- `cmd /c start-service.cmd /?` passed after consolidating root launch scripts
- root `*.cmd` listing now contains only `rebuild-triton.cmd` and `start-service.cmd`
- `git diff --check` passed after consolidating root launch scripts
- Docker GPU runtime validation passed with `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`; both RTX 2080 Ti and RTX 4090 were visible inside the container
- Triton segmenter was restarted with `--gpus all`, `SHADOWGEN_TRITON_SEGMENTER_DEVICE=cuda:0`, `SHADOWGEN_TRITON_SEGMENTER_RESOLUTION=512`, and `SHADOWGEN_TRITON_SEGMENTER_COMPILE_ENABLED=false`
- `docker exec shadowgen-triton-segmenter python3 -c "import torch; ..."` confirmed CUDA is available and container `cuda:0` is `NVIDIA GeForce RTX 4090`
- `.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010 --wait-seconds 300` passed after GPU Triton restart
- `.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --output-dir artifacts\triton-gpu-smoke-compile-off` passed; full render reported `segmentation_ms=1057` and Triton logs showed the warm 512px infer executing in about `0.31s`
- `.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010 shadowgen_detector --wait-seconds 300` passed after adding `shadowgen_detector`
- `.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010 shadowgen_segmenter --wait-seconds 300` passed after adding `shadowgen_detector`
- `.venv\Scripts\python.exe tools\smoke_triton_detector.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --output-dir artifacts\triton-detector-smoke --timeout-ms 300000` passed after adding live Triton detector support
- `.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --output-dir artifacts\triton-segmenter-after-detector-smoke --direct-only --timeout-ms 300000` passed after adding live Triton detector support
- `.venv\Scripts\python.exe -m pytest -q` passed after adding live Triton detector support: `96 passed, 4 warnings`
- `.venv\Scripts\python.exe -m pytest -q` passed after native Triton transport, benchmark helper, RMBG-2.0 preparation, and GroundingDINO ONNX export tooling: `96 passed, 4 warnings`
- `.venv\Scripts\python.exe tools\prepare_rmbg2_onnx_triton.py --source var\tmp\rmbg14\onnx\model_fp16.onnx --target var\tmp\rmbg-test\shadowgen_segmenter_rmbg2\1\model.onnx` passed as an ONNX config-generation smoke using the non-gated RMBG-1.4 ONNX file
- `.venv\Scripts\python.exe tools\prepare_rmbg2_onnx_triton.py --filename onnx/model.onnx` initially failed cleanly with a gated-model access message because `HF_TOKEN` was not loaded from `.env`
- `.venv\Scripts\python.exe tools\prepare_rmbg2_onnx_triton.py --filename onnx/model.onnx` passed after teaching the tool to read `HF_TOKEN` from `.env`; generated ignored `shadowgen_segmenter_rmbg2` model/config files with input `pixel_values` and output `alphas`
- Local ONNX Runtime smoke for `shadowgen_segmenter_rmbg2` passed on CPU fallback with output shape `[1,1,1024,1024]` and alpha range `0..0.9999997`; Windows ORT CUDA provider was unavailable due a missing `cublasLt64_12.dll`
- `.venv\Scripts\python.exe tools\export_detector_onnx.py --height 512 --width 512` passed and generated ignored `shadowgen_detector_onnx` model/config files
- Local ONNX Runtime smoke for `shadowgen_detector_onnx` passed with `CUDAExecutionProvider`, output shapes `logits=[1,900,256]`, `pred_boxes=[1,900,4]`, bbox `[235,172,391,365]`, confidence `0.3508`, and warm pure ORT inference about `164 ms`
- `cmd /c start-triton.cmd help`, `cmd /c rebuild-triton.cmd help`, and `cmd /c start-service.cmd help` passed after splitting Triton rebuild, Triton start, and FastAPI start
- `.venv\Scripts\python.exe -m compileall src tools tests` passed after ONNX detector wiring and launch-script split
- `.venv\Scripts\python.exe -m pytest tests/test_triton_transport.py tests/test_api.py tests/test_runtime.py -q` passed: `62 passed, 3 warnings`
- `.venv\Scripts\python.exe -m pytest -q` passed: `96 passed, 4 warnings`
- `cmd /c rebuild-triton.cmd` passed after Docker Desktop was repaired
- `cmd /c start-triton.cmd` passed and loaded all live Triton models as READY: `shadowgen_detector`, `shadowgen_segmenter`, `shadowgen_detector_onnx`, `shadowgen_segmenter_rmbg2`
- `.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --variant rmbg-2.0 --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --direct-only --timeout-ms 300000 --output-dir artifacts\triton-rmbg2-smoke` passed with `mask_extrema=[0,254]`
- `.venv\Scripts\python.exe tools\smoke_triton_detector.py --variant grounding-dino-onnx --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --direct-only --timeout-ms 300000 --output-dir artifacts\triton-detector-onnx-smoke` passed with bbox `[353,257,586,545]` and confidence about `0.366`
- full FastAPI render smoke with `detector=triton/grounding-dino-onnx` passed and reported capability `backend_kind=triton`, `endpoint=http://127.0.0.1:8010`
- full FastAPI render smoke with `segmenter=triton/rmbg-2.0` passed and reported capability `backend_kind=triton`, `endpoint=http://127.0.0.1:8010`
- combined FastAPI render smoke with `detector=triton/grounding-dino-onnx` and `segmenter=triton/rmbg-2.0` passed; measured latency is functional but still unstable and remains a performance follow-up
- Playground now exposes detector variants `grounding-dino` / `grounding-dino-onnx` and segmenter variants `birefnet` / `rmbg-2.0`; choosing an ONNX variant automatically switches the stage to `triton`
- `start-triton.cmd` now defaults to `TRITON_GPU_DEVICE=1`, so Docker exposes only the host RTX 4090 to Triton; inside the container that GPU is addressed as `cuda:0`
- ONNX model configs now pin `instance_group.gpus: [0]`; after Docker GPU remapping this points detector/segmenter ONNX execution at the RTX 4090 path instead of letting Triton instantiate on both visible GPUs
- `.venv\Scripts\python.exe tools\smoke_triton_detector.py --variant grounding-dino-onnx --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --direct-only --timeout-ms 300000 --output-dir artifacts\triton-detector-onnx-gpu1-smoke` passed after GPU pinning
- `.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --variant rmbg-2.0 --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg --direct-only --timeout-ms 300000 --output-dir artifacts\triton-rmbg2-gpu1-smoke` passed after GPU pinning
- Default local and Triton-Python segmenter model id changed from `ZhengPeng7/BiRefNet-matting` to `ZhengPeng7/BiRefNet` after visual comparison on the project examples
- RMBG-2.0 Triton adapter keeps feeding the current ONNX model at `1024x1024`; a `512x512` smoke failed on an internal ONNXRuntime `Reshape`, so the model metadata is dynamic but this export is not practically resolution-flexible
- `docker compose config --no-interpolate` passed after adding the two-container local Docker stack
- `cmd /c rebuild-service-container.cmd` passed and built `shadowgen-ml-service:local`
- `cmd /c start-docker-stack.cmd` passed and started `shadowgen-triton-segmenter` plus `shadowgen-ml-service`
- `GET http://127.0.0.1:8000/health` returned `status=ok`, `async_enabled=true`, `accepting_jobs=true`, and `preferred_submit_mode=async` from the service container
- `GET http://127.0.0.1:8010/v2/health/ready` returned HTTP 200 from the Triton container
- `start-docker-stack.cmd` was corrected to avoid implicit rebuilds and now waits for the Triton healthcheck before starting the service container
- service-container `GET /v1/capabilities` reported Triton detector variants `grounding-dino` / `grounding-dino-onnx` and segmenter variants `birefnet` / `rmbg-2.0` as `available=True` via `http://triton:8000`
- documentation was updated to make the two-container Docker stack the recommended local workflow and keep `start-triton.cmd` + `start-service.cmd` as advanced split debug mode
- `docker compose config --no-interpolate` passed after the documentation refresh
- `git diff --check` passed for the updated docs and tracking files
- `Dockerfile.service` now defaults to production-safe `SHADOWGEN_DEV_API_ENABLED=0` and `SHADOWGEN_DEV_SHUTDOWN_ENABLED=0`
- `docker-compose.service.yml` was added for service-only production replacement with local backends and GPU selection via `SERVICE_GPU_DEVICE`
- `start-service-container.cmd help` and `rebuild-service-container.cmd help` passed
- `docker compose -f docker-compose.service.yml config --no-interpolate` passed
- full Docker image rebuild was not rerun because Docker Desktop daemon was unavailable in this session
- `start-service-container.cmd` now reads only `SERVICE_GPU_DEVICE` from `.env` before applying the default and does not print other environment values
- `.env.example` documents `SERVICE_GPU_DEVICE=1` and `SHADOWGEN_TARGET_DEVICE=cuda:0` without containing secrets
- the ignored local `.env` was configured with host GPU `1`, mapped to `cuda:0` inside the service container
- `docs/service-contract.md` now captures the public topology, health/capabilities handshake, render schema, sync/async responses, idempotency, errors, model controls, and frontend migration checklist
- the frontend shadow supplement was corrected: current `V2-DIFF` may use `elevation_deg` for a coarse prompt/view bucket and does not apply `opacity`

## Remaining Bootstrap Gaps

- current BiRefNet ONNX export is blocked in this environment by `torchvision::deform_conv2d`, so the live `segmenter` bridge currently depends on the temporary Triton Python backend
- `V2-DIFF` Triton backend is scaffolded but not implemented
- compatibility shims still remain in the repository
- ONNX Triton latency is not yet consistently faster than local inference; next work should profile ONNXRuntime execution provider placement, warmup, input resolution, and transport overhead
