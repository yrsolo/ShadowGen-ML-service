# SGML-BOOTSTRAP Evidence

## In Progress

- Repository initialized locally with branch `dev`
- Bootstrap tracking files created

## Captured Evidence

- `python -m compileall src tests` passed
- `python -m pytest` passed: `26 passed`
- FastAPI app exposes `/health`, `/v1/capabilities`, `/v1/render`
- FastAPI app exposes `/playground`, `/v1/dev/pipeline/run-all`, `/v1/dev/pipeline/run-stage/{stage_key}`
- Request contract updated with `preprocess.padding_px`
- Browser playground added for stage-by-stage testing with previews and `mock/real` mode switches
- `geometry_estimator` now has a real GeoCalib adapter path, runtime fallback behavior, and stage-level debug metadata
- Playground `Geometry` card now shows numeric camera data and `geometry_overlay` preview
- GeoCalib was installed into the active `.venv` and `GET /v1/capabilities` now reports `geometry_estimator` with `implementation=real`
- Real geometry smoke path now runs end-to-end with preview overlays and numeric camera metadata
- Geometry overlay now includes a synthetic floor grid, and debug metadata exposes active GeoCalib runtime settings (`weights`, `camera_model`, `shared_intrinsics`)
- GroundingDINO was installed into the active `.venv` and `GET /v1/capabilities` now reports `detector` with `implementation=real`
- Real detector smoke path now runs end-to-end with bbox details plus `detection_overlay` and `crop_for_resize` previews
- Debug/playground stage overrides now execute real mock adapters for `detector` and `geometry_estimator`, instead of only relabeling the active backend
- The prepared working crop now preserves outer margins via `SHADOWGEN_WORKING_CONTENT_SCALE`, so the detected object no longer fills the entire square and shadow projection has room
- Stage 5 segmentation is now wired with a real `ZhengPeng7/BiRefNet_lite-matting` adapter, deterministic mock fallback, and playground previews for `working_crop`, `mask`, and `cutout`
- On this workstation, BiRefNet real mode is intentionally kept on mock by default because CUDA is unavailable; CPU execution requires explicit opt-in through `SHADOWGEN_BIREFNET_ALLOW_CPU=true`
- `python -m pytest` passed: `34 passed`
- Hard-cut architecture refactor completed:
  - `core/`, `application/`, `infrastructure/`, `interfaces/`, and `bootstrap/` now exist as the primary code layout
  - render/debug orchestration moved out of the old service module into dedicated use cases
  - stage adapters are split per stage package instead of living in multi-stage `real.py` / `mock.py` implementations
  - HTTP schemas/routes and playground UI are separated from pipeline internals
  - architecture boundary tests now enforce clean-layer imports
- `python -m pytest` passed after the architecture cut: `37 passed`

## Pending Follow-Ups

- Bring up the remaining real model wrappers against local NVIDIA environment
- Add Docker packaging and deployment docs
