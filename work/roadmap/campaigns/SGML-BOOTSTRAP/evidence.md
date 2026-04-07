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

## Pending Follow-Ups

- Bring up the remaining real model wrappers against local NVIDIA environment
- Add Docker packaging and deployment docs
