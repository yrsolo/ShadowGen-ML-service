# SGML-BOOTSTRAP Evidence

## Scope Achieved

- repository initialized on `dev`
- layered architecture established
- service API implemented
- playground UI implemented
- multiple ML stages wired with mock and real paths
- project docs rewritten into a usable current-state reference set

## Captured Evidence

- `GET /health`, `GET /v1/capabilities`, `POST /v1/render` are implemented
- `GET /playground`, `POST /v1/dev/pipeline/run-all`, and `POST /v1/dev/pipeline/run-stage/{stage_key}` are implemented
- request contract supports `preprocess.padding_px`
- repository architecture is split into:
  - `core`
  - `application`
  - `bootstrap`
  - `infrastructure`
  - `interfaces`

### Stage Wiring Evidence

- `geometry_estimator`
  - GeoCalib real backend
  - mock backend
  - playground overlay and numeric details

- `detector`
  - GroundingDINO real backend
  - mock backend
  - bbox previews and working-crop preview

- `segmenter`
  - BiRefNet real backend
  - mock backend
  - working crop, mask, and cutout previews

- `foreground_refiner`
  - Fast Foreground Colour Estimation stage is standalone
  - no longer hidden inside segmentation

- `depth_estimator`
  - Depth Anything V2 real backend
  - mock backend

- `normal_estimator`
  - StableNormal real backend
  - `from-depth` fallback backend
  - mock backend

- `shadow_generator`
  - deterministic mock backend
  - `V1-GAN` backend migrated from legacy pix2pix inference
  - `V2-DIFF` scaffold class and runtime slot prepared

### Shadow-Specific Evidence

- local `V1-GAN` checkpoint stored under ignored `.models/shadow/AveragedModel.pth`
- real shadow backend successfully loaded on local GPU
- live smoke verified:
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
- API summary in `docs/api.md`
- repository workflow rules in `docs/workflow.md`

### Validation Evidence

- `py -3.11 -m pytest` passed: `56 passed`
- earlier live shadow smoke succeeded through the debug endpoint with real `V1-GAN`

## Remaining Bootstrap Gaps

- `V2-DIFF` shadow backend is scaffolded but not implemented
- compatibility shims still remain in the repository
- Docker/deployment documentation is not yet part of bootstrap
