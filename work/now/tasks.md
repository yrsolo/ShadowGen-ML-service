# Current Tasks

## Active

- [x] Initialize git repository and branch `dev`
- [x] Create baseline repo structure and tracking files
- [x] Implement HTTP service and pipeline runtime
- [x] Add tests and run validation checks
- [x] Finalize docs and evidence for bootstrap campaign
- [x] Add mini web UI for pipeline stage testing
- [x] Implement real/mocked Geometry step wiring with GeoCalib adapter, fallback, and UI diagnostics
- [x] Activate GeoCalib in the local `.venv` and verify the real Geometry path end-to-end
- [x] Improve Geometry preview with floor-grid overlay and expose GeoCalib runtime settings in debug details
- [x] Implement real/mocked Detection step wiring with GroundingDINO, runtime fallback, and bbox debug previews
- [x] Fix playground stage-mode switching so `mock` really executes mock adapters for Geometry and Detection
- [x] Keep outer margins in the post-detection working crop so shadows have room instead of stretching the object to the full square
- [x] Implement stage 5 segmentation with real/mocked BiRefNet wiring, fallback behavior, and playground mask previews
- [x] Execute hard-cut architecture refactor into `core/application/infrastructure/interfaces/bootstrap`
- [x] Replace service-centered orchestration with use cases, stage runner, backend selector, preview registry, and composition root
- [x] Add architecture boundary tests so new code cannot leak FastAPI/Pydantic into core/application again
- [x] Normalize Python packaging so `pyproject.toml` carries the real non-torch ML dependencies and the local runbook documents the explicit CUDA `torch` install flow
- [x] Add a dedicated foreground colour refinement stage based on Fast Foreground Colour Estimation so semi-transparent edge colour correction is decoupled from the segmenter
- [x] Implement stage 06 `Depth` with real `Depth Anything V2` wiring and stage 07 `Normals` as a proper runtime-selectable module with mock/real behavior
- [x] Upgrade stage 07 `Normals` with a neural `StableNormal` backend while keeping the explicit `from-depth` fallback as a separate runtime module

## Notes

- Models, image inputs, caches, and generated artifacts stay outside git tracking via `.gitignore`.
