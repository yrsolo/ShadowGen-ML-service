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

## Notes

- Models, image inputs, caches, and generated artifacts stay outside git tracking via `.gitignore`.
