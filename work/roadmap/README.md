# Roadmap

## Current Campaigns

- `SGML-BOOTSTRAP`: bootstrap the new ML service repository, API contract, local workflow, and first executable pipeline skeleton.
  - hard-cut architecture refactor completed inside this campaign to establish the long-term layered service structure before more real-model stages land
  - packaging/runbook normalized so CUDA `torch` installation is explicit and `pyproject.toml` only owns the project and non-torch ML stack
  - foreground colour refinement is now a standalone post-segmentation stage instead of being hidden inside matting backends
  - depth estimation and normals are now first-class pipeline stages with runtime selection, previews, and diagnostics
  - `normal_estimator` now prefers a neural `StableNormal` backend and falls back to an explicit `from-depth` backend instead of hiding that logic inside depth

## Next Likely Campaigns

- Docker packaging and container runtime
- Real model bring-up on target NVIDIA hardware
- Worker adapter integration
- Performance profiling and cache tuning
