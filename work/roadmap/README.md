# Roadmap

## Current Campaigns

- `SGML-BOOTSTRAP`: bootstrap the new ML service repository, API contract, local workflow, and first executable pipeline skeleton.
  - hard-cut architecture refactor completed inside this campaign to establish the long-term layered service structure before more real-model stages land
  - packaging/runbook normalized so CUDA `torch` installation is explicit and `pyproject.toml` only owns the project and non-torch ML stack
  - foreground colour refinement is now a standalone post-segmentation stage instead of being hidden inside matting backends

## Next Likely Campaigns

- Docker packaging and container runtime
- Real model bring-up on target NVIDIA hardware
- Worker adapter integration
- Performance profiling and cache tuning
