# Current Tasks

## Active

- [x] Bootstrap the service repository, API, workflow, and tracking
- [x] Build the layered architecture (`core/application/bootstrap/infrastructure/interfaces`)
- [x] Add browser playground for stage-by-stage testing
- [x] Bring up real and mock Geometry
- [x] Bring up real and mock Detection
- [x] Bring up real and mock Segmentation
- [x] Add standalone Foreground Refinement stage
- [x] Bring up real and mock Depth
- [x] Bring up real and fallback Normals
- [x] Add shadow generation backends:
  - [x] `mock`
  - [x] `V1-GAN`
  - [x] `V2-DIFF` scaffold
- [x] Refactor runtime to Triton-ready backend registry architecture
- [x] Correct the Triton-ready architecture:
  - [x] switch Triton transport to standard tensor infer payloads
  - [x] make `stage_io` the actual heavy-stage boundary
  - [x] tighten fallback and architecture tests
- [x] Harden Triton-ready execution boundaries:
  - [x] convert heavy-stage contracts to real `RasterAsset` payloads
  - [x] normalize stage runtime faults into debug-safe failed-stage executions
  - [x] validate Triton tensor schema rank/layout metadata
  - [x] derive compatibility `actual_mode` strictly from execution outcome
- [x] Add async render job API and in-memory job backend
- [x] Update project documentation to match the Triton-ready repository state
- [x] Document the worker <-> ML-core contract:
  - [x] define sync compatibility and async-native worker modes
  - [x] define capabilities handshake for batching and async support
  - [x] define responsibility split between worker concurrency and ML-core batching
- [x] Harden the ML core for optimal worker integration:
  - [x] add bounded concurrent async dispatcher with capacity snapshot
  - [x] add worker-facing readiness and submit-mode metadata
  - [x] add async idempotency via `request_id`
  - [x] add internal Triton stage micro-batching for heavy downstream stages

## Next Likely

- [ ] Implement working Triton execution for one heavy stage end-to-end against a real Triton server
- [ ] Implement the real `V2-DIFF` shadow backend
- [ ] Add shadow-stage input previews for `img`, `mask`, `depth`, and `normal`
- [ ] Continue replacing compatibility shims with direct imports from the layered structure
- [ ] Add Docker packaging and deployment-oriented runtime docs

## Notes

- Models, checkpoints, caches, debug artifacts, and generated outputs stay outside git tracking.
- `.models/` is the expected local home for heavy checkpoints, including the current `V1-GAN` shadow weights.
- Heavy stages now expose execution through `backend_kind` (`mock|local|triton`) plus `model_variant`.
