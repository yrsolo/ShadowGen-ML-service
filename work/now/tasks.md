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
- [x] Bring up the first live Triton stage path for `segmenter`:
  - [x] define ONNX-first segmenter contract
  - [x] add Triton model repository scaffold for `shadowgen_segmenter`
  - [x] add BiRefNet ONNX export tool
  - [x] switch segmenter Triton adapter to live mask-first postprocess flow

## Next Likely

- [x] Switch `segmenter` to a temporary Triton Python backend while BiRefNet ONNX export remains blocked
- [x] Add opt-in Torch runtime acceleration knobs for BiRefNet (`torch.compile`, matmul precision)
- [x] Add live Triton bring-up helpers and readiness checks for the temporary Python backend
- [x] Fix Triton local helper ports and add a Windows `.cmd` launcher
- [x] Add a Windows service launcher that starts FastAPI with Triton segmenter defaults
- [ ] Run the first live end-to-end Triton smoke against a real `shadowgen_segmenter` server
- [ ] Implement the real `V2-DIFF` shadow backend
- [ ] Add shadow-stage input previews for `img`, `mask`, `depth`, and `normal`
- [x] Rework the playground into a horizontal stage scroller with wheel-driven horizontal navigation, shift-wheel vertical card scroll, and compact desktop controls
- [x] Document the training, export, serving, and acceptance contract for the future `V2-DIFF` shadow model
- [x] Prepare a reproducible Shadow V2 sample-pack generator and local 10-image sample pack
- [x] Replace unsuitable Shadow V2 sample-pack sources with product-case foreground objects and add pack README
- [ ] Continue replacing compatibility shims with direct imports from the layered structure
- [ ] Extend the temporary Triton segmenter bridge with a verified custom image smoke run

## Notes

- Models, checkpoints, caches, debug artifacts, and generated outputs stay outside git tracking.
- `.models/` is the expected local home for heavy checkpoints, including the current `V1-GAN` shadow weights.
- Heavy stages now expose execution through `backend_kind` (`mock|local|triton`) plus `model_variant`.
