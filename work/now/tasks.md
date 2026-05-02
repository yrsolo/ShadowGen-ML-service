# Current Tasks

## Active

- [x] Add a Russian GitHub Pages style academic project page with product, service, dataset, and training repository links
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

- [x] Bring up live Triton detector path for `detector`:
  - [x] add `shadowgen_detector` Triton Python backend scaffold
  - [x] wire local launcher to request detector through Triton
  - [x] add direct detector Triton smoke helper
  - [x] rebuild image and validate live detector inference
## Next Likely

- [x] Switch `segmenter` to a temporary Triton Python backend while BiRefNet ONNX export remains blocked
- [x] Add opt-in Torch runtime acceleration knobs for BiRefNet (`torch.compile`, matmul precision)
- [x] Add live Triton bring-up helpers and readiness checks for the temporary Python backend
- [x] Fix Triton local helper ports and add a Windows `.cmd` launcher
- [x] Add a Windows service launcher that starts FastAPI with Triton segmenter defaults
- [x] Run the first live end-to-end Triton smoke against a real `shadowgen_segmenter` server
- [x] Implement the real local `V2-DIFF` shadow backend
- [x] Fix V2-DIFF inpaint mask semantics so the model edits background/shadow area instead of the object
- [ ] Add shadow-stage input previews for `img`, `mask`, `depth`, and `normal`
- [x] Rework the playground into a horizontal stage scroller with wheel-driven horizontal navigation, shift-wheel vertical card scroll, and compact desktop controls
- [x] Prevent tablet playground stage area from collapsing by giving the horizontal pipeline a minimum height and page-level vertical scroll
- [x] Document the training, export, serving, and acceptance contract for the future `V2-DIFF` shadow model
- [x] Profile and reduce local `V2-DIFF` shadow latency by lowering the default denoising step count
- [x] Connect the accelerated `V2-DIFF` production bundle with fast LCM inference defaults
- [x] Disable the unused Geometry stage by default and hide it from the playground flow
- [x] Make `from-depth-v2` the default normals variant and remove the expensive inpaint step from depth-derived normals
- [x] Prepare a reproducible Shadow V2 sample-pack generator and local 10-image sample pack
- [x] Replace unsuitable Shadow V2 sample-pack sources with product-case foreground objects and add pack README
- [x] Temporarily simplify `V2-DIFF` shadow integration to a control-free `img + mask -> shadow_image` contract while keeping `V1-GAN` as the controllable rot/top-view model
- [x] Switch shadow stage output contract from separate `shadow_rgba` layer to full `shadow_image`
- [x] Add public `shadow.model` contract and frontend-facing model selection documentation
- [x] Clamp BiRefNet cutout alpha by the cleaned foreground mask so shadow inputs do not include matting-alpha leaks
- [x] Switch default BiRefNet segmenter from lite matting to the quality matting checkpoint
- [x] Restore larger default working-canvas margins for shadow generation by lowering `SHADOWGEN_WORKING_CONTENT_SCALE` to `0.68`
- [ ] Continue replacing compatibility shims with direct imports from the layered structure
- [x] Extend the temporary Triton segmenter bridge with a verified custom image smoke run
- [x] Add visible Windows FastAPI launchers and a Playground shutdown control for the current ML-service process
- [x] Simplify local operations to two root launch scripts:
  - [x] `rebuild-triton.cmd`
  - [x] `start-service.cmd`
- [x] Switch the local Triton segmenter launch path to GPU-first defaults and verify CUDA serving

## Notes

- Models, checkpoints, caches, debug artifacts, and generated outputs stay outside git tracking.
- `.models/` is the expected local home for heavy checkpoints, including the current `V1-GAN` shadow weights.
- Heavy stages now expose execution through `backend_kind` (`mock|local|triton`) plus `model_variant`.
