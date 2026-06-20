# Roadmap

## Current Campaigns

- `CODEX-PET`
  Hatch a local Codex-compatible animated pet package for this workstation.

- `SGML-BOOTSTRAP`
  Build the repository, service architecture, local workflow, playground UI, and first working model stack.

  Current result inside this campaign:
  - layered service architecture is in place
  - architecture audit remediation tightened dev API exposure, dependency boundaries, fallback policy, shared stage invocation, settings loading, and imaging utility ownership
  - runtime is now registry-based and Triton-ready
  - sync and async execution paths coexist behind the same orchestration model
  - Russian GitHub Pages style academic project page exists under `docs/project-page/` and links the product, ML service, dataset, and training repositories
  - project page includes generated architecture illustrations with reproducible prompts and interactive media tabs
  - heavy stages are prepared for `mock|local|triton`
  - detector has a temporary Triton Python backend launcher using GroundingDINO
  - segmenter has a temporary Triton Python backend launcher with local offset Triton ports
  - Triton transport now defaults to native binary tensor payloads, with JSON kept only as a debug fallback
  - segmenter has been live-smoked through Triton on `http://127.0.0.1:8010`
  - `grounding-dino-onnx` is wired as a live experimental Triton ONNX detector variant with ML-core postprocess
  - `rmbg-2.0` is wired as a live experimental Triton ONNX segmenter variant after gated Hugging Face weights preparation
  - recommended production replacement launch is a service-only Docker container through `rebuild-service-container.cmd`, `start-service-container.cmd`, and `stop-service-container.cmd`
  - production container GPU selection is configured through `.env` with `SERVICE_GPU_DEVICE`; the selected host GPU is exposed as `cuda:0` inside the container
  - production container HTTP port is configured through `.env` with `SERVICE_HTTP_PORT`; default is `9001`
  - `docs/service-contract.md` is the authoritative frontend/backend/worker integration handoff
  - optional Triton/debug launch is a two-container Docker stack through `rebuild-triton.cmd`, `rebuild-service-container.cmd`, `start-docker-stack.cmd`, and `stop-docker-stack.cmd`
  - advanced split debug launch remains available through `start-triton.cmd` and `start-service.cmd`
  - playground uses horizontal stage navigation with per-card vertical scroll
  - shadow stage is prepared for named model generations:
    - `mock`
    - `V1-GAN` as the current controllable rot/top-view model
    - `V2-DIFF` as the current control-free `img + mask -> shadow_image` Triton slot
  - documentation now has both a fast overview and deeper module-level reference

## Next Likely Campaigns

- `SGML-TRITON-BRINGUP`
  Connect one or more heavy stages to a real Triton Inference Server and harden transport behavior.

- `SGML-SHADOW-V2`
  Implement the real `V2-DIFF` shadow backend and add stage-specific debug inputs/outputs.

- `SGML-CLEANUP`
  Remove remaining compatibility shims, restore dependency boundaries, and simplify import paths.

- `SGML-DEPLOY`
  Harden Docker packaging, production environment defaults, and deployment observability.

- `SGML-PERF`
  Profile runtime hot paths, batching opportunities, and cache behavior.
