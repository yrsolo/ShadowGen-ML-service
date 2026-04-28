# Roadmap

## Current Campaigns

- `SGML-BOOTSTRAP`
  Build the repository, service architecture, local workflow, playground UI, and first working model stack.

  Current result inside this campaign:
  - layered service architecture is in place
  - runtime is now registry-based and Triton-ready
  - sync and async execution paths coexist behind the same orchestration model
  - Russian GitHub Pages style academic project page exists under `docs/project-page/` and links the product, ML service, dataset, and training repositories
  - heavy stages are prepared for `mock|local|triton`
  - segmenter has a temporary Triton Python backend launcher with local offset Triton ports
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
  Remove remaining compatibility shims and simplify import paths.

- `SGML-DEPLOY`
  Add Docker packaging, deployment docs, and environment hardening.

- `SGML-PERF`
  Profile runtime hot paths, batching opportunities, and cache behavior.
