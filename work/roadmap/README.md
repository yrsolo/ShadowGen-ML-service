# Roadmap

## Current Campaigns

- `SGML-BOOTSTRAP`
  Build the repository, service architecture, local workflow, playground UI, and first working model stack.

  Current result inside this campaign:
  - layered service architecture is in place
  - major pipeline stages already exist with runtime wiring
  - shadow stage is prepared for named model generations:
    - `mock`
    - `V1-GAN`
    - `V2-DIFF`
  - documentation now has both a fast overview and deeper module-level reference

## Next Likely Campaigns

- `SGML-SHADOW-V2`
  Implement the real `V2-DIFF` shadow backend and add stage-specific debug inputs/outputs.

- `SGML-CLEANUP`
  Remove remaining compatibility shims and simplify import paths.

- `SGML-DEPLOY`
  Add Docker packaging, deployment docs, and environment hardening.

- `SGML-PERF`
  Profile runtime hot paths and tune cache and model initialization behavior.
