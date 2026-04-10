# Workflow

## Branching

- default branch for active work: `dev`
- do not merge to `main` without explicit approval
- keep changes small, testable, and reversible

## Tracking

Keep these files updated during meaningful work:

- [work/now/tasks.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/work/now/tasks.md)
- [work/roadmap/README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/work/roadmap/README.md)
- [evidence.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/work/roadmap/campaigns/SGML-BOOTSTRAP/evidence.md)

## Commit Discipline

Before commit:

- run tests for changed areas
- update docs if behavior changed
- do not commit secrets
- do not commit model weights, caches, or generated artifacts

## Folder Discipline

### Put code here

- `core/`
  internal contracts, canonical stage I/O, typed models, job contracts, service errors

- `application/`
  orchestration, stage runner, backend selection, pipeline state, async jobs

- `bootstrap/`
  runtime wiring, probes, backend registry defaults, capability assembly

- `infrastructure/`
  concrete stage adapters, Triton subsystem, cache, encoding, preview logic, job backends

- `interfaces/http/`
  FastAPI routes, schemas, HTTP mappers

- `interfaces/dev/`
  playground UI

### Do not put new logic here

- `pipeline/`
- `adapters/`
- root compatibility exports like `web_ui.py` and `schemas.py`

These are transition shims, not the architectural center.

## Execution Design Rules

- heavy stages must be designed around `backend_kind`, not vague `real`
- supported backend kinds are:
  - `mock`
  - `local`
  - `triton`
- model family selection is separate from execution kind
- a new backend must expose machine-readable metadata:
  - model variant
  - model name
  - model version
  - device
  - endpoint when relevant
  - batching and async support flags

## Stage Design Rules

- each stage should have a clear interface
- stage-specific logic should live in that stage package
- one stage must not silently own another stage's responsibility
- heavy stages should accept canonical inputs suitable for local or Triton execution

Examples:

- foreground colour correction is a standalone stage, not segmentation internals
- normals have their own stage and explicit fallback path
- shadow model families are explicit variants, not a vague single `real`

## Model Integration Rules

When integrating a new model:

1. add or update contracts only if the public or internal stage interface truly changed
2. add the backend inside the correct `infrastructure/stages/<stage>/` package
3. keep training code out of this repo unless explicitly needed
4. import only the minimum inference path from legacy projects
5. put weights in ignored local storage such as `.models/`
6. wire the backend in `bootstrap/container.py`
7. expose useful runtime metadata in capabilities and debug UI
8. add tests
9. update docs

If the backend is remote:

1. use the shared Triton subsystem
2. keep transport details out of `application/`
3. register the backend in the runtime registry
4. expose fallback behavior explicitly

## Shadow-Specific Rule

For shadow generation:

- keep model families explicit:
  - `mock`
  - `V1-GAN`
  - `V2-DIFF`
- real backends should consume `softness` as model input
- coarse post-blur softness is allowed only in the mock shadow backend

## Async Execution Rule

- sync render remains the primary debug-friendly path
- async jobs must reuse the same stage runner and backend registry
- async infrastructure must stay replaceable
- job persistence and queue logic belong under `infrastructure/jobs/`

## Local-Only Data

Use ignored directories for local runtime data:

- `.models/`
- `.cache/`
- `var/cache/`
- `var/tmp/`
- `artifacts/`
- `data/input/`
- `data/output/`
- `data/debug/`

## Documentation Rule

After changing architecture, execution model, module ownership, model lineup, runtime behavior, or workflow:

- update [README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/README.md)
- update [docs/README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/README.md)
- update the relevant detailed docs

The docs should stay useful for:

- a fast first read
- teammate onboarding
- future backend migration, especially to Triton
