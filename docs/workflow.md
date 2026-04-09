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
  internal contracts, commands, typed models, service errors

- `application/`
  orchestration, stage runner, backend selection, pipeline state

- `bootstrap/`
  runtime wiring, probes, capability assembly

- `infrastructure/`
  concrete model adapters, cache, encoding, preview logic

- `interfaces/http/`
  FastAPI routes, schemas, HTTP mappers

- `interfaces/dev/`
  playground UI

### Do not put new logic here

- `pipeline/`
- `adapters/`
- root compatibility exports like `web_ui.py` and `schemas.py`

These are transition shims, not the architectural center.

## Stage Design Rules

- each stage should have a clear interface
- each stage should have explicit mock and real behavior when practical
- stage-specific logic should live in that stage package
- do not bury one stage’s responsibility inside another

Examples:

- foreground colour correction is a standalone stage, not segmentation internals
- normals have their own stage and fallback path
- shadow model families are explicit named variants, not a vague single `real`

## Model Integration Rules

When integrating a new model:

1. add or update contracts only if the public or internal stage interface truly changed
2. add the model inside the correct `infrastructure/stages/<stage>/` package
3. keep training code out of this repo unless explicitly needed
4. import only the minimum inference path from legacy projects
5. put weights in ignored local storage such as `.models/`
6. wire the backend in `bootstrap/container.py`
7. expose useful runtime metadata in capabilities and debug UI
8. add tests
9. update docs

## Shadow-Specific Rule

For shadow generation:

- keep model families explicit:
  - `mock`
  - `V1-GAN`
  - `V2-DIFF`
- real models should consume `softness` as model input
- coarse post-blur softness is allowed only in the mock shadow backend

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

After changing architecture, module ownership, model lineup, runtime behavior, or workflow:

- update [README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/README.md)
- update [docs/README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/README.md)
- update the relevant detailed docs

The docs should stay useful for:

- a fast first read
- a teammate onboarding into the repo
- a future refactor that needs to understand the current boundaries
