# Workflow

## Branching

- Default working branch: `dev`
- Small reversible changes
- Tests for changed areas before commit

## Tracking

Update:

- `work/now/tasks.md`
- `work/roadmap/README.md`
- `work/roadmap/campaigns/SGML-BOOTSTRAP/evidence.md`

## Git hygiene

- Track only code, docs, configs, and tests
- Keep model weights, caches, images, and generated artifacts in ignored directories
- Do not commit secrets
- Use `var/cache`, `var/tmp`, `artifacts`, `.models`, and `data/*` for local-only runtime data

## Structure discipline

- Put business commands, models, and ports in `core/`
- Put orchestration only in `application/`
- Put model backends and technical persistence in `infrastructure/`
- Put FastAPI, schemas, and route handlers in `interfaces/http/`
- Put playground and debug presentation in `interfaces/dev/`
- Keep post-processing modules such as foreground colour refinement as standalone stages; do not bury them inside the segmenter implementation
- Keep legacy root or `pipeline/` modules as compatibility shims only, not as places for new logic
