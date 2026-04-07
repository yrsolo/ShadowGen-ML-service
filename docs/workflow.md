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
