# AGENTS.md

Operational rules for AI agents working in this repository.

> Primary runtime control document: `agent/OPERATING_CONTRACT.md`.

## Start Gate (mandatory)
Every agent session must start with:
1) read `agent/OPERATING_CONTRACT.md`
2) read `AGENTS.md`
3) confirm: `CONTRACT CHECK: OK`

If missing, execution/planning must not continue.

## Scope
- Applies to the whole repository unless overridden by a deeper `AGENTS.md`.
- Owner instruction in chat overrides this file.

## Working mode
- Default working branch: `dev`.
- Never push/merge to `main` without explicit owner approval.
- Keep changes small, reversible, testable.
- Do not delete or move large folders without explicit owner approval.

## Tracking
- keep local tracking:
  - `work/now/tasks.md`
  - `work/roadmap/README.md`
  - `work/roadmap/campaigns/<CAM>/{charter,plan,evidence}.md`


## Quality gate before commit
- No secret leaks.
- `npm test` / `npm run build` (or relevant) passes for changed areas.
- Docs updated when behavior/config/process changes.

## Clean architecture rules 
- SOLID
- DDD
- Hexagonal
