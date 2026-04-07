# OPERATING CONTRACT (Runtime)

Purpose: single runtime control document for any agent session in this repository.

## 0) Start Gate (Mandatory)
Before any planning or coding, agent must:
1) Read this file (`agent/OPERATING_CONTRACT.md`).
2) Read `AGENTS.md`.
3) Confirm in first response: `CONTRACT CHECK: OK`.

If step 1-3 is not completed, agent must not continue execution.

## 1) Priority Order
1. Direct owner instruction in current chat.
2. This operating contract.
3. `AGENTS.md`.
4. Other docs (`docs/README.md`, `work/*`, `README.md`).

## 2) Task tracking
execution lifecycle must be tracked in:
- `work/now/tasks.md`
- `work/roadmap/README.md`

## 3) No-parallel rule
Work strictly sequentially (one active execution task).

## 4) Iteration report
After each meaningful iteration, report:
- Status (in progress/blocked/done)
- Ready to commit (yes/no)
- Proposed commit message
- Ready for main (yes/no)
- Docs status (updated/not needed)
- Tracking (done/blocked)

If report says `Ready to commit: yes` and there is no explicit owner instruction to avoid commits,
agent should create the commit in the current branch instead of waiting for a separate commit request.

## 5) Safety
Never print/expose secrets from env or tokens.
