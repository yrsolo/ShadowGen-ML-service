# Documentation Index

This folder is split into two kinds of material:

- active documentation for the current service
- historical bootstrap material preserved for reference

## Read In This Order

1. [README.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/README.md)
2. [architecture.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/architecture.md)
3. [modules.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/modules.md)
4. [worker-core-contract.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/worker-core-contract.md)
5. [runbook-local.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/runbook-local.md)
6. [api.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
7. [workflow.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/workflow.md)

## Active Docs

- [architecture.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/architecture.md)
  Control plane vs execution plane, layered structure, sync and async flows, Triton-ready stage boundaries.

- [modules.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/modules.md)
  Folder-by-folder explanation of the codebase, including the Triton subsystem and async job backend.

- [runbook-local.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/runbook-local.md)
  Local setup, GPU environment, startup commands, backend-kind settings, Triton settings, and troubleshooting.

- [worker-core-contract.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/worker-core-contract.md)
  Integration contract for `ShadowGen-v2` worker authors, including sync vs async modes, capability handshake, and batching boundaries.

- [api.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md)
  Public API, async jobs API, dev/playground request model, and execution metadata semantics.

- [workflow.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/workflow.md)
  Repository workflow, tracking expectations, and architectural placement rules.

## Historical Source Material

- [docs/first/breef.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/first/breef.md)
- [docs/first/overview.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/first/overview.md)
- [docs/first/models.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/first/models.md)
- [docs/first/contract_new-ml-service.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/first/contract_new-ml-service.md)

These are no longer the primary source of truth for the running service, but they remain useful for understanding the initial design intent and early model selection reasoning.
