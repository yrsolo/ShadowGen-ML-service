# ShadowGen ML Service

Stateless synchronous ML service for ShadowGen v2.

## What is in scope

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/render`
- Explicit request and response schema
- One final artifact per successful request
- Optional debug artifacts
- Mock and real adapter slots for every ML stage

## Local development

Primary runtime notes and workflow live in:

- `docs/README.md`
- `docs/workflow.md`
- `docs/runbook-local.md`

Quick start:

```powershell
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m uvicorn shadowgen_ml_service.main:app --reload
.venv/Scripts/python.exe -m pytest
```

Playground UI:

- open `http://127.0.0.1:8000/playground`

## Current status

This repository now contains the service skeleton, pipeline contracts, deterministic mock implementations, runtime adapter selection, tests, and primary project documentation.
