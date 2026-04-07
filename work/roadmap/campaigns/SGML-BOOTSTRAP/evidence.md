# SGML-BOOTSTRAP Evidence

## In Progress

- Repository initialized locally with branch `dev`
- Bootstrap tracking files created

## Captured Evidence

- `python -m compileall src tests` passed
- `python -m pytest` passed: `10 passed`
- FastAPI app exposes `/health`, `/v1/capabilities`, `/v1/render`
- Request contract updated with `preprocess.padding_px`

## Pending Follow-Ups

- Bring up real model wrappers against local NVIDIA environment
- Add Docker packaging and deployment docs
