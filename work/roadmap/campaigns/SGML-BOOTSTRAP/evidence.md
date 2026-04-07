# SGML-BOOTSTRAP Evidence

## In Progress

- Repository initialized locally with branch `dev`
- Bootstrap tracking files created

## Captured Evidence

- `python -m compileall src tests` passed
- `python -m pytest` passed: `13 passed`
- FastAPI app exposes `/health`, `/v1/capabilities`, `/v1/render`
- FastAPI app exposes `/playground`, `/v1/dev/pipeline/run-all`, `/v1/dev/pipeline/run-stage/{stage_key}`
- Request contract updated with `preprocess.padding_px`
- Browser playground added for stage-by-stage testing with previews and `mock/real` mode switches

## Pending Follow-Ups

- Bring up real model wrappers against local NVIDIA environment
- Add Docker packaging and deployment docs
