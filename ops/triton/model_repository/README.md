# Triton Model Repository

This directory contains git-tracked Triton model repository scaffolds.

Current live-first models:

- `shadowgen_segmenter`
- `shadowgen_detector`

Current repository shape:

- `ops/triton/model_repository/shadowgen_segmenter/config.pbtxt`
- `ops/triton/model_repository/shadowgen_segmenter/1/model.py`
- `ops/triton/model_repository/shadowgen_detector/config.pbtxt`
- `ops/triton/model_repository/shadowgen_detector/1/model.py`

Current operational path:

- temporary Triton `python` backend for BiRefNet
- temporary Triton `python` backend for GroundingDINO
- default model id: `ZhengPeng7/BiRefNet-matting`
- default detector model id: `IDEA-Research/grounding-dino-base`
- model returns a mask-first tensor contract
- `cutout`, `crop`, and compatibility `bbox` remain ML-core postprocess responsibilities
- detector returns bbox/confidence tensors; detection overlay and crop remain ML-core responsibilities

Optional ONNX tooling is still tracked, but current BiRefNet export is blocked in this environment by `torchvision::deform_conv2d`.

Relevant files:

- `ops/triton/Dockerfile.segmenter-python`
- `rebuild-triton.cmd`
- `start-service.cmd`
- `tools/check_triton_segmenter_ready.py`
- `tools/smoke_triton_segmenter.py`
- `tools/export_segmenter_onnx.py`

`Dockerfile.segmenter-python` copies this repository into `/models` during image build. `rebuild-triton.cmd` is the canonical way to refresh that baked-in model repository after code changes.

`start-service.cmd` maps Triton container ports to offset host ports so FastAPI can keep using local port `8000`.

Host ports used by the helper:

- `8010`: Triton HTTP API, used by ML-core (`SHADOWGEN_TRITON_URL=http://127.0.0.1:8010`)
- `8011`: Triton gRPC API
- `8012`: Triton metrics

The temporary Python backend uses `KIND_CPU` in `config.pbtxt` so Triton can load the model without Docker GPU runtime during local bring-up. When `start-service.cmd` is run with `TRITON_GPU=1`, the Python model code still chooses CUDA if PyTorch can see it.

The launcher also provides runtime overrides through environment variables consumed by `model.py`:

- `SHADOWGEN_TRITON_SEGMENTER_MODEL_ID`
- `SHADOWGEN_TRITON_SEGMENTER_RESOLUTION`
- `SHADOWGEN_TRITON_SEGMENTER_DEVICE`
- `SHADOWGEN_TRITON_DETECTOR_MODEL_ID`
- `SHADOWGEN_TRITON_DETECTOR_PROMPT`
- `SHADOWGEN_TRITON_DETECTOR_BOX_THRESHOLD`
- `SHADOWGEN_TRITON_DETECTOR_TEXT_THRESHOLD`
- `SHADOWGEN_TRITON_DETECTOR_DEVICE`

Default `start-service.cmd` behavior:

- `TRITON_GPU=1`
- `device=cuda:0`
- `resolution=512`
- `compile_enabled=false`
- HuggingFace cache is mounted from the host into `/root/.cache/huggingface`

For CPU-only bring-up, set `TRITON_GPU=0` before running `start-service.cmd`. CPU mode is intentionally slow and should not be used for performance checks.

Validation command:

```powershell
.venv\Scripts\python.exe tools\smoke_triton_detector.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
```

Readiness checks:

```powershell
.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010 shadowgen_segmenter
.venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:8010 shadowgen_detector
```
