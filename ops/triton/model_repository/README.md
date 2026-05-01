# Triton Model Repository

This directory contains git-tracked Triton model repository scaffolds.

Current live-first model:

- `shadowgen_segmenter`

Current repository shape for `shadowgen_segmenter`:

- `ops/triton/model_repository/shadowgen_segmenter/config.pbtxt`
- `ops/triton/model_repository/shadowgen_segmenter/1/model.py`

Current operational path:

- temporary Triton `python` backend for BiRefNet
- default model id: `ZhengPeng7/BiRefNet-matting`
- model returns a mask-first tensor contract
- `cutout`, `crop`, and compatibility `bbox` remain ML-core postprocess responsibilities

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

Default behavior:

- without `TRITON_GPU=1`: `device=cpu`, `resolution=512`
- with `TRITON_GPU=1`: `device=cuda`, `resolution=1024`
- HuggingFace cache is mounted from the host into `/root/.cache/huggingface`

Validation command:

```powershell
.venv\Scripts\python.exe tools\smoke_triton_segmenter.py --base-url http://127.0.0.1:8010 --image C:\Users\solofarm\Pictures\Screenshots\1.jpg
```
