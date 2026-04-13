# Triton Model Repository

This directory contains git-tracked Triton model repository scaffolds.

Current live-first model:

- `shadowgen_segmenter`

Current repository shape for `shadowgen_segmenter`:

- `ops/triton/model_repository/shadowgen_segmenter/config.pbtxt`
- `ops/triton/model_repository/shadowgen_segmenter/1/model.py`

Current operational path:

- temporary Triton `python` backend for BiRefNet
- model returns a mask-first tensor contract
- `cutout`, `crop`, and compatibility `bbox` remain ML-core postprocess responsibilities

Optional ONNX tooling is still tracked, but current BiRefNet export is blocked in this environment by `torchvision::deform_conv2d`.

Relevant files:

- `ops/triton/Dockerfile.segmenter-python`
- `tools/run_triton_segmenter_python.cmd`
- `tools/run_triton_segmenter_python.ps1`
- `tools/check_triton_segmenter_ready.py`
- `tools/export_segmenter_onnx.py`

`Dockerfile.segmenter-python` copies this repository into `/models` during image build. The run helper defaults to that baked-in model repository because Windows bind mounts from non-standard workspace drives can be unreliable. Use the helper's `-BindModelRepository` switch only when bind mounts are known to work in your Docker Desktop setup.

The helper maps Triton container ports to offset host ports so FastAPI can keep using local port `8000`.

Host ports used by the helper:

- `8010`: Triton HTTP API, used by ML-core (`SHADOWGEN_TRITON_URL=http://127.0.0.1:8010`)
- `8011`: Triton gRPC API
- `8012`: Triton metrics

The temporary Python backend uses `KIND_CPU` in `config.pbtxt` so Triton can load the model without Docker GPU runtime during local bring-up. When the launcher is run with `-Gpu`, the Python model code still chooses CUDA if PyTorch can see it.
