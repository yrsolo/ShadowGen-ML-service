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
- `tools/export_segmenter_onnx.py`
