# Triton Model Repository

This directory contains git-tracked Triton model repository scaffolds.

Current live-first model:

- `shadowgen_segmenter`

Export the ONNX model into:

- `ops/triton/model_repository/shadowgen_segmenter/1/model.onnx`

Recommended command:

```powershell
.venv\Scripts\python.exe tools\export_segmenter_onnx.py
```
