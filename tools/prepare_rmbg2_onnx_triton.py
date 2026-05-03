from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "ops" / "triton" / "model_repository" / "shadowgen_segmenter_rmbg2" / "1" / "model.onnx"

ONNX_TO_TRITON_DTYPE = {
    1: "TYPE_FP32",
    10: "TYPE_FP16",
}


def _download(repo_id: str, filename: str, target: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install the project with the ml extra first.") from exc
    try:
        return Path(hf_hub_download(repo_id, filename, local_dir=target.parent / "_hf_download"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download {repo_id}/{filename}. RMBG-2.0 is gated; run `huggingface-cli login` "
            "or provide HF_TOKEN in the environment, then retry."
        ) from exc


def _tensor_shape(value) -> list[int | str]:
    shape = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append(-1)
    return shape


def _load_contract(model_path: Path) -> tuple[str, str, str, str]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required to inspect the downloaded model.") from exc
    model = onnx.load(str(model_path))
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"Expected one image input and one alpha output, got {len(inputs)} inputs and {len(outputs)} outputs")
    input_tensor = inputs[0]
    output_tensor = outputs[0]
    input_dtype = ONNX_TO_TRITON_DTYPE.get(input_tensor.type.tensor_type.elem_type)
    output_dtype = ONNX_TO_TRITON_DTYPE.get(output_tensor.type.tensor_type.elem_type)
    if input_dtype is None or output_dtype is None:
        raise RuntimeError(
            f"Unsupported ONNX dtypes: input={input_tensor.type.tensor_type.elem_type}, output={output_tensor.type.tensor_type.elem_type}"
        )
    input_shape = _tensor_shape(input_tensor)
    output_shape = _tensor_shape(output_tensor)
    if len(input_shape) != 4 or input_shape[1] != 3:
        raise RuntimeError(f"Expected NCHW RGB input, observed {input_tensor.name}: {input_shape}")
    if len(output_shape) != 4:
        raise RuntimeError(f"Expected NCHW alpha output, observed {output_tensor.name}: {output_shape}")
    return input_tensor.name, input_dtype, output_tensor.name, output_dtype


def _write_config(model_dir: Path, *, input_name: str, input_dtype: str, output_name: str, output_dtype: str) -> None:
    config = f'''name: "shadowgen_segmenter_rmbg2"
platform: "onnxruntime_onnx"
max_batch_size: 4
input [
  {{
    name: "{input_name}"
    data_type: {input_dtype}
    dims: [ 3, 1024, 1024 ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: {output_dtype}
    dims: [ 1, 1024, 1024 ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 2, 4 ]
  max_queue_delay_microseconds: 15000
}}
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
'''
    (model_dir / "config.pbtxt").write_text(config, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare gated BRIA RMBG-2.0 ONNX for the Triton model repository.")
    parser.add_argument("--repo-id", default="briaai/RMBG-2.0")
    parser.add_argument("--filename", default="onnx/model.onnx")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source", type=Path, default=None, help="Use an already downloaded ONNX file instead of Hugging Face download.")
    args = parser.parse_args(argv)

    target = args.target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve() if args.source else _download(args.repo_id, args.filename, target).resolve()
    shutil.copy2(source, target)
    input_name, input_dtype, output_name, output_dtype = _load_contract(target)
    _write_config(target.parents[1], input_name=input_name, input_dtype=input_dtype, output_name=output_name, output_dtype=output_dtype)
    print(
        "\n".join(
            [
                f"prepared_model={target}",
                f"config={target.parents[1] / 'config.pbtxt'}",
                f"input={input_name}:{input_dtype}",
                f"output={output_name}:{output_dtype}",
                "next=rebuild-triton.cmd",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
