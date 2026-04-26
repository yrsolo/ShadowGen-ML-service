# Shadow V2-DIFF Model Contract

This document defines the technical requirements for the next-generation shadow model used by the `shadow_generator` stage.

Current integration decision:

- `V1-GAN` remains the controllable local model for top-view/rotated shadows.
- `V2-DIFF` is temporarily simplified to a control-free diffusion slot.
- Current `V2-DIFF` inference input is only `img + mask`.
- `angle`, `elevation`, `softness`, `reflection`, `depth`, and `normal` remain in the wider pipeline/sample-pack contract for future controlled checkpoints, but the current `V2-DIFF` backend must ignore them.

The target model variant is:

- stage key: `shadow_generator`
- model variant: `v2-diff`
- primary execution backend: `triton`
- fallback during development: `local` or `mock`

The document is intentionally written as a training, export, and serving contract. A model checkpoint is not considered integration-ready until it satisfies this contract or explicitly documents an approved deviation.

## Goal

The current `V2-DIFF` model generates a plausible standalone shadow layer for a foreground object already extracted from the source image.

The current model must condition on:

- object appearance
- object alpha mask

The full controlled model is deferred. A later checkpoint may additionally learn to condition on:

- estimated depth
- estimated normals
- light azimuth
- light elevation
- softness
- optional reflection control

The output is not the final composed image. The output is a standalone shadow layer that the ML core compositor can place under the cutout.

## Non-Goals

The model must not own:

- object detection
- segmentation
- foreground colour refinement
- final background composition
- output image encoding
- request parsing or API validation

Those responsibilities remain in the ML core pipeline.

## Canonical Input Contract

The ML core passes the shadow model a canonical working-canvas crop after detection, crop, pad, resize, segmentation, foreground refinement, depth, and normals.

All image-like inputs must share the same spatial size.

Recommended first production size:

- `512 x 512`

The model implementation should avoid hard-coding this size internally unless the exported artifact clearly declares the supported shape.

### Tensor Inputs

| Name | Required | Shape | Dtype | Range | Layout | Meaning |
| --- | --- | --- | --- | --- | --- | --- |
| `img` | yes | `[N, 4, H, W]` | `FP32` | `0..1` | `NCHW` | Refined RGBA object cutout on transparent background. |
| `mask` | yes | `[N, 1, H, W]` | `FP32` | `0..1` | `NCHW` | Foreground alpha/object mask. |
| `depth` | future | `[N, 1, H, W]` | `FP32` | `0..1` | `NCHW` | Relative depth map on the canonical crop. Not required by the current control-free `V2-DIFF`. |
| `normal` | future | `[N, 3, H, W]` | `FP32` | `0..1` | `NCHW` | Normal map encoded as RGB-like XYZ remap. Not required by the current control-free `V2-DIFF`. |
| `angle` | future | `[N]` or `[N, 1]` | `FP32` | degrees | scalar | Light azimuth in degrees. Current `V2-DIFF` ignores it; `V1-GAN` uses it for rot/top-view shadow generation. |
| `elevation` | future | `[N]` or `[N, 1]` | `FP32` | degrees | scalar | Light elevation above horizon in degrees. Current `V2-DIFF` ignores it. |
| `softness` | future | `[N]` or `[N, 1]` | `FP32` | `0..1` | scalar | Learned shadow edge softness control for a future controlled checkpoint. Current `V2-DIFF` ignores it. |
| `reflection` | future | `[N]` or `[N, 1]` | `FP32` | `0..1` | scalar | Optional glossy/reflection control for a future controlled checkpoint. Current `V2-DIFF` ignores it. |

### Angle Semantics

`angle` is an azimuth in degrees in the product contract.

Training data and inference code must use one documented convention and keep it stable. The recommended convention is:

- `0` means light comes from the right side of the image
- `90` means light comes from the top side of the image
- values increase counter-clockwise in image coordinates
- valid public range is `0..360`

If the training pipeline uses a different convention internally, the conversion must happen before export or inside the model adapter, not in the product API.

### Elevation Semantics

`elevation` is the light height above the horizon in degrees.

Recommended range:

- minimum: `5`
- maximum: `85`

Expected behavior:

- lower elevation produces longer shadows
- higher elevation produces shorter, more compact shadows

### Softness Semantics

`softness` is not active in the current control-free `V2-DIFF` model. It remains reserved for a future controlled checkpoint.

Important:

- a future controlled `v2-diff` model must not rely on a post-model Gaussian blur to implement softness
- coarse post-blur is allowed only in the `mock` backend
- `softness=0` should produce the sharpest supported shadow
- `softness=1` should produce the softest supported shadow

### Reflection Semantics

`reflection` is reserved for reflective or glossy-floor behavior.

The current control-free checkpoint ignores this value. A future controlled checkpoint must:

- keep output stable when reflection is unused
- document whether reflection is active

## Output Contract

The model must produce a standalone shadow tensor.

| Name | Required | Shape | Dtype | Range | Layout | Meaning |
| --- | --- | --- | --- | --- | --- | --- |
| `shadow` | yes | `[N, 4, H, W]` | `FP32` | `0..1` | `NCHW` | RGBA shadow layer aligned to the canonical working canvas. |

Recommended channel semantics:

- `R,G,B`: shadow colour/tint in straight alpha convention
- `A`: shadow opacity mask

The compositor may use the alpha channel as the primary opacity signal. If RGB is not meaningful in the first checkpoint, RGB should still be valid and preferably represent black or softly tinted shadow colour.

The output must:

- be spatially aligned with all inputs
- keep transparent regions numerically stable
- not include the foreground object
- not include the final background
- not apply final product opacity; global `opacity` remains a compositor-level control

## Training Data Requirements

Each training sample for the current control-free checkpoint should contain, or be reproducibly converted into:

- foreground RGBA object crop
- foreground mask
- target shadow RGBA or target shadow alpha plus colour policy

The sample pack may additionally contain future-control fields:

- depth map in the same crop coordinates
- normal map in the same crop coordinates
- light azimuth metadata
- light elevation metadata
- softness metadata or estimated softness label
- reflection metadata, even if always `0` initially

Dataset samples must be normalized into the same canonical crop contract used by inference.

### Dataset Quality Gates

Training data should be rejected or quarantined when:

- foreground mask and target shadow are misaligned
- object crop has inconsistent scale or padding
- light labels are missing or visibly wrong
- target shadow includes the object body
- target shadow is baked into the background and cannot be separated reliably
- alpha values are clipped too aggressively for soft edges

### Recommended Dataset Splits

Use object-level and scene-level separation:

- train: `80%`
- validation: `10%`
- test: `10%`

The same object instance should not appear in both train and test unless the explicit goal is same-object relighting.

## Training Behavior Requirements

The current model does not need to be controllable. It should produce plausible shadows from object appearance and mask only.

The future controlled model must be controllable.

Validation should include controlled sweeps where only one input changes:

- same object, same depth/normal, different `angle`
- same object, same angle, different `elevation`
- same object, same angle/elevation, different `softness`
- same object, same controls, small mask perturbations

Expected behavior:

- changing `angle` changes shadow direction
- changing `elevation` changes shadow length and placement
- changing `softness` changes penumbra softness without simply blurring the whole output
- transparent and semi-transparent foreground regions produce believable shadow transitions
- output remains stable for thin structures, ears, legs, handles, fur, and other silhouette details

## Recommended Losses and Metrics

Exact architecture is not fixed by this contract, but the training objective should measure both pixel quality and control fidelity.

Recommended losses:

- masked L1 or Charbonnier loss on shadow alpha
- perceptual loss on RGB/tint when RGB shadow colour is meaningful
- gradient or edge-aware loss for contact shadow and penumbra boundaries
- direction consistency loss or auxiliary evaluator for `angle`
- length/scale consistency loss or evaluator for `elevation`
- optional adversarial/diffusion objective for realism

Recommended validation metrics:

- alpha MAE on shadow region
- soft-edge IoU or boundary F-score
- direction error in degrees
- shadow extent error
- contact-region error near object footprint
- LPIPS or similar perceptual metric for RGB shadow layer if applicable
- latency and GPU memory at target batch sizes

## Inference Requirements

The model must support:

- batch size `1`
- batch size `4` as the first Triton dynamic batching target
- deterministic or bounded-stochastic inference mode for production
- stable output for repeated identical inputs when seed/config is fixed

Recommended latency target for first integration:

- batch `1`: less than `1000 ms` on target local GPU
- batch `4`: better throughput than four serial batch-1 calls

These latency numbers are acceptance targets, not hard blockers for early research checkpoints.

## Triton Serving Contract

The long-term serving target is Triton using the standard tensor infer protocol.

Preferred export order:

1. `ONNX` when the architecture supports clean export
2. `TensorRT` optimization after ONNX behavior is stable
3. Triton Python backend only when export is blocked or for research bring-up

The Triton model repository entry should use:

- model name: `shadowgen_shadow_v2`
- model variant: `v2-diff`
- input tensor names exactly matching this contract
- output tensor name: `shadow`
- `max_batch_size`: `4` for first production integration
- dynamic batching enabled with a short queue delay

The ML core currently expects this `V2-DIFF` binding shape:

```text
inputs:
  img:        FP32 [N, 4, H, W]
  mask:       FP32 [N, 1, H, W]

outputs:
  shadow:     FP32 [N, 4, H, W]
```

A future controlled `V2-DIFF` may extend the binding with `depth`, `normal`, `angle`, `elevation`, `softness`, and `reflection`. The Triton adapter sends only tensors declared by the active binding, so this extension does not require a public API change.

## Export Artifact Requirements

Every model release must include:

- model weights or exported model artifact
- model config
- preprocessing contract version
- postprocessing contract version
- input tensor names
- output tensor names
- expected tensor shapes
- normalization rules
- supported image size policy
- supported batch size
- expected runtime backend
- training dataset/version identifier
- validation report
- known limitations

Recommended artifact layout:

```text
shadowgen_shadow_v2/
  1/
    model.onnx
  config.pbtxt
  model-card.md
  validation-report.json
  preprocessing.json
```

For a temporary Python backend:

```text
shadowgen_shadow_v2/
  1/
    model.py
    model/
      weights.safetensors
      config.json
  config.pbtxt
  model-card.md
  validation-report.json
```

Weights and large generated artifacts must not be committed to git.

## Versioning

Use semantic model versions in metadata even if Triton stores numeric version folders.

Recommended naming:

- `v2-diff-research-YYYYMMDD`
- `v2-diff-alpha.1`
- `v2-diff-beta.1`
- `v2-diff-prod.1`

The ML core capability metadata should expose:

- `model_name`
- `model_version`
- `model_variant`
- `backend_kind`
- `device`
- `endpoint`

## Acceptance Checklist

A checkpoint is ready for ML-core integration when:

- it accepts `img` and `mask` by the exact contract names
- it returns `shadow` with shape `[N, 4, H, W]`
- output values are finite and clipped or naturally bounded to `0..1`
- batch `1` works
- batch `4` works or limitation is documented
- current control-free output is plausible on product-like object cutouts
- if a future controlled checkpoint declares control tensors, angle/elevation/softness/reflection sweeps must visibly affect the output
- inference does not require product API fields beyond this contract
- model card and validation report are included
- at least one smoke sample can be run through the ML core debug path

## Integration Smoke Test

The first integration smoke should run:

1. full sync pipeline with `shadow_generator` requested as `triton/v2-diff`
2. debug stage rerun for `shadow_generator`
3. async render job using the same model
4. fallback test with Triton unavailable

Expected debug metadata:

- `requested_backend_kind = triton`
- `actual_backend_kind = triton`
- `model_variant = v2-diff`
- `model_name = shadowgen_shadow_v2`
- `endpoint` is filled
- `fallback_reason = null`

## Sample Pack For Model Developers

Use the repository helper to prepare a small model-input pack from curated product-like external object photos:

```powershell
.venv\Scripts\python.exe tools\prepare_shadow_v2_sample_pack.py --count 10 --backend-kind local --normal-variant from-depth-v2 --output-dir artifacts\shadow-v2-sample-pack
```

The generated folder is intentionally under `artifacts/`, so it is ignored by git and can be shared separately with model developers. The default curated set is meant to match the product scenario, not stress segmentation: one meaningful foreground object, large enough in the frame, photographed on a phone or in a simple product/studio setup.

Each sample folder contains:

- `source.png`
- `img.png`
- `mask.png`
- `depth.png`
- `normal.png`
- `controls.json`
- `shadow_input.npz`
- `source.json`
- `stages.json`

`shadow_input.npz` contains ready-to-load `FP32` tensors. The current control-free `V2-DIFF` should consume only `img` and `mask`; the remaining tensors are included so the same sample pack can support future controlled checkpoints.

- `img`: `[1, 4, H, W]`
- `mask`: `[1, 1, H, W]`
- `depth`: `[1, 1, H, W]`
- `normal`: `[1, 3, H, W]`
- `angle`: `[1]`
- `elevation`: `[1]`
- `softness`: `[1]`
- `reflection`: `[1]`

Depth sample rule:

- raw depth is treated as float
- values are normalized using only pixels inside `mask`
- pixels outside `mask` are written as zero
- exported `depth.png` is an 8-bit grayscale contract preview

## Open Decisions

The following decisions may be finalized after first training experiments:

- whether RGB shadow colour is predicted directly or fixed by compositor policy
- whether `normal` should use `0..1` encoded XYZ or `-1..1` internal conversion
- whether the first production model is diffusion, conditional U-Net, or another architecture
- whether reflection becomes an active learned control in the first checkpoint
- whether canonical size remains `512x512` or moves to a larger working canvas
