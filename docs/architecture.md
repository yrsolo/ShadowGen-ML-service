# Architecture

## Layered layout

- `core/`: internal value objects, commands, runtime metadata, errors, and ports
- `application/`: use cases, pipeline context, stage runner, stage catalog, backend selection
- `infrastructure/`: stage implementations, filesystem cache, artifact encoding, preview builders
- `interfaces/http/`: FastAPI app factory, route modules, request/response schemas, mappers
- `interfaces/dev/`: playground UI and dev-facing presentation surface
- `bootstrap/`: composition root, probes, runtime descriptor assembly

## Architectural rules

- `core` does not import FastAPI, Pydantic, or interface modules
- `application` depends on `core`, never on HTTP schemas or route modules
- stage implementations are split per stage package instead of `real.py` / `mock.py` god files
- public and dev HTTP contracts stay stable while internal orchestration changes behind mappers and use cases
- preview generation and cache persistence are separate subsystems, not embedded into the main render use case

## Pipeline stages

1. Decode and validate image
2. Estimate geometry on full image
3. Detect main object
4. Crop, pad, and resize with `preprocess.padding_px`
5. Segment the prepared working crop
6. Estimate depth
7. Compute normals
8. Generate shadow
9. Compose final output
10. Encode artifacts and metrics

## Geometry step

- `geometry_estimator` runs on the original full image before crop and segmentation
- the real backend is `GeoCalib` when available in the active virtual environment
- the result is normalized into:
  - `camera_fov`
  - `camera_pitch`
  - `camera_roll`
  - `confidence`
- the playground exposes both numeric geometry details and an overlay preview

## Detection step

- `detector` runs on the original full image after geometry and before crop / resize
- the real backend is `IDEA-Research/grounding-dino-base` through the Hugging Face `transformers` implementation
- the detector uses a fixed prompt, currently `object.`, to find the primary foreground object
- candidate boxes are ranked by confidence, and when scores are close the larger area wins
- the downstream crop is fitted into the square working canvas with a configurable content scale, so the object no longer touches the output edges and there is room for shadow projection
- the normalized result is:
  - `bbox`
  - `confidence`
- the playground exposes:
  - `detection_overlay` with the selected bbox
  - `crop_for_resize` preview for the downstream working crop
  - numeric bbox coordinates and backend metadata

## Segmentation step

- `segmenter` runs after crop / pad / resize on the prepared working crop, not on the full source image
- the real backend is `ZhengPeng7/BiRefNet_lite-matting` through Hugging Face remote code
- preprocessing follows the model reference path:
  - resize to the configured BiRefNet resolution
  - convert to tensor
  - normalize with ImageNet statistics
- postprocessing returns:
  - `mask`
  - `cutout_rgba`
  - `bbox`
- the binary mask keeps only the largest connected component to suppress small noise islands
- the playground exposes:
  - `working_crop`
  - `mask`
  - `cutout`
  - stage metadata for mask size and backend mode
