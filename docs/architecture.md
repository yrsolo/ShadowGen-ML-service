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
6. Refine semi-transparent foreground colours with Fast Foreground Colour Estimation
7. Estimate depth
8. Compute normals
9. Generate shadow
10. Compose final output
11. Encode artifacts and metrics

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

## Foreground refinement step

- `foreground_refiner` is a dedicated post-segmentation stage, not logic embedded into the segmenter adapter
- it takes the prepared working crop plus the segmentation alpha matte and corrects edge colours for semi-transparent pixels
- the real backend uses the Fast Foreground Colour Estimation method from `Photoroom/fast-foreground-estimation`
- this stage exists to reduce matte fringing and background colour contamination before depth, shadow, and final composition
- the fallback backend is a passthrough refiner that preserves the incoming alpha but skips colour correction
- the playground exposes:
  - `segmenter_cutout`
  - `foreground_cutout`
  - backend metadata for the refinement stage

## Depth step

- `depth_estimator` runs on the refined foreground cutout after segmentation and foreground colour correction
- the real backend is `depth-anything/Depth-Anything-V2-Small-hf` through Hugging Face `transformers`
- the model output is normalized into a single-channel depth map and resized back to the working crop resolution
- the binary foreground mask is applied again after inference so background pixels stay zeroed
- the playground exposes:
  - `depth`
  - `working_cutout`
  - backend, map size, and device metadata

## Normals step

- `normal_estimator` is a separate pipeline stage and does not live inside the depth adapter
- the preferred real backend is the neural `Stable-X/StableNormal` estimator, which predicts normals from the refined foreground cutout
- when the neural backend is unavailable or fails to initialize, the service keeps the stage operational through the explicit `from-depth` fallback backend
- the fallback backend computes image-space gradients from depth and converts them into RGB normal vectors
- the mock backend returns a flat neutral normal map, which keeps the playground switch meaningful and preserves stage symmetry
- the playground exposes:
  - `normals`
  - backend, variant, map size, and device metadata
