# Architecture

## Layers

- API layer: FastAPI endpoints and request validation
- Application layer: render orchestration and result assembly
- Domain contracts: pipeline stage protocols and typed models
- Infrastructure layer: mock adapters, real adapter slots, filesystem cache

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
- the normalized result is:
  - `bbox`
  - `confidence`
- the playground exposes:
  - `detection_overlay` with the selected bbox
  - `crop_for_resize` preview for the downstream working crop
  - numeric bbox coordinates and backend metadata
