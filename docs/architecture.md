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
4. Crop and pad with `preprocess.padding_px`
5. Segment foreground
6. Estimate depth
7. Compute normals
8. Generate shadow
9. Compose final output
10. Encode artifacts and metrics
