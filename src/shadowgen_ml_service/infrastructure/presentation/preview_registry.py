from __future__ import annotations

from shadowgen_ml_service.core.assets import RasterAsset
from shadowgen_ml_service.core.contracts import PreviewBuilderRegistry
from shadowgen_ml_service.utils.images import draw_detection_overlay, draw_geometry_overlay, ensure_pil, ensure_asset


class DefaultPreviewBuilderRegistry(PreviewBuilderRegistry):
    def build(self, stage_key: str, stage_value: object, context: object) -> dict[str, RasterAsset]:
        source_rgba = getattr(context, "source_rgba", None)
        working_crop = getattr(context, "working_crop", None)
        if stage_key == "decode" and source_rgba is not None:
            return {"source": ensure_asset(source_rgba)}
        if stage_key == "geometry_estimator" and source_rgba is not None:
            return {
                "geometry_input": ensure_asset(source_rgba),
                "geometry_overlay": ensure_asset(draw_geometry_overlay(
                    ensure_pil(source_rgba),
                    stage_value.camera_fov,
                    stage_value.camera_pitch,
                    stage_value.camera_roll,
                    stage_value.confidence,
                )),
            }
        if stage_key == "detector" and source_rgba is not None:
            previews = {"detection_overlay": ensure_asset(draw_detection_overlay(ensure_pil(source_rgba), stage_value.bbox, stage_value.confidence))}
            if working_crop is not None:
                previews["crop_for_resize"] = ensure_asset(working_crop)
            return previews
        if stage_key == "segmenter":
            return {"working_crop": stage_value.crop_rgba, "cutout": stage_value.cutout_rgba, "mask": stage_value.mask}
        if stage_key == "foreground_refiner":
            previews = {"foreground_cutout": stage_value.cutout_rgba}
            pre_refinement_cutout = getattr(context, "pre_refinement_cutout", None)
            if pre_refinement_cutout is not None:
                previews["segmenter_cutout"] = pre_refinement_cutout
            return previews
        if stage_key == "depth_estimator":
            previews = {"depth": stage_value.depth_map}
            segmentation = getattr(context, "segmentation", None)
            if segmentation is not None:
                previews["working_cutout"] = segmentation.cutout_rgba
            return previews
        if stage_key == "normal_estimator":
            return {"normals": stage_value.normal_map}
        if stage_key == "shadow_generator":
            return {"shadow": stage_value.shadow_rgba}
        if stage_key == "composer":
            return {"final": stage_value.final_image}
        return {}
