from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import probe_grounding_dino
from shadowgen_ml_service.infrastructure.stages.geometry.geocalib import probe_geocalib
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import probe_birefnet
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe


def probe_depth_anything() -> RealAdapterProbe:
    from shadowgen_ml_service.infrastructure.stages.shared.model_support import module_available

    available = module_available("torch") and module_available("transformers")
    detail = None if available else "requires torch, transformers, and a local Depth Anything integration"
    return RealAdapterProbe("depth-anything/Depth-Anything-V2-Small-hf", "bootstrap-probe", available, detail)
