from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch
from ..errors import ModelUnavailableError
from ..ml.gradcam import GradCAMResult, compute_gradcam, layer_magnitude_map
from ..ml.registry import ModelRegistry
from ..utils.images import apply_colormap, make_heatmap_overlay, resize_array
logger = logging.getLogger(__name__)
class ExplainabilityService:
    def __init__(self, registry: ModelRegistry, config):
        self.registry = registry
        self.config = config
    def available_methods(self) -> dict:
        entries = []
        for model_id in self.registry.segmentation_model_ids():
            loaded = self.registry.load_model(model_id)
            entries.append(
                {
                    "model_id": model_id,
                    "display_name": loaded.descriptor.display_name,
                    "available": loaded.is_ready,
                    "method": "Grad-CAM",
                    "target_layer": self._target_layer_name(loaded) if loaded.is_ready else None,
                    "reason": None if loaded.is_ready else loaded.message,
                }
            )
        classifier_path = Path(self.config.CLASSIFIER_ARTIFACT_DIR)
        return {
            "segmentation": entries,
            "classification": {
                "method": "permutation importance",
                "available": bool(list(classifier_path.glob("*.joblib"))) if classifier_path.exists() else False,
                "reason": (
                    None
                    if classifier_path.exists() and list(classifier_path.glob("*.joblib"))
                    else "No validated radiomics classifier artefact is present, so feature "
                    "attributions cannot be computed."
                ),
            },
            "disclaimer": (
                "Feature attribution and saliency maps describe how a model produced its output. "
                "They are not evidence of pathology and must not be read as a diagnosis."
            ),
        }
    def _target_layer_name(self, loaded) -> str:
        module = loaded.module
        if module is not None and hasattr(module, "target_layer_name"):
            return module.target_layer_name()
        defaults = {
            "segnet_brats_4ch": "encoder.4",
            "unet_dynamic_512": "conv9_2",
        }
        return defaults.get(loaded.descriptor.model_id, "")
    def explain_segmentation(
        self,
        model_id: str,
        tensor: torch.Tensor,
        *,
        display_image: np.ndarray,
        predicted_mask: np.ndarray | None = None,
        target_layer: str | None = None,
        include_activation_map: bool = False,
    ) -> dict:
        loaded = self.registry.load_model(model_id)
        if not loaded.is_ready or loaded.module is None:
            raise ModelUnavailableError(
                f"Grad-CAM requires a loaded model; '{model_id}' is {loaded.status}: "
                f"{loaded.message}"
            )
        module = loaded.module
        layer = target_layer or self._target_layer_name(loaded)
        if not layer:
            raise ModelUnavailableError(
                f"No Grad-CAM target layer is registered for '{model_id}'. Adding another "
                "architecture requires extending `ExplainabilityService._target_layer_name`."
            )
        output_is_logit = getattr(module, "output_semantics", "probability") == "logit"
        if predicted_mask is None and hasattr(module, "output_semantics"):
            with torch.no_grad():
                raw = module(tensor.to(loaded.device))
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            probability = (torch.sigmoid(raw) if output_is_logit else raw)[0, 0].cpu().numpy()
            predicted_mask = probability > loaded.descriptor.fixed_threshold
        result: GradCAMResult = compute_gradcam(
            module,
            tensor,
            model_id=model_id,
            target_layer=layer,
            region_of_interest=predicted_mask,
            output_is_logit=output_is_logit,
        )
        gray = display_image
        heatmap_rgb = apply_colormap(result.heatmap)
        blended = make_heatmap_overlay(gray, result.heatmap, alpha=0.5)
        payload = result.summary()
        payload.update(
            {
                "images": {
                    "heatmap_gray": result.heatmap,
                    "heatmap_rgb": heatmap_rgb,
                    "overlay": blended,
                },
                "display_size": list(gray.shape),
                "target_layer_semantics": _layer_semantics(model_id, layer),
            }
        )
        if include_activation_map:
            try:
                activation = layer_magnitude_map(module, tensor.to(loaded.device), layer)
                payload["activation_map"] = resize_array(activation, result.heatmap.shape)
                payload["activation_map_note"] = (
                    "Mean absolute activation at the same layer, without gradients. Shown for "
                    "comparison with the gradient-weighted map."
                )
            except Exception as exc:
                payload["activation_map"] = None
                payload["activation_map_note"] = f"unavailable ({type(exc).__name__})"
        return payload
def _layer_semantics(model_id: str, layer: str) -> str:
    if model_id == "segnet_brats_4ch":
        return (
            f"'{layer}' is the last encoder convolution of SegNet. SegNet has no skip connections, "
            "so the resolution of this feature map is one quarter of the input and the heatmap is "
            "upsampled to the input grid for display."
        )
    if model_id == "unet_dynamic_512":
        return (
            f"'{layer}' is the last convolution of the final decoder block of the DynamicUNet, at "
            "the full input resolution."
        )
    if model_id == "raunet_brats_4ch":
        return (
            f"'{layer}' is the last decoder block of RAUNet, at the full input resolution. The "
            "block is preceded by an attention gate, so the highlighted regions reflect the gated "
            "skip features rather than raw encoder activations."
        )
    return f"Target layer: {layer}."
