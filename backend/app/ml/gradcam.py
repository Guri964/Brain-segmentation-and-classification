from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from ..utils.images import resize_array
logger = logging.getLogger(__name__)
@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    model_id: str
    target_layer: str
    target_score: str
    method: str
    explanation: str
    activation_shape: list[int]
    warnings: list[str]
    def summary(self) -> dict:
        return {
            "model_id": self.model_id,
            "target_layer": self.target_layer,
            "target_score": self.target_score,
            "method": self.method,
            "explanation": self.explanation,
            "activation_shape": self.activation_shape,
            "heatmap_shape": list(self.heatmap.shape),
            "heatmap_mean": float(self.heatmap.mean()),
            "heatmap_max": float(self.heatmap.max()),
            "warnings": self.warnings,
        }
class ActivationCapture:
    def __init__(self, module: torch.nn.Module, layer_name: str):
        self.module = module
        self.layer_name = layer_name
        self.activation: torch.Tensor | None = None
        self._handle = None
    def _resolve(self) -> torch.nn.Module:
        target = self.module
        for part in self.layer_name.split("."):
            if part.isdigit() and isinstance(target, (torch.nn.Sequential, torch.nn.ModuleList)):
                target = target[int(part)]
            elif hasattr(target, part):
                target = getattr(target, part)
            else:
                raise ValueError(f"target layer '{self.layer_name}' not found in the model")
        return target
    def __enter__(self):
        layer = self._resolve()
        self._handle = layer.register_forward_hook(self._hook)
        return self
    def _hook(self, _module, _inputs, output):
        self.activation = output
    def __exit__(self, *exc_info):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
def compute_gradcam(
    module: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    model_id: str,
    target_layer: str,
    region_of_interest: np.ndarray | None = None,
    output_is_logit: bool = True,
    relu: bool = True,
) -> GradCAMResult:
    warnings: list[str] = []
    module.eval()
    tensor = tensor.detach().clone().requires_grad_(True)
    with ActivationCapture(module, target_layer) as capture:
        output = module(tensor)
    logits = output[0] if isinstance(output, (list, tuple)) else output
    activation = capture.activation
    if activation is None:
        raise RuntimeError(f"no activation captured at layer '{target_layer}'")
    probabilities = torch.sigmoid(logits) if output_is_logit else logits
    if region_of_interest is not None:
        roi = torch.as_tensor(np.asarray(region_of_interest).astype(np.float32))
        if roi.shape[-2:] != probabilities.shape[-2:]:
            roi = F.interpolate(
                roi[None, None], size=probabilities.shape[-2:], mode="nearest"
            )[0, 0]
        roi = (roi > 0.5).to(probabilities.dtype)
        denominator = roi.sum().clamp(min=1.0)
        score = (probabilities * roi).sum() / denominator
        target_score_name = "mean predicted probability inside the predicted region"
        if float(roi.sum()) == 0:
            warnings.append(
                "The predicted region is empty, so the explanation falls back to the whole image."
            )
            score = probabilities.mean()
            target_score_name = "mean predicted probability over the whole image"
    else:
        score = probabilities.mean()
        target_score_name = "mean predicted probability over the whole image"
    gradient = torch.autograd.grad(
        outputs=score,
        inputs=activation,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    if gradient is None:
        raise RuntimeError("Grad-CAM gradient is None; the target layer is not on the forward path")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1)
    if relu:
        cam = torch.relu(cam)
    cam = cam[0].detach().cpu().numpy().astype(np.float32)
    cam = cam - float(cam.min())
    maximum = float(cam.max())
    if maximum > 1e-8:
        cam = cam / maximum
    else:
        warnings.append("The Grad-CAM map is constant; the model output does not depend on this layer.")
        cam = np.zeros_like(cam)
    heatmap = resize_array(cam, tuple(probabilities.shape[-2:]))
    return GradCAMResult(
        heatmap=heatmap.astype(np.float32),
        model_id=model_id,
        target_layer=target_layer,
        target_score=target_score_name,
        method="Grad-CAM (gradient-weighted activation mapping, Selvaraju et al. 2017)",
        explanation=(
            "Warm colours mark image regions whose activation contributed most strongly to the "
            "model's own predicted region for this case. The map explains the model's behaviour on "
            "this input; it does not establish that a tumour is present, and it is not a "
            "radiological finding."
        ),
        activation_shape=list(activation.shape),
        warnings=warnings,
    )
def layer_magnitude_map(module: torch.nn.Module, tensor: torch.Tensor, layer_name: str) -> np.ndarray:
    with ActivationCapture(module, layer_name) as capture:
        with torch.no_grad():
            module(tensor)
    activation = capture.activation
    if activation is None:
        raise RuntimeError(f"no activation captured at layer '{layer_name}'")
    magnitude = activation.abs().mean(dim=1)[0].detach().cpu().numpy()
    maximum = float(magnitude.max())
    if maximum > 1e-8:
        magnitude = magnitude / maximum
    return magnitude.astype(np.float32)
def permutation_importance(
    predict_proba,
    features: np.ndarray,
    feature_names: list[str],
    *,
    target_index: int,
    n_repeats: int = 8,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    features = np.asarray(features, dtype=np.float32)
    baseline = float(np.asarray(predict_proba(features.reshape(1, -1))).ravel()[target_index])
    importances = []
    for index, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            perturbed = features.copy()
            perturbed[index] = float(rng.permutation(perturbed)[0]) if perturbed.size > 1 else perturbed[index]
            score = float(np.asarray(predict_proba(perturbed.reshape(1, -1))).ravel()[target_index])
            deltas.append(baseline - score)
        importances.append(
            {
                "feature": name,
                "importance": float(np.mean(deltas)),
                "std": float(np.std(deltas)),
                "direction": "increases score" if np.mean(deltas) > 0 else "decreases score",
            }
        )
    importances.sort(key=lambda item: abs(item["importance"]), reverse=True)
    return {
        "method": "permutation importance (single case)",
        "baseline_score": baseline,
        "importances": importances,
        "note": (
            "Permutation importance measures how much the model's score for this case changes when "
            "one feature value is shuffled. It describes the model, not the biology."
        ),
    }
