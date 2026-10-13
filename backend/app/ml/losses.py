from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, from_logits: bool = False, smooth: float = 1.0):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            predicted = torch.sigmoid(predicted)
        dims = tuple(range(1, predicted.ndim))
        intersection = (predicted * target).sum(dim=dims)
        denom = predicted.sum(dim=dims) + target.sum(dim=dims)
        coefficient = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1 - coefficient.mean()
class BCEDiceLoss(nn.Module):
    def __init__(self, device=None, from_logits: bool = False, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.dice_loss = DiceLoss(from_logits=from_logits)
        self.from_logits = from_logits
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if device is not None:
            self.to(device)
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(predicted, target) if self.from_logits else F.binary_cross_entropy(predicted, target)
        return self.bce_weight * bce + self.dice_weight * self.dice_loss(predicted, target)
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.from_logits = from_logits
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            predicted = torch.sigmoid(predicted)
        dims = tuple(range(1, predicted.ndim))
        tp = (predicted * target).sum(dim=dims)
        fp = (predicted * (1 - target)).sum(dim=dims)
        fn = ((1 - predicted) * target).sum(dim=dims)
        tversky = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        return ((1 - tversky) ** self.gamma).mean()
def distance_weight_map(target: torch.Tensor, beta: float = 5.0, sigma_ratio: float = 0.25) -> torch.Tensor:
    from ...utils.metrics import _edt
    array = target.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError("distance_weight_map expects a (B,H,W) or (B,1,H,W) tensor")
    weights = np.ones_like(array, dtype=np.float32)
    for index in range(array.shape[0]):
        mask = array[index] > 0.5
        if not mask.any():
            continue
        distance_inside = _edt(mask)
        distance_outside = _edt(~mask)
        sigma = max(float(np.sqrt(mask.sum())) * sigma_ratio, 1.0)
        w_in = 1.0 + beta * np.exp(-distance_inside / sigma)
        w_out = 1.0 + beta * np.exp(-distance_outside / sigma)
        weights[index] = np.where(mask, w_in, w_out)
    tensor = torch.from_numpy(weights).to(target.device).unsqueeze(1)
    return tensor
class BoundaryAwareLoss(nn.Module):
    def __init__(
        self,
        from_logits: bool = True,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        beta: float = 5.0,
    ):
        super().__init__()
        self.from_logits = from_logits
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.beta = beta
        self.dice = DiceLoss(from_logits=from_logits)
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(predicted, target)
        else:
            bce = F.binary_cross_entropy(predicted, target)
        weights = distance_weight_map(target, beta=self.beta)
        if self.from_logits:
            boundary = F.binary_cross_entropy_with_logits(predicted, target, weight=weights)
        else:
            boundary = F.binary_cross_entropy(predicted, target, weight=weights)
        return (
            self.bce_weight * bce
            + self.dice_weight * self.dice(predicted, target)
            + self.boundary_weight * boundary
        )
class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, aux_weights: tuple[float, ...] = (0.4, 0.2, 0.1)):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights
    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            primary, *aux = outputs
        else:
            return self.base_loss(outputs, target)
        total = self.base_loss(primary, target)
        for weight, aux_output in zip(self.aux_weights, aux):
            total = total + weight * self.base_loss(aux_output, target)
        return total
def build_loss(
    name: str = "bce_dice",
    *,
    from_logits: bool = True,
    deep_supervision: bool = False,
    aux_weights: tuple[float, ...] = (0.4, 0.2, 0.1),
) -> nn.Module:
    name = name.lower()
    if name == "dice":
        base: nn.Module = DiceLoss(from_logits=from_logits)
    elif name == "bce":
        base = nn.BCEWithLogitsLoss() if from_logits else nn.BCELoss()
    elif name == "bce_dice":
        base = BCEDiceLoss(from_logits=from_logits)
    elif name == "focal_tversky":
        base = FocalTverskyLoss(from_logits=from_logits)
    elif name == "boundary_aware":
        base = BoundaryAwareLoss(from_logits=from_logits)
    else:
        raise ValueError(f"unknown loss '{name}'")
    if deep_supervision:
        return DeepSupervisionLoss(base, aux_weights=aux_weights)
    return base
def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predicted = (probabilities > threshold).float()
    dims = tuple(range(1, predicted.ndim))
    intersection = (predicted * target).sum(dim=dims)
    denom = predicted.sum(dim=dims) + target.sum(dim=dims)
    return ((2 * intersection + 1e-6) / (denom + 1e-6)).mean()
