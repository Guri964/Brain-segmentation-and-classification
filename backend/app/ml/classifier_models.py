from __future__ import annotations
import torch
import torch.nn as nn
class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.n_classes = n_classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
class RadiomicsDeepFusion(nn.Module):
    def __init__(
        self,
        radiomics_dim: int,
        deep_dim: int,
        n_classes: int = 2,
        hidden_dim: int = 64,
        fusion: str = "concat",
    ):
        super().__init__()
        self.radiomics_branch = nn.Sequential(
            nn.Linear(radiomics_dim, hidden_dim), nn.ReLU(inplace=True)
        )
        self.deep_branch = nn.Sequential(nn.Linear(deep_dim, hidden_dim), nn.ReLU(inplace=True))
        self.fusion = fusion
        if fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, 2), nn.Softmax(dim=1))
        self.head = nn.Linear(2 * hidden_dim if fusion != "sum" else hidden_dim, n_classes)
    def forward(self, radiomics: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        r = self.radiomics_branch(radiomics)
        d = self.deep_branch(deep)
        if self.fusion == "gated":
            weights = self.gate(torch.cat([r, d], dim=1))
            combined = torch.cat([r * weights[:, :1], d * weights[:, 1:]], dim=1)
        elif self.fusion == "sum":
            combined = r + d
        else:
            combined = torch.cat([r, d], dim=1)
        return self.head(combined)
def build_classifier(architecture: str, *, input_dim: int, **kwargs) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "mlp":
        return MLPClassifier(
            input_dim,
            n_classes=kwargs.get("n_classes", 2),
            hidden_dims=tuple(kwargs.get("hidden_dims", (128, 64))),
            dropout=kwargs.get("dropout", 0.0),
            batch_norm=kwargs.get("batch_norm", False),
        )
    if architecture == "linear":
        return LinearClassifier(input_dim, n_classes=kwargs.get("n_classes", 2))
    if architecture == "feature_fusion":
        return RadiomicsDeepFusion(
            radiomics_dim=input_dim,
            deep_dim=kwargs.get("deep_dim", 32),
            n_classes=kwargs.get("n_classes", 2),
            hidden_dim=kwargs.get("hidden_dim", 64),
            fusion=kwargs.get("fusion", "concat"),
        )
    raise ValueError(f"unknown classifier architecture '{architecture}'")
