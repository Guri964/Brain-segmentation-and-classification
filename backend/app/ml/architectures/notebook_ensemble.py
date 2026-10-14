"""Architectures defined in ``FINAL ENSEMBLE NB.ipynb``.

Transcribed **verbatim** from the notebook so the checkpoints it produced load with an exact
key/shape match.  Three networks live here:

``NotebookSegNet``
    ``enc`` = Conv(4->64)-BN-ReLU-MaxPool-Conv(64->128)-BN-ReLU-MaxPool,
    ``dec`` = Upsample-Conv(128->64)-BN-ReLU-Upsample-Conv(64->1).  Two pooling stages, no skip
    connections, bilinear upsampling, **no sigmoid** (the notebook applies ``torch.sigmoid`` to
    the output), so :attr:`output_semantics` is ``"logit"``.

``NotebookRAUNet`` (+ :class:`AttentionGate`)
    Residual-free attention U-Net: four encoder stages ``[32, 64, 128, 256]``, an additive
    attention gate on each skip connection, ``ConvTranspose2d`` upsampling followed by an
    ``F.interpolate`` correction, and a 1x1 output convolution.  ``"logit"`` for the same reason.

``NotebookRadiomicsCNN``
    ``Linear(46->256)-BN-ReLU-Dropout(0.3) -> Linear(256->128)-BN-ReLU-Dropout(0.2) ->
    Linear(128->64)-ReLU -> Linear(64->2)`` over the 46 handcrafted features of the notebook.
    ``"class_logits"``: the notebook wraps it in ``torch.softmax``/``CrossEntropyLoss``.

``build_monai_unet``
    The notebook's U-Net is ``monai.networks.nets.UNet(spatial_dims=2, in_channels=4,
    out_channels=1, channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2,
    dropout=0.1)``.  Re-implementing MONAI's residual units would risk a key mismatch, so the
    real class is used and MONAI stays an optional dependency: without it the checkpoint is
    reported as *dependency missing* with the install command instead of being loaded wrongly.

Nothing here trains anything and nothing here invents a prediction; these are only the
container classes the checkpoints were saved from.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- SegNet (notebook 2)


class NotebookSegNet(nn.Module):
    """Two-stage encoder / bilinear decoder, no skip connections (notebook cell 11)."""

    output_semantics = "logit"

    def __init__(self, in_ch: int = 4, out_ch: int = 1) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 - notebook definition
        return self.dec(self.enc(x))


# ------------------------------------------------------------------- RAUNet (notebook 2)


class AttentionGate(nn.Module):
    """Additive attention gate (notebook cell 13)."""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:], mode="bilinear", align_corners=True)
        return x * self.psi(F.relu(g_up + self.W_x(x), inplace=True))


class NotebookRAUNet(nn.Module):
    """Residual attention U-Net, ``base=32`` (notebook cell 13), 1,994,794 parameters."""

    output_semantics = "logit"

    def __init__(self, in_ch: int = 4, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()

        def cb(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.ReLU(True),
                nn.Conv2d(oc, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.ReLU(True),
            )

        self.e1 = cb(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = cb(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = cb(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.bot = cb(base * 4, base * 8)

        self.ag3 = AttentionGate(base * 8, base * 4, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = cb(base * 8, base * 4)

        self.ag2 = AttentionGate(base * 4, base * 2, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = cb(base * 4, base * 2)

        self.ag1 = AttentionGate(base * 2, base, base)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = cb(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 - notebook definition
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b = self.bot(self.p3(e3))

        up3 = self.up3(b)
        up3 = F.interpolate(up3, size=e3.shape[2:], mode="bilinear", align_corners=True)
        d3 = self.d3(torch.cat([self.ag3(b, e3), up3], 1))

        up2 = self.up2(d3)
        up2 = F.interpolate(up2, size=e2.shape[2:], mode="bilinear", align_corners=True)
        d2 = self.d2(torch.cat([self.ag2(d3, e2), up2], 1))

        up1 = self.up1(d2)
        up1 = F.interpolate(up1, size=e1.shape[2:], mode="bilinear", align_corners=True)
        d1 = self.d1(torch.cat([self.ag1(d2, e1), up1], 1))

        return self.out(d1)


# --------------------------------------------------- radiomics feature CNN (notebook 2)


class NotebookRadiomicsCNN(nn.Module):
    """``RadiomicsCNN`` from notebook cell 27: 46 features -> 2 logits."""

    output_semantics = "class_logits"

    def __init__(self, inp: int = 46, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        self.input_dim = inp
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 - notebook definition
        return self.net(x)


# ---------------------------------------------------------------- MONAI U-Net (optional)

MONAI_UNET_CONFIG = {
    "spatial_dims": 2,
    "in_channels": 4,
    "out_channels": 1,
    "channels": (32, 64, 128, 256),
    "strides": (2, 2, 2),
    "num_res_units": 2,
    "dropout": 0.1,
}


def monai_available() -> bool:
    """True when the version-agnostic import can be satisfied."""

    import importlib.util

    return importlib.util.find_spec("monai") is not None


def build_monai_unet():
    """The notebook's U-Net, built by MONAI itself (cell 12).

    Raises :class:`RuntimeError` with the install command when MONAI is absent so the registry
    can report *dependency missing* rather than *incompatible checkpoint*.
    """

    try:
        from monai.networks.nets import UNet as MonaiUNet
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the notebook's U-Net checkpoint is a MONAI model and MONAI is not importable; "
            "install it with 'pip install monai'"
        ) from exc
    return MonaiUNet(**MONAI_UNET_CONFIG)


__all__ = [
    "AttentionGate",
    "MONAI_UNET_CONFIG",
    "NotebookRAUNet",
    "NotebookRadiomicsCNN",
    "NotebookSegNet",
    "build_monai_unet",
    "monai_available",
]
