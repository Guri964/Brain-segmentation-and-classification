from __future__ import annotations
import torch
import torch.nn as nn
class SegNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1):
        super(SegNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x
    @property
    def output_semantics(self) -> str:
        return "logit"
    def target_layer_name(self) -> str:
        return "encoder.4"
