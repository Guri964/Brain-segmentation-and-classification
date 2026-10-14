from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class RAUNetConfig:
    in_channels: int = 4
    out_channels: int = 1
    base_filters: int = 16
    depth: int = 4
    use_attention: bool = True
    use_residual: bool = True
    deep_supervision: bool = True
    dropout: float = 0.0
    filters: list[int] = field(default_factory=list)
    def channel_list(self) -> list[int]:
        if self.filters:
            return self.filters
        return [self.base_filters * (2**i) for i in range(self.depth + 1)]
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)
class PlainBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))
class AttentionGate(nn.Module):
    def __init__(self, f_x: int, f_g: int, f_int: int | None = None):
        super().__init__()
        f_int = f_int or max(f_x // 2, 1)
        self.theta_x = nn.Conv2d(f_x, f_int, kernel_size=1, bias=True)
        self.phi_g = nn.Conv2d(f_g, f_int, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(f_int, 1, kernel_size=1, bias=True)
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        additive = F.relu(self.theta_x(x) + self.phi_g(g))
        attention = torch.sigmoid(self.psi(additive))
        return x * attention
class RAUNet(nn.Module):
    def __init__(self, config: RAUNetConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = RAUNetConfig(**kwargs)
        self.config = config
        channels = config.channel_list()
        self.filters = channels
        block = ResidualBlock if config.use_residual else PlainBlock
        self.stem = block(config.in_channels, channels[0])
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for index in range(config.depth):
            self.encoder_blocks.append(block(channels[index], channels[index + 1]))
            self.pools.append(nn.MaxPool2d(2))
        self.bottleneck = block(channels[-1], channels[-1])
        self.upsamples = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.heads = nn.ModuleList()
        for index in reversed(range(config.depth)):
            in_ch = channels[index + 1]
            skip_ch = channels[index]
            self.upsamples.append(nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2))
            self.attention_gates.append(
                AttentionGate(skip_ch, skip_ch, max(skip_ch // 2, 1))
                if config.use_attention
                else nn.Identity()
            )
            self.decoder_blocks.append(block(skip_ch * 2, skip_ch))
            self.dropouts.append(nn.Dropout2d(p=config.dropout) if config.dropout > 0 else nn.Identity())
        for index in reversed(range(config.depth)):
            self.heads.append(nn.Conv2d(channels[index], config.out_channels, kernel_size=1))
    def forward(self, x: torch.Tensor, return_aux: bool = False, return_logits: bool = True):
        skips = []
        out = self.stem(x)
        skips.append(out)
        for block, pool in zip(self.encoder_blocks, self.pools):
            out = pool(out)
            out = block(out)
            skips.append(out)
        out = self.bottleneck(out)
        decoder_outputs: list[torch.Tensor] = []
        for index in range(len(self.upsamples)):
            skip = skips[-1 - index - 1]
            out = self.upsamples[index](out)
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            gate = self.attention_gates[index]
            skip_gated = gate(skip, out) if not isinstance(gate, nn.Identity) else skip
            out = torch.cat([out, skip_gated], dim=1)
            out = self.decoder_blocks[index](out)
            out = self.dropouts[index](out)
            decoder_outputs.append(out)
        primary = self.heads[-1](decoder_outputs[-1])
        logits = primary if return_logits else torch.sigmoid(primary)
        if not return_aux:
            return logits
        aux = []
        for index in range(len(decoder_outputs) - 1):
            head_index = index
            aux_map = self.heads[head_index](decoder_outputs[index])
            if aux_map.shape[-2:] != primary.shape[-2:]:
                aux_map = F.interpolate(
                    aux_map, size=primary.shape[-2:], mode="bilinear", align_corners=False
                )
            aux.append(aux_map)
        return logits, aux
    @property
    def output_semantics(self) -> str:
        return "logit"
    def target_layer_name(self) -> str:
        return f"decoder_blocks.{len(self.decoder_blocks) - 1}"
    def has_dropout(self) -> bool:
        return self.config.dropout > 0
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
def build_raunet(**kwargs) -> RAUNet:
    known = set(RAUNetConfig.__dataclass_fields__.keys())
    config_kwargs = {k: v for k, v in kwargs.items() if k in known}
    return RAUNet(RAUNetConfig(**config_kwargs))
