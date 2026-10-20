from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        pad = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    """Baseline CNN for 128x128 grayscale phononic crystal images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(1, 32, 3, 2),
            ConvBNAct(32, 64, 3, 2),
            ConvBNAct(64, 128, 3, 2),
            ConvBNAct(128, 256, 3, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SobelEdge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        kernel = torch.stack([kx, ky]).unsqueeze(1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = F.conv2d(x, self.kernel, padding=1)
        return torch.sqrt(edge[:, :1].square() + edge[:, 1:].square() + 1e-6)


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = ConvBNAct(channels, mid, 1)
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        y = self.conv1(torch.cat([x_h, x_w], dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        attn = torch.sigmoid(self.conv_h(y_h)) * torch.sigmoid(self.conv_w(y_w.transpose(2, 3)))
        return x * attn


class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int, use_multiscale: bool = True, use_attention: bool = True) -> None:
        super().__init__()
        self.use_multiscale = use_multiscale
        if use_multiscale:
            branch_ch = channels // 3
            self.b3 = ConvBNAct(channels, branch_ch, 3, groups=1)
            self.b5 = ConvBNAct(channels, branch_ch, 5, groups=1)
            self.b7 = ConvBNAct(channels, channels - 2 * branch_ch, 7, groups=1)
            self.fuse = ConvBNAct(channels, channels, 1)
        else:
            self.single = ConvBNAct(channels, channels, 3)
        self.attn = CoordAttention(channels) if use_attention else nn.Identity()
        self.out = ConvBNAct(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.use_multiscale:
            x = self.fuse(torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1))
        else:
            x = self.single(x)
        x = self.attn(x)
        return self.out(x) + identity


class HoleBandNet(nn.Module):
    """Edge-aware multi-scale network designed for bandgap median regression.

    The model keeps a fixed Sobel branch for pore boundaries, multi-kernel blocks
    for hole-size sensitivity, and coordinate attention for periodic layout cues.
    """

    def __init__(
        self,
        use_edge: bool = True,
        use_multiscale: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge = use_edge
        self.edge = SobelEdge() if use_edge else nn.Identity()
        in_ch = 2 if use_edge else 1
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 32, 3, 1),
            ConvBNAct(32, 64, 3, 2),
        )
        self.stage1 = nn.Sequential(
            MultiScaleBlock(64, use_multiscale, use_attention),
            ConvBNAct(64, 128, 3, 2),
        )
        self.stage2 = nn.Sequential(
            MultiScaleBlock(128, use_multiscale, use_attention),
            ConvBNAct(128, 192, 3, 2),
        )
        self.stage3 = nn.Sequential(
            MultiScaleBlock(192, use_multiscale, use_attention),
            ConvBNAct(192, 256, 3, 2),
        )
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(512, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(192, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_edge:
            x = torch.cat([x, self.edge(x)], dim=1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = torch.cat([self.pool_avg(x), self.pool_max(x)], dim=1)
        return self.head(x)


def _adapt_first_conv_to_gray(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        1,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def build_model(name: str, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNN()
    if name == "holebandnet":
        return HoleBandNet()
    if name == "holebandnet_no_edge":
        return HoleBandNet(use_edge=False)
    if name == "holebandnet_no_multiscale":
        return HoleBandNet(use_multiscale=False)
    if name == "holebandnet_no_attention":
        return HoleBandNet(use_attention=False)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.conv1 = _adapt_first_conv_to_gray(model.conv1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.features[0][0] = _adapt_first_conv_to_gray(model.features[0][0])
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model
    if name == "convnext_tiny_reg":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.features[0][0] = _adapt_first_conv_to_gray(model.features[0][0])
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 1),
        )
        return model
    if name == "mobilenetv3_large_reg":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.features[0][0] = _adapt_first_conv_to_gray(model.features[0][0])
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(384, 1),
        )
        return model
    raise ValueError(
        f"Unknown model {name!r}. Use simple_cnn, holebandnet, holebandnet_no_edge, "
        "holebandnet_no_multiscale, holebandnet_no_attention, resnet18, efficientnet_b0, "
        "convnext_tiny_reg, or mobilenetv3_large_reg."
    )
