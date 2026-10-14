"""ResNet-18 (CIFAR variant) and a small ViT.

Both take images in [0, 1] and normalise internally, so attribution methods
operate directly in pixel space. Every ReLU is its own non-inplace module
(no module reuse) so hook-based attribution (LRP, Guided Backprop) sees
each activation separately. Residual additions go through captum's
parameter-free Addition_Module so LRP conserves relevance across skips.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from captum.attr._utils.custom_modules import Addition_Module

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    def __init__(self, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# ---------------------------------------------------------------- ResNet-18

class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.add = Addition_Module()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(self.add(out, self.shortcut(x)))


class ResNet18(nn.Module):
    """ResNet-18 layout (4 stages x 2 BasicBlocks). `width` = channels of stage 1 (64 = standard)."""

    def __init__(self, num_classes: int = 10, width: int = 64):
        super().__init__()
        self.norm = Normalize()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        widths, strides = (width, 2 * width, 4 * width, 8 * width), (1, 2, 2, 2)
        layers, cin = [], width
        for w, s in zip(widths, strides):
            layers.append(nn.Sequential(BasicBlock(cin, w, s), BasicBlock(w, w, 1)))
            cin = w
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 * width, num_classes)

    def forward(self, x):
        x = self.stem(self.norm(x))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.pool(x), 1))


# ---------------------------------------------------------------- ViT
#
# `lrp_mode` (off during training and normal inference) implements
# conservative propagation (CP-LRP, Ali et al. 2022, "XAI for Transformers"):
# attention probabilities, the LayerNorm scale and the GELU gate are treated
# as constants. The network is then linear in its input given those
# constants, and LRP-0 equals gradient x input. Forward values are unchanged.

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.lrp_mode = False

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * (d // self.heads) ** -0.5).softmax(dim=-1)
        if self.lrp_mode:
            attn = attn.detach()
        out = attn @ v
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class LayerNorm(nn.LayerNorm):
    """nn.LayerNorm (same parameters) whose scale can be frozen for CP-LRP."""

    lrp_mode = False

    def forward(self, x):
        if not self.lrp_mode:
            return super().forward(x)
        mean = x.mean(-1, keepdim=True)
        std = (x.var(-1, unbiased=False, keepdim=True) + self.eps).sqrt().detach()
        return (x - mean) / std * self.weight + self.bias


class GELU(nn.GELU):
    """nn.GELU whose gate gelu(x)/x can be frozen for CP-LRP."""

    lrp_mode = False

    def forward(self, x):
        y = super().forward(x)
        if not self.lrp_mode:
            return y
        safe = torch.where(x.abs() > 1e-6, x, torch.full_like(x, 1e-6))
        return x * (y / safe).detach()


def set_lrp_mode(model: nn.Module, on: bool) -> nn.Module:
    for m in model.modules():
        if hasattr(m, "lrp_mode"):
            m.lrp_mode = on
    return model


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, patch: int = 4, dim: int = 192, depth: int = 6, heads: int = 3,
                 mlp_ratio: float = 2, num_classes: int = 10, image_size: int = 32):
        super().__init__()
        self.norm = Normalize()
        self.patch = patch
        self.grid = image_size // patch
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid ** 2 + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(self.norm(x)).flatten(2).transpose(1, 2)       # (B, 64, dim)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], dim=1) + self.pos_embed
        x = self.final_norm(self.blocks(x))
        return self.head(x[:, 0])


def build_model(cfg: dict, name: str) -> nn.Module:
    m = cfg["models"][name]
    if m["arch"] == "resnet18":
        return ResNet18(width=m.get("width", 64))
    if m["arch"] == "vit":
        return ViT(patch=m["patch"], dim=m["dim"], depth=m["depth"], heads=m["heads"], mlp_ratio=m["mlp_ratio"])
    raise ValueError(f"unknown arch {m['arch']!r}")
