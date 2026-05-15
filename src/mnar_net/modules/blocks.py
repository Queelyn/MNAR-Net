from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1)
        return x * weights


class DownsampleBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ShuffleResidualBlock1D(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, groups: int) -> None:
        super().__init__()
        if hidden_channels % groups != 0:
            raise ValueError("hidden_channels must be divisible by groups.")

        self.groups = groups
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=9,
            padding=4,
            groups=channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.expand = nn.Conv1d(channels, hidden_channels, kernel_size=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.shrink = nn.Conv1d(hidden_channels, channels, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se = SEBlock1D(channels)

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, length)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, channels, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.bn1(self.depthwise(x)))
        x = F.gelu(self.bn2(self.expand(x)))
        x = self._channel_shuffle(x)
        x = F.gelu(self.bn3(self.shrink(x)))
        x = self.se(x)
        return x + residual


class ShuffleStem1D(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, groups: int, depth: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ShuffleResidualBlock1D(channels, hidden_channels, groups) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MultiScaleFeatureBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4.")

        branch_channels = out_channels // 4
        self.branch_1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )
        self.branch_7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )
        self.branch_15 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.se = SEBlock1D(out_channels)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.branch_1(x), self.branch_3(x), self.branch_7(x), self.branch_15(x)],
            dim=1,
        )
        fused = self.se(self.fuse(features))
        return F.gelu(fused + self.skip(x))
