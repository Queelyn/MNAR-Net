from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDilatedRateConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Iterable[int] = (1, 3, 5),
        branch_fuse: str = "cat",
        use_fuse_conv: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        self.branch_fuse = branch_fuse
        self.use_fuse_conv = use_fuse_conv
        self.use_residual = use_residual

        if not self.dilations:
            raise ValueError("At least one dilation branch is required.")
        if self.branch_fuse not in {"sum", "cat"}:
            raise ValueError("branch_fuse must be either 'sum' or 'cat'.")

        branch_channels = out_channels if self.branch_fuse == "sum" else max(out_channels // len(self.dilations), 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for dilation in self.dilations
            ]
        )

        fused_channels = out_channels if self.branch_fuse == "sum" else branch_channels * len(self.dilations)
        if self.use_fuse_conv:
            self.fuse = nn.Sequential(
                nn.Conv1d(fused_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
        else:
            self.fuse = (
                nn.Sequential(
                    nn.Conv1d(fused_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
                if fused_channels != out_channels
                else nn.Identity()
            )

        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        if len(features) == 1:
            fused = features[0]
        elif self.branch_fuse == "sum":
            fused = torch.stack(features, dim=0).sum(dim=0)
        else:
            fused = torch.cat(features, dim=1)

        fused = self.fuse(fused)
        if self.use_residual:
            fused = fused + self.skip(x)
        return F.gelu(fused)


class MultiScaleDilatedFusion(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        channels_scale_1: int = 64,
        channels_scale_2: int = 128,
        channels_scale_3: int = 192,
        target_length: int = 64,
        dilations: Iterable[int] = (1, 3, 5),
        apply_scales: Iterable[int] = (1, 2, 3),
        branch_fuse: str = "cat",
        use_fuse_conv: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.apply_scales = tuple(sorted(set(int(value) for value in apply_scales)))

        self.block_s1 = MultiDilatedRateConvolutionBlock(
            channels_scale_1,
            channels_scale_1,
            dilations=dilations,
            branch_fuse=branch_fuse,
            use_fuse_conv=use_fuse_conv,
            use_residual=use_residual,
        )
        self.block_s2 = MultiDilatedRateConvolutionBlock(
            channels_scale_2,
            channels_scale_2,
            dilations=dilations,
            branch_fuse=branch_fuse,
            use_fuse_conv=use_fuse_conv,
            use_residual=use_residual,
        )
        self.block_s3 = MultiDilatedRateConvolutionBlock(
            channels_scale_3,
            channels_scale_3,
            dilations=dilations,
            branch_fuse=branch_fuse,
            use_fuse_conv=use_fuse_conv,
            use_residual=use_residual,
        )

        self.proj_s1 = nn.Sequential(nn.Conv1d(channels_scale_1, dim, kernel_size=1, bias=False), nn.BatchNorm1d(dim), nn.GELU())
        self.proj_s2 = nn.Sequential(nn.Conv1d(channels_scale_2, dim, kernel_size=1, bias=False), nn.BatchNorm1d(dim), nn.GELU())
        self.proj_s3 = nn.Sequential(nn.Conv1d(channels_scale_3, dim, kernel_size=1, bias=False), nn.BatchNorm1d(dim), nn.GELU())

        self.pool_s1 = nn.AdaptiveAvgPool1d(target_length)
        self.pool_s2 = nn.AdaptiveAvgPool1d(target_length)
        self.pool_s3 = nn.AdaptiveAvgPool1d(target_length)

    def _maybe_apply(self, scale_id: int, block: nn.Module, features: torch.Tensor) -> torch.Tensor:
        return block(features) if scale_id in self.apply_scales else features

    def forward(
        self,
        features_scale_1: torch.Tensor,
        features_scale_2: torch.Tensor,
        features_scale_3: torch.Tensor,
    ) -> torch.Tensor:
        f1 = self._maybe_apply(1, self.block_s1, features_scale_1)
        f2 = self._maybe_apply(2, self.block_s2, features_scale_2)
        f3 = self._maybe_apply(3, self.block_s3, features_scale_3)

        p1 = self.pool_s1(self.proj_s1(f1))
        p2 = self.pool_s2(self.proj_s2(f2))
        p3 = self.pool_s3(self.proj_s3(f3))

        return (p1 + p2 + p3) / 3.0
