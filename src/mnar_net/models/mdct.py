from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from mnar_net.modules.blocks import DownsampleBlock1D, MultiScaleFeatureBlock1D, ShuffleStem1D
from mnar_net.modules.mdrc import MultiScaleDilatedFusion
from mnar_net.modules.transformer import TransformerEncoder


class MDCTBackbone(nn.Module):
    def __init__(
        self,
        num_classes: int,
        signal_length: int = 1024,
        input_channels: int = 3,
        dim: int = 192,
        depth: int = 1,
        heads: int = 8,
        mlp_dim: int = 768,
        classifier_dim: int = 64,
        dropout: float = 0.2,
        dilations: tuple[int, ...] = (1, 3, 5),
        apply_scales: tuple[int, ...] = (1, 2, 3),
        branch_fuse: str = "cat",
        use_fuse_conv: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.signal_length = signal_length

        self.downsample_1 = DownsampleBlock1D(input_channels, 32, kernel_size=7, stride=2)
        self.shuffle_stem = ShuffleStem1D(32, hidden_channels=128, groups=4, depth=1)

        self.downsample_2 = DownsampleBlock1D(32, 64, kernel_size=5, stride=2)
        self.feature_block_2 = MultiScaleFeatureBlock1D(64, 64)

        self.downsample_3 = DownsampleBlock1D(64, 128, kernel_size=3, stride=2)
        self.feature_block_3 = MultiScaleFeatureBlock1D(128, 128)

        self.downsample_4 = DownsampleBlock1D(128, dim, kernel_size=3, stride=2)
        self.patch_projection = nn.Conv1d(dim, dim, kernel_size=1, stride=1)

        token_length = self._infer_token_length(signal_length, input_channels, dim)
        self.mdrc_fusion = MultiScaleDilatedFusion(
            dim=dim,
            channels_scale_1=64,
            channels_scale_2=128,
            channels_scale_3=dim,
            target_length=token_length,
            dilations=dilations,
            apply_scales=apply_scales,
            branch_fuse=branch_fuse,
            use_fuse_conv=use_fuse_conv,
            use_residual=use_residual,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.empty(1, token_length + 1, dim))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer = TransformerEncoder(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.4),
            nn.Linear(dim, classifier_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_dim, num_classes),
        )

    def _infer_token_length(self, signal_length: int, input_channels: int, dim: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, signal_length)
            out = self.shuffle_stem(self.downsample_1(dummy))
            out = self.feature_block_2(self.downsample_2(out))
            out = self.feature_block_3(self.downsample_3(out))
            out = self.patch_projection(self.downsample_4(out))
        return int(out.shape[-1])

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.shuffle_stem(self.downsample_1(x))
        features_scale_1 = self.feature_block_2(self.downsample_2(out))
        features_scale_2 = self.feature_block_3(self.downsample_3(features_scale_1))
        features_scale_3 = self.patch_projection(self.downsample_4(features_scale_2))

        fused = self.mdrc_fusion(features_scale_1, features_scale_2, features_scale_3)
        tokens = rearrange(fused, "b d t -> b t d")

        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        tokens = self.transformer(tokens, mask=mask)
        return tokens[:, 0]

    def get_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_features(x, mask=mask)

    def expand_classifier(self, new_num_classes: int) -> None:
        old_fc = self.classifier[-1]
        old_num_classes = old_fc.out_features
        if new_num_classes <= old_num_classes:
            return

        new_fc = nn.Linear(
            old_fc.in_features,
            new_num_classes,
            device=old_fc.weight.device,
            dtype=old_fc.weight.dtype,
        )
        with torch.no_grad():
            new_fc.weight[:old_num_classes].copy_(old_fc.weight)
            new_fc.bias[:old_num_classes].copy_(old_fc.bias)
            nn.init.normal_(new_fc.weight[old_num_classes:], std=0.01)
            nn.init.zeros_(new_fc.bias[old_num_classes:])
        self.classifier[-1] = new_fc

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.forward_features(x, mask=mask)
        return self.classifier(features)
