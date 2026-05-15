from __future__ import annotations

import torch
import torch.nn as nn

from mnar_net.models.mdct import MDCTBackbone
from mnar_net.replay.naer import NAERMemory


class MNARNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        signal_length: int = 1024,
        memory_per_class: int = 20,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.backbone = MDCTBackbone(
            num_classes=num_classes,
            signal_length=signal_length,
            **backbone_kwargs,
        )
        self.memory = NAERMemory(max_per_class=memory_per_class)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone(x, mask=mask)

    def get_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone.get_features(x, mask=mask)

    def expand_classifier(self, new_num_classes: int) -> None:
        self.backbone.expand_classifier(new_num_classes)

    def update_memory(
        self,
        x_clean: torch.Tensor,
        x_noisy: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device | None = None,
    ) -> None:
        self.memory.update(
            model=self.backbone,
            x_clean=x_clean,
            x_noisy=x_noisy,
            labels=labels,
            device=device,
        )

    def sample_replay(
        self,
        snr_db: float | None = None,
        seed: int = 0,
        use_adaptive_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.memory.sample(
            snr_db=snr_db,
            seed=seed,
            use_adaptive_noise=use_adaptive_noise,
        )
