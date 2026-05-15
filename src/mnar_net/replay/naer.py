from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from mnar_net.data.preprocessing import add_awgn, zscore_per_channel


@dataclass
class MemoryEntry:
    clean: np.ndarray | None = None
    noisy: np.ndarray | None = None
    labels: np.ndarray | None = None


class NAERMemory:
    def __init__(self, max_per_class: int = 20) -> None:
        self.max_per_class = max_per_class
        self._store: Dict[int, MemoryEntry] = {}

    def _extract_features(
        self,
        model: nn.Module,
        inputs: np.ndarray,
        device: torch.device,
        batch_size: int = 64,
    ) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
                features = model.get_features(batch).cpu().numpy()
                outputs.append(features)
        return np.concatenate(outputs, axis=0)

    def _herding(self, features: np.ndarray, n_select: int) -> np.ndarray:
        if len(features) <= n_select:
            return np.arange(len(features))

        class_mean = features.mean(axis=0)
        selected_indices = []
        remaining_indices = list(range(len(features)))
        running_mean = np.zeros_like(class_mean)

        for _ in range(n_select):
            target = class_mean - running_mean
            distances = ((features[remaining_indices] - target) ** 2).sum(axis=1)
            best_index = remaining_indices[int(np.argmin(distances))]
            selected_indices.append(best_index)
            running_mean = features[selected_indices].mean(axis=0)
            remaining_indices.remove(best_index)

        return np.array(selected_indices, dtype=np.int64)

    def update(
        self,
        model: nn.Module,
        x_clean: torch.Tensor | np.ndarray,
        x_noisy: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        device: torch.device | None = None,
    ) -> None:
        if isinstance(x_clean, torch.Tensor):
            x_clean = x_clean.detach().cpu().numpy()
        if isinstance(x_noisy, torch.Tensor):
            x_noisy = x_noisy.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        if x_clean.shape != x_noisy.shape:
            raise ValueError("x_clean and x_noisy must have the same shape.")

        if device is None:
            first_parameter = next(model.parameters())
            device = first_parameter.device

        for class_id in np.unique(labels):
            class_mask = labels == class_id
            clean_class = x_clean[class_mask].astype(np.float32)
            noisy_class = x_noisy[class_mask].astype(np.float32)
            class_labels = labels[class_mask].astype(np.int64)

            features = self._extract_features(model, noisy_class, device=device)
            selected = self._herding(features, self.max_per_class)

            self._store[int(class_id)] = MemoryEntry(
                clean=clean_class[selected],
                noisy=noisy_class[selected],
                labels=class_labels[selected],
            )

    def sample(
        self,
        snr_db: float | None = None,
        seed: int = 0,
        use_adaptive_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._store:
            return (
                torch.empty(0, 3, 1024, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
            )

        replay_inputs = []
        replay_labels = []
        rng = np.random.RandomState(seed)

        for class_id, entry in self._store.items():
            if entry.labels is None:
                continue

            if use_adaptive_noise and snr_db is not None and entry.clean is not None:
                augmented = np.stack([add_awgn(sample, snr_db=snr_db, rng=rng) for sample in entry.clean], axis=0)
                augmented = np.stack([zscore_per_channel(sample) for sample in augmented], axis=0)
                replay_inputs.append(augmented.astype(np.float32))
            elif entry.noisy is not None:
                replay_inputs.append(entry.noisy.astype(np.float32))
            else:
                continue

            replay_labels.append(entry.labels.astype(np.int64))

        if not replay_inputs:
            return (
                torch.empty(0, 3, 1024, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
            )

        x = torch.from_numpy(np.concatenate(replay_inputs, axis=0))
        y = torch.from_numpy(np.concatenate(replay_labels, axis=0))
        return x, y

    def __len__(self) -> int:
        total = 0
        for entry in self._store.values():
            if entry.labels is not None:
                total += int(len(entry.labels))
        return total
