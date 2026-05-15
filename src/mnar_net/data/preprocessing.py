from __future__ import annotations

import math
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.RandomState | None = None) -> np.ndarray:
    if signal.ndim != 2:
        raise ValueError(f"Expected shape (channels, length), got {signal.shape}.")
    rng = rng or np.random.RandomState()
    noisy = np.empty_like(signal, dtype=np.float32)
    for channel_index in range(signal.shape[0]):
        channel = signal[channel_index].astype(np.float32)
        signal_power = float(np.mean(channel ** 2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(loc=0.0, scale=math.sqrt(noise_power), size=channel.shape).astype(np.float32)
        noisy[channel_index] = channel + noise
    return noisy


def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected shape (channels, length), got {x.shape}.")
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def sliding_windows(signal: np.ndarray, window_size: int, step: int) -> np.ndarray:
    if signal.ndim != 2:
        raise ValueError(f"Expected shape (channels, length), got {signal.shape}.")
    channels, length = signal.shape
    if length < window_size:
        return np.zeros((0, channels, window_size), dtype=np.float32)

    starts = list(range(0, length - window_size + 1, step))
    windows = np.empty((len(starts), channels, window_size), dtype=np.float32)
    for index, start in enumerate(starts):
        windows[index] = signal[:, start : start + window_size]
    return windows
