from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads.")

        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = rearrange(
            self.to_qkv(x),
            "b n (qkv h d) -> qkv b h n d",
            qkv=3,
            h=self.heads,
        )
        attention_logits = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            attention_mask = torch.nn.functional.pad(mask.flatten(1), (1, 0), value=True)
            attention_mask = attention_mask[:, None, :] * attention_mask[:, :, None]
            attention_logits = attention_logits.masked_fill(~attention_mask, float("-inf"))

        attention = attention_logits.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.dropout(self.proj(out))


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim, MultiHeadSelfAttention(dim, heads=heads, dropout=dropout))),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            x = attention(x, mask=mask)
            x = feed_forward(x)
        return x
