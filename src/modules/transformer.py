"""
Переиспользуемая реализация блока трансформера
Возможно в будущем будет дополняться другими реализациями
"""
import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.mha = MultiHeadSelfAttention(n_heads, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
