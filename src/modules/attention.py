"""
Модуль где будут храниться переиспользуемые реализации внимания
"""
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int, embedding_size: int):
        super().__init__()
        assert embedding_size % n_heads == 0
        self.head_dim = embedding_size // n_heads
        self.n_heads = n_heads
        self.embedding_size = embedding_size

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)
        self.W_O = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # [B, N, D]
        B, N, D = x.shape
        Q = Q.view((B, N, self.n_heads, self.head_dim)).transpose(1, 2)  # [B, H, N, d_K]
        K = K.view((B, N, self.n_heads, self.head_dim)).transpose(1, 2)  # [B, H, N, d_K]
        V = V.view((B, B, self.n_heads, self.head_dim)).transpose(1, 2)  # [B, H, N, d_K]

        qk = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(qk, dim=-1)

        head_out = attn_weights @ V
        head_out = head_out.transpose(1, 2).contiguous().view((b, n, self.embedding_size))

        out = self.W_O(head_out)
        return out
