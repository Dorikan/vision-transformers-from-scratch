"""
Модуль патч эмбеддера.
Возможно в будущем будет расширяться другими реализациями
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        assert image_size % patch_size == 0

        self.embed_dim = embed_dim
        self.n_patches = (image_size // patch_size) ** 2

        self.conv = nn.Conv2d(3, self.embed_dim, patch_size, stride=patch_size) # (B, C, H, W)
        self.cls = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, self.embed_dim))

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        cls_tokens = self.cls.expand(x.shape[0], -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)

        assert x.shape[1] == self.n_patches + 1

        x = x + self.pos_embed
        return x
