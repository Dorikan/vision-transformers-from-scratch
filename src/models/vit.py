"""
Реализация ViT на PyTorch
"""
import torch.nn as nn

from src.modules import TransformerBlock, PatchEmbedding


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, n_heads, num_classes=10, n_blocks=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embedder = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)

        n_patches = self.patch_embedder.n_patches

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio)
            for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedder(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0, :]
        x = self.classifier(cls)
        return x
