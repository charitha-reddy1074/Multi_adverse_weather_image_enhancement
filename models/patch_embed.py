"""Convolutional patch embedding that keeps a dense restoration grid."""
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=32, patch_size=3, stride=1,
                 padding=1, norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size,
                              stride=stride, padding=padding)
        self.norm = nn.GroupNorm(1, embed_dim) if norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))
