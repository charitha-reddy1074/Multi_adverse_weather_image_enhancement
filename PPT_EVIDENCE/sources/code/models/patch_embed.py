"""Convolutional patch embedding that keeps a dense restoration grid."""
from torch import nn


class PatchEmbed(nn.Module):
    """Chosen dense 3x3 convolution + GroupNorm(1,C) patch projection.

    Figure 4 names patch embeddings at each level; its kernels, strides and
    feature widths are not specified. This class alone does not reproduce them.
    """
    def __init__(self, in_channels=3, embed_dim=32, patch_size=3, stride=1,
                 padding=1, norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size,
                              stride=stride, padding=padding)
        self.norm = nn.GroupNorm(1, embed_dim) if norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))
