"""Hierarchical PoolFormer-style feature encoder."""
import torch.nn.functional as F
from torch import nn


class FrequencyGuidedBlock(nn.Module):
    """PoolFormer token mixer plus a depthwise channel MLP."""
    def __init__(self, channels, mlp_ratio=2.0, pool_size=3):
        super().__init__()
        hidden = max(channels, int(channels * mlp_ratio))
        self.norm1 = nn.GroupNorm(1, channels)
        self.pool_size = pool_size
        self.norm2 = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(), nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        z = self.norm1(x)
        pooled = F.avg_pool2d(z, self.pool_size, stride=1,
                              padding=self.pool_size // 2)
        x = x + (pooled - z)
        return x + self.mlp(self.norm2(x))


class HierarchicalEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128), depths=(2, 2, 2)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(*(FrequencyGuidedBlock(c) for _ in range(d)))
            for c, d in zip(channels, depths)
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1)
            for i in range(len(channels)-1)
        ])

    def forward(self, x, supplementary_modules=None):
        features, priors = [], []
        previous = None
        for i, stage in enumerate(self.stages):
            if i:
                x = self.downsamples[i-1](x)
            x = stage(x)
            if supplementary_modules is not None:
                previous = supplementary_modules[i](x, previous)
                priors.append(previous)
                x = x + previous
            features.append(x)
        return features, priors
