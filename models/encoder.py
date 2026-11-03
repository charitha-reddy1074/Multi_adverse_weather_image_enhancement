"""Hierarchical PoolFormer-style feature encoder."""
import math
import torch
import torch.nn.functional as F
from torch import nn


class FrequencyGuidedBlock(nn.Module):
    """Fig. 4-style GN/mixer/residual/GN/MLP/residual, with chosen internals.

    AvgPool(z)-z follows the PoolFormer-inspired interpretation; Fig. 4 only
    labels 'Pooling'. Group count, MLP ratio, activations and edge padding are
    explicit reconstruction assumptions, not specified AIRFormer hyperparameters.
    """
    def __init__(self, channels, mlp_ratio=2.0, pool_size=3,
                 token_mixer="pooling", attention_heads=4,max_spatial_tokens=1024):
        super().__init__()
        if token_mixer not in {"pooling","self_attention"}:
            raise ValueError("token_mixer must be pooling or self_attention")
        hidden = max(channels, int(channels * mlp_ratio))
        self.norm1 = nn.GroupNorm(1, channels)
        self.pool_size = pool_size
        self.token_mixer=token_mixer
        self.max_spatial_tokens=max_spatial_tokens
        self.attention=(nn.MultiheadAttention(channels,
            num_heads=math.gcd(channels,attention_heads),batch_first=True)
            if token_mixer=="self_attention" else None)
        self.norm2 = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(), nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        # Normalize before local token mixing, then add the first residual.
        z = self.norm1(x)
        if self.token_mixer=="pooling":
            pooled = F.avg_pool2d(z, self.pool_size, stride=1,
                                  padding=self.pool_size // 2)
            x = x + (pooled - z)
        else:
            b,c,h,w=z.shape
            query=z.flatten(2).transpose(1,2)
            if h*w>self.max_spatial_tokens:
                side=max(1,int(self.max_spatial_tokens**.5))
                kv=F.adaptive_avg_pool2d(z,(side,side)).flatten(2).transpose(1,2)
            else: kv=query
            mixed=self.attention(query,kv,kv,need_weights=False)[0]
            x=x+mixed.transpose(1,2).reshape(b,c,h,w)
        # The second pre-normalized branch mixes channels and local neighborhoods.
        return x + self.mlp(self.norm2(x))


class HierarchicalEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), depths=(2, 2, 2, 2),
                 token_mixer="pooling",attention_heads=4,max_spatial_tokens=1024):
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("Figure 4 requires four encoder positions")
        self.stages = nn.ModuleList([
            nn.Sequential(*(FrequencyGuidedBlock(c,token_mixer=token_mixer,
                attention_heads=attention_heads,max_spatial_tokens=max_spatial_tokens)
                for _ in range(d)))
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
                preceding = x
                x = self.downsamples[i-1](x)
                if supplementary_modules is not None:
                    # Eq. 10: first SP sees zero feature and zero prior. Later
                    # SPs see the preceding enhanced stage and preceding SP.
                    source = torch.zeros_like(preceding) if i == 1 else preceding
                    prev = torch.zeros_like(preceding) if i == 1 else previous
                    previous = supplementary_modules[i-1](source, prev)
                    if previous.shape != x.shape:
                        raise RuntimeError("Supplementary prior and patch embedding must align")
                    priors.append(previous)
                    x = x + previous
            x = stage(x)
            features.append(x)
        return features, priors
