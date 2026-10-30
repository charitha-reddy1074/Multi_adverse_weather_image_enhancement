"""Task-biased channel, spatial, and wavelet queried attention."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .supplementary_prior import dwt2, idwt2


class QueriedAttention(nn.Module):
    def __init__(self, channels, mode, heads=4, max_spatial_tokens=1024,
                 query_combine="add", use_task_query=True):
        super().__init__()
        if mode not in {"channel","spatial","wavelet"}:
            raise ValueError(f"unsupported attention mode: {mode}")
        if heads<1 or max_spatial_tokens<1:
            raise ValueError("heads and max_spatial_tokens must be positive")
        if query_combine not in {"add","gate"}:
            raise ValueError("query_combine must be 'add' or 'gate'")
        self.q_channels = channels * 4 if mode == "wavelet" else channels
        if self.q_channels % heads:
            heads = math.gcd(self.q_channels, heads)
        self.channels, self.heads, self.dim = channels, heads, self.q_channels//heads
        self.mode, self.max_tokens, self.query_combine = mode, max_spatial_tokens, query_combine
        self.use_task_query=use_task_query
        self.norm1 = nn.LayerNorm(self.q_channels)
        self.q_proj = nn.Conv2d(self.q_channels, self.q_channels, 1)
        self.k_proj = nn.Conv2d(self.q_channels, self.q_channels, 1)
        self.v_proj = nn.Conv2d(self.q_channels, self.q_channels, 1)
        self.out_proj = nn.Conv2d(self.q_channels, self.q_channels, 1)
        query_shape = (1, heads, self.dim, 1) if mode == "channel" else (1, heads, 1, self.dim)
        self.task_query = nn.Parameter(torch.zeros(*query_shape))
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * 2
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                                  nn.GELU(), nn.Conv2d(hidden, channels, 1))

    def _split(self, x):
        b, _, h, w = x.shape
        return x.reshape(b, self.heads, self.dim, h*w).transpose(-1,-2)

    @staticmethod
    def _layer_norm(x, layer_norm):
        return layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

    def forward(self, x):
        residual=x
        if self.mode == "wavelet":
            coeffs, size = dwt2(x)
            b, c, bands, hh, ww = coeffs.shape
            kv_in = coeffs.reshape(b, c*bands, hh, ww)
            q_in = self._layer_norm(kv_in,self.norm1)
        else:
            q_in=self._layer_norm(x,self.norm1); kv_in=q_in
        q=self._split(self.q_proj(q_in))
        k=self._split(self.k_proj(kv_in))
        v=self._split(self.v_proj(kv_in))
        if self.mode == "channel":
            # Channel tokens attend over channel tokens after spatial whitening.
            q=F.normalize(q.transpose(-1,-2),dim=-1)
            k=F.normalize(k.transpose(-1,-2),dim=-1)
            bias=self.task_query
            if self.use_task_query and self.query_combine == "add": q=q+bias
            elif self.use_task_query and self.query_combine == "gate": q=q*(1+torch.tanh(bias))
            attn=torch.softmax((q @ k.transpose(-1,-2))*self.dim**-.5,dim=-1)
            y=attn @ v.transpose(-1,-2)
            y=y.transpose(-1,-2)
        else:
            if k.shape[-2] > self.max_tokens:
                # Reduced-resolution K/V retains full-resolution queries.
                side=max(1,int(math.sqrt(self.max_tokens)))
                small=F.adaptive_avg_pool2d(kv_in,(side,side))
                k=self._split(self.k_proj(small)); v=self._split(self.v_proj(small))
            if self.use_task_query:
                q=q*(1+torch.tanh(self.task_query)) if self.query_combine=="gate" else q+self.task_query
            attn=torch.softmax((q @ k.transpose(-1,-2))*self.dim**-.5,dim=-1)
            y=attn @ v
        b,_,h,w=x.shape
        if self.mode == "wavelet":
            hh, ww = coeffs.shape[-2:]
            y=y.transpose(-1,-2).reshape(b,self.q_channels,hh,ww)
            y=self.out_proj(y).reshape(b,self.channels,4,hh,ww)
            update=idwt2(y,size)
        else:
            y=y.transpose(-1,-2).reshape(b,self.channels,h,w)
            update=self.out_proj(y)
        if self.mode == "wavelet":
            # Attention output gates/refines the real wavelet coefficients;
            # inverse transform returns a spatial feature at original size.
            update=update + residual
            x=update
        else:
            x=residual+update
        return x+self.mlp(self._layer_norm(x,self.norm2))


class ChannelQueriedSelfAttention(QueriedAttention):
    def __init__(self, channels, **kwargs): super().__init__(channels,"channel",**kwargs)
class SpaceQueriedSelfAttention(QueriedAttention):
    def __init__(self, channels, **kwargs): super().__init__(channels,"spatial",**kwargs)
class WaveletQueriedSelfAttention(QueriedAttention):
    def __init__(self, channels, **kwargs): super().__init__(channels,"wavelet",**kwargs)
