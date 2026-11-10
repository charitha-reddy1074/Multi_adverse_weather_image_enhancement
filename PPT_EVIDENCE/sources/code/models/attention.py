"""Paper-guided queried attention; task-query composition remains an assumption."""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from .supplementary_prior import dwt2, idwt2


class QueriedAttention(nn.Module):
    """Figure 5 axes and wavelet branches with explicit assumed query modes.

    add/gate combine projected features with a broadcast learned task vector.
    Neither composition is established by the available author specification.
    spatial_kv_pooling is an opt-in resource approximation; strict is default.
    """
    def __init__(self, channels, mode, heads=4, max_spatial_tokens=1024,
                 query_combine="add", use_task_query=True, spatial_kv_pooling=False):
        super().__init__()
        if mode not in {"channel", "spatial", "wavelet"}:
            raise ValueError(f"unsupported attention mode: {mode}")
        if heads < 1 or max_spatial_tokens < 1 or channels % heads:
            raise ValueError("heads must divide channels; token cap must be positive")
        if mode == "wavelet" and channels % 4:
            raise ValueError("Figure 5(c) IDWT branch requires channels divisible by four")
        if query_combine not in {"add", "gate"}:
            raise ValueError("query_combine must be 'add' or 'gate'")
        self.channels, self.heads, self.dim = channels, heads, channels // heads
        self.mode, self.max_tokens, self.query_combine = mode, max_spatial_tokens, query_combine
        self.spatial_kv_pooling = spatial_kv_pooling
        self.use_task_query = use_task_query
        # Bound temporary logits on builds without fused attention. Chunk only
        # queries: every softmax still contains ALL keys, so no graph pooling.
        self.max_attention_elements = 8 * 1024**2
        self.norm1 = nn.LayerNorm(channels)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        if mode == "wavelet":
            # DWT packs 4C; shared projection gives C at HW/4. Its separate
            # IDWT route produces C/4 at HW, as labeled in Figure 5(c).
            self.wavelet_proj = nn.Conv2d(4 * channels, channels, 1)
        self.out_proj = nn.Conv2d(channels + (channels // 4 if mode == "wavelet" else 0), channels, 1)
        shape = (1, heads, self.dim, 1) if mode == "channel" else (1, heads, 1, self.dim)
        self.task_query = nn.Parameter(torch.zeros(*shape))
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * 2
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(), nn.Conv2d(hidden, channels, 1))

    def _split(self, x):
        b, _, h, w = x.shape
        return x.reshape(b, self.heads, self.dim, h*w).transpose(-1, -2)

    @staticmethod
    def _layer_norm(x, layer_norm):
        return layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        b, c, h, w = x.shape
        # All three blocks normalize the full-resolution input before attention.
        q_in = self._layer_norm(x, self.norm1)
        kv_in = q_in
        inverse = None
        if self.mode == "wavelet":
            coeffs, size = dwt2(q_in)
            hh, ww = coeffs.shape[-2:]
            kv_in = self.wavelet_proj(coeffs.reshape(b, 4*c, hh, ww))
            inverse = idwt2(kv_in.reshape(b, c//4, 4, hh, ww), size)
        if self.mode != "channel" and self.spatial_kv_pooling and kv_in.shape[-2] * kv_in.shape[-1] > self.max_tokens:
            side = max(1, int(self.max_tokens**.5))
            kv_in = F.adaptive_avg_pool2d(kv_in, (side, side))
        q = self._split(self.q_proj(q_in))
        k = self._split(self.k_proj(kv_in))
        v = self._split(self.v_proj(kv_in))
        if self.mode == "channel":
            q = F.normalize(q.transpose(-1, -2), dim=-1)
            k = F.normalize(k.transpose(-1, -2), dim=-1)
            v = v.transpose(-1, -2)
        if self.use_task_query:
            q = q * (1 + torch.tanh(self.task_query)) if self.query_combine == "gate" else q + self.task_query
        self.last_attention_shape = (*q.shape[:-1], k.shape[-2])
        # SDPA computes full attention with softmax over keys without requiring
        # a materialized HW-by-HW matrix on supported CUDA backends.
        def attend(query, key, value):
            # Channel attention uses HW as the embedding dimension. Full-size
            # images can exceed CUTLASS kernel limits even though its attention
            # matrix is only (C/heads)-by-(C/heads). The math backend preserves
            # the same attention and scale without those fused-kernel limits.
            if self.mode == "channel" and query.is_cuda:
                with sdpa_kernel(SDPBackend.MATH):
                    return F.scaled_dot_product_attention(query, key, value,
                        dropout_p=0.0, scale=self.dim**-.5)
            return F.scaled_dot_product_attention(query, key, value,
                dropout_p=0.0, scale=self.dim**-.5)
        count = q.shape[0] * q.shape[1] * q.shape[-2] * k.shape[-2]
        if self.mode != "channel" and count > self.max_attention_elements:
            chunk_size = max(1, self.max_attention_elements // (q.shape[0]*q.shape[1]*k.shape[-2]))
            chunks = []
            for query in q.split(chunk_size, dim=-2):
                # Recompute per-chunk probabilities in backward instead of
                # retaining the full quadratic set of attention activations.
                if torch.is_grad_enabled() and query.requires_grad:
                    chunks.append(checkpoint(attend, query, k, v, use_reentrant=False))
                else:
                    chunks.append(attend(query, k, v))
            y = torch.cat(chunks, dim=-2)
        else:
            y = attend(q, k, v)
        if self.mode == "channel":
            y = y.transpose(-1, -2)
        y = y.transpose(-1, -2).reshape(b, c, h, w)
        if inverse is not None:
            y = torch.cat((y, inverse), dim=1)
        x = x + self.out_proj(y)
        return x + self.mlp(self._layer_norm(x, self.norm2))


class ChannelQueriedSelfAttention(QueriedAttention):
    def __init__(self, channels, **kwargs): super().__init__(channels, "channel", **kwargs)
class SpaceQueriedSelfAttention(QueriedAttention):
    def __init__(self, channels, **kwargs): super().__init__(channels, "spatial", **kwargs)
class WaveletQueriedSelfAttention(QueriedAttention):
    def __init__(self, channels, **kwargs): super().__init__(channels, "wavelet", **kwargs)
