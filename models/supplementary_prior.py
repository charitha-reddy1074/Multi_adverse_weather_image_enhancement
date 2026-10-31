"""Haar DWT/IDWT and recurrent Supplementary Prior stages."""
import torch
from torch import nn
import torch.nn.functional as F


def dwt2(x):
    """Orthonormal Haar transform. Returns coeffs [B,C,4,ceil(H/2),ceil(W/2)] and original size."""
    h, w = x.shape[-2:]
    # A08: fixed Haar basis and replicate extension are engineering choices.
    # Eq. (9) defines a general wavelet transform, not this specific family.
    ph, pw = h % 2, w % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    bands = torch.stack(((a+b+c+d)*.5, (-a-b+c+d)*.5,
                         (-a+b-c+d)*.5, (a-b-c+d)*.5), dim=2)
    return bands, (h, w)


def idwt2(coeffs, output_size=None):
    """Inverse of :func:`dwt2`, cropping replicate padding for odd sizes."""
    ll, lh, hl, hh = coeffs.unbind(dim=2)
    a=(ll-lh-hl+hh)*.5; b=(ll-lh+hl-hh)*.5
    c=(ll+lh-hl-hh)*.5; d=(ll+lh+hl+hh)*.5
    out=torch.empty((*ll.shape[:2], ll.shape[-2]*2, ll.shape[-1]*2),
                    device=ll.device, dtype=ll.dtype)
    out[...,0::2,0::2]=a; out[...,0::2,1::2]=b
    out[...,1::2,0::2]=c; out[...,1::2,1::2]=d
    return out[..., :output_size[0], :output_size[1]] if output_size else out


class SupplementaryPrior(nn.Module):
    """Wavelet-guided prior approximation; not a literal Fig. 4 SP graph.

    The code filters only detail bands, resizes packed coefficients to feature
    resolution and fuses current/bands/previous features. Figure 4 instead draws
    fused-filter branches concatenated before refinement. These chosen tensors
    and filters are not established as equivalent by Eq. (10). See A02/A03.
    """
    def __init__(self, channels, previous_channels=None):
        super().__init__()
        # III-A supports preserving low frequencies while filtering details.
        # The exact 3x3 detail filter and 6C fusion below are chosen internals.
        self.detail_filter = nn.Conv2d(channels*3, channels*3, 3, padding=1,
                                       groups=channels*3)
        self.fuse = nn.Sequential(nn.Conv2d(channels*6, channels, 1),
                                  nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))
        self.refine = nn.Sequential(nn.GroupNorm(1, channels),
                                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                    nn.GELU(), nn.Conv2d(channels, channels, 1))
        self.prev_proj = nn.Conv2d(previous_channels or channels, channels, 1)

    def filter_wavelet_bands(self, encoder_feature):
        b,c=encoder_feature.shape[:2]
        bands,_=dwt2(encoder_feature)
        approximation=bands[:,:,0]
        details=bands[:,:,1:4].flatten(1,2)
        details=self.detail_filter(details)
        details=details.reshape(b,c,3,*details.shape[-2:])
        return torch.cat((approximation.unsqueeze(2),details),dim=2)

    def forward(self, encoder_feature, previous_prior=None):
        b, c, h, w = encoder_feature.shape
        # LL passes unchanged inside filter_wavelet_bands. The following fusion
        # can still change it; preservation here does not mean an identity prior.
        bands = self.filter_wavelet_bands(encoder_feature).flatten(1,2)
        # A03: resize coefficients, do not IDWT them. This is not proof of a
        # spatially equivalent depth-wise wavelet convolution from the paper.
        bands = F.interpolate(bands, size=(h,w), mode="bilinear", align_corners=False)
        prev = torch.zeros_like(encoder_feature) if previous_prior is None else previous_prior
        if prev.shape[-2:] != (h,w):
            prev = F.interpolate(prev, size=(h,w), mode="bilinear", align_corners=False)
        if prev.shape[1] != c:
            prev = self.prev_proj(prev)
        prior = self.fuse(torch.cat((encoder_feature, bands, prev), dim=1))
        return prior + self.refine(prior)
