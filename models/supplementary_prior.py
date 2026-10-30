"""Haar DWT/IDWT and recurrent Supplementary Prior stages."""
import torch
from torch import nn
import torch.nn.functional as F


def dwt2(x):
    """Orthonormal Haar transform. Returns coeffs [B,C,4,ceil(H/2),ceil(W/2)] and original size."""
    h, w = x.shape[-2:]
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
    """Stage-wise recurrent frequency feature guidance (Equation 10 form)."""
    def __init__(self, channels, previous_channels=None):
        super().__init__()
        # Eq. (9)-(10): preserve the low-pass approximation coefficients and
        # apply depthwise filtering only to the three high-frequency bands.
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
        # LL passes unchanged; the following pointwise fusion combines LL with
        # filtered LH/HL/HH and the preceding supplementary feature.
        bands = self.filter_wavelet_bands(encoder_feature).flatten(1,2)
        bands = F.interpolate(bands, size=(h,w), mode="bilinear", align_corners=False)
        prev = torch.zeros_like(encoder_feature) if previous_prior is None else previous_prior
        if prev.shape[-2:] != (h,w):
            prev = F.interpolate(prev, size=(h,w), mode="bilinear", align_corners=False)
        if prev.shape[1] != c:
            prev = self.prev_proj(prev)
        prior = self.fuse(torch.cat((encoder_feature, bands, prev), dim=1))
        return prior + self.refine(prior)
