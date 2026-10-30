import torch
from torch import nn
import torch.nn.functional as F


def dct2(x):
    """Orthonormal 2D DCT-II using differentiable cosine basis matrices."""
    h, w = x.shape[-2:]
    # Trigonometric basis construction and matrix products use fp32 under AMP;
    # gradients still flow back through the cast to half precision inputs.
    x=x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
    device, dtype = x.device, x.dtype
    ky = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)
    ny = torch.arange(h, device=device, dtype=dtype).unsqueeze(0)
    cy = torch.cos(torch.pi * (2 * ny + 1) * ky / (2 * h))
    cy[0] *= (1.0 / h) ** 0.5
    if h > 1:
        cy[1:] *= (2.0 / h) ** 0.5
    kx = torch.arange(w, device=device, dtype=dtype).unsqueeze(1)
    nx = torch.arange(w, device=device, dtype=dtype).unsqueeze(0)
    cx = torch.cos(torch.pi * (2 * nx + 1) * kx / (2 * w))
    cx[0] *= (1.0 / w) ** 0.5
    if w > 1:
        cx[1:] *= (2.0 / w) ** 0.5
    return torch.matmul(torch.matmul(cy, x), cx.transpose(0, 1))


class MultiDomainConsistencyLoss(nn.Module):
    def __init__(self, lambda_freq=0.04, eps=0.001):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.eps = eps
    def forward(self, pred, target):
        spatial = F.smooth_l1_loss(pred.float(), target.float())
        freq_pred, freq_target = dct2(pred), dct2(target)
        # Explicit reduction policy: one Frobenius norm over all RGB/frequency
        # coefficients per image, then average the batch. This follows the
        # equation while making the batch reduction deterministic.
        diff = (freq_pred - freq_target).flatten(1)
        frequency = torch.sqrt(diff.square().sum(dim=1) + self.eps ** 2).mean()
        return spatial + self.lambda_freq * frequency, {
            "smooth_l1": spatial.detach(),
            "dct_frequency": frequency.detach()
        }
