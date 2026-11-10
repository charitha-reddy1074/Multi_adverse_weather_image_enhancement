"""DCT consistency plus conventional Smooth L1; printed Eq. 12 differs (A09)."""
import torch
from torch import nn
import torch.nn.functional as F


def dct2(x):
    """Orthonormal DCT-II on the final two dimensions, differentiable in x."""
    if x.ndim < 2:
        raise ValueError("dct2 expects at least two dimensions")
    h,w=x.shape[-2:]
    # Build basis and multiply in fp32 for fp16/bfloat16 inputs. The cast is
    # differentiable and keeps the frequency norm stable under AMP.
    work=x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
    device,dtype=work.device,work.dtype
    ky=torch.arange(h,device=device,dtype=dtype).unsqueeze(1)
    ny=torch.arange(h,device=device,dtype=dtype).unsqueeze(0)
    cy=torch.cos(torch.pi*(2*ny+1)*ky/(2*h))
    cy[0]*=(1.0/h)**0.5
    if h>1: cy[1:]*=(2.0/h)**0.5
    kx=torch.arange(w,device=device,dtype=dtype).unsqueeze(1)
    nx=torch.arange(w,device=device,dtype=dtype).unsqueeze(0)
    cx=torch.cos(torch.pi*(2*nx+1)*kx/(2*w))
    cx[0]*=(1.0/w)**0.5
    if w>1: cx[1:]*=(2.0/w)**0.5
    # Explicitly leave autocast for the basis multiplications so fp16 autocast
    # cannot reduce DCT precision. Disable autocast only for supported devices.
    with torch.autocast(device_type=device.type,enabled=False):
        return torch.matmul(torch.matmul(cy,work),cx.transpose(0,1))


class MultiDomainConsistencyLoss(nn.Module):
    """L_smooth1 + lambda_freq * sqrt(||DCT(p)-DCT(t)||_2^2 + eps^2).

    Reduction policy: PyTorch elementwise Smooth L1 mean across B,C,H,W;
    frequency coefficients use one Frobenius norm per image across C,H,W,
    followed by a mean over batch. A09: printed Eq. (12) defines E as a spatial
    mean residual *before* the piecewise penalty; this implementation averages
    elementwise penalties instead. They differ on cancellation examples. The
    conventional choice is retained, but must not be claimed literally exact.
    """
    def __init__(self,lambda_freq=0.04,eps=0.001):
        super().__init__()
        self.lambda_freq=float(lambda_freq)
        self.eps=float(eps)

    def forward(self,pred,target):
        if pred.shape!=target.shape or pred.ndim!=4:
            raise ValueError("pred and target must have equal NCHW shapes")
        # A09: mean SmoothL1(error), not SmoothL1(mean(error)) in printed Eq. 12.
        spatial=F.smooth_l1_loss(pred.float(),target.float(),reduction="mean")
        difference=(dct2(pred)-dct2(target)).flatten(1)
        frequency=torch.sqrt(difference.square().sum(dim=1)+self.eps**2).mean()
        total=spatial+self.lambda_freq*frequency
        return total,{"smooth_l1":spatial.detach(),"dct_frequency":frequency.detach()}
