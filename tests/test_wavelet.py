import pytest
import torch
from models.supplementary_prior import dwt2,idwt2,SupplementaryPrior
from models.encoder import HierarchicalEncoder
from torch import nn


@pytest.mark.parametrize("shape",[(2,3,16,20),(1,4,17,19),(1,2,16,19)])
def test_dwt_idwt_shape_roundtrip_and_gradient(shape):
    x=torch.randn(*shape,requires_grad=True)
    coeffs,size=dwt2(x)
    assert coeffs.shape==(shape[0],shape[1],4,(shape[2]+1)//2,(shape[3]+1)//2)
    y=idwt2(coeffs,size)
    assert y.shape==x.shape
    torch.testing.assert_close(y,x,atol=1e-6,rtol=1e-6)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_prior_channel_scale_alignment():
    prior=SupplementaryPrior(16,previous_channels=8)
    x=torch.randn(2,16,15,17,requires_grad=True)
    prev=torch.randn(2,8,8,9)
    y=prior(x,prev)
    assert y.shape==x.shape and torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None


def test_supplementary_prior_preserves_approximation_band():
    x=torch.randn(1,4,12,14)
    prior=SupplementaryPrior(4)
    reference,_=dwt2(x)
    filtered=prior.filter_wavelet_bands(x)
    torch.testing.assert_close(filtered[:,:,0],reference[:,:,0])


def test_encoder_starts_first_supplementary_prior_from_zeros():
    class Recorder(nn.Module):
        def __init__(self, channels):
            super().__init__(); self.inputs=None; self.channels=channels
        def forward(self,x,previous):
            self.inputs=(x.detach().clone(),previous.detach().clone())
            return torch.zeros(x.shape[0], self.channels, (x.shape[-2]+1)//2,
                               (x.shape[-1]+1)//2, device=x.device)
    enc=HierarchicalEncoder((4,8,16,32),(1,1,1,1))
    priors=nn.ModuleList([Recorder(8),Recorder(16),Recorder(32)])
    features,_=enc(torch.randn(1,4,16,16),priors)
    assert len(features)==4
    assert torch.count_nonzero(priors[0].inputs[0])==0
    assert torch.count_nonzero(priors[0].inputs[1])==0
    torch.testing.assert_close(priors[1].inputs[0], features[1])
    torch.testing.assert_close(priors[2].inputs[0], features[2])
