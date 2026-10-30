import pytest
import torch
from models.supplementary_prior import dwt2,idwt2,SupplementaryPrior


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
