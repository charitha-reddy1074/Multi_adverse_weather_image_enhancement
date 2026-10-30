import pytest
import torch
from losses.multidomain import dct2,MultiDomainConsistencyLoss


def test_dct_constant_and_gradient():
    x=torch.ones(2,3,8,10,requires_grad=True)
    coeff=dct2(x)
    assert torch.allclose(coeff[...,1:,:],torch.zeros_like(coeff[...,1:,:]),atol=1e-5)
    assert torch.allclose(coeff[...,:,1:],torch.zeros_like(coeff[...,:,1:]),atol=1e-5)
    coeff.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_orthonormal_dct_preserves_energy():
    x=torch.randn(2,3,9,12)
    coeff=dct2(x)
    torch.testing.assert_close(coeff.square().sum((-2,-1)),x.square().sum((-2,-1)),atol=2e-5,rtol=2e-5)


def test_loss_identical_finite_and_nonzero_gradient():
    loss_fn=MultiDomainConsistencyLoss(.04,.001)
    target=torch.rand(2,3,16,16)
    pred=target.clone().requires_grad_()
    loss,parts=loss_fn(pred,target)
    assert torch.isfinite(loss) and abs(loss.item()-.00004)<1e-7
    (loss+pred.square().mean()).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    pred2=(target+.05).requires_grad_()
    loss2,_=loss_fn(pred2,target)
    loss2.backward()
    assert loss2.item()>0 and pred2.grad.abs().sum()>0


def test_half_input_loss_stays_finite():
    pred=torch.rand(1,3,8,8,dtype=torch.float16,requires_grad=True)
    target=torch.rand_like(pred)
    loss,_=MultiDomainConsistencyLoss()(pred,target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


@pytest.mark.parametrize("device",["cpu"]+( ["cuda"] if torch.cuda.is_available() else [] ))
def test_loss_device_compatibility(device):
    pred=torch.rand(1,3,12,10,device=device,requires_grad=True)
    target=torch.rand_like(pred)
    loss,_=MultiDomainConsistencyLoss()(pred,target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_cpu_bfloat16_autocast_compatibility():
    pred=torch.rand(1,3,8,8,requires_grad=True)
    target=torch.rand_like(pred)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        loss,_=MultiDomainConsistencyLoss()(pred,target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
