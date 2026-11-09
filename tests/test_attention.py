import pytest
import torch
from models.attention import (ChannelQueriedSelfAttention,
    SpaceQueriedSelfAttention,WaveletQueriedSelfAttention)


@pytest.mark.parametrize("cls",[ChannelQueriedSelfAttention,
    SpaceQueriedSelfAttention,WaveletQueriedSelfAttention])
def test_attention_shape_query_gradient_and_effect(cls):
    module=cls(8,heads=2,max_spatial_tokens=16)
    x=torch.randn(1,8,9,11,requires_grad=True)
    y=module(x)
    assert y.shape==x.shape and not torch.allclose(y,x)
    y.mean().backward()
    assert module.task_query.grad is not None
    assert torch.isfinite(module.task_query.grad).all()


def test_decoder_default_attention_order():
    from models.decoder import FrequencyRefinementBlock
    block=FrequencyRefinementBlock(8,heads=2)
    assert block.order==("channel","space","wavelet")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_full_resolution_cuda_inference():
    # This full-image shape previously failed in channel SDPA with
    # "cutlassF: no kernel found to launch!" after crop-based training passed.
    from models.airformer import AIRFormer
    from utils.inference import _restore_tensor

    device = torch.device("cuda")
    model = AIRFormer(base_channels=16, stage_blocks=(1, 1, 1, 1),
        attention_heads=2, max_spatial_tokens=256, spatial_kv_pooling=True).to(device).eval()
    x = torch.rand(1, 3, 384, 512, device=device)
    y, _ = _restore_tensor(model, x, device, 128)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_channel_attention_matches_cpu_output_and_gradients():
    torch.manual_seed(42)
    cpu = ChannelQueriedSelfAttention(8, heads=2).double()
    gpu = ChannelQueriedSelfAttention(8, heads=2).cuda().double()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(1, 8, 33, 47, dtype=torch.float64, requires_grad=True)
    cuda_x = x.detach().cuda().requires_grad_()
    expected, actual = cpu(x), gpu(cuda_x)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(cuda_x.grad.cpu(), x.grad, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(gpu.task_query.grad.cpu(), cpu.task_query.grad,
                               rtol=1e-3, atol=1e-6)
