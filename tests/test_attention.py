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
