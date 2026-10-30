import itertools
import pytest
import torch
from models.encoder import FrequencyGuidedBlock
from models.decoder import FrequencyRefinementBlock
from models.airformer import AIRFormer


ORDERS=list(itertools.permutations(("channel","space","wavelet")))


@pytest.mark.parametrize("order",ORDERS)
def test_all_table_xii_orders_construct_and_run(order):
    block=FrequencyRefinementBlock(8,order=order,heads=2,max_spatial_tokens=16)
    x=torch.randn(1,8,8,10)
    y=block(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    assert block.order==order


@pytest.mark.parametrize("mode,expected_blocks",[("spatial",1),("frequency",3),("none",0)])
def test_decoder_ablation_modes(mode,expected_blocks):
    block=FrequencyRefinementBlock(8,heads=2,mode=mode)
    assert len(block.blocks)==expected_blocks
    x=torch.randn(1,8,8,8)
    y=block(x)
    assert y.shape==x.shape
    if mode=="none": torch.testing.assert_close(y,x)


def test_encoder_token_mixer_ablation_shapes():
    x=torch.randn(1,8,8,10)
    for mixer in ("pooling","self_attention"):
        block=FrequencyGuidedBlock(8,token_mixer=mixer,attention_heads=2,max_spatial_tokens=16)
        y=block(x)
        assert y.shape==x.shape and torch.isfinite(y).all()


def test_model_exposes_component_ablation_switches():
    model=AIRFormer(base_channels=4,stage_blocks=(1,1,1),attention_heads=1,
        encoder_token_mixer="self_attention",use_supplementary_prior=False,
        use_task_queries=False,decoder_mode="spatial")
    y=model(torch.rand(1,3,17,19))
    assert y.shape==(1,3,17,19) and torch.isfinite(y).all()
