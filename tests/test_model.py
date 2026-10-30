import torch
from models.airformer import AIRFormer


def test_model_shape_backward_odd_input():
    model=AIRFormer(base_channels=8,stage_blocks=(1,1,1),attention_heads=2,
                    max_spatial_tokens=64)
    x=torch.rand(1,3,33,47,requires_grad=True)
    y=model(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(m.task_query.grad is not None for m in model.modules()
               if hasattr(m,"task_query"))


def test_checkpoint_state_roundtrip(tmp_path):
    model=AIRFormer(base_channels=4,stage_blocks=(1,1,1),attention_heads=1,
                    max_spatial_tokens=16)
    path=tmp_path/"state.pt"
    torch.save({"model":model.state_dict(),"model_config":{"base_channels":4,
        "stage_blocks":(1,1,1),"attention_heads":1,"max_spatial_tokens":16}},path)
    state=torch.load(path,map_location="cpu",weights_only=False)
    clone=AIRFormer(**state["model_config"])
    clone.load_state_dict(state["model"])
    for p1,p2 in zip(model.parameters(),clone.parameters()):
        torch.testing.assert_close(p1,p2)
