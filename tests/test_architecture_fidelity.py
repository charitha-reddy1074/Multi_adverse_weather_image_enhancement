"""Regression checks for confirmed graph corrections, not author equivalence."""
import torch
from models.airformer import AIRFormer
from models.attention import QueriedAttention


def test_prior_previous_feature_and_preblock_injection():
    model = AIRFormer(base_channels=4, stage_blocks=(1,1,1,1), attention_heads=1)
    stage_in, stage_out, patch_out, prior_in, prior_out = {}, {}, {}, {}, {}
    handles = []
    def before(storage, index):
        def capture(_, args):
            storage[index] = tuple(t.detach().clone() for t in args)
        return capture
    def after(storage, index):
        def capture(_, args, result):
            storage[index] = result.detach().clone()
        return capture
    for i, stage in enumerate(model.encoder.stages):
        handles.append(stage.register_forward_pre_hook(before(stage_in, i)))
        handles.append(stage.register_forward_hook(after(stage_out, i)))
    for i, patch in enumerate(model.encoder.downsamples):
        handles.append(patch.register_forward_hook(after(patch_out, i)))
    for i, prior in enumerate(model.supplementary_priors):
        handles.append(prior.register_forward_pre_hook(before(prior_in, i)))
        handles.append(prior.register_forward_hook(after(prior_out, i)))
    model(torch.rand(1,3,16,24))
    for handle in handles:
        handle.remove()
    assert torch.count_nonzero(prior_in[0][0]) == torch.count_nonzero(prior_in[0][1]) == 0
    for i in range(3):
        torch.testing.assert_close(stage_in[i+1][0], patch_out[i]+prior_out[i])
        if i:
            torch.testing.assert_close(prior_in[i][0], stage_out[i])
            torch.testing.assert_close(prior_in[i][1], prior_out[i-1])


def test_wavelet_full_grid_normalization_concat_and_both_branch_gradients():
    module = QueriedAttention(8, 'wavelet', heads=2, max_spatial_tokens=1)
    seen = {}
    def capture(_, args):
        seen['concat'] = args[0]
        args[0].retain_grad()
    handle = module.out_proj.register_forward_pre_hook(capture)
    x = torch.randn(1,8,7,9,requires_grad=True)
    y = module(x)
    y.square().mean().backward()
    handle.remove()
    assert module.norm1.normalized_shape == (8,)
    assert module.last_attention_shape == (1,2,63,20)
    assert seen['concat'].shape == (1,10,7,9)
    for branch in (seen['concat'].grad[:,:8], seen['concat'].grad[:,8:]):
        assert torch.isfinite(branch).all() and branch.abs().sum() > 0
    for layer in (module.q_proj, module.k_proj, module.v_proj, module.wavelet_proj):
        assert layer.weight.grad.abs().sum() > 0


def test_pooling_requires_explicit_resource_opt_in():
    strict = QueriedAttention(8, 'spatial', heads=2, max_spatial_tokens=4)
    pooled = QueriedAttention(8, 'spatial', heads=2, max_spatial_tokens=4, spatial_kv_pooling=True)
    x = torch.randn(1,8,7,9)
    strict(x); pooled(x)
    assert strict.last_attention_shape == (1,2,63,63)
    assert pooled.last_attention_shape == (1,2,63,4)


def test_query_chunking_preserves_full_attention_outputs_and_gradients():
    for mode in ('spatial', 'wavelet'):
        full = QueriedAttention(8, mode, heads=2)
        chunked = QueriedAttention(8, mode, heads=2)
        chunked.load_state_dict(full.state_dict())
        chunked.max_attention_elements = 256
        a = torch.randn(1,8,7,9,requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        y, z = full(a), chunked(b)
        torch.testing.assert_close(y, z, atol=1e-6, rtol=1e-5)
        y.square().mean().backward(); z.square().mean().backward()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
        for p, q in zip(full.parameters(), chunked.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
