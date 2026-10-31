import tarfile
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from airformer.data import PairedWeatherDataset
from datasets.archive import safe_extract
from datasets.pairing import assert_disjoint_datasets, image_map
from losses.multidomain import MultiDomainConsistencyLoss
from models.airformer import AIRFormer
from models.attention import QueriedAttention
from utils.inference import load_model, restore_image, symmetric_pad
from utils.metrics import image_metrics


def small_model(**kwargs):
    return AIRFormer(base_channels=4, stage_blocks=(1, 1, 1),
                     attention_heads=1, max_spatial_tokens=16, **kwargs)


@pytest.mark.parametrize("crop,augment,mixup", [(16, True, 1), (16, False, 0), (32, True, 1), (None, False, 0)])
def test_dataset_augmentation_and_mixup_preserve_pairs(tmp_path, crop, augment, mixup):
    for folder in ("degraded", "clean"):
        (tmp_path / folder).mkdir()
        Image.fromarray(np.arange(20*18*3, dtype=np.uint8).reshape(18, 20, 3)).save(tmp_path / folder / "a.png")
    ds = PairedWeatherDataset(tmp_path, crop, augment, mixup)
    for _ in range(4):
        x, y, _ = ds[0]
        torch.testing.assert_close(x, y)
        if crop:
            assert x.shape == (3, crop, crop)


def test_folder_rejects_unpaired_data(tmp_path):
    (tmp_path / "clean").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "clean" / "a.png")
    with pytest.raises(ValueError, match="Unpaired"):
        PairedWeatherDataset(tmp_path)


def test_renamed_cross_source_clean_scene_leakage(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "renamed.png"
    Image.new("RGB", (9, 10), (100, 50, 90)).save(first)
    Image.new("RGB", (9, 10), (100, 50, 90)).save(second)
    a = SimpleNamespace(records=[(first, first, "rain", "sourceA", "train", "a")])
    b = SimpleNamespace(records=[(second, second, "haze", "sourceB", "val", "b")])
    with pytest.raises(ValueError, match="clean_pixels"):
        assert_disjoint_datasets(a, b)


@pytest.mark.parametrize("name", ["../escape.txt", "C:/escape.txt", "a/../../escape.txt", "a:stream", "..\\escape.txt"])
def test_archive_traversal_rejected_before_writing(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("valid.txt", "must not be written")
        z.writestr(name, "bad")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "valid.txt").exists()


def test_archive_rejects_tar_links(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../escape"
        tar.addfile(info)
    with pytest.raises(ValueError, match="Unsupported"):
        safe_extract(archive, tmp_path / "out")


def test_archive_safe_resume_and_no_overwrite(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("nested/data.txt", "hello")
    safe_extract(archive, tmp_path / "out")
    safe_extract(archive, tmp_path / "out")
    (tmp_path / "out/nested/data.txt").write_text("other")
    with pytest.raises(ValueError, match="differs"):
        safe_extract(archive, tmp_path / "out")


@pytest.mark.parametrize("mode", ["channel", "spatial", "wavelet"])
def test_task_queries_update_and_affect_output(mode):
    torch.manual_seed(42)
    module = QueriedAttention(8, mode, heads=2, max_spatial_tokens=16)
    x = torch.randn(1, 8, 7, 9)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    before = module.task_query.detach().clone()
    module(x).square().mean().backward()
    assert module.task_query.grad.abs().sum() > 0
    assert torch.isfinite(module.task_query.grad).all()
    optimizer.step()
    assert not torch.equal(module.task_query, before)
    enabled = module(x)
    module.use_task_query = False
    assert not torch.allclose(enabled, module(x), atol=1e-8, rtol=1e-8)


def test_sp_changes_output_and_receives_gradients():
    torch.manual_seed(43)
    model = small_model()
    x = torch.rand(1, 3, 16, 16)
    enabled = model(x)
    enabled.square().mean().backward()
    for stage in model.supplementary_priors:
        grads = [p.grad for p in stage.parameters() if p.grad is not None]
        assert sum(g.abs().sum() for g in grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)
    model.use_supplementary_prior = False
    assert not torch.allclose(enabled, model(x))


def test_checkpointed_decoder_equivalent_outputs_and_gradients():
    torch.manual_seed(44)
    regular, checkpointed = small_model(), small_model(gradient_checkpointing=True)
    checkpointed.load_state_dict(regular.state_dict())
    x = torch.rand(1, 3, 16, 20)
    a, b = regular(x), checkpointed(x)
    torch.testing.assert_close(a, b)
    a.square().mean().backward(); b.square().mean().backward()
    for p, q in zip(regular.parameters(), checkpointed.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tiny_batch_overfit(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(45)
    model = small_model().to(device)
    x = torch.rand(1, 3, 12, 12, device=device)
    target = x * 0.8
    loss_fn = MultiDomainConsistencyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    initial = loss_fn(model(x), target)[0].item()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), target)[0]
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    final = loss_fn(model(x), target)[0].item()
    assert final < initial * 0.65, (initial, final)


def test_real_checkpoint_loader_and_inference_dimensions(tmp_path):
    model = small_model()
    path = tmp_path / "state.pt"
    torch.save({"model": model.state_dict(), "model_config": dict(base_channels=4,
        stage_blocks=(1, 1, 1), attention_heads=1, max_spatial_tokens=16)}, path)
    loaded, device = load_model(path, "cpu")
    image = Image.new("RGB", (19, 17), (100, 80, 90))
    output, seconds = restore_image(loaded, image, device)
    assert output.size == image.size and seconds > 0


def test_symmetric_padding_geometry_and_crop():
    x = torch.rand(1, 3, 17, 19)
    padded, (left, top, width, height) = symmetric_pad(x)
    assert padded.shape[-2:] == (128, 128)
    torch.testing.assert_close(padded[..., top:top+height, left:left+width], x)


def test_metrics_known_noise_and_shape_rejection():
    target = np.full((12, 14, 3), 0.5, dtype=np.float32)
    psnr, ssim = image_metrics(target + 0.1, target)
    assert abs(psnr - 20) < 1e-4 and 0 < ssim < 1


def test_duplicate_extension_identity_rejected(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    with pytest.raises(ValueError, match="duplicate"):
        image_map(tmp_path)


@pytest.mark.parametrize("mode,expected", [("channel", (1, 2, 4, 4)),
    ("spatial", (1, 2, 63, 16)), ("wavelet", (1, 2, 20, 16))])
def test_attention_axes_softmax_normalization_and_kv_shapes(monkeypatch, mode, expected):
    original = torch.softmax
    seen = []

    def capture(tensor, dim, *args, **kwargs):
        result = original(tensor, dim, *args, **kwargs)
        seen.append((tuple(tensor.shape), dim, result.detach()))
        return result

    monkeypatch.setattr(torch, "softmax", capture)
    module = QueriedAttention(8, mode, heads=2, max_spatial_tokens=16)
    module(torch.randn(1, 8, 7, 9))
    shape, dim, probabilities = seen[0]
    assert shape == expected and dim == -1
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)))


def test_spatial_attention_has_nonlocal_effect():
    torch.manual_seed(72)
    module = QueriedAttention(8, "spatial", heads=2, max_spatial_tokens=128)
    x = torch.randn(1, 8, 7, 9)
    changed = x.clone()
    changed[:, 0, -1, -1] += 5
    assert not torch.allclose(module(x)[..., 0, 0], module(changed)[..., 0, 0], atol=1e-8, rtol=1e-8)
