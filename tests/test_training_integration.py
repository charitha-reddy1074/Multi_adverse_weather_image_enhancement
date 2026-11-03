import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from datasets.pairing import write_manifest


def test_cli_train_resume_rng_and_safe_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for split, color in (("train", (100, 80, 90)), ("val", (80, 90, 110))):
        path = tmp_path / f"{split}.png"
        Image.new("RGB", (12, 14), color).save(path)
        write_manifest([dict(input_path=str(path), target_path=str(path),
            weather_type="rain", source_dataset="fixture", split=split, scene_id=split)], tmp_path / f"{split}.csv")
    cfg = dict(seed=42, epochs=1, batch_size=1, crop_size=12, lr=0.0002,
        min_lr=1e-7, base_channels=4, stage_blocks=[1, 1, 1, 1], attention_heads=1,
        max_spatial_tokens=16, dataset_root=str(tmp_path), num_workers=0,
        train_manifest=str(tmp_path / "train.csv"), val_manifest=str(tmp_path / "val.csv"),
        mixup_prob=0, mixed_precision=torch.cuda.is_available())
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    command = [sys.executable, str(root / "train.py"), "--config", str(config), "--output", str(tmp_path / "checkpoints")]
    first = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint = tmp_path / "checkpoints/latest.pt"
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved["epoch"] == 1 and saved["rng"]["torch"].dtype == torch.uint8
    assert saved["optimizer"]["state"] and saved["scheduler"]["last_epoch"] == 1
    cfg["epochs"] = 2
    config.write_text(yaml.safe_dump(cfg))
    resumed = subprocess.run(command + ["--resume", str(checkpoint)], cwd=root,
                             capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved["epoch"] == 2 and saved["scheduler"]["last_epoch"] == 2


def test_train_test_only_uses_final_model_without_test_checkpoint_selection(tmp_path):
    import json
    for split, color in (("train", (20,40,60)), ("test", (80,100,120))):
        image = tmp_path / f"{split}.png"
        Image.new("RGB", (12,14), color).save(image)
        write_manifest([dict(input_path=str(image), target_path=str(image), weather_type="snow",
            source_dataset="fixture", split=split, scene_id=split)], tmp_path / f"{split}.csv")
    cfg = dict(seed=42, epochs=1, batch_size=1, crop_size=12, lr=2e-4, min_lr=1e-7,
        base_channels=4, stage_blocks=[1,1,1,1], attention_heads=1, num_workers=0,
        dataset_root=str(tmp_path), train_manifest=str(tmp_path/'train.csv'),
        test_manifest=str(tmp_path/'test.csv'), validation_enabled=False, mixup_prob=0)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'train.py'), '--config', str(config),
        '--output', str(tmp_path/'checkpoints')], cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'no validation/model selection' in result.stdout
    assert result.stdout.count('final_test=') == 1
    assert not (tmp_path/'checkpoints/best.pt').exists()
    saved = torch.load(tmp_path/'checkpoints/latest.pt', weights_only=True, map_location='cpu')
    assert saved['val'] is None
    assert 'psnr' in json.loads((tmp_path/'checkpoints/final_test.json').read_text())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_overflow_recovers_without_corrupting_weights():
    from models.airformer import AIRFormer
    from losses.multidomain import MultiDomainConsistencyLoss
    torch.manual_seed(42)
    model = AIRFormer(base_channels=8, stage_blocks=(1, 1, 1, 1), attention_heads=2,
                      max_spatial_tokens=32).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=2**20)
    x = torch.rand(1, 3, 32, 32, device="cuda")
    successful = False
    skipped = False
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss = MultiDomainConsistencyLoss()(model(x), x * 0.8)[0]
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=False)
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if scaler.get_scale() < old_scale:
            skipped = True
        else:
            successful = True
            break
    assert skipped and successful
