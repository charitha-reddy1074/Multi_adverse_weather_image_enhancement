"""Real-pair CLI integration and repeatable pilot runtime/VRAM measurements.

Creates a deliberately tiny randomly initialized checkpoint under outputs/.
It is a pipeline diagnostic, never a pretrained restoration model or paper score.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.pairing import write_manifest
from losses.multidomain import MultiDomainConsistencyLoss
from models.airformer import AIRFormer


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "audit_smoke"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((root / "configs/pilot.yaml").read_text())
    model_cfg = dict(base_channels=cfg["base_channels"], stage_blocks=cfg["stage_blocks"],
                     attention_heads=cfg["attention_heads"], max_spatial_tokens=cfg["max_spatial_tokens"])
    model = AIRFormer(**model_cfg).to(device)
    x = torch.rand(1, 3, 128, 128, device=device)
    target = x * 0.8
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = MultiDomainConsistencyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    def sync():
        if use_amp:
            torch.cuda.synchronize()

    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            model(x)
        sync()
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(3):
            y = model(x)
        sync()
        inference_time = (time.perf_counter() - start) / 3
        inference_memory = torch.cuda.max_memory_allocated() if use_amp else None
        numeric = dict(sum=y.sum().item(), mean=y.mean().item(), first=y.flatten()[:8].tolist())
    model.train()
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    sync(); start = time.perf_counter()
    losses = []
    # GradScaler may skip initial overflowed steps while adapting its scale.
    successful_steps = 0
    attempts = 0
    while successful_steps < 3 and attempts < 16:
        attempts += 1
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            loss = loss_fn(model(x), target)[0]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if finite:
            successful_steps += 1
            losses.append(loss.item())
        else:
            assert use_amp and scaler.get_scale() < old_scale
    if successful_steps != 3:
        raise FloatingPointError("AMP failed to produce finite gradients after scale adaptation")
    sync()
    training_time = (time.perf_counter() - start) / attempts
    report = dict(torch=torch.__version__, device=str(device),
        gpu=torch.cuda.get_device_name() if use_amp else None,
        input_shape=list(x.shape), output_shape=list(y.shape),
        parameters=sum(p.numel() for p in model.parameters()),
        inference_seconds=inference_time, inference_peak_allocated_bytes=inference_memory,
        training_step_seconds=training_time, training_amp=use_amp,
        training_peak_allocated_bytes=torch.cuda.max_memory_allocated() if use_amp else None,
        losses=losses, amp_attempts=attempts, amp_final_scale=scaler.get_scale(), fixed_input_output=numeric,
        note="Random pilot model diagnostic; no trained benchmark performance claim.")
    del model, optimizer, x, y, loss
    if use_amp:
        torch.cuda.empty_cache()
    (output / "benchmark.json").write_text(json.dumps(report, indent=2))
    subsets = {}
    for split in ("train", "val", "test"):
        with (root / "data/manifests" / f"{split}.csv").open(newline="", encoding="utf8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"Empty {split}")
        subsets[split] = rows[:2] if split == "train" else rows[:1]
        write_manifest(subsets[split], output / f"{split}.csv")
    cfg.update(base_channels=4, max_spatial_tokens=16, crop_size=32,
               batch_size=1, epochs=2, num_workers=0, gradient_accumulation_steps=2,
               train_manifest=str(output / "train.csv"), val_manifest=str(output / "val.csv"))
    config = output / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    log = output / "commands.log"

    def run(*arguments):
        with log.open("a", encoding="utf8") as stream:
            stream.write("\n" + " ".join(arguments) + "\n"); stream.flush()
            subprocess.run([sys.executable, *arguments], cwd=root, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    # Exercise checkpoint restoration (including scheduler/RNG/scaler). This
    # short changed-horizon smoke does not claim uninterrupted-run equivalence.
    cfg["epochs"] = 1; config.write_text(yaml.safe_dump(cfg))
    checkpoint_dir = output / "checkpoints"
    run("train.py", "--config", str(config), "--output", str(checkpoint_dir))
    cfg["epochs"] = 2; config.write_text(yaml.safe_dump(cfg))
    run("train.py", "--config", str(config), "--resume", str(checkpoint_dir / "latest.pt"), "--output", str(checkpoint_dir))
    # Evaluate a genuine held-out Rain12 pair with the diagnostic checkpoint.
    row = next((r for r in rows if r["source_dataset"] == "Rain12"), rows[0])
    from PIL import Image
    for folder, key in (("heldout_input", "input_path"), ("heldout_target", "target_path")):
        destination = output / folder
        destination.mkdir(exist_ok=True)
        with Image.open(row[key]) as image:
            image.convert("RGB").save(destination / "sample.png")
    run("inference.py", "--input", str(output / "heldout_input"), "--checkpoint", str(checkpoint_dir / "latest.pt"), "--output", str(output / "restored"))
    run("evaluate.py", "--pred", str(output / "restored"), "--target", str(output / "heldout_target"), "--csv", str(output / "heldout_metrics.csv"))
    print(json.dumps(report, indent=2))
    print(f"CLI smoke passed; diagnostic checkpoint/metrics/log: {output}")


if __name__ == "__main__":
    main()
