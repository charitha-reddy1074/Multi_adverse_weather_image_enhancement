import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airformer import AIRFormer, MultiDomainConsistencyLoss
from airformer.data import PairedWeatherDataset
from utils.metrics import image_metrics
from utils.inference import _restore_tensor
from datasets.pairing import assert_disjoint_datasets


def validate(model, loader, loss_fn, device):
    model.eval()
    losses, psnrs, ssims = [], [], []
    with torch.no_grad():
        for x, target, _ in loader:
            x, target = x.to(device), target.to(device)
            pred, _ = _restore_tensor(model, x, device, 128)
            losses.append(loss_fn(pred, target)[0].item())
            for p, t in zip(pred.clamp(0, 1).cpu().numpy(), target.cpu().numpy()):
                p, t = np.moveaxis(p, 0, -1), np.moveaxis(t, 0, -1)
                psnr,ssim=image_metrics(p,t)
                psnrs.append(psnr); ssims.append(ssim)
    if not losses:
        raise RuntimeError("Validation dataset is empty")
    return {"loss": float(np.mean(losses)), "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims))}


def seed_worker(worker_id):
    worker_seed=torch.initial_seed()%2**32
    np.random.seed(worker_seed); random.seed(worker_seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.yaml")
    p.add_argument("--train-root")
    p.add_argument("--val-root")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--crop-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--base-channels", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--encoder-token-mixer",choices=("pooling","self_attention"))
    p.add_argument("--decoder-mode",choices=("frequency","spatial","none"))
    p.add_argument("--attention-order",help="comma-separated permutation, e.g. channel,space,wavelet")
    p.add_argument("--disable-sp",action="store_true")
    p.add_argument("--disable-task-queries",action="store_true")
    p.add_argument("--lambda-freq",type=float)
    p.add_argument("--train-manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--resume")
    p.add_argument("--output", default="checkpoints")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf8"))
    overrides = {"dataset_root": a.train_root, "epochs": a.epochs,
                 "batch_size": a.batch_size, "crop_size": a.crop_size,
                 "lr": a.lr, "base_channels": a.base_channels,
                 "num_workers": a.workers}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if a.train_manifest: cfg["train_manifest"]=a.train_manifest
    if a.val_manifest: cfg["val_manifest"]=a.val_manifest
    if a.encoder_token_mixer: cfg["encoder_token_mixer"]=a.encoder_token_mixer
    if a.decoder_mode: cfg["decoder_mode"]=a.decoder_mode
    if a.attention_order: cfg["attention_order"]=tuple(a.attention_order.split(","))
    if a.disable_sp: cfg["use_supplementary_prior"]=False
    if a.disable_task_queries: cfg["use_task_queries"]=False
    if a.lambda_freq is not None: cfg["lambda_freq"]=a.lambda_freq
    train_root = Path(cfg["dataset_root"])
    val_root = Path(a.val_root) if a.val_root else train_root.parent / "val"
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = int(cfg.get("num_workers", 0))
    train_manifest=Path(cfg.get("train_manifest",""))
    val_manifest=Path(cfg.get("val_manifest",""))
    for name in ("train_manifest", "val_manifest"):
        if cfg.get(name) and not Path(cfg[name]).is_file():
            raise FileNotFoundError(f"Configured {name} not found: {cfg[name]}; build validated manifests first")
    train_ds=PairedWeatherDataset(train_root,cfg["crop_size"],True,
        cfg.get("mixup_prob",.5),manifest=train_manifest if train_manifest.is_file() else None,
        balance_weather=bool(cfg.get("balance_weather",False)))
    val_ds=PairedWeatherDataset(val_root,None,False,
        manifest=val_manifest if val_manifest.is_file() else None)
    assert_disjoint_datasets(train_ds, val_ds)
    for dataset, expected in ((train_ds, "train"), (val_ds, "val")):
        if any(row[4] not in (expected, "unknown", "") for row in dataset.records):
            raise ValueError(f"Expected {expected} records; official test records cannot be used for model selection")
    sampler=None
    if train_ds.sample_weights is not None:
        sampler=torch.utils.data.WeightedRandomSampler(train_ds.sample_weights,
            num_samples=len(train_ds),replacement=True)
    generator=torch.Generator().manual_seed(seed)
    train = DataLoader(train_ds,
                       batch_size=cfg["batch_size"], shuffle=sampler is None,sampler=sampler,
                       num_workers=workers, pin_memory=device.type == "cuda",
                       drop_last=False,worker_init_fn=seed_worker,generator=generator)
    val = DataLoader(val_ds, batch_size=1, shuffle=False,
                     num_workers=workers,worker_init_fn=seed_worker)
    model_cfg = {"base_channels": int(cfg["base_channels"]),
                 "stage_blocks": tuple(cfg.get("stage_blocks", (2, 2, 2))),
                 "attention_heads": int(cfg.get("attention_heads",4)),
                 "max_spatial_tokens": int(cfg.get("max_spatial_tokens",1024)),
                 "use_supplementary_prior": bool(cfg.get("use_supplementary_prior",True)),
                 "use_task_queries": bool(cfg.get("use_task_queries",True)),
                 "use_frequency_decoder": bool(cfg.get("use_frequency_decoder",True)),
                 "residual_image": bool(cfg.get("residual_image",True)),
                 "gradient_checkpointing": bool(cfg.get("gradient_checkpointing",False)),
                 "query_combine": cfg.get("query_combine","add"),
                 "attention_order": tuple(cfg.get("attention_order",("channel","space","wavelet")))}
    model_cfg["encoder_token_mixer"]=cfg.get("encoder_token_mixer","pooling")
    model_cfg["decoder_mode"]=cfg.get("decoder_mode",
        "frequency" if cfg.get("use_frequency_decoder",True) else "none")
    model = AIRFormer(**model_cfg).to(device)
    loss_fn = MultiDomainConsistencyLoss(float(cfg.get("lambda_freq", .04)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            betas=tuple(cfg.get("betas", (.9, .999))),
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg["epochs"]), eta_min=float(cfg["min_lr"]))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    logdir=out/"runs"; writer=SummaryWriter(log_dir=str(logdir))
    amp_enabled=device.type=="cuda" and bool(cfg.get("mixed_precision",True))
    scaler=torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch=0; best=float("inf")
    if a.resume:
        saved=torch.load(a.resume,map_location="cpu",weights_only=True)
        if saved.get("model_config", model_cfg) != model_cfg:
            raise ValueError("Resume model_config differs from current config; use the checkpoint's architecture")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        if "scheduler" in saved: sched.load_state_dict(saved["scheduler"])
        if "scaler" in saved: scaler.load_state_dict(saved["scaler"])
        start_epoch=int(saved.get("epoch",0)); best=float(saved.get("best_val",float("inf")))
        if "rng" in saved:
            rng=saved["rng"]
            random.setstate(rng["python"])
            np.random.set_state((rng["numpy"][0],np.array(rng["numpy"][1],dtype=np.uint32),
                                 *rng["numpy"][2:]))
            torch.set_rng_state(rng["torch"])
            generator.set_state(rng["loader"])
            if torch.cuda.is_available() and rng["cuda"]:
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"resuming at epoch {start_epoch+1} from {a.resume}")
    accum=max(1,int(cfg.get("gradient_accumulation_steps",1)))
    if device.type=="cuda":
        print(f"CUDA device={torch.cuda.get_device_name(0)} VRAM={torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")
    else: print("CUDA unavailable; using CPU with mixed precision disabled")
    for epoch in range(start_epoch,int(cfg["epochs"])):
        model.train(); running = []
        bar = tqdm(train, desc=f"train {epoch+1}/{cfg['epochs']}")
        opt.zero_grad(set_to_none=True)
        for step,(x, target, _) in enumerate(bar):
            x, target = x.to(device), target.to(device)
            with torch.autocast(device_type=device.type,enabled=amp_enabled):
                pred = model(x); raw_loss = loss_fn(pred, target)[0]
                # Normalize a short final accumulation window by its actual
                # number of microbatches instead of underweighting it.
                window_start=(step//accum)*accum
                window_size=min(accum,len(train)-window_start)
                loss=raw_loss/window_size
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"non-finite training loss at epoch={epoch+1} step={step}")
            scaler.scale(loss).backward()
            do_step=(step+1)%accum==0 or step+1==len(train)
            if do_step:
                scaler.unscale_(opt)
                # FP16 overflow is handled by GradScaler: skip the update and
                # lower the scale. FP32 nonfinite gradients remain fatal.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp_enabled)
                previous_scale=scaler.get_scale()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if scaler.get_scale()<previous_scale:
                    print(f"AMP overflow: skipped optimizer update; scale={scaler.get_scale()}")
            running.append(raw_loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        metrics = validate(model, val, loss_fn, device)
        train_loss=float(np.mean(running))
        writer.add_scalar("loss/train",train_loss,epoch+1)
        writer.add_scalar("loss/validation",metrics["loss"],epoch+1)
        writer.add_scalar("metrics/psnr",metrics["psnr"],epoch+1)
        writer.add_scalar("metrics/ssim",metrics["ssim"],epoch+1)
        writer.add_scalar("learning_rate",opt.param_groups[0]["lr"],epoch+1)
        is_best=metrics["loss"]<best
        if is_best: best=metrics["loss"]
        state = {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                 "scaler":scaler.state_dict(),"val": metrics,"best_val":best,
                 "model_config": model_cfg, "config": cfg}
        np_state=np.random.get_state()
        state["rng"]={"python":random.getstate(),
            "numpy":(np_state[0],np_state[1].tolist(),*np_state[2:]),
            "torch":torch.get_rng_state(),"loader":generator.get_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        torch.save(state, out / "latest.pt")
        if is_best: torch.save(state, out / "best.pt")
        writer.add_scalar("loss/best_validation",best,epoch+1)
        print(f"epoch={epoch+1} train_loss={train_loss:.6f} "
              f"val_loss={metrics['loss']:.6f} PSNR={metrics['psnr']:.4f} "
              f"SSIM={metrics['ssim']:.6f} best_val={best:.6f}")
    writer.close()


if __name__ == "__main__":
    main()
