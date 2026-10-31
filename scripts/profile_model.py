"""Profile a configured AIRFormer at Table XIII's 256x256 input size."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from models.airformer import AIRFormer

PAPER_PARAMS_M=58.34
PAPER_FLOPS_G=5.746


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/paper.yaml")
    p.add_argument("--checkpoint")
    p.add_argument("--height",type=int,default=256)
    p.add_argument("--width",type=int,default=256)
    p.add_argument("--device",default="auto")
    p.add_argument("--iterations",type=int,default=20)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--skip-flops",action="store_true")
    a=p.parse_args()
    if a.iterations<1 or a.warmup<0 or a.height<1 or a.width<1:
        p.error("iterations/height/width must be positive and warmup nonnegative")
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                        if a.device=="auto" else a.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        p.error("CUDA was selected but is unavailable")
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf8"))
    model_cfg={"base_channels":int(cfg.get("base_channels",32)),
        "stage_blocks":tuple(cfg.get("stage_blocks",(2,2,2))),
        "attention_heads":int(cfg.get("attention_heads",4)),
        "max_spatial_tokens":int(cfg.get("max_spatial_tokens",1024)),
        "attention_order":tuple(cfg.get("attention_order",("channel","space","wavelet"))),
        "query_combine":cfg.get("query_combine","add"),
        "encoder_token_mixer":cfg.get("encoder_token_mixer","pooling"),
        "decoder_mode":cfg.get("decoder_mode","frequency"),
        "use_supplementary_prior":bool(cfg.get("use_supplementary_prior",True)),
        "use_task_queries":bool(cfg.get("use_task_queries",True)),
        "residual_image":bool(cfg.get("residual_image",True))}
    model=AIRFormer(**model_cfg).to(device)
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
        if "model_config" in state:
            model_cfg=state["model_config"]
            model=AIRFormer(**model_cfg).to(device)
        model.load_state_dict(state.get("model",state))
    model.eval()
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"profile_input={a.height}x{a.width} device={device}")
    print(f"trainable_parameters={params:,} ({params/1e6:.3f} M)")
    print(f"Table XIII AIRFormer reference: {PAPER_PARAMS_M:.2f} M params; {PAPER_FLOPS_G:.3f} GFLOPs")
    print(f"parameter_delta_vs_reference={params/1e6-PAPER_PARAMS_M:+.3f} M")
    if not a.skip_flops:
        try:
            from thop import profile
            sample=torch.zeros(1,3,a.height,a.width,device=device)
            macs,_=profile(model,inputs=(sample,),verbose=False)
            flops=2*macs
            print(f"THOP estimate={flops/1e9:.3f} GFLOPs (2 FLOPs/MAC convention)")
            print(f"FLOPs_delta_vs_reference={flops/1e9-PAPER_FLOPS_G:+.3f} G")
            print("THOP may omit unsupported attention operations; treat this as an estimate.")
        except ImportError:
            print("THOP unavailable; install requirements.txt or pass --skip-flops.")
    x=torch.rand(1,3,a.height,a.width,device=device)
    if device.type=="cuda":
        for _ in range(a.warmup):
            with torch.inference_mode(): model(x)
        torch.cuda.synchronize(device)
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(a.iterations): model(x)
        end.record(); torch.cuda.synchronize(device)
        elapsed=start.elapsed_time(end)/1000/a.iterations
    else:
        for _ in range(a.warmup):
            with torch.inference_mode(): model(x)
        start=time.perf_counter()
        with torch.inference_mode():
            for _ in range(a.iterations): model(x)
        elapsed=(time.perf_counter()-start)/a.iterations
    print(f"mean_inference_seconds={elapsed:.6f} iterations={a.iterations} warmup={a.warmup}")


if __name__=="__main__": main()
