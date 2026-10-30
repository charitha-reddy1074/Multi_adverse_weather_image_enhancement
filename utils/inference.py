"""Symmetric multiple-of-128 padding and tiled-free single-image inference."""
import time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor,to_pil_image
from PIL import Image
from pathlib import Path


def symmetric_pad(x,multiple=128):
    h,w=x.shape[-2:]
    ph,pw=(-h)%multiple,(-w)%multiple
    top,bottom=ph//2,ph-ph//2
    left,right=pw//2,pw-pw//2
    if not ph and not pw: return x,(left,top,w,h)
    mode="reflect" if h>max(top,bottom) and w>max(left,right) and min(h,w)>1 else "replicate"
    return F.pad(x,(left,right,top,bottom),mode=mode),(left,top,w,h)


def _restore_tensor(model,x,device,pad_multiple):
    x,pad=symmetric_pad(x,pad_multiple)
    if device.type=="cuda": torch.cuda.synchronize(device)
    start=time.perf_counter()
    with torch.inference_mode(): y=model(x)
    if device.type=="cuda": torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    left,top,w,h=pad
    return y[...,top:top+h,left:left+w],elapsed


def _starts(length,tile,step):
    if length<=tile: return [0]
    values=list(range(0,length-tile+1,step))
    if values[-1]!=length-tile: values.append(length-tile)
    return values


def restore_image(model,image,device=None,pad_multiple=128,tile_size=0,overlap=32):
    device=torch.device(device or next(model.parameters()).device)
    x=to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    if not tile_size:
        y,elapsed=_restore_tensor(model,x,device,pad_multiple)
    else:
        if tile_size<=2*overlap: raise ValueError("tile_size must exceed twice overlap")
        h,w=x.shape[-2:]; step=tile_size-2*overlap
        ys,xs=_starts(h,tile_size,step),_starts(w,tile_size,step)
        accum=torch.zeros_like(x); weights=torch.zeros((1,1,h,w),device=device,dtype=x.dtype)
        elapsed=0.0
        for top in ys:
            for left in xs:
                bottom,right=min(top+tile_size,h),min(left+tile_size,w)
                tile=x[...,top:bottom,left:right]
                output,dt=_restore_tensor(model,tile,device,pad_multiple); elapsed+=dt
                th,tw=output.shape[-2:]
                wy=torch.ones(th,device=device,dtype=x.dtype); wx=torch.ones(tw,device=device,dtype=x.dtype)
                oy=min(overlap,th//2); ox=min(overlap,tw//2)
                if top>0 and oy: wy[:oy]=torch.linspace(1/(oy+1),oy/(oy+1),oy,device=device,dtype=x.dtype)
                if bottom<h and oy: wy[-oy:]=torch.linspace(oy/(oy+1),1/(oy+1),oy,device=device,dtype=x.dtype)
                if left>0 and ox: wx[:ox]=torch.linspace(1/(ox+1),ox/(ox+1),ox,device=device,dtype=x.dtype)
                if right<w and ox: wx[-ox:]=torch.linspace(ox/(ox+1),1/(ox+1),ox,device=device,dtype=x.dtype)
                weight=(wy[:,None]*wx[None,:])[None,None]
                accum[...,top:bottom,left:right]+=output*weight
                weights[...,top:bottom,left:right]+=weight
        y=accum/weights.clamp_min(1e-8)
    y=y.squeeze(0).clamp(0,1).cpu()
    return to_pil_image(y),elapsed


def load_model(checkpoint,device="auto",base_channels=None):
    if device=="auto": device="cuda" if torch.cuda.is_available() else "cpu"
    device=torch.device(device)
    state=torch.load(checkpoint,map_location=device)
    from models.airformer import AIRFormer
    cfg=dict(state.get("model_config",{}))
    cfg.setdefault("base_channels",base_channels or state.get("base_channels",32))
    if base_channels is not None: cfg["base_channels"]=base_channels
    model=AIRFormer(**cfg).to(device)
    model.load_state_dict(state.get("model",state))
    model.eval()
    return model,device


def run_paths(model,paths,output_dir,device):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in paths:
        path=Path(path)
        image=Image.open(path).convert("RGB")
        restored,elapsed=restore_image(model,image,device)
        dest=(output_dir/path.name).with_suffix(".png")
        dest.parent.mkdir(parents=True,exist_ok=True); restored.save(dest)
        records.append((dest,elapsed))
    return records
