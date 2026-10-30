import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.metrics import image_metrics
from utils.inference import load_model


EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
def image_map(root):
    root=Path(root)
    return {x.relative_to(root).with_suffix("").as_posix():x for x in root.rglob("*")
            if x.suffix.lower() in EXTS}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pred",required=True); p.add_argument("--target",required=True)
    p.add_argument("--csv",default="results/per_image_metrics.csv")
    p.add_argument("--checkpoint"); p.add_argument("--inference-times")
    a=p.parse_args(); preds=image_map(a.pred); targets=image_map(a.target)
    keys=sorted(set(preds)&set(targets))
    if not keys: raise SystemExit("No paired prediction/target paths; relative paths must match.")
    rows=[]; grouped=defaultdict(list)
    for key in keys:
        x=np.asarray(Image.open(preds[key]).convert("RGB"),dtype=np.float32)/255
        y=np.asarray(Image.open(targets[key]).convert("RGB"),dtype=np.float32)/255
        if x.shape!=y.shape:
            print(f"skip size mismatch {key}: pred={x.shape} target={y.shape}"); continue
        psnr,ssim=image_metrics(x,y)
        dataset=Path(key).parts[0] if len(Path(key).parts)>1 else "all"
        rows.append((key,dataset,psnr,ssim)); grouped[dataset].append((psnr,ssim))
    if not rows: raise SystemExit("No same-size image pairs found.")
    output=Path(a.csv); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f); writer.writerow(("image","dataset","psnr_db","ssim")); writer.writerows(rows)
    for name,values in sorted(grouped.items()):
        print(f"{name}: images={len(values)} PSNR={np.mean([v[0] for v in values]):.4f} dB SSIM={np.mean([v[1] for v in values]):.6f}")
    print(f"overall: images={len(rows)} PSNR={np.mean([r[2] for r in rows]):.4f} dB SSIM={np.mean([r[3] for r in rows]):.6f}")
    if a.checkpoint:
        model,_=load_model(a.checkpoint)
        print(f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        try:
            from thop import profile
            sample=next(iter(preds.values()))
            im=Image.open(sample).convert("RGB"); h,w=im.height,im.width
            macs,_=profile(model,inputs=(__import__('torch').rand(1,3,h,w),),verbose=False)
            print(f"FLOPs_estimate={2*macs:.0f} at input={h}x{w}")
        except ImportError: print("FLOPs estimate skipped (install thop).")
    if a.inference_times:
        with Path(a.inference_times).open(newline="",encoding="utf8") as f:
            times=[float(r["seconds"]) for r in csv.DictReader(f)]
        if times: print(f"mean_inference_seconds={np.mean(times):.6f}")


if __name__=="__main__": main()
