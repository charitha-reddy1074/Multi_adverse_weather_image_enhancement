import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from utils.inference import load_model,restore_image


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",default="outputs/restored"); p.add_argument("--device",default="auto")
    p.add_argument("--base-channels",type=int); p.add_argument("--tile-size",type=int,default=0)
    a=p.parse_args(); model,device=load_model(a.checkpoint,a.device,a.base_channels)
    root=Path(a.input); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    extensions={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
    files=[root] if root.is_file() else sorted(f for f in root.rglob("*") if f.suffix.lower() in extensions)
    records=[]
    for f in files:
        if f.suffix.lower() not in extensions: continue
        image=Image.open(f).convert("RGB")
        # Tiling is not enabled implicitly; a too-large image should fail clearly
        # rather than produce seams from an undocumented overlap strategy.
        restored,elapsed=restore_image(model,image,device,tile_size=a.tile_size)
        rel=Path(f.name) if root.is_file() else f.relative_to(root)
        dest=(out/rel).with_suffix(".png"); dest.parent.mkdir(parents=True,exist_ok=True)
        restored.save(dest); records.append((str(rel),elapsed)); print(dest)
    with (out/"inference_times.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f); writer.writerow(("image","seconds")); writer.writerows(records)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"images={len(records)} params={params:,} mean_inference_seconds={sum(t for _,t in records)/max(1,len(records)):.6f}")


if __name__=="__main__": main()
