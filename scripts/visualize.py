"""Write paired input/prediction/target comparison strips."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--pred",required=True)
    p.add_argument("--target",required=True); p.add_argument("--output",default="results/visuals")
    a=p.parse_args(); roots=[Path(a.input),Path(a.pred),Path(a.target)]; out=Path(a.output)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
    inputs={x.relative_to(roots[0]).as_posix():x for x in roots[0].rglob("*") if x.suffix.lower() in exts}
    for key,source in sorted(inputs.items()):
        peers=[r/Path(key) for r in roots[1:]]
        if not all(x.is_file() for x in peers): print("missing pair",key); continue
        images=[Image.open(x).convert("RGB") for x in (source,*peers)]
        size=images[0].size
        if any(im.size!=size for im in images): print("size mismatch",key); continue
        labels=("degraded","restored","target")
        canvas=Image.new("RGB",(size[0]*3,size[1]+24),"white")
        draw=ImageDraw.Draw(canvas)
        for i,(im,label) in enumerate(zip(images,labels)):
            canvas.paste(im,(i*size[0],24)); draw.text((i*size[0]+6,5),label,fill="black")
        dest=out/Path(key); dest.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(dest.with_suffix(".png"))
    print(f"visuals saved under {out}")


if __name__=="__main__": main()
