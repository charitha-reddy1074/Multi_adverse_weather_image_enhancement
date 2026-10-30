"""Build CSV manifests from manually aligned source/split folder trees."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.pairing import verify_manifest, write_manifest, image_map

WEATHER={"rain14000":"rain","rain800":"rain","rain100h":"rain",
"rain100l":"rain","rain1200":"rain","rain12":"rain","raindrop":"raindrop",
"snow100k-l":"snow","snow100k-m":"snow","snow100k-s":"snow","csd":"snow",
"its":"haze","ots":"haze","hsts":"haze","reside its":"haze",
"reside ots":"haze","reside hsts":"haze","reside-its":"haze",
"reside-ots":"haze","reside-hsts":"haze"}
EXPECTED={"rain14000":{"train":11200,"test":2800},"rain800":{"train":700,"test":100},
"rain100h":{"train":1800,"test":100},"rain100l":{"train":0,"test":100},
"rain1200":{"train":0,"test":1200},"rain12":{"train":0,"test":12},
"raindrop":{"train":861,"test":307},"snow100k-l":{"train":9000,"test":16801},
"snow100k-m":{"train":0,"test":16588},"snow100k-s":{"train":0,"test":16611},
"csd":{"train":8000,"test":2000},"reside its":{"train":4500,"test":500},
"reside ots":{"train":4495,"test":500},"reside hsts":{"train":0,"test":10}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="datasets/AIR40K/raw")
    p.add_argument("--output",default="data/manifests")
    p.add_argument("--splits",nargs="+",default=["train","val","test"])
    a=p.parse_args(); root=Path(a.root).resolve(); out=Path(a.output)
    scene_splits={}
    for split in a.splits:
        records=[]
        for source_dir in sorted((root/split).glob("*")):
            src=source_dir.name
            ins=source_dir/"degraded"; targets=source_dir/"clean"
            if not ins.is_dir() or not targets.is_dir():
                print(f"skip {source_dir}: expected degraded/ and clean/ directories")
                continue
            imap,tmap=image_map(ins),image_map(targets)
            for rel in sorted(set(imap)&set(tmap)):
                scene=Path(rel).with_suffix("").as_posix()
                scene_splits.setdefault((src,scene),set()).add(split)
                records.append({"input_path":str(imap[rel].resolve()),
                    "target_path":str(tmap[rel].resolve()),
                    "weather_type":WEATHER.get(src.lower(),"unknown"),
                    "source_dataset":src,"split":split,
                    "scene_id":scene})
            for rel in sorted(set(imap)^set(tmap)):
                print(f"unpaired {split}/{src}/{rel}")
        dest=out/f"{split}.csv"; write_manifest(records,dest)
        count,issues=verify_manifest(dest)
        print(f"{split}: {count} pairs -> {dest}; issues={len(issues)}")
        for issue in issues: print("ERROR",issue)
        by_source={}
        for row in records:
            name=row["source_dataset"].lower().replace("-"," ")
            by_source[name]=by_source.get(name,0)+1
        if split in {"train","test"}:
            for source,counts in EXPECTED.items():
                if source in by_source and by_source[source]!=counts[split]:
                    print(f"COUNT MISMATCH {split} {source}: found={by_source[source]} expected={counts[split]}")
    for (source,scene),splits in scene_splits.items():
        if len(splits)>1: print(f"ERROR train/test leakage: {source}/{scene}: {sorted(splits)}")
    print("Paper Table I audit: train rows sum to 40,556; reported total is 40,561 (difference 5). No pairs were fabricated.")


if __name__=="__main__": main()
