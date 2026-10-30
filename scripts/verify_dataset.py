import argparse
from pathlib import Path
from sys import path
path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.pairing import image_map,validate_pair

p=argparse.ArgumentParser()
p.add_argument("--root", default="datasets/train")
args=p.parse_args()
root=Path(args.root)
degraded=root/"degraded"
if not degraded.is_dir(): degraded=root/"rainy"
a,b=image_map(degraded),image_map(root/"clean")
print("degraded images:",len(a),"clean images:",len(b))
print("paired:",len(set(a)&set(b)))
print("degraded without clean:",len(set(a)-set(b)))
print("clean without degraded:",len(set(b)-set(a)))
if not (set(a)&set(b)): raise SystemExit("ERROR: no matching basenames.")
bad=[]
for key in sorted(set(a)&set(b)):
    ok,sz_a,sz_b=validate_pair(a[key],b[key])
    if not ok: bad.append((key,sz_a,sz_b))
for row in bad: print("INVALID PAIR",*row)
print("decode/size mismatches:",len(bad))
if bad: raise SystemExit(2)
