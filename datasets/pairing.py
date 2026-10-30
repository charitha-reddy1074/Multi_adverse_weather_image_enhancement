"""Manifest helpers. Pair by known identity, never by independently sorting files."""
import csv
from pathlib import Path
from PIL import Image

FIELDS=("input_path","target_path","weather_type","source_dataset","split","scene_id")
EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}


def image_map(root):
    root=Path(root); result={}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTS:
            key=path.relative_to(root).with_suffix("").as_posix()
            if key in result: raise ValueError(f"duplicate relative image identity {key} under {root}")
            result[key]=path
    return result


def validate_pair(input_path,target_path):
    try:
        with Image.open(input_path) as im_a, Image.open(target_path) as im_b:
            im_a.verify(); im_b.verify()
        with Image.open(input_path) as im_a, Image.open(target_path) as im_b:
            return im_a.size==im_b.size, im_a.size, im_b.size
    except Exception as e:
        return False, None, str(e)


def write_manifest(records, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader()
        writer.writerows(records)
    return path


def verify_manifest(path, check_overlap=True):
    path=Path(path); seen={}; issues=[]; count=0
    with path.open(newline="",encoding="utf8") as f:
        for line,row in enumerate(csv.DictReader(f),start=2):
            count+=1
            for col in ("input_path","target_path"):
                p=Path(row[col]); p=p if p.is_absolute() else path.parent/p
                if not p.is_file(): issues.append(f"line {line}: missing {col}: {p}")
            ip=Path(row["input_path"]); tp=Path(row["target_path"])
            ip=ip if ip.is_absolute() else path.parent/ip
            tp=tp if tp.is_absolute() else path.parent/tp
            if ip.is_file() and tp.is_file():
                valid,size_a,size_b=validate_pair(ip,tp)
                if not valid: issues.append(f"line {line}: decode/size mismatch input={size_a} target={size_b}")
            if row.get("split") and row.get("scene_id"):
                key=(row.get("source_dataset"),row["scene_id"])
                if key in seen and seen[key]!=row["split"] and check_overlap:
                    issues.append(f"scene leakage for {key}: {seen[key]} and {row['split']}")
                seen[key]=row["split"]
    return count,issues
