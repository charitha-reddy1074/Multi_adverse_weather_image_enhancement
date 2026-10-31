"""Manifest helpers. Pair by known identity, never by independently sorting files."""
import csv
import hashlib
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
            im_a.load(); im_b.load()
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
    path=Path(path); seen={}; issues=[]; count=0; input_paths=set()
    with path.open(newline="",encoding="utf8") as f:
        for line,row in enumerate(csv.DictReader(f),start=2):
            count+=1
            for col in ("input_path","target_path"):
                p=Path(row[col]); p=p if p.is_absolute() else path.parent/p
                if not p.is_file(): issues.append(f"line {line}: missing {col}: {p}")
            ip=Path(row["input_path"]); tp=Path(row["target_path"])
            ip=ip if ip.is_absolute() else path.parent/ip
            tp=tp if tp.is_absolute() else path.parent/tp
            identity=str(ip.resolve()).casefold()
            if identity in input_paths: issues.append(f"line {line}: duplicate input: {ip}")
            input_paths.add(identity)
            if row.get("split") not in {"train", "val", "test"}:
                issues.append(f"line {line}: invalid split: {row.get('split')}")
            if row.get("weather_type") not in {"rain", "raindrop", "snow", "haze", "rain_haze", "mixed_weather", "sand_dust"}:
                issues.append(f"line {line}: invalid weather_type: {row.get('weather_type')}")
            if ip.is_file() and tp.is_file():
                valid,size_a,size_b=validate_pair(ip,tp)
                if not valid: issues.append(f"line {line}: decode/size mismatch input={size_a} target={size_b}")
            if row.get("split") and row.get("scene_id"):
                key=(row.get("source_dataset"),row["scene_id"])
                if key in seen and seen[key]!=row["split"] and check_overlap:
                    issues.append(f"scene leakage for {key}: {seen[key]} and {row['split']}")
                seen[key]=row["split"]
    return count,issues


def assert_disjoint_datasets(*datasets):
    """Reject cross-split scene/path/pixel-identical ground-truth leakage.

    Clean pixel hashes catch renamed copies and many-input/one-clean RESIDE
    scenes. Perceptually similar scenes still require source provenance review.
    """
    seen = {}
    cache = {}
    for split_index, dataset in enumerate(datasets):
        for input_path, target_path, _, source, _, scene in dataset.records:
            input_path, target_path = Path(input_path), Path(target_path)
            keys = [("scene", source.casefold(), scene)]
            keys.extend(("path", str(p.resolve()).casefold()) for p in (input_path, target_path))
            target_path = target_path.resolve()
            if target_path not in cache:
                with Image.open(target_path) as image:
                    rgb = image.convert("RGB")
                    cache[target_path] = hashlib.sha256(
                        str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            keys.append(("clean_pixels", cache[target_path]))
            for key in keys:
                if key in seen and seen[key] != split_index:
                    raise ValueError(f"Cross-split leakage: {key[0]} at {target_path}")
                seen[key] = split_index
