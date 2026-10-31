"""Prepare official Rain14000/Rain12 releases without copying or re-splitting tests.

Scene-disjoint validation is a documented 5% holdout from official training.
This is a source-release setup, not the unpublished AIR40K sampling manifest.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.archive import safe_extract
from datasets.pairing import write_manifest, validate_pair, assert_disjoint_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw, output = Path(args.raw).resolve(), Path(args.output)
    records = []
    archives = {}
    for name in ("Rain14000", "Rain12"):
        archive = raw / f"{name}.zip"
        if not archive.is_file():
            continue
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        root = raw / "extracted" / name
        safe_extract(archive, root)
        archives[name] = {"path": str(archive), "sha256": digest, "bytes": archive.stat().st_size}
        if name == "Rain14000":
            base = root / "rainy_image_dataset"
            scenes = sorted(p.stem for p in (base / "training" / "ground_truth").glob("*.jpg"))
            random.Random(args.seed).shuffle(scenes)
            validation = set(scenes[:max(1, len(scenes) // 20)])
            for release_split in ("training", "testing"):
                for image in sorted((base / release_split / "rainy_image").glob("*.jpg")):
                    scene = image.stem.rsplit("_", 1)[0]
                    target = base / release_split / "ground_truth" / f"{scene}.jpg"
                    split = "test" if release_split == "testing" else ("val" if scene in validation else "train")
                    records.append(dict(input_path=str(image), target_path=str(target), weather_type="rain",
                                        source_dataset=name, split=split, scene_id=scene))
        else:
            for image in sorted(root.glob("*_in.png")):
                scene = image.stem.removesuffix("_in")
                records.append(dict(input_path=str(image), target_path=str(root / f"{scene}_GT.png"),
                                    weather_type="rain", source_dataset=name, split="test", scene_id=scene))
    if not records:
        raise SystemExit("No supported archives present")
    issues = []; input_hashes = {}; duplicates = []
    for row in records:
        valid, size_a, size_b = validate_pair(row["input_path"], row["target_path"])
        if not valid:
            issues.append(f"{row['input_path']}: {size_a} vs {size_b}")
        with Path(row["input_path"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in input_hashes:
            previous = input_hashes[digest]
            duplicates.append([row["input_path"], previous["input_path"]])
            if row["split"] != previous["split"]:
                issues.append(f"Duplicate input across splits: {row['input_path']}")
        else:
            input_hashes[digest] = row
    # Use the same cross-split guard as training; no augmentation or random crops.
    from types import SimpleNamespace
    sets = []
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        sets.append(SimpleNamespace(records=[tuple(r[k] for k in
            ("input_path", "target_path", "weather_type", "source_dataset", "split", "scene_id")) for r in subset]))
    try:
        assert_disjoint_datasets(*sets)
    except ValueError as error:
        issues.append(str(error))
    result = {"seed": args.seed, "archives": archives, "pairs": len(records),
              "counts": dict(Counter(f"{r['source_dataset']}/{r['split']}" for r in records)),
              "issues": issues, "duplicate_inputs": duplicates,
              "duplicate_policy": "Preserve documented within-split source-release duplicates; cross-split copies are fatal.",
              "status": "READY" if not issues else "PAIRING FAILED"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "integrity.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    if issues:
        raise SystemExit(json.dumps(result, indent=2))
    for split in ("train", "val", "test"):
        write_manifest([r for r in records if r["split"] == split], output / f"{split}.csv")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
