"""Prepare the four-weather run from audited local manifests, without re-splitting."""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.pairing import write_manifest, validate_pair


def main():
    proof = json.loads(Path("docs/RESIDE_MIRROR_VERIFICATION.json").read_text())
    if proof["status"] != "MATCH" or proof["checked_files"] != 11000 or proof["mismatches"]:
        raise ValueError("Run verify_reside_mirror.py successfully first")
    output = Path("data/manifests/four_weather")
    splits = {}
    identities = {}
    files = set()
    samples = {}
    for split in ("train", "test", "official_test"):
        source = Path(f"data/manifests/available/{split}.csv")
        with source.open(newline="", encoding="utf8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            expected = "test" if split == "official_test" else split
            if row["split"] != expected:
                raise ValueError(f"Unexpected split in {source}")
            if row["source_dataset"] == "UNVERIFIED-haze-release":
                row["source_dataset"] = "RESIDE-ITS-HF-val-custom"
            key = (row["source_dataset"], row["scene_id"])
            if key in identities and identities[key] != split:
                raise ValueError(f"Cross-split scene {key}")
            identities[key] = split
            files.update((row["input_path"], row["target_path"]))
            if split == "train":
                samples.setdefault(row["weather_type"], row)
        splits[split] = rows
    if set(samples) != {"rain", "raindrop", "snow", "haze"}:
        raise ValueError(f"Missing weather types: {set(samples)}")
    missing = [name for name in files if not Path(name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    for weather, row in samples.items():
        valid, a, b = validate_pair(row["input_path"], row["target_path"])
        if not valid:
            raise ValueError(f"Invalid {weather} sample: {a}, {b}")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_manifest(rows, output / f"{split}.csv")
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  counts={split: dict(Counter(r["weather_type"] for r in rows))
                          for split, rows in splits.items()},
                  totals={split: len(rows) for split, rows in splits.items()},
                  unique_files=len(files), missing_files=0, scene_overlap=0,
                  decoded_sample_weather=sorted(samples),
                  scope="Fresh file-existence, scene-ID and four sample decode checks; prior full decode/pixel identity audit retained; trainer repeats clean-pixel split checks before optimization.",
                  manifests={split: hashlib.sha256((output / f"{split}.csv").read_bytes()).hexdigest()
                             for split in splits},
                  haze_mirror_revision=proof["revision"],
                  split_policy="Existing custom 80/20 train/test; no validation; official tests separate.")
    (output / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    cfg = yaml.safe_load(Path("configs/local_available.yaml").read_text())
    cfg.update(train_manifest=str(output / "train.csv"), test_manifest=str(output / "test.csv"))
    Path("configs/four_weather.yaml").write_text(
        "# Custom four-weather run; 80/20 train/test, no validation.\n"
        "# Haze matches the pinned unofficial HF ITS val mirror; not a SOTS benchmark.\n"
        + yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
