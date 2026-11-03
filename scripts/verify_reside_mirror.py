"""Compare existing haze/clean files with a pinned Hugging Face mirror revision.

Only metadata is downloaded. A successful comparison identifies the mirror
content, not the authenticity of its claimed original RESIDE release.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    repo = "dronefreak/RESIDE-ITS"
    api = "https://huggingface.co/api/datasets/" + repo
    session = requests.Session()
    response = session.get(api, timeout=60)
    response.raise_for_status()
    revision = response.json()["sha"]
    root = Path("data/raw/extracted/haze-candidates")
    records = []
    for remote, local in (("val/clear", "clear"), ("val/haze", "haze")):
        url = f"{api}/tree/{revision}/{remote}?limit=1000"
        entries = []
        while url:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            entries.extend(item for item in response.json() if item["type"] == "file")
            url = response.links.get("next", {}).get("url")
        remote_names = {Path(item["path"]).name for item in entries}
        local_names = {p.name for p in (root / local).glob("*.png")}
        if remote_names != local_names:
            raise ValueError(f"File membership differs for {remote}: local={len(local_names)}, remote={len(remote_names)}")
        records.extend((root / local / Path(item["path"]).name,
                        item["lfs"]["oid"], item["size"]) for item in entries)
        print(f"Listed {remote}: {len(entries)} files", flush=True)

    def check(record):
        path, expected, size = record
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return None if digest == expected and path.stat().st_size == size else str(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        mismatches = [result for result in pool.map(check, records) if result]
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), repository=repo,
                  revision=revision, remote_directories=["val/clear", "val/haze"],
                  local_root=str(root.resolve()), checked_files=len(records),
                  checked_bytes=sum(r[2] for r in records), mismatches=mismatches,
                  status="MATCH" if not mismatches else "MISMATCH",
                  meaning="All local PNG files match mirror LFS SHA-256 and size; original release attribution remains the mirror publisher's claim.",
                  split_policy="Custom scene-disjoint 80/20 split of mirror val content; not official ITS training or SOTS testing.")
    Path("docs/RESIDE_MIRROR_VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise ValueError("Mirror verification failed")


if __name__ == "__main__":
    main()
