"""Custom scene-disjoint 80/20 experiments; preserve known official tests."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from .pairing import write_manifest


def _clean_hash(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return path, hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return path, hashlib.file_digest(stream, 'sha256').hexdigest()


def write_custom_manifests(records, output, report, seed=42, report_path=None):
    """Group scenes, identical clean pixels and input bytes before splitting.

    Official tests stay outside this custom experiment. Haze pairs with unknown
    original splits enter only this explicitly user-authorized custom experiment.
    Grouping can make the pair ratio differ slightly from exactly 80 percent.
    """
    output = Path(output)
    official = [r for r in records if r['split'] == 'test']
    pool = [dict(r) for r in records if r['split'] != 'test']
    all_records = pool + official
    print('Hashing scenes and inputs for custom 80/20 split', flush=True)
    with ThreadPoolExecutor(max_workers=8) as workers:
        clean_hashes = dict(workers.map(_clean_hash, sorted({r['target_path'] for r in all_records})))
        input_hashes = dict(workers.map(_file_hash, sorted({r['input_path'] for r in all_records})))

    def identities(row):
        return [('scene', row['source_dataset'], row['scene_id']),
                ('pixels', clean_hashes[row['target_path']]),
                ('input_bytes', input_hashes[row['input_path']])]

    # Exclude any pool member matching a known official test identity.
    forbidden = {key for row in official for key in identities(row)}
    excluded = [r for r in pool if any(key in forbidden for key in identities(r))]
    pool = [r for r in pool if not any(key in forbidden for key in identities(r))]
    parent = list(range(len(pool)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(pool):
        for key in identities(row):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(pool):
        groups[find(i)].append(row)
    by_source = defaultdict(list)
    for group in groups.values():
        sources = '+'.join(sorted({r['source_dataset'] for r in group}))
        by_source[sources].append(group)
    rng = random.Random(seed)
    for source in sorted(by_source):
        bucket = sorted(by_source[source], key=lambda group: min(r['input_path'] for r in group))
        rng.shuffle(bucket)
        train_count = int(.8*len(bucket))
        for index, group in enumerate(bucket):
            for row in group:
                row['split'] = 'train' if index < train_count else 'test'
    # Validate all identities from the cached hashes, without decoding twice.
    assigned = {}
    for row in pool:
        for key in identities(row):
            if key in assigned and assigned[key] != row['split']:
                raise ValueError(f'Cross-split leakage: {key[0]}')
            assigned[key] = row['split']
    for split in ('train', 'test'):
        write_manifest([r for r in pool if r['split'] == split], output / f'{split}.csv')
    write_manifest(official, output / 'official_test.csv')
    write_manifest(excluded, output / 'excluded_test_overlap.csv')
    # No separate validation set: a test set must not select the checkpoint.
    validation = (output / 'val.csv').resolve()
    if validation.parent != output.resolve():
        raise ValueError('Invalid validation cleanup path')
    validation.unlink(missing_ok=True)
    quarantine = (output / 'haze_quarantine.csv').resolve()
    if quarantine.parent != output.resolve():
        raise ValueError('Invalid quarantine cleanup path')
    quarantine.unlink(missing_ok=True)
    counts = dict(Counter(f"{r['source_dataset']}/{r['split']}" for r in pool))
    total = len(pool)
    train = sum(r['split'] == 'train' for r in pool)
    weather = sorted({r['weather_type'] for r in pool if r['split']=='train'})
    report.update(finished_at=datetime.now(timezone.utc).isoformat(), counts=counts,
        status=('READY FOR CUSTOM FOUR-WEATHER EXPERIMENT; NOT AIR40K'
                if set(weather) == {'rain','raindrop','snow','haze'} else 'PARTIAL WEATHER COVERAGE'), seed=seed,
        split_policy='User requested train/test only, 80/20 per-source clean-identity groups; no validation split',
        custom_pairs=total, training_pairs=train, testing_pairs=total-train,
        training_fraction=train/total if total else None, validation_pairs=0,
        official_test_pairs=len(official), excluded_test_overlap=len(excluded),
        training_weather=weather,
        haze_policy='User-authorized custom 80/20 split; original release and split unverified; not a RESIDE/AIR40K benchmark claim',
        identity_checks='No scene, identical clean pixels or identical input bytes cross custom train/test; known official tests excluded',
        source_membership='Custom local experiment; not AIR40K')
    report.pop('quarantine_pairs', None)
    report['limitations'] = [
        'Exact AIR40K membership and Snow100K-L training selection unavailable.',
        'Haze original source/split unknown: custom experiment only, no canonical benchmark claims.',
        'ITS_clear has no matching local ITS haze archive; trans is not RGB clean ground truth.',
        'Exact duplicates are grouped; near-duplicate scenes may still require source metadata review.']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'integrity.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({key: report[key] for key in ('status','counts','training_pairs','testing_pairs','official_test_pairs')}, indent=2))
    return report
