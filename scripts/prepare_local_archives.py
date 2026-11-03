"""Extract/audit local archives and create the user's custom 80/20 train/test split.

Retains original archives and official tests. Writes separate all-available
manifests; never silently replaces the existing Rain14000 experiment manifests.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.archive import safe_extract
from datasets.pairing import validate_pair
from datasets.splitting import write_custom_manifests


def main():
    raw = Path('data/raw').resolve()
    output = Path('data/manifests/available')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), archives={}, issues=[],
                  limitations=[
                      'AIR40K membership and Snow100K-L training subset are unavailable.',
                      'Haze/clear original release and split unknown: custom experiment only.',
                      'ITS_clear has no matching local ITS haze archive; trans is not RGB clean ground truth.',
                      'Source release duplicates within a split are preserved and reported.'])
    destinations = {'Rain14000.zip': 'Rain14000', 'Rain12.zip': 'Rain12',
                    'train.zip': 'Raindrop', 'Snow100K-training.tar.gz': 'Snow100K-training',
                    'clear.zip': 'haze-candidates', 'haze.zip': 'haze-candidates',
                    'trans.zip': 'haze-candidates', 'ITS_clear.zip': 'ITS-unpaired'}
    for filename, folder in destinations.items():
        archive = raw / filename
        if not archive.is_file():
            report['issues'].append(f'Missing archive: {filename}')
            continue
        print(f'Extracting and checking {filename}', flush=True)
        try:
            safe_extract(archive, raw / 'extracted' / folder)
            with archive.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            report['archives'][filename] = dict(status='EXTRACTED', bytes=archive.stat().st_size,
                sha256=digest, destination=str(raw / 'extracted' / folder))
        except Exception as error:
            report['archives'][filename] = dict(status='INVALID - EXCLUDED', error=str(error))
            report['issues'].append(f'{filename}: {error}')
        (output / 'integrity.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    # Never reuse stale extracted directories after a required archive failed
    # verification. Retain existing manifests until their inputs are trustworthy.
    required = ('Rain14000.zip', 'Rain12.zip', 'train.zip',
                'Snow100K-training.tar.gz', 'clear.zip', 'haze.zip')
    unavailable = [name for name in required
                   if report['archives'].get(name, {}).get('status') != 'EXTRACTED']
    if unavailable:
        report.update(status='BLOCKED - REQUIRED ARCHIVES NOT VERIFIED',
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      manifests_updated=False, unavailable_archives=unavailable)
        (output / 'integrity.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        raise RuntimeError('Required archives not verified; existing manifests retained: '
                           + ', '.join(unavailable))
    records = []
    # These manifests were produced by the source-specific Rain preparation.
    for split in ('train', 'val', 'test'):
        with Path(f'data/manifests/{split}.csv').open(newline='', encoding='utf8') as stream:
            records.extend(csv.DictReader(stream))

    def add_pairs(source, weather, pairs):
        pairs = sorted(pairs)
        for image, target, scene in pairs:
            records.append(dict(input_path=str(image), target_path=str(target), weather_type=weather,
                source_dataset=source, scene_id=scene, split='candidate'))

    if report['archives'].get('train.zip', {}).get('status') == 'EXTRACTED':
        base = raw / 'extracted/Raindrop/train'
        add_pairs('Raindrop', 'raindrop', [(p, base / 'gt' / (p.stem.removesuffix('_rain')+'_clean.png'),
            p.stem.removesuffix('_rain')) for p in (base / 'data').glob('*_rain.png')])
    if report['archives'].get('Snow100K-training.tar.gz', {}).get('status') == 'EXTRACTED':
        base = raw / 'extracted/Snow100K-training/all'
        add_pairs('Snow100K-training', 'snow', [(p, base / 'gt' / p.name, p.stem)
            for p in (base / 'synthetic').glob('*.jpg')])
    # User authorized a custom split; do not invent an official source identity.
    base = raw / 'extracted/haze-candidates'
    candidates = [dict(input_path=str(p), target_path=str(base / 'clear' / (p.stem.split('_')[0]+'.png')),
        weather_type='haze', source_dataset='UNVERIFIED-haze-release', split='candidate',
        scene_id=p.stem.split('_')[0]) for p in sorted((base / 'haze').glob('*.png'))]
    print(f'Validating {len(records)} eligible and {len(candidates)} local haze candidate pairs', flush=True)
    invalid, valid = [], []
    for index, row in enumerate(records + candidates):
        image, target = Path(row['input_path']), Path(row['target_path'])
        ok, a, b = validate_pair(image, target)
        if not ok:
            invalid.append(dict(input=str(image), target=str(target), sizes=[a, b]))
            continue
        valid.append(row)
        if index % 5000 == 0:
            print(f'Validated {index}/{len(records)+len(candidates)}', flush=True)
    report.update(invalid_pairs=invalid)
    write_custom_manifests(valid, output, report, report_path=Path('docs/LOCAL_DATASET_AUDIT.json'))


if __name__ == '__main__':
    main()
