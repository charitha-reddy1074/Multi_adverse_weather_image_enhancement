import csv
from PIL import Image
from datasets.splitting import write_custom_manifests


def test_custom_split_groups_renamed_clean_duplicates_and_keeps_official_tests(tmp_path):
    records = []
    for i in range(12):
        image = tmp_path/f'{i}.png'
        # Last two are renamed copies of the first two clean scenes.
        color = i if i < 10 else i-10
        Image.new('RGB',(8,8),(color,40,70)).save(image)
        records.append(dict(input_path=str(image), target_path=str(image), weather_type='haze',
            source_dataset='local', split='quarantine', scene_id=str(i)))
    official = tmp_path/'official.png'
    Image.new('RGB',(8,8),(250,50,80)).save(official)
    records.append(dict(input_path=str(official), target_path=str(official), weather_type='rain',
        source_dataset='official', split='test', scene_id='official'))
    output = tmp_path/'manifests'
    report = write_custom_manifests(records, output, {})
    assert report['official_test_pairs'] == 1 and report['validation_pairs'] == 0
    assert not (output/'val.csv').exists()
    assignments = {}
    for split in ('train','test'):
        with (output/f'{split}.csv').open() as stream:
            for row in csv.DictReader(stream):
                assignments[int(row['scene_id'])] = split
    assert assignments[0] == assignments[10]
    assert assignments[1] == assignments[11]
    assert set(assignments.values()) == {'train','test'}
