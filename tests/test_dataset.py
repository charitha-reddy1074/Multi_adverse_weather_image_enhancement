import csv
from PIL import Image
import torch
from airformer.data import PairedWeatherDataset
from datasets.pairing import verify_manifest


def test_manifest_pair_decoding_alignment(tmp_path):
    degraded=tmp_path/"raw"/"rain"/"a.png"
    clean=tmp_path/"raw"/"clean"/"a.png"
    degraded.parent.mkdir(parents=True); clean.parent.mkdir(parents=True)
    image=Image.new("RGB",(20,18),(120,80,30))
    image.save(degraded); image.save(clean)
    manifest=tmp_path/"train.csv"
    fields=["input_path","target_path","weather_type","source_dataset","split","scene_id"]
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerow({"input_path":str(degraded),"target_path":str(clean),
            "weather_type":"rain","source_dataset":"toy","split":"train","scene_id":"a"})
    ds=PairedWeatherDataset(tmp_path,crop_size=16,augment=False,manifest=manifest)
    x,y,_=ds[0]
    assert x.shape==y.shape==(3,16,16)
    torch.testing.assert_close(x,y)
    count,issues=verify_manifest(manifest)
    assert count==1 and not issues


def test_manifest_detects_split_leakage(tmp_path):
    image=tmp_path/"x.png"; Image.new("RGB",(8,8)).save(image)
    manifest=tmp_path/"pairs.csv"
    fields=["input_path","target_path","weather_type","source_dataset","split","scene_id"]
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for split in ("train","test"):
            w.writerow({"input_path":str(image),"target_path":str(image),
                "source_dataset":"s","split":split,"scene_id":"same"})
    _,issues=verify_manifest(manifest)
    assert any("leakage" in e for e in issues)
