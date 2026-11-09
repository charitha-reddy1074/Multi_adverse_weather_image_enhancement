import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from datasets.pairing import write_manifest
from models.airformer import AIRFormer
from PIL import Image
from scripts.test_restoration import mask_counts, mask_scores


def test_auxiliary_mask_semantics_and_undefined_scores():
    # Four pixels, one each of TN, FP, FN, and TP. The predicted mask measures
    # applied correction relative to the input, not residual output error.
    degraded = np.array([0.0, 0.0, 0.8, 0.8], dtype=np.float32)[None, :, None].repeat(3, 2)
    target = np.zeros_like(degraded)
    restored = np.array([0.0, 0.8, 0.8, 0.0], dtype=np.float32)[None, :, None].repeat(3, 2)
    counts = mask_counts(degraded, restored, target, .1)
    assert counts == dict(tn=1, fp=1, fn=1, tp=1)
    scores = mask_scores(counts)
    assert scores["accuracy"] == scores["precision"] == scores["recall"] == scores["f1"] == .5
    assert scores["confusion_matrix"] == [[1, 1], [1, 1]]
    empty = mask_scores(dict(tn=4, fp=0, fn=0, tp=0))
    assert empty["accuracy"] == 1
    assert empty["precision"] is empty["recall"] is empty["f1"] is None


def test_evaluation_preflight_full_report_and_resume(tmp_path):
    records = []
    for i, (split, weather) in enumerate((("train", "rain"), ("test", "rain"), ("test", "haze"))):
        source, target = tmp_path/f"input{i}.png", tmp_path/f"target{i}.png"
        Image.new("RGB", (12, 14), (50+i*20, 90, 110)).save(source)
        Image.new("RGB", (12, 14), (70+i*20, 100, 120)).save(target)
        records.append(dict(input_path=str(source), target_path=str(target), split=split,
            weather_type=weather, source_dataset="fixture", scene_id=str(i)))
    write_manifest(records[:1], tmp_path/"train.csv")
    write_manifest(records[1:], tmp_path/"test.csv")
    model_config = dict(base_channels=4, stage_blocks=(1,1,1,1), attention_heads=1,
                        max_spatial_tokens=16, spatial_kv_pooling=True)
    checkpoint = tmp_path/"model.pt"
    torch.save(dict(epoch=1, model=AIRFormer(**model_config).state_dict(), model_config=model_config,
        config=dict(epochs=1, train_manifest=str(tmp_path/"train.csv"),
                    test_manifest=str(tmp_path/"test.csv"), test_pad_multiple=8)), checkpoint)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/"results"
    command = [sys.executable, str(root/"scripts/test_restoration.py"),
               "--checkpoint", str(checkpoint), "--output", str(output)]
    for extra in (["--preflight-only"], ["--resume"], ["--resume"]):
        result = subprocess.run(command+extra, cwd=root, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
    status = json.loads((output/"status.json").read_text())
    assert status["state"] == "completed" and status["completed"] == 2
    summary = json.loads((output/"summary.json").read_text())["groups"]
    assert summary["overall"]["images"] == 2
    assert summary["rain"]["images"] == summary["haze"]["images"] == 1
    assert sum(summary["overall"]["auxiliary_masks"]["0.10"][k] for k in ("tn","fp","fn","tp")) == 2*12*14
    assert (output/"report.html").is_file()
    assert (output/"visualizations/auxiliary_confusion_matrix.png").is_file()
    assert (output/"visualizations/examples_haze.svg").is_file()
    # Changing the checkpoint must not mix new predictions with saved results.
    saved = torch.load(checkpoint, weights_only=True)
    saved["model"]["reconstruct.bias"] += .1
    torch.save(saved, checkpoint)
    result = subprocess.run(command+["--resume"], cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "different code/data/checkpoint" in result.stderr
