# Local setup and training

Status: **ready for a Rain14000 source-release training experiment**.
**Not ready for a verified AIR40K all-weather reproduction.** Rain800,
Rain100H/L, Rain1200, raindrop, snow and haze data are still blocked/partial.
No full training was launched by this audit.

Architecture re-verification is now complete and finds the implementation is
**not the exact AIRFormer graph**. See `architecture/README.md` for extracted
figures and differences. The ready pipeline trains the current reconstruction.
For the user's strict-paper objective, resolve those architecture differences
before investing in full training; later graph corrections may change checkpoint
compatibility. Working training does not establish paper fidelity.

The isolated `.venv/` has Python 3.14.6, torch 2.13.0+cu130,
torchvision 0.28.0+cu130, TensorBoard and the project dependencies. CUDA was
tested on the NVIDIA RTX 2000 Ada Generation (16 GiB). Python/PyTorch/CUDA
versions are local choices, not claims about the paper. The complete installed
package snapshot is `requirements-lock-windows-cu130.txt`.

Available manifests under `data/manifests/`:

- `train.csv`: 11,970 Rain14000 input/target pairs.
- `val.csv`: 630 pairs from 45 disjoint official-training clean scenes.
- `test.csv`: 1,400 official Rain14000 pairs plus 12 Rain12 pairs.
- `integrity.json`: full decoding, image-size and clean-pixel split checks.

From this repository in PowerShell:

```powershell
# Start the resource-efficient experiment; this will take substantial time.
./.venv/Scripts/python train.py --config configs/pilot.yaml --output checkpoints/pilot

# Resume the same experiment/configuration.
./.venv/Scripts/python train.py --config configs/pilot.yaml --resume checkpoints/pilot/latest.pt --output checkpoints/pilot

# After training, demonstrate the saved weights without training again.
./.venv/Scripts/python inference.py --input path/to/images --checkpoint checkpoints/pilot/best.pt --output outputs/demo --device cuda

# Quantitative evaluation requires matching held-out clean references.
./.venv/Scripts/python evaluate.py --pred outputs/demo --target path/to/matching/clean/images --csv results/demo_metrics.csv
```

The same trained weights serve all weather inputs; quality on a weather type
requires appropriate training coverage. Rain-only training is not proof of
effective raindrop/snow/haze restoration. Once weights are trained and saved,
testing/demonstration requires only loading the checkpoint and inference. You
do **not** retrain for each demonstration or program restart. Keep both the
checkpoint and its saved `model_config`. Retrain or fine-tune only to change
learned behavior or improve performance.

`configs/paper.yaml` retains the requested optimizer/loss/augmentation settings,
but original width/depth/batch/epoch/query/fusion details are unresolved, and
pooled K/V is an adaptation. Do not describe it as exact paper reproduction.
It is independent of `configs/pilot.yaml`; no hardware adaptation overwrites it.
Paper config batch 8 training memory has not been measured on this GPU. Begin
with pilot rather than assuming the paper config fits.

| Setting | Paper-oriented config | Resource-efficient pilot |
|---|---|---|
| crop / batch | 256 / 8 | 128 / 2 |
| base width / blocks | 32 / [2,2,2] (assumed) | 16 / [1,1,1] |
| heads / K,V token cap | 4 / 1024 (assumed/adapted) | 2 / 256 |
| epochs | 300 (assumed) | 20 |
| cosine minimum LR | 1e-7 | 1e-6 |
| gradient accumulation | 1 | 4 |
| mixed precision | enabled locally | enabled locally |

Both use AdamW (.9,.999), weight decay1e-4, initial LR2e-4, MixUp probability.5,
the same Smooth L1 + .04*DCT-frequency loss, and the same three-domain decoder
mechanisms. These differences are explicit; neither config is overwritten.

Rebuild manifests on another machine (absolute local file paths are regenerated):

```powershell
./scripts/download_required.ps1 -Datasets Rain14000,Rain12
./.venv/Scripts/python scripts/prepare_available_data.py
./.venv/Scripts/python -m pytest tests -q
```

Downloads retain partial files and preserve original archives. Dataset licenses,
sizes and blocked access conditions are in `datasets/downloads.csv` and
`docs/DATASET_STATUS.md`. Do not mix official test data into training/validation.
Original AIR40K sampling is unavailable; do not silently force Table I counts.

For a fresh Windows/Python-3.14 environment with this supported CUDA driver,
create a venv, install the explicit CUDA-tagged torch/torchvision pair from
`https://download.pytorch.org/whl/cu130`, then install the lock file with
`--extra-index-url https://download.pytorch.org/whl/cu130`. Other platforms need
a compatible PyTorch build; this Windows snapshot is not a universal lockfile.

`outputs/audit_smoke/checkpoints/` contains only a two-step-per-epoch integration
fixture trained for two tiny epochs. Its visual output is poor, as expected.
It verifies CLI wiring, serialization/resume, padding/cropping, held-out RGB
metrics and CUDA execution. **Do not use or distribute it as a pretrained model.**
