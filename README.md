# AIRFormer research reimplementation

Independent PyTorch implementation guided by Tao Gao et al., “Frequency-Oriented
Efficient Transformer for All-in-One Weather-Degraded Image Restoration,” IEEE
TCSVT 34(3), 2024, DOI: 10.1109/TCSVT.2023.3299324.

This is not the authors' unavailable source code and includes no original
pretrained weights. Read [architecture assumptions](docs/architecture_assumptions.md)
before interpreting model internals or results. No benchmark results are
claimed by this repository.

## Included

- Shared restoration model with DWT supplementary priors and query-guided
  channel, spatial, and wavelet refinement.
- Paired image dataset loader, train/validation loop, inference, PSNR/SSIM
  evaluation, and dataset pair verification.
- Paper-oriented and memory-conscious pilot configs.
- Shape, gradient, and Haar reconstruction architecture tests.
- Configurable Table XI component and Table XII attention-order ablations (see
  [ablation protocol](docs/ablation_protocol.md)).
- A 256×256 Table XIII parameter/FLOPs/inference-time profiler.

## Data layout

Keep each degraded/clean pair under matching relative paths:

```text
datasets/AIR40K/raw/train/<source>/degraded/<relative-name>.png
datasets/AIR40K/raw/train/<source>/clean/<relative-name>.png
datasets/AIR40K/raw/test/<source>/degraded/<relative-name>.png
datasets/AIR40K/raw/test/<source>/clean/<relative-name>.png
```

The paper composes AIR40K from public rain, raindrop, snow, and haze datasets,
but the exact manifest and consolidated download are not available in this
workspace. Follow each source dataset's license and preserve its official
splits. The loader accepts the older `rainy/clean` folder naming as well.

## Setup and run

Install a compatible PyTorch build first, then install `requirements.txt`.

```bash
python scripts/build_manifests.py --root datasets/AIR40K/raw --output data/manifests
python scripts/train.py --config configs/pilot.yaml
```

For the paper-oriented settings:

```bash
python scripts/train.py --config configs/paper.yaml
```

The paper config records paper settings and explicitly marks unresolved model
width/depth, epoch count, and batch size. The pilot reduces crop size, width,
depth, and epochs to fit smaller hardware. Resume with
`python scripts/train.py --config configs/pilot.yaml --resume checkpoints/latest.pt`.

```bash
python scripts/infer.py --input datasets/AIR40K/test/degraded \
  --checkpoint checkpoints/best.pt --output outputs/restored
python scripts/evaluate.py --pred outputs/restored \
  --target datasets/AIR40K/test/clean
```

Evaluation pairs images by relative path and reports per-image and per-dataset
RGB PSNR (dB) and SSIM. Keep test folders separate from training data. To
check source project pages (not archive downloads), run
`python scripts/download_datasets.py`.

Create a disjoint validation subset from training scenes and save it as
`data/manifests/val.csv` (or `datasets/AIR40K/val/`). The paper table gives no
validation split; never select checkpoints on the official test set.

Use `python scripts/visualize.py --input <degraded> --pred outputs/restored
--target <clean>` to write side-by-side result strips. `configs/paper.yaml`
sets the paper crop size and training values; no performance scores are bundled.
