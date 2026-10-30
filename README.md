# AIRFormer: weather restoration reproduction

This repository contains an independent PyTorch reconstruction of **AIRFormer** by Tao Gao et al., *Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration*, IEEE TCSVT 34(3), 2024 ([DOI](https://doi.org/10.1109/TCSVT.2023.3299324)). The authors' source repository and pretrained weights were unavailable during this implementation. This code follows the paper and its figures/equations, while recording unresolved details as explicit engineering assumptions in [`docs/architecture_assumptions.md`](docs/architecture_assumptions.md). It does not claim exact equivalence to unpublished code or report fabricated results.

## What is implemented

- One RGB restoration model for rain streaks, adherent raindrops, snow, and haze.
- Convolutional patch embedding, a hierarchical PoolFormer-style encoder, Haar-wavelet Supplementary Prior modules, and hierarchical reconstruction.
- Decoder refinement in the paper's preferred **Channel → Space → Wavelet** order, with learned task-biased query parameters.
- Smooth L1 plus weighted DCT frequency loss, paired-image loading, matched crops/augmentations, training/validation, checkpoint resume, inference, PSNR/SSIM evaluation, and result visualization.
- Paper-oriented and memory-oriented YAML configurations, dataset provenance catalog, manifest construction/verification, and component ablation controls.

Read the [architecture assumptions](docs/architecture_assumptions.md), [dataset provenance](docs/dataset_provenance.md), and [reproduction status](docs/reproduction_report.md) before interpreting the implementation or its output.

## Readiness and items you must supply

The repository is ready to install and run once a compatible Python/PyTorch environment and paired image data are supplied. It does **not** include dataset archives, generated manifests, pretrained weights, or a trained checkpoint. No training run or benchmark score is claimed. The current development environment did not have PyTorch/Pillow/scikit-image/pytest available, so runtime tests and a training smoke test have not yet been confirmed here.

You will need:

1. Python 3.10 or 3.11 recommended.
2. A PyTorch build compatible with your OS and CPU/CUDA setup. Install it using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) before the remaining requirements.
3. The paired datasets you are permitted to use. See [`datasets/sources.csv`](datasets/sources.csv) and [`datasets/README.md`](datasets/README.md). Dataset source pages are not all direct download links; follow each owner's access and license terms.
4. A disjoint validation split drawn from training scenes. Do not use test images for validation or training.

## Installation

Clone the repository and enter its directory. Create and activate a virtual environment:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux/macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the appropriate PyTorch/torchvision pair from the official selector, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the environment and architecture tests:

```bash
python -c "import torch, torchvision, PIL, skimage; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m pytest tests/ -q
```

The tests need to pass before starting a long training run. For a CPU-only smoke run, use a very small dataset and override the pilot batch/worker values as needed; full training on CPU will be slow.

## Prepare paired data

The manifest builder pairs images by **matching relative paths**, never by sorted order. Normalize permitted data into this layout (same filename and relative path for input and target):

```text
datasets/AIR40K/raw/
  train/<source>/degraded/<scene>/<image>.png
  train/<source>/clean/<scene>/<image>.png
  val/<source>/degraded/<scene>/<image>.png
  val/<source>/clean/<scene>/<image>.png
  test/<source>/degraded/<scene>/<image>.png
  test/<source>/clean/<scene>/<image>.png
```

The paper does not specify a validation split. Reserve validation **scenes** from the official training set before manifest generation. Keep related variants and repeated views of a scene in the same split (especially RESIDE and RainCityscapes). Never move official test images into train or validation. Review [`docs/dataset_provenance.md`](docs/dataset_provenance.md) for count discrepancies and unpaired datasets such as ACDC.

Build and check manifests:

```bash
python scripts/build_manifests.py --root datasets/AIR40K/raw --output data/manifests
```

The script prints missing pairs, split leakage, and known benchmark count mismatches. It writes `train.csv`, `val.csv`, and `test.csv`; training uses `train.csv` and `val.csv`. The generated CSVs and raw data are ignored by Git, so create them on the machine that will train. If filenames differ between degraded and clean images, create a reviewed CSV manually with columns `input_path,target_path,weather_type,source_dataset,split,scene_id`; do not infer pairs by order. The default folder-based verifier is:

```bash
python scripts/verify_dataset.py --root datasets/AIR40K/train
```

For manifest-backed checks, use the `verify_manifest` helper in `datasets/pairing.py` or run the dataset tests after installing requirements.

## Train

Start with the pilot config. It preserves AIRFormer's core modules while lowering crop size, channel width, and block count. Adjust `batch_size`, `num_workers`, `base_channels`, and `gradient_accumulation_steps` to fit available memory.

```bash
python train.py --config configs/pilot.yaml
```

The default output folder is `checkpoints/`; it writes `latest.pt`, `best.pt`, and TensorBoard logs. Resume from the latest checkpoint:

```bash
python train.py --config configs/pilot.yaml --resume checkpoints/latest.pt
```

To view logs:

```bash
tensorboard --logdir checkpoints/runs
```

`configs/paper.yaml` uses 256×256 crops and the paper's reported optimizer/loss settings, but model width/depth, batch size, and epoch count are unresolved or engineering defaults. It is not a verified exact reproduction and may exceed modest GPU memory. Inspect the YAML comments and assumption document before using it.

## Inference and evaluation

Restore a single image or all images in a folder. The inference script pads symmetrically to dimensions divisible by 128, crops the result back to the original size, and saves an inference-time CSV. Tiling is optional and is an engineering approximation.

```bash
python inference.py --input path/to/degraded_images --checkpoint checkpoints/best.pt --output outputs/restored --device auto
```

For a single image, `--input` may be its file path. To enable tiling for large images, pass `--tile-size 512` (choose a size that fits your GPU).

Prediction paths must match target paths relative to their respective folders:

```bash
python evaluate.py --pred outputs/restored --target path/to/clean_images --csv results/per_image_metrics.csv --checkpoint checkpoints/best.pt
```

This reports per-image CSV entries and aggregate RGB PSNR/SSIM. FLOPs are estimated only when `thop` can profile the model and may omit some operations. Results are meaningful only when the exact benchmark split, pair alignment, color range, and evaluation protocol are recorded. For side-by-side comparisons:

```bash
python scripts/visualize.py --input path/to/degraded_images --pred outputs/restored --target path/to/clean_images --output results/visuals
```

## Dataset catalog and acquisition check

`datasets/sources.csv` lists the requested sources, their weather label, availability notes, and whether they are paired restoration data. ACDC's corresponding normal-condition images are weak scene correspondences, not pixel-aligned restoration targets. `scripts/download_datasets.py` checks listed project pages for reachability only; it does not download or verify archives. After downloading, verify archive integrity and pairing locally and record archive name, URL, terms/license, date, and SHA-256.

## Repository map

- `models/` — patch embedding, encoder, Supplementary Prior, queried attention, decoder, full model.
- `losses/` — multi-domain Smooth L1 + DCT objective.
- `datasets/`, `scripts/build_manifests.py` — paired data loading and manifest tooling.
- `configs/` — pilot and paper-oriented settings.
- `train.py`, `inference.py`, `evaluate.py` — command-line entrypoints.
- `tests/` — wavelet, attention, model, loss, data, and ablation checks.
- `docs/` — architecture assumptions, dataset provenance, ablation protocol, reproduction status.

See [`docs/ablation_protocol.md`](docs/ablation_protocol.md) for the available component and attention-order comparisons. The reported AIRFormer parameter/FLOPs figures are references, not guaranteed values for this independently reconstructed model.
