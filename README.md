# AIRFormer-inspired weather restoration reconstruction

**Architecture re-verification: NOT EXACT.** The main-paper comparison is complete
and found confirmed graph differences. Read the illustrated
[architecture comparison README](docs/architecture/README.md) before describing
this as a paper implementation. Three encoder levels, SP placement, pooled
spatial K/V, the wavelet attention/IDWT graph, assumed query/tail details, and
literal Equation 12 reduction prevent an exact reproduction claim. The paper
configuration has 1,146,883 parameters versus the paper's reported 58.34 million.

The model's tensor logic is unchanged by this documentation/comment review.
CUDA setup, 62 tests, real-pair training/resume/inference/evaluation smoke checks
pass. Rain14000 and Rain12 are downloaded and validated; other required datasets
are blocked/partial. Figures 4/5, Equations 9-13 and Tables XII/XIII were inspected
in the supplied 14-page PDF. Original code/supplementary implementation details
remain unavailable locally; the main-paper review cannot certify those settings.
Start with [local training instructions](docs/TRAINING_READY.md),
[audit findings](docs/AUDIT_REPORT.md), [paper compliance](docs/PAPER_COMPLIANCE.md),
and [actual dataset status](docs/DATASET_STATUS.md).

After training once, load the saved checkpoint for testing/demonstrations;
**retraining is not required each time**. The tiny audit checkpoint is diagnostic
only and is not a pretrained restoration model.

This repository contains an independent PyTorch reconstruction inspired by **AIRFormer** by Tao Gao et al., *Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration*, IEEE TCSVT 34(3), 2024 ([DOI](https://doi.org/10.1109/TCSVT.2023.3299324)). The authors' source repository and pretrained weights were unavailable during implementation. The code implements selected paper ideas with documented topology differences and engineering assumptions. It is not the exact architecture depicted in the paper and does not report fabricated reproduction results.

## What is implemented

- One RGB restoration model for rain streaks, adherent raindrops, snow, and haze.
- Convolutional patch embedding, a three-level PoolFormer-style encoder, Haar-wavelet prior approximations, and a chosen U-shaped reconstruction tail.
- Decoder refinement in the paper's preferred **Channel → Space → Wavelet** order, with learned task-biased query parameters.
- Smooth L1 plus weighted DCT frequency loss, paired-image loading, matched crops/augmentations, training/validation, checkpoint resume, inference, PSNR/SSIM evaluation, and result visualization.
- Paper-oriented and memory-oriented YAML configurations, dataset provenance catalog, manifest construction/verification, and component ablation controls.

Read the [architecture assumptions](docs/architecture_assumptions.md), [dataset provenance](docs/dataset_provenance.md), and [reproduction status](docs/reproduction_report.md) before interpreting the implementation or its output.

## Readiness and items you must supply

The local audit environment is installed under `.venv/` with CUDA-enabled PyTorch.
Validated Rain14000/Rain12 archives are under `data/raw/`, with manifests under
`data/manifests/`. Full training and a useful trained restoration checkpoint are
not yet available. See `docs/TRAINING_READY.md` for the exact local commands and
`docs/DATASET_STATUS.md` for unavailable all-weather sources. The instructions
below also describe setup on another machine.

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

### Requested datasets and source links

| Dataset | Source / download instructions | Use status |
|---|---|---|
| CDD-11 | [OneRestore project](https://github.com/gy65896/OneRestore) links the [CDD-11 Hugging Face dataset](https://huggingface.co/datasets/gy65896/CDD-11). | Paired archive is documented. Filter low-light-only and other non-weather classes if the experiment is limited to AIRFormer's weather domains. |
| Rain200H | [Joint Rain Removal project page](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) | Paired; 1,800 train / 200 test. Confirm archive naming and avoid any duplicate Rain100H/Rain1800 scenes. |
| Rain200L | [Joint Rain Removal project page](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) | Paired; 1,800 train / 200 test. |
| Snow100K-L | [DesnowNet author page](https://sites.google.com/view/yunfuliu/desnownet) | Paired synthetic subset; 16,801 test images. |
| Snow100K-S | [DesnowNet author page](https://sites.google.com/view/yunfuliu/desnownet) | Paired synthetic subset; 16,611 test images. |
| Outdoor-Rain | [Histoformer project](https://github.com/sunshangquan/Histoformer) and its documented benchmark setup | Paired joint rain/haze data; 9,000 train. Check which test protocol/archive you obtained (Test1 commonly 750; other reports use 1,500). |
| Raindrop | [Official DeRaindrop repository](https://github.com/rui1996/DeRaindrop) (README links the owner-hosted archive) | 861 train; Test-A is a subset of Test-B. The archive's documented Test-B has 239 images; do not concatenate Test-A and Test-B as independent images. AIR40K's 307 count needs archive-specific reconciliation. |
| SPA-Data | [Deraining project page](https://stevewongv.github.io/derain-project.html); [public implementation with documented dataset links](https://github.com/Yizhou-Li-CV/ECNet) | Paired real rain; standard test is 1,000. Confirm whether a training count refers to full-resolution images or extracted patches. |
| RESIDE-S | [Dust to Detail paper](https://openaccess.thecvf.com/content/CVPR2025W/WiCV/html/Mishra_Dust_to_Detail_Restoring_Sand-dust_Images_with_Frequency-Guided_Attention_and_Multi-Scale_Features_CVPRW_2025_paper.html) | Paper-described synthetic sand-dust derivative. No canonical standalone owner download was verified; this is not RESIDE's haze dataset. Treat as unavailable until its archive/generation recipe and source-scene split are confirmed. |
| Rain Cityscapes | [Rain Cityscapes paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Single_Image_Rain_Removal_Using_Classification_Guided_Network_CVPR_2019_paper.html); [Cityscapes source data](https://www.cityscapes-dataset.com/) | Paired synthetic variants; Cityscapes access/terms may apply. Preserve source-scene identity so variants from one scene cannot cross splits. |
| ACDC20 | [Official ACDC page](https://acdc.vision.ee.ethz.ch/) | 4,006 adverse-weather images, annotations, and same-scene normal-condition correspondences. Not pixel-aligned restoration pairs; exclude from supervised paired training and PSNR/SSIM. |

“Ready” here means the catalog and generic paired-manifest workflow can ingest verified pairs; it does not mean the archives are in this Git repository. **Before tomorrow's training**, download data from the links above where terms permit, normalize only validated pairs to the directory layout, create a scene-disjoint validation split, build manifests, and run `python -m pytest tests/ -q`. RESIDE-S requires separate provenance resolution and ACDC20 is not eligible for this paired supervised run.

## Repository map

- `models/` — patch embedding, encoder, Supplementary Prior, queried attention, decoder, full model.
- `losses/` — multi-domain Smooth L1 + DCT objective.
- `datasets/`, `scripts/build_manifests.py` — paired data loading and manifest tooling.
- `configs/` — pilot and paper-oriented settings.
- `train.py`, `inference.py`, `evaluate.py` — command-line entrypoints.
- `tests/` — wavelet, attention, model, loss, data, and ablation checks.
- `docs/` — architecture assumptions, dataset provenance, ablation protocol, reproduction status.

See [`docs/ablation_protocol.md`](docs/ablation_protocol.md) for the available component and attention-order comparisons. The reported AIRFormer parameter/FLOPs figures are references, not guaranteed values for this independently reconstructed model.
