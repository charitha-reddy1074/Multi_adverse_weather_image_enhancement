# Reproduction status

## Implemented

- One configurable model with dense patch embedding, hierarchical
  PoolFormer-style encoder, recurrent Haar supplementary priors, ordered
  channel/space/wavelet queried decoder, skip reconstruction, and RGB output.
- Differentiable DCT loss, paired-folder/CSV loader, augmentations, train and
  validation loop, checkpoint/resume, AMP, gradient accumulation, TensorBoard,
  symmetric inference padding, PSNR/SSIM, and result CSV generation.
- Paper-oriented and pilot configurations, architecture assumptions, dataset
  source catalog, manifest builder, and CPU/GPU-ready tests.
- Encoder pooling/self-attention, SP, spatial/frequency decoder, task-query,
  loss, and all six attention-order ablation controls.

## Not achieved in this environment

- No public dataset archives were downloaded or locally present, so no training
  run, checkpoint, PSNR/SSIM score, visual result, or FLOPs result is claimed.
- PyTorch, torchvision, Pillow, and scikit-image are not installed in the active
  Python environment. Syntax compilation passed; runtime architecture and loss
  tests could not run here.
- The checked-in paper text extraction supports the Figure 4 top-level mapping.
  The figure pixels were not visually rendered, so internal details visible
  only in the drawing remain documented assumptions.
- Upload to the requested GitHub repository is pending remote repository access
  and a safe local Git checkout; the working directory itself has no `.git`
  directory and its parent `.git` points at the user's home directory.

## Next executable checks

Install a compatible PyTorch build from pytorch.org, then run:

```bash
pip install -r requirements.txt
python -m pytest tests/
python scripts/build_manifests.py --root datasets/AIR40K/raw --output data/manifests
python scripts/train.py --config configs/pilot.yaml
```
