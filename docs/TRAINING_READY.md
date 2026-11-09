# Local training readiness - 2026-11-02

The corrected model and **58,766 training / 14,695
test pairs** are ready for a custom experiment with four weather types. The user
requested **80/20 training/testing only**, so there is no validation set.
[Dataset details and missing components](DATASET_STATUS.md).

## Start from scratch

```powershell
./.venv/Scripts/python train.py --config configs/local_available.yaml --output checkpoints/local80_20
```

This uses the corrected four-position hierarchy, prior routing and wavelet decoder.
It is a resource-adapted paper-guided experiment: base width 16, one block at each
of four encoder positions, 128x128 crops, explicit pooled spatial/wavelet K/V,
batch 2, gradient accumulation 4, 20 assumed epochs and balanced weather sampling.
The 612,435-parameter model is not the authors' 58.34M model. Haze source/split
is unverified, so no canonical RESIDE/AIR40K claim can be made.

The trainer saves `latest.pt` after each epoch. With validation disabled, it does
not select a best checkpoint. It evaluates the custom test set only after the
fixed epoch schedule and writes `final_test.json`. Keep that test set held out;
do not tune hyperparameters against repeated test runs. The separate known
official tests are not consumed by this config.

The current task performed fresh diagnostic optimization and feasibility checks,
**not a full 20-epoch dataset training run**. No old checkpoint was resumed.
The old three-position diagnostic checkpoint is incompatible with this model.
Resume only a checkpoint created with the current config and graph:

```powershell
./.venv/Scripts/python train.py --config configs/local_available.yaml --resume checkpoints/local80_20/latest.pt --output checkpoints/local80_20
```

## Strict spatial attention on this machine

`configs/paper.yaml` disables K/V pooling and uses full-key attention. It still
contains unverified numerical architecture/reconstruction/query assumptions.
The RTX 2000 Ada (16 GiB) has passed a fresh strict batch-one 256x256
forward/backward/optimizer step: **99.77 seconds**, **0.947 GiB peak allocated
CUDA memory**. Query chunking and recomputation preserve every key; they do not
approximate the attention axis. The initial unchunked attempt ran out of memory
because this Windows PyTorch build has no compiled Flash Attention kernel.
[Measured evidence](architecture/system_validation.json).

For a custom strict experiment, copy the local config, set
`spatial_kv_pooling: false`, and choose the crop/width/depth/batch deliberately.
The measured paper-guided batch-one path is substantially slower than the pooled
pilot; a completed long run has not been benchmarked. Do not assume the paper's
runtime or FLOPs apply to the chosen widths/strides here.

**Exact reproduction still requires** the supplementary/original model settings,
query/SP/tail details, loss-reduction clarification and AIR40K membership.
See [current architecture review](architecture/README.md) and [activity log](LOG.md).
