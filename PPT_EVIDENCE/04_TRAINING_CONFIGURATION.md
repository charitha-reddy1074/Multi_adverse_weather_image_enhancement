# Final training configuration and recorded history

The configuration below comes from the **saved epoch-20 checkpoint**, not a larger paper-oriented YAML. `sources/checkpoint_verification.json` records the strict-load result and all provenance hashes.

| Setting | Verified final setting |
|---|---|
| Parameters | 612,435 trainable |
| Encoder widths / depths | [16,32,64,128] / [1,1,1,1] |
| Attention | 2 heads, add learned query, Channel -> Space -> Wavelet |
| Resource approximation | Spatial/wavelet K/V pooling enabled; 256-token cap |
| Crop | 128 x 128 RGB |
| Batch / accumulation / effective batch | 8 / 1 / 8; short final batch may be smaller |
| Epochs / seed | 20 completed / 42 |
| Optimizer | AdamW, betas .9/.999, weight decay 1e-4 |
| Learning rate | Initial 2e-4, cosine schedule T_max=20, minimum 1e-6 |
| Loss | Mean elementwise Smooth L1 + .04 smoothed per-image orthonormal DCT-II norm |
| Frequency epsilon | .001 |
| Augmentation | Paired random crop, flips, quarter turns, uniform-lambda MixUp p=.5 |
| Sampling | Weather-balanced inverse-count sampling with replacement |
| AMP / clipping | CUDA autocast + GradScaler; clip gradient norm at 1.0 |
| Gradient checkpointing flag | False; attention query-chunk fallback can independently recompute probabilities |
| Workers | 2 |
| Validation | Disabled; fixed final epoch, no validation model selection |
| Hardware | NVIDIA RTX 2000 Ada Generation; log reports 16.00 GiB VRAM |
| Final checkpoint | `checkpoints\four_weather_resume_epoch17\latest.pt` |

Sources: checkpoint `config`/`model_config`; `scripts/train.py:137-225`; `airformer/data.py`; retained training logs. GPU identity also recorded by completed evaluation provenance. Training uses AMP; completed final evaluation uses **FP32 without autocast**.

## Recorded history

All 20 loss and learning-rate points were read from TensorBoard scalar events in the four retained training sessions. `training/epoch_history.csv` retains exact values, event wall timestamps and source directory. The loss plot is the actual logged mean of training batch losses, not a fitted curve. It may be nonmonotonic because of augmentation, balanced sampling and optimization.

- First recorded epoch loss: **0.864306629**.
- Final recorded epoch loss: **0.494056374**.
- Completed epoch: **20**.
- Logged final LR: **9.99999997475e-07**.
- Learning-rate values are logged **after** `scheduler.step()`. The first epoch uses initial 2e-4 during updates; its post-epoch logged rate is lower. No calculated schedule is presented as logged.

## Pauses, resume and actual time

| Retained run | Epochs retained | Start IST | Finish IST | Session wall seconds |
|---|---|---|---|---:|
| four_weather_20261104_fast | 1-3 | 2026-11-04T13:10:47.472094+05:30 | 2026-11-04T16:35:54.068804+05:30 | 12306.596710 |
| four_weather_20261105_resume | 4-10 | 2026-11-05T10:06:51.216907+05:30 | 2026-11-05T16:46:03.828664+05:30 | 23952.611757 |
| four_weather_resume_epoch11 | 11-16 | 2026-11-06T10:36:40.083155+05:30 | 2026-11-06T16:21:51.743638+05:30 | 20711.660483 |
| four_weather_resume_epoch17 | 17-20 | 2026-11-07T11:57:44.951584+05:30 | 2026-11-07T15:46:44.110053+05:30 | 13739.158469 |

Sum of recorded retained-session wall intervals: **70710.027419 s**, approximately 19h 38m 30s. Calendar span: **268556.637959 s**, 3d 2h 35m 57s. Session wall time includes startup, discarded partial epochs and the original failed final-test attempt. It excludes overnight pauses and abandoned earlier runs. **Pure training compute was not separately measured.** Source: final `duration_summary.json` cross-referencing session `status.json` files.

Resume chain: `four_weather_20261104_fast/epoch_003.pt` -> `four_weather_20261105_resume/epoch_010.pt` -> `four_weather_resume_epoch11/epoch_016.pt` -> `four_weather_resume_epoch17/latest.pt` (20). Resumption restores weights, optimizer, scheduler, scaler and RNG, not just the epoch label. Original checkpoint files remain untouched.

The last session's failure status describes a CUDA attention-kernel crash in testing **after** the epoch-20 save. The final log excerpt contains epoch 17-20 completion lines. The later separate full test completed. Do not describe the failed supervisor as failed training, and do not claim the model was retrained to fix evaluation.

Graphics: `training_loss_curve.png`, `learning_rate_curve.png`, `training_configuration.png`, `training_completion_evidence.png` (accurately labeled rendered excerpt), and `checkpoint_summary.png`.

## Dataset composition and preparation

| Weather | Local source | Train | Custom test | Total |
|---|---|---:|---:|---:|
| rain | Rain14000 | 10,080 | 2,520 | 12,600 |
| raindrop | Raindrop | 688 | 173 | 861 |
| snow | Snow100K local training archive | 39,998 | 10,002 | 50,000 |
| haze | Unofficial RESIDE-ITS mirror val | 8,000 | 2,000 | 10,000 |

Total 73,461 paired records, 58,766 train and 14,695 test. These count pairs, not unique underlying scenes or unique files. Sources: exact four-weather manifests and `preflight.json` (4 November).

Pairing uses explicit `input_path`/`target_path` mapping from source-specific preparation logic, not sorted filenames. `datasets/splitting.py::write_custom_manifests` groups source scene IDs, SHA-256 of decoded clean RGB pixels including dimensions, and SHA-256 of input file bytes via connected identity groups. Groups are shuffled by seed 42 within source buckets, with 80% of groups assigned to train. This can deviate slightly from exactly 80% of pairs. Known official tests and overlapping identities are excluded from the custom pool. The separate `official_test.csv` has 1,412 rain pairs and is not part of the final 14,695 evaluation.

Fresh historical preflight reports 127,746 unique referenced files, zero missing files, and zero scene overlap, plus one decoded pair per weather. The preparation/integrity records include earlier pixel-hash and decoding checks. This extraction re-counted all manifest rows, rechecked manifest hashes, checked selected sample identities and decoded the twelve chosen pairs. It did not repeat all-image decoding or rehash the full datasets.

Training loader `airformer/data.py` converts RGB to [0,1] tensors, checks paired sizes, pads small images, applies the same random 128x128 crop to each pair, paired horizontal/vertical flips, random quarter-turn rotations and MixUp p=0.5. The MixUp coefficient is uniform in the local code. Weather-balanced sampling uses inverse domain counts with replacement, so an epoch is not a guarantee that every distinct training image is visited exactly once. Test images retain full dimensions, with symmetric padding to 128 multiples and crop-back, without augmentation.

Provenance: owner/project links are catalogued in `datasets/sources.csv`. Local archive identity and preparation are in `docs/LOCAL_DATASET_AUDIT.json` and `scripts/prepare_local_archives.py`. Haze files match `dronefreak/RESIDE-ITS`, revision `39300fc7deb9138a59d5489f91a2f86d14d62eb6`, `val/clear` and `val/haze`: 11,000 files, no byte/size mismatch in `docs/RESIDE_MIRROR_VERIFICATION.json`. That confirms mirror identity, not original release provenance. It is not a canonical SOTS test. Exact hashes do not rule out near-duplicate scenes. Canonical AIR40K membership, CSD and official benchmark releases remain missing.

Genuine local pair paths and scene IDs are in `restoration_samples/sample_manifest.csv`. Dataset graphics are in `datasets/`, including `dataset_details.csv`. No external image or dataset was downloaded.


## Evidence hierarchy and identifiers

1. Saved epoch-20 checkpoint: `checkpoints\four_weather_resume_epoch17\latest.pt`. SHA-256 `523b2f26b123d97e6b139729f378172cb6e46b328c1b9d1d5a6250e69f76403f`.
2. Completed evaluation: `outputs/four_weather_test_epoch20_20261109/status.json`, `summary.json`, `provenance.json`, and `per_image_metrics.csv`.
3. Stored checkpoint `config` and `model_config`, checked against `configs/four_weather_fast.yaml`.
4. Executable source in `models/`, `losses/`, `airformer/`, `datasets/`, `scripts/`, `utils/`.
5. Retained training logs and TensorBoard events, followed by dated documentation. Historical diagnostic runs are separate.

Paths are relative to `D:\Multi_adverse_weather_image_enhancement` unless explicitly absolute. Local source hashes and modification timestamps are in `sources/source_inventory.csv`. Packaged snapshots are in `sources/`. This extraction generated documentation/assets and twelve inference examples only. It did not train, change weights, alter datasets, or repeat full evaluation.

