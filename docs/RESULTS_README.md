# Four-weather restoration: completed results

Updated 9 November 2026. Training completed **20/20 epochs**, and testing completed on **all 14,695 held-out image pairs**. The final model achieved **25.6356 dB PSNR and 0.805444 SSIM**, improving mean PSNR by **4.9635 dB** over the degraded inputs.

This is an independent AIRFormer-inspired experiment on a custom split. It is not an exact reproduction of the authors' architecture or their AIR40K benchmark. See [paper comparison](PAPER_COMPARISON.md) for the differences.

## Final restoration results

Higher PSNR and SSIM are better. Lower MAE and RMSE are better; errors below use RGB values in [0,1].

| Weather | Test pairs | Input PSNR (dB) | Restored PSNR (dB) | PSNR gain (dB) | Input SSIM | Restored SSIM | Restored MAE | Restored RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rain | 2,520 | 21.770 | 26.494 | +4.724 | 0.6854 | 0.7764 | 0.03557 | 0.04976 |
| Raindrop | 173 | 20.290 | 21.279 | +0.989 | 0.7394 | 0.7430 | 0.06815 | 0.09066 |
| Snow | 10,002 | 22.256 | 26.394 | +4.139 | 0.7420 | 0.8140 | 0.03458 | 0.05076 |
| Haze | 2,000 | 11.402 | 21.136 | +9.735 | 0.6248 | 0.8044 | 0.08388 | 0.10358 |
| **Overall** | **14,695** | **20.672** | **25.636** | **+4.963** | **0.7163** | **0.8054** | **0.04185** | **0.05824** |

Overall MAE fell from 0.08332 to 0.04185, a **49.77% reduction**. Mean per-image RMSE fell from 0.11544 to 0.05824, a **49.54% reduction**. PSNR improved on **92.40% of test images**; some images became worse, so the galleries include worst, median, and best gains.

The largest mean PSNR gain was on haze. Raindrop removal had the smallest gain and only a small SSIM improvement, making it a useful target for further work. Snow accounts for approximately 68% of test images, so the overall mean is strongly influenced by snow performance.

![Quality before and after restoration](../outputs/four_weather_test_epoch20_20261109/visualizations/quality_by_weather.png)

![Distribution of improvements](../outputs/four_weather_test_epoch20_20261109/visualizations/improvement_distribution.png)

## Accuracy, precision, recall, F1, and confusion matrix

The model outputs restored RGB images, not predicted weather classes. These requested scores therefore describe an **auxiliary pixel-error/correction-mask diagnostic**, not weather classification or the percentage of successfully restored images.

- True positive class: a pixel's mean absolute RGB difference between degraded input and clean target exceeds the threshold.
- Predicted positive class: a pixel's mean absolute RGB difference between degraded input and restored output exceeds the same threshold.
- Counts are pooled across pixels, so larger images have more weight. These masks assess where corrections were applied; PSNR/SSIM assess restoration quality.

| Fixed threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0.05 | 79.73% | 72.85% | 82.43% | 77.35% |
| **0.10 (primary)** | **87.52%** | **79.46%** | **67.29%** | **72.87%** |
| 0.20 | 94.30% | 82.60% | 62.03% | 70.85% |

Thresholds were fixed before evaluation, not selected for the best test result. The 87.52% value must not be described as weather-classification accuracy.

Primary confusion matrix, threshold 0.10:

| True input-error mask | Predicted: no large correction | Predicted: large correction |
|---|---:|---:|
| Low input error | TN = 2,740,703,119 | FP = 167,746,083 |
| High input error | FN = 315,579,901 | TP = 649,102,404 |

![Auxiliary confusion matrix](../outputs/four_weather_test_epoch20_20261109/visualizations/auxiliary_confusion_matrix.png)

## Data and trained model

| Weather | Local source | Training pairs | Testing pairs |
|---|---|---:|---:|
| Rain | Rain14000 local archive | 10,080 | 2,520 |
| Raindrop | DeRaindrop training archive | 688 | 173 |
| Snow | Snow100K training archive | 39,998 | 10,002 |
| Haze | Pinned unofficial RESIDE ITS mirror, `val` content | 8,000 | 2,000 |
| **Total** | Custom scene-disjoint 80/20 split | **58,766** | **14,695** |

There was no validation set and no test-based checkpoint selection. The fixed final epoch was evaluated. This haze split is not official RESIDE SOTS; it must not be labeled as a SOTS result. See [dataset provenance](TRAINING_README.md#1-know-which-data-the-run-uses).

- **Checkpoint:** [latest.pt](../checkpoints/four_weather_resume_epoch17/latest.pt), saved epoch 20.
- **Trainable parameters:** 612,435, measured from the checkpoint's architecture configuration. The larger paper-oriented repository configuration has a different count.
- **Training:** 20 epochs, batch size 8, 128 x 128 crops, mixed precision, weather-balanced sampling; AdamW, learning rate 0.0002, cosine schedule with configured minimum 0.000001.
- **Architecture settings:** base width 16; four stage depths [1,1,1,1]; two attention heads; pooling encoder; Channel -> Space -> Wavelet refinement; spatial/wavelet K/V pooling enabled, cap 256 tokens.
- **Hardware:** NVIDIA RTX 2000 Ada Generation, approximately 16 GiB VRAM as reported by PyTorch.
- **Testing:** original-size images, FP32, batch size one, no tiling; symmetric padding to multiples of 128, then crop back. RGB skimage PSNR/SSIM, no border crop; SSIM window seven where possible.

Restoration scores are arithmetic means over images. Overall scores weight each image equally, rather than averaging the four weather means. RMSE is averaged per image. Auxiliary mask scores use pooled pixel counts instead.

Checkpoint SHA-256: `523b2f26b123d97e6b139729f378172cb6e46b328c1b9d1d5a6250e69f76403f`.

## Measured durations

All timestamps below use India Standard Time (UTC+05:30). Durations are rounded to the nearest second.

| Retained training session | Completed epochs retained | Start | End | Session wall time |
|---|---|---|---|---|
| 4 November 2026 | 1-3 | 13:10:47 | 16:35:54 | 3 h 25 min 7 s |
| 5 November 2026 | 4-10 | 10:06:51 | 16:46:03 | 6 h 39 min 13 s |
| 6 November 2026 | 11-16 | 10:36:40 | 16:21:51 | 5 h 45 min 12 s |
| 7 November 2026 | 17-20 | 11:57:44 | 15:46:44 | 3 h 48 min 59 s |
| **Sum of session wall times** | **1-20** | | | **19 h 38 min 30 s** |

The session sum includes startup/data checks, discarded partial epochs after pauses, and the original failed final-test attempt. It excludes overnight pauses and earlier abandoned runs. It is **not a measurement of pure training compute**. Total time is computed before rounding individual rows.

Training calendar span, including overnight pauses: **3 days 2 h 35 min 57 s**, from the first retained run's start to the final run's exit. Training loss decreased from **0.864307 at epoch 1** to **0.494056 at epoch 20**; training loss alone does not establish generalization.

| Final testing measurement | Duration / timing |
|---|---|
| Start | 9 November 2026, 10:23:20 IST |
| Completed, including report generation | 9 November 2026, 12:34:25 IST |
| **Total testing job wall time** | **2 h 11 min 5 s** |
| Full image/metric evaluation loop | 2 h 10 min 7 s |
| Preflight inside this final job | 22.06 s |
| Sum of model inference timings | 1 h 33 min 16 s |
| Mean model inference timing | 0.380816 s/image, variable original resolutions |

Inference timing covers the synchronized model call. It excludes padding, image decoding, input/output transfers outside that call, CPU metrics, and report generation. The total job includes startup, preflight, metrics, and final visualizations. It does not include earlier debugging or separate preflight/regression runs. The earlier 80-minute estimate was superseded by these actual measurements.

![Training loss over completed epochs](../outputs/four_weather_test_epoch20_20261109/visualizations/training_loss.png)

## Error recovery and verification

The 7 November supervisor's `failed` state describes the combined training/testing job. Epoch 20 had already been saved when final testing hit `cutlassF: no kernel found to launch!`. CUDA channel attention was switched to PyTorch's math SDPA backend, keeping the trained weights and attention scale.

The repaired evaluator decoded all 14,695 pairs, checked dimensions and manifest separation, and passed the largest padded shape for each weather. Thirteen targeted regression/evaluator tests passed. Full evaluation then completed with an empty error log. No retraining was required. Historical status files remain unchanged; the [new evaluation status](../outputs/four_weather_test_epoch20_20261109/status.json) records successful completion.

## Reports, examples, and exact evidence

- [Interactive browser report](../outputs/four_weather_test_epoch20_20261109/report.html), [detailed result report](reports/four_weather_test_epoch20_20261109/report.md), and [full summary/provenance](../outputs/four_weather_test_epoch20_20261109/summary.json).
- [Per-image metrics](../outputs/four_weather_test_epoch20_20261109/per_image_metrics.csv), [duration calculations](../outputs/four_weather_test_epoch20_20261109/duration_summary.json), and [preflight evidence](../outputs/four_weather_test_epoch20_20261109/preflight.json).
- Worst/median/best examples and error maps: [rain](../outputs/four_weather_test_epoch20_20261109/visualizations/examples_rain.png), [raindrop](../outputs/four_weather_test_epoch20_20261109/visualizations/examples_raindrop.png), [snow](../outputs/four_weather_test_epoch20_20261109/visualizations/examples_snow.png), [haze](../outputs/four_weather_test_epoch20_20261109/visualizations/examples_haze.png). Each chart also has an SVG version in the same folder.
- [Paper comparison](PAPER_COMPARISON.md), [architecture assumptions](architecture/README.md), [training configuration](../configs/four_weather_fast.yaml), and [activity log](LOG.md).

To run a separate repeat evaluation without retraining, use a new output directory. The completed output preserves the original results, durations, and evaluator-code signature. An interrupted repeat can be continued by adding `--resume` with the same checkpoint, data, and code.

```powershell
.venv/Scripts/python -B -u scripts/test_restoration.py --checkpoint checkpoints/four_weather_resume_epoch17/latest.pt --output outputs/four_weather_test_epoch20_repeat
```
