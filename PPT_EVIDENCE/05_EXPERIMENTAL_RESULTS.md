# Completed experimental results

Checkpoint: `checkpoints\four_weather_resume_epoch17\latest.pt`, saved epoch 20. Completed final job: 9 November 2026, 10:23:20-12:34:25 IST.

Sources: `outputs/four_weather_test_epoch20_20261109/summary.json`, `per_image_metrics.csv`, `status.json`, `provenance.json`, `duration_summary.json`. All group means for ten input/output metrics were recomputed from 14,695 CSV rows and matched to better than 1e-8.

| Group | N | Input PSNR | Output PSNR | Gain dB | Input SSIM | Output SSIM | Improved PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 14,695 | 20.672171 | 25.635622 | +4.963452 | 0.716301 | 0.805444 | 92.3988% |
| rain | 2,520 | 21.770011 | 26.493647 | +4.723635 | 0.685421 | 0.776417 | 97.5397% |
| raindrop | 173 | 20.289706 | 21.279203 | +0.989497 | 0.739357 | 0.743045 | 78.6127% |
| snow | 10,002 | 22.255925 | 26.394444 | +4.138518 | 0.741970 | 0.814045 | 89.8220% |
| haze | 2,000 | 11.401618 | 21.136477 | +9.734859 | 0.624843 | 0.804400 | 100.0000% |

## Error metrics

All are RGB float, range [0,1]. Each image has equal weight; these are means of per-image values.

| Group | Input MAE | Output MAE | Input MSE | Output MSE | Input RMSE | Output RMSE |
|---|---:|---:|---:|---:|---:|---:|
| overall | 0.083317875 | 0.041852197 | 0.020642788 | 0.004655871 | 0.115437557 | 0.058244781 |
| rain | 0.066287092 | 0.035567088 | 0.009139321 | 0.002751878 | 0.088709771 | 0.049755267 |
| raindrop | 0.065497076 | 0.068147929 | 0.011932619 | 0.009049213 | 0.103195470 | 0.090656109 |
| snow | 0.055138657 | 0.034576055 | 0.009942162 | 0.002894423 | 0.088645920 | 0.050757678 |
| haze | 0.247242428 | 0.083884835 | 0.089404422 | 0.015483879 | 0.284158486 | 0.103580993 |

Mean MAE reduction = 100 x (1 - mean restored MAE / mean input MAE) = **49.768046%**. It is not the mean of per-image percentage reductions. Mean per-image RMSE differs from sqrt(mean MSE), and the mean PSNR is not PSNR computed from the dataset mean MSE.

## Timing and protocol

- Mean recorded model call: **0.380816395 s/image**. Sum of calls: **5596.096920 s**.
- Test loop: **7807.230104 s**; complete job **7865 s**, 2h 11m 5s. Preflight **22.062997 s**.
- RTX 2000 Ada; PyTorch 2.13.0+cu130; FP32, batch one, native image sizes, symmetric multiple-of-128 padding, crop-back, no tiling.
- `_restore_tensor` synchronizes CUDA around the model call. Decoding, metric computation and work outside the timed call are excluded from inference time. It is not full application latency.
- PSNR/SSIM: skimage RGB floats, data_range=1, SSIM channel_axis=2, window at most 7. Source `utils/metrics.py`.

## Expected cross-checks

| Prompt expectation | Evidence | Resolution |
|---|---|---|
| 20 epochs | checkpoint epoch=20 and scalar history 1-20 | Matches |
| 14,695 pairs | completed status, CSV row count and manifest | Matches |
| PSNR about 25.6356 | 25.6356223575 | Matches rounding |
| SSIM about .805444 | 0.8054439391 | Matches rounding |
| Gain about +4.9635 | 4.9634516253 | Matches rounding |
| MAE reduction about 49.77% | 49.76804584% | Matches rounding |

No expected value was substituted for missing measurements.

## Interpretations and failure cases

All weather groups improve in average PSNR/SSIM. Haze has the greatest PSNR gain, though the input baseline is particularly poor. Raindrop is weakest and its mean MAE increases from .065497 to .068148; this must remain visible. 1,117 images have nonpositive PSNR gain. Snow contributes 68.06% of the overall image-weighted mean.

The twelve examples are selected by minimum, upper-median and maximum **PSNR gain** within each entire weather group. They are not selected by restored PSNR or aesthetics. Haze's worst-gain sample still improves because all haze samples have positive gain. Best/worst labels describe this recorded criterion, not a universal perceptual ranking.

Existing galleries remain unchanged in the original `outputs/four_weather_test_epoch20_20261109/visualizations/examples_*.png` paths. Large duplicate galleries are excluded from this package. Individual native predictions were not stored separately by the full-evaluation script, so the selected twelve held-out inputs were inferred once with the verified final checkpoint. Float PSNR/SSIM were recomputed before display quantization and agree with original rows within .005 dB and .0001 SSIM. Native aspect ratio is retained. Every identity and numerical delta is in `restoration_samples/sample_manifest.csv`.

No full-dataset evaluation was repeated. Auxiliary accuracy/F1/confusion matrices are correction/error-mask diagnostics, not weather classification or substitutes for PSNR/SSIM. They are not headline presentation results.

## Evidence hierarchy and identifiers

1. Saved epoch-20 checkpoint: `checkpoints\four_weather_resume_epoch17\latest.pt`. SHA-256 `523b2f26b123d97e6b139729f378172cb6e46b328c1b9d1d5a6250e69f76403f`.
2. Completed evaluation: `outputs/four_weather_test_epoch20_20261109/status.json`, `summary.json`, `provenance.json`, and `per_image_metrics.csv`.
3. Stored checkpoint `config` and `model_config`, checked against `configs/four_weather_fast.yaml`.
4. Executable source in `models/`, `losses/`, `airformer/`, `datasets/`, `scripts/`, `utils/`.
5. Retained training logs and TensorBoard events, followed by dated documentation. Historical diagnostic runs are separate.

Paths are relative to `D:\Multi_adverse_weather_image_enhancement` unless explicitly absolute. Local source hashes and modification timestamps are in `sources/source_inventory.csv`. Packaged snapshots are in `sources/`. This extraction generated documentation/assets and twelve inference examples only. It did not train, change weights, alter datasets, or repeat full evaluation.

