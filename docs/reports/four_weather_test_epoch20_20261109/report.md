# Epoch-20 restoration test results

Completed: 2026-11-09T12:34:25+05:30

Evaluated **14,695 paired test images** at their original resolution using FP32 inference.
This is the custom four-weather split; haze uses the pinned unofficial ITS mirror and is not an official SOTS benchmark.

| Group | Images | Input PSNR | Restored PSNR | PSNR gain | Input SSIM | Restored SSIM | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 14695 | 20.672 | 25.636 | +4.963 | 0.7163 | 0.8054 | 0.04185 | 0.05824 |
| haze | 2000 | 11.402 | 21.136 | +9.735 | 0.6248 | 0.8044 | 0.08388 | 0.10358 |
| rain | 2520 | 21.770 | 26.494 | +4.724 | 0.6854 | 0.7764 | 0.03557 | 0.04976 |
| raindrop | 173 | 20.290 | 21.279 | +0.989 | 0.7394 | 0.7430 | 0.06815 | 0.09066 |
| snow | 10002 | 22.256 | 26.394 | +4.139 | 0.7420 | 0.8140 | 0.03458 | 0.05076 |

PSNR and SSIM are computed on clipped float RGB [0,1], full image, no border crop. SSIM uses the existing skimage RGB protocol (7-pixel window where possible).
Each image has equal weight in restoration means; weather macro means are not substituted for image means. RMSE is averaged per image. Timing excludes disk loading, CPU metrics, and report creation.

## Auxiliary pixel-error/correction masks

These are not weather classification scores. True positive class: mean absolute RGB(input − clean target) > threshold. Predicted positive class: mean absolute RGB(input − restored output) > threshold. Thus the scores describe where the model applied corrections to corrupted pixels, not whether restoration was successful. PSNR/SSIM measure restoration quality.
Scores pool pixel counts across images; large images weigh more. Confusion matrix rows are true [low input error, high input error], columns predicted [no large correction, large correction]. Zero denominators are reported as undefined.

| Fixed threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0.05 | 0.7973 | 0.7285 | 0.8243 | 0.7735 |
| 0.10 | 0.8752 | 0.7946 | 0.6729 | 0.7287 |
| 0.20 | 0.9430 | 0.8260 | 0.6203 | 0.7085 |

Threshold 0.10 is the primary diagnostic; 0.05 and 0.20 are fixed sensitivity checks, not tuned on the test set.

The trained model and training schedule were fixed before testing. No model selection uses test scores.

![quality by weather](../../../outputs/four_weather_test_epoch20_20261109/visualizations/quality_by_weather.png)

![improvement distribution](../../../outputs/four_weather_test_epoch20_20261109/visualizations/improvement_distribution.png)

![auxiliary confusion matrix](../../../outputs/four_weather_test_epoch20_20261109/visualizations/auxiliary_confusion_matrix.png)

![examples haze](../../../outputs/four_weather_test_epoch20_20261109/visualizations/examples_haze.png)

![examples rain](../../../outputs/four_weather_test_epoch20_20261109/visualizations/examples_rain.png)

![examples raindrop](../../../outputs/four_weather_test_epoch20_20261109/visualizations/examples_raindrop.png)

![examples snow](../../../outputs/four_weather_test_epoch20_20261109/visualizations/examples_snow.png)
