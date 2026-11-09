# Reference AIRFormer paper compared with this project

Updated 9 November 2026. Reference: Tao Gao et al., *Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration*, IEEE TCSVT 34(3), 2024, [DOI](https://doi.org/10.1109/TCSVT.2023.3299324). The supplied main-paper PDF was checked directly, including Tables I-V, XI-XIII and Section IV-B.

**This project completed a smaller, independent AIRFormer-inspired implementation and a custom four-weather experiment.** It does not establish exact author-code reproduction or a like-for-like benchmark against the paper. The completed local result is **25.6356 dB PSNR / 0.805444 SSIM** on **14,695 custom held-out pairs**.

## What the authors did and what this project did

| Area | Reference paper | This completed project |
|---|---|---|
| Goal | One model for deraining, raindrop removal, desnowing, and dehazing | One trained RGB restoration model covering the same four weather types |
| Training dataset | AIR40K, 40,561 training samples in Table I | 58,766 pairs from four usable local sources; custom scene-disjoint 80/20 split |
| Test collection | Table I lists 57,629 samples across benchmark test collections; some subsets overlap, including Raindrop Test-A/Test-B | 14,695 custom held-out pairs; not the paper's benchmark collection |
| Rain data | Multiple sources: Rain14000, Rain800, Rain100H/L, Rain1200, Rain12; six test benchmarks | Rain14000 local archive: 10,080 train / 2,520 test |
| Raindrop data | Raindrop: 861 train; Test-A and Test-B evaluated | Split the local training archive into 688 train / 173 test; not Test-A/Test-B |
| Snow data | Selected Snow100K training samples plus CSD; evaluated SnowL/M/S and Snow2000 | Local Snow100K training archive: 39,998 train / 10,002 custom test; no CSD benchmark |
| Haze data | ITS/OTS training content and HazeIn/HazeOut/Haze10 tests | Pinned unofficial ITS mirror `val` content: 8,000 train / 2,000 custom test; not SOTS |
| Architecture | Frequency-guided hierarchical encoder, Supplementary Prior, and frequency-refined queried-attention decoder | Four-position pooling encoder, recurrent Haar prior interpretation, Channel -> Space -> Wavelet decoder, chosen skip-fusion/reconstruction operations |
| Author-equivalence | Authors' method | Independent implementation; numerical widths, task-query composition, SP internals, tail, and literal loss reduction retain documented assumptions |
| Trainable parameters | 58.34 million, Table XIII | **612,435** in the actually trained checkpoint; the repository's larger paper-oriented configuration was not used for this run |
| Spatial/wavelet attention | Paper's depicted attention graph | Resource approximation: K/V pooling enabled, 256-token cap; full-resolution queries retained |
| Training crops | 256 x 256, Section IV-B | 128 x 128 |
| Epochs and batch size | Not specified in the supplied main paper; supplementary material is referenced | Fixed 20 epochs; batch size 8, accumulation 1; no validation-based checkpoint selection |
| Optimizer | AdamW, beta values 0.9/0.999, weight decay 0.0001 | Same optimizer, beta values, and weight decay |
| Learning rate | Initial 0.0002; cosine annealing to 0.0000001 | Initial 0.0002; configured cosine minimum 0.000001 |
| Data augmentation | Horizontal/vertical flips, 90-degree rotations, MixUp probability 0.5 | These augmentation types and MixUp probability 0.5; weather-balanced training sampling also enabled |
| Training precision | Not established from the supplied main paper | CUDA mixed precision enabled |
| Loss | Spatial/frequency consistency; frequency weight 0.04 | Smooth L1 elementwise mean plus DCT-II per-image frequency norm, weight 0.04; printed Equation 12 reduction differs and remains an explicit assumption |
| Hardware | NVIDIA GeForce RTX 3090, Section IV-B | NVIDIA RTX 2000 Ada Generation; approximately 16 GiB VRAM reported locally |
| Evaluation padding | Symmetric padding to multiples of 128, crop back afterward | Same size policy; full original-size FP32 evaluation, batch one, no tiling |
| Evaluation measures | skimage PSNR/SSIM; other experiments include NIQE/SSEQ | skimage RGB PSNR/SSIM, MAE/MSE/RMSE, baseline gains, loss/timing; extra explicitly labeled mask diagnostics |
| Additional experiments | Component/order/loss ablations, benchmark baselines, single-weather and real-world evaluations | Ablation switches exist, but no completed matched multi-run ablation or competitor retraining is claimed for this experiment |

## Published scores and local scores

The following numbers are **context, not a controlled leaderboard**. The paper's weather summaries average benchmark datasets; this project's summaries average individual images within a different custom split. Dataset membership, model size, training schedule, and aggregation differ. No cross-paper percentage improvement or numerical deficit is inferred.

| Weather | Paper AIRFormer PSNR / SSIM | Paper evidence | This project PSNR / SSIM | Local evaluation |
|---|---|---|---|---|
| Rain | 30.25 / 0.949 | Table II average; Table XI final row | 26.494 / 0.7764 | 2,520 custom Rain14000 pairs |
| Raindrop | 28.26 / 0.911 | Table III average; Table XI final row | 21.279 / 0.7430 | 173 held-out local training-archive pairs |
| Haze | 32.33 / 0.989 | Table IV average; Table XI final row | 21.136 / 0.8044 | 2,000 custom ITS-mirror pairs |
| Snow | 31.75 / 0.954 | Table XI final row, benchmark-weather average | 26.394 / 0.8140 | 10,002 custom Snow100K pairs |
| Overall local image-weighted result | No equivalent single overall value used here | Different test composition | **25.636 / 0.8054** | **14,695 custom pairs** |

Table III's printed heading says dehazing, but its columns are RainDA/RainDB and Section IV-C.2 discusses it as raindrop removal. The raindrop interpretation above follows the columns and surrounding text. We use all-in-one results rather than mixing in the separately trained single-weather results from Tables VI-IX.

The valid measured improvement here compares the **same held-out images before and after restoration**: overall PSNR rose from 20.6722 to 25.6356 dB (**+4.9635 dB**), and SSIM rose from 0.716301 to 0.805444 (**+0.089143**). Mean MAE decreased by **49.77%**. This supports improvement over the degraded-input baseline on this split, not superiority over the published AIRFormer.

## Duration and efficiency comparison

| Measurement | Reference paper | This project | Interpretation |
|---|---|---|---|
| Total training duration | Not reported in the supplied main paper | **19 h 38 min 30 s** summed wall time for the four retained training sessions | Includes startup, discarded partial epochs, and the original failed final-test attempt; excludes overnight pauses and earlier abandoned runs |
| Pure training compute | Not established | Not separately measured | Do not present the session sum as pure GPU compute |
| Training calendar span | Not established | **3 days 2 h 35 min 57 s**, 4-7 November 2026 | Includes user-requested overnight pauses |
| Total full-test job duration | Not reported for the entire benchmark collection | **2 h 11 min 5 s**, 9 November 2026, 10:23:20-12:34:25 IST | Includes final-job preflight and report generation |
| Full test image/metric loop | Not established | **2 h 10 min 7 s** for 14,695 images | Includes per-image processing, metrics, and durable progress writes |
| Model inference time | **0.086 s/image**, Table XIII, 256 x 256 images on RTX 3090 | **0.380816 s/image** averaged across variable full-resolution test images on RTX 2000 Ada | Different hardware, image sizes, implementation, and timing protocol; no speed ratio is claimed |
| Summed model inference intervals | Not reported for the whole test collection | **1 h 33 min 16 s** | Excludes decoding, CPU metrics, report generation, and transfers/padding outside the timed model call |
| FLOPs | **5.746 billion**, Table XIII, 256 x 256 | Not measured for this trained checkpoint in the completed evaluation | No unsupported FLOPs or speed advantage claimed |
| Parameter count | **58.34 million**, Table XIII | **0.612435 million** | Different architectures; smaller size does not establish equivalent quality or speed |

Training sessions, rounded to the nearest second:

| Date (IST) | Epochs retained | Wall time | Status evidence |
|---|---|---|---|
| 4 November | 1-3 | 3 h 25 min 7 s | [Session status](../checkpoints/four_weather_20261104_fast/status.json) |
| 5 November | 4-10 | 6 h 39 min 13 s | [Session status](../checkpoints/four_weather_20261105_resume/status.json) |
| 6 November | 11-16 | 5 h 45 min 12 s | [Session status](../checkpoints/four_weather_resume_epoch11/status.json) |
| 7 November | 17-20 | 3 h 48 min 59 s | [Session status](../checkpoints/four_weather_resume_epoch17/status.json) |

The last training supervisor says `failed` because testing crashed after epoch 20 was saved. The saved final model was verified, the CUDA attention issue was repaired, and the separate [full evaluation completed successfully](../outputs/four_weather_test_epoch20_20261109/status.json). No training was repeated to recover these results.

## What was achieved and what remains

This project delivered a functioning four-weather restoration pipeline, trained final weights, scene-disjoint custom evaluation, resumable testing, exact metric/provenance files, and visual reports. Across the held-out images, restoration improved average PSNR/SSIM in every weather group, and 92.40% of images gained PSNR. Raindrop removal remains the weakest improvement area.

The requested extra accuracy (**87.52%**), precision (**79.46%**), recall (**67.29%**), and F1 (**72.87%**) at threshold 0.10 describe auxiliary pixel-error/correction masks. The paper does not report this diagnostic, so those values have no paper classification counterpart. See [definitions and confusion matrix](RESULTS_README.md#accuracy-precision-recall-f1-and-confusion-matrix).

A controlled paper comparison would require matching architecture and supplementary specifications, AIR40K membership and official tests, training settings, metric details, and hardware/timing protocol. Controlled ablations and competitor runs would also be needed to attribute gains to individual components.

## Sources and verification

- Supplied [publisher PDF](<../Frequency-Oriented_Efficient_Transformer_for_All-in-One_Weather-Degraded_Image_Restoration (1).pdf>), Table I (PDF page 5 / printed 1890), Tables II-III (page 7 / 1892), Tables IV-V and Section IV-B (page 8 / 1893), Tables XI-XIII (page 10 / 1895). [DOI](https://doi.org/10.1109/TCSVT.2023.3299324).
- Locally rendered paper pages checked visually: [page 5](../outputs/paper_review/comparison_pages/page_5.png), [page 7](../outputs/paper_review/comparison_pages/page_7.png), [page 8](../outputs/paper_review/comparison_pages/page_8.png), [page 10](../outputs/paper_review/comparison_pages/page_10.png).
- Local [final summary](../outputs/four_weather_test_epoch20_20261109/summary.json), [duration calculations](../outputs/four_weather_test_epoch20_20261109/duration_summary.json), [training config](../configs/four_weather_fast.yaml), [manifest](../data/manifests/four_weather/test.csv), [architecture assumptions](architecture/README.md), and [activity log](LOG.md).
- [Results README and visualization links](RESULTS_README.md) and [full browser report](../outputs/four_weather_test_epoch20_20261109/report.html).
