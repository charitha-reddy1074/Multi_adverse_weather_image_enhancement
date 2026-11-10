# Project audit for the research presentation

Audit date: 9 November 2026. Project: **AIRFormer-inspired all-in-one adverse-weather image restoration**.

## Research objective and problem

Train one supervised RGB-to-RGB restoration function for rain streaks, adherent raindrops, snow and haze. These degradations obscure scene structure and detail. One deployed model avoids selecting among four separate weather-specific models. This experiment measures paired restoration quality; it does not demonstrate downstream driving safety, detection accuracy or weather classification.

## Base paper

Tao Gao, Yuanbo Wen, Kaihao Zhang, Jing Zhang, Ting Chen, Lidong Liu, and Wenhan Luo. Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration. IEEE Transactions on Circuits and Systems for Video Technology, 34(3), March 2024. DOI: 10.1109/TCSVT.2023.3299324.

Primary evidence: supplied `Frequency-Oriented_Efficient_Transformer_for_All-in-One_Weather-Degraded_Image_Restoration (1).pdf`, first page, Figures 4/5, Eq.9-13, Section IV-B and Tables I-XIII. The MDPI paper suggested in the earlier layout is not the AIRFormer base paper. Figure provenance is retained in `sources/paper_figure_provenance.json`.

## Completed work

- Implemented a four-position pooling encoder, three recurrent Haar-guided supplementary priors, channel/space/wavelet queried decoder refinement, skip reconstruction and a residual RGB head. Sources: `models/airformer.py`, `encoder.py`, `supplementary_prior.py`, `attention.py`, `decoder.py`.
- Prepared and checked 58,766 training pairs and 14,695 custom held-out pairs from Rain14000, Raindrop, Snow100K and local haze. Sources: `data/manifests/four_weather/train.csv`, `test.csv`, `preflight.json`, `datasets/splitting.py`.
- Completed fixed training through epoch 20 across four retained sessions, 4-7 November. Exact stored configuration: base 16, stage depths [1,1,1,1], two heads, 256-token pooled K/V. Re-counted **612,435 trainable parameters** by strict loading of final checkpoint.
- Recovered interrupted sessions using saved model, optimizer, cosine scheduler, AMP scaler and random states. Sources: `scripts/train.py`, session training logs, checkpoint state keys.
- Completed independent full-resolution FP32 testing on 9 November, 10:23:20-12:34:25 IST. Sources: `outputs/four_weather_test_epoch20_20261109/status.json`, `provenance.json`, `duration_summary.json`.
- Produced per-image metrics and visual reports. Existing examples are preserved, with twelve additional package-only predictions matching recorded metrics within documented tolerance.

## Verified final outcomes

| Measure | Degraded input | Restored output |
|---|---:|---:|
| Mean PSNR (dB) | 20.672170732 | 25.635622358 |
| Mean SSIM | 0.716300933 | 0.805443939 |
| Mean MAE | 0.083317875 | 0.041852197 |
| Mean MSE | 0.020642788 | 0.004655871 |
| Mean per-image RMSE | 0.115437557 | 0.058244781 |

PSNR gain **+4.963451625 dB**. Mean MAE reduction **49.7680%**. Positive PSNR gains on **92.3988%** of images. Mean timed model call **0.380816395 s/image**, across variable image sizes on RTX 2000 Ada. Sources: final summary and reconciled per-image CSV (`sources/metric_reconciliation.csv`).

## Outdated claims resolved

- Earlier `docs/dataset_provenance.md` says no local images. Current manifests, file checks and 14,695 completed evaluation records supersede it.
- Older `docs/DATASET_STATUS.md` says no full research training result. Final checkpoint/events and completed evaluation supersede this historical status.
- `docs/architecture_assumptions.md` quotes 2,762,691 parameters for a larger configuration. It is not the 612,435-parameter completed run.
- Historical three-position pilot/audit parameter counts and diagnostic timings in `docs/OPTIMIZATION_LOG.md` and `AUDIT_BENCHMARK.json` are not final-model results.
- `checkpoints/four_weather_resume_epoch17/status.json` reports failure from post-training testing. The epoch-20 checkpoint and events prove training completion; `outputs/four_weather_test_epoch20_20261109/status.json` proves the later successful full test.
- `docs/MASTER_ARCHITECTURE_AND_MODEL_LIFECYCLE.md` and `docs/PAPER_COMPARISON.md`, dated 9 November, agree with final numeric artifacts. They are explanations, not substitutes for metric files.

## Remaining limitations and work

- Independent paper-guided reconstruction, not certified author-equivalent AIRFormer. Complete supplementary numerical settings and author implementation are unavailable in the repository evidence.
- Four encoder positions and three prior routes are verified, but widths/depths, additive learned queries, Haar choice, prior fusion and reconstruction tail are local interpretations.
- Trained spatial/wavelet attention pools K/V to 256 tokens, changing the attention graph. No matched ablation attributes quality gains to individual components.
- Custom grouped local split, not AIR40K or the paper's official benchmark tests. Snow dominates the test mean (10,002 of 14,695). No validation split or validation-based checkpoint selection.
- Haze bytes match a pinned unofficial RESIDE-ITS mirror `val` revision, not proof of canonical release attribution or official ITS/SOTS split identity. Near-duplicate leakage is not ruled out by exact hashes.
- Raindrop mean PSNR and SSIM rise slightly, but mean MAE increases. A good aggregate does not guarantee improvement in each sample or metric.
- Local FLOPs are unavailable. Variable-resolution local inference time cannot be compared as a speed ratio with paper 256x256 RTX 3090 timing.
- Unweighted orthonormal DCT L2 preserves error energy. It does not mathematically weight high frequencies more heavily.
- Evaluator source hash differs from final provenance after documented report-routing edits. Original evaluator bytes were not recovered. Checkpoint, both manifests, attention and inference hashes match; recorded metrics reconcile and selected predictions reproduce them.


Future work: obtain complete author specification and official memberships, use a documented validation protocol for a future experiment, run matched component ablations and baselines, address raindrop failure cases, test real-world generalization and profile the trained model under a fixed resolution/hardware protocol. These are proposals, not completed work.

## Evidence hierarchy and identifiers

1. Saved epoch-20 checkpoint: `checkpoints\four_weather_resume_epoch17\latest.pt`. SHA-256 `523b2f26b123d97e6b139729f378172cb6e46b328c1b9d1d5a6250e69f76403f`.
2. Completed evaluation: `outputs/four_weather_test_epoch20_20261109/status.json`, `summary.json`, `provenance.json`, and `per_image_metrics.csv`.
3. Stored checkpoint `config` and `model_config`, checked against `configs/four_weather_fast.yaml`.
4. Executable source in `models/`, `losses/`, `airformer/`, `datasets/`, `scripts/`, `utils/`.
5. Retained training logs and TensorBoard events, followed by dated documentation. Historical diagnostic runs are separate.

Paths are relative to `D:\Multi_adverse_weather_image_enhancement` unless explicitly absolute. Local source hashes and modification timestamps are in `sources/source_inventory.csv`. Packaged snapshots are in `sources/`. This extraction generated documentation/assets and twelve inference examples only. It did not train, change weights, alter datasets, or repeat full evaluation.

