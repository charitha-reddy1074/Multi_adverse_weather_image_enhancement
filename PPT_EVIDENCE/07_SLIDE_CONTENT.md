# Fifteen-slide research presentation content

Target: 12-15 minutes. Use one main visual per slide, white background, navy headings and teal highlights. Personal/guide/institute names and actual presentation date are not supplied. Add them only when confirmed. Sources and details belong in speaker notes, not dense slide bodies.

## 01. Title and Project Overview

**One key message:** One trained model restores four weather categories.

**Slide content:**

- AIRFormer-inspired multi-weather RGB restoration
- Independent implementation guided by Gao et al., IEEE TCSVT 2024
- Completed: 20 training epochs and 14,695 held-out tests
- Presenter / department / institute / guide: add confirmed details

**Verified values and sources:** Checkpoint epoch=20; outputs/four_weather_test_epoch20_20261109/status.json; supplied paper first page

**Recommended visual:** diagrams/model_lifecycle.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\diagrams\model_lifecycle.png`

**Speaker notes:** Introduce the work as an independent AIRFormer-inspired implementation. The completed work includes data preparation, architecture implementation, training, testing and evidence reporting. Personal and institutional details were not supplied, so they must be filled accurately. Use 9 November 2026 as evidence cutoff, not an invented presentation date.

**Possible guide questions:**

- What work is completed?
- Is this an exact AIRFormer reproduction?

## 02. Problem Statement and Motivation

**One key message:** One restoration function handles four distinct degradations.

**Slide content:**

- Rain, raindrops, snow and haze obscure image detail
- Learn from degraded/clean images of the same scene
- Use one RGB-to-RGB model without a weather-label input
- Evaluate reconstruction quality against clean targets

**Verified values and sources:** airformer/data.py; models/airformer.py::forward; supplied paper Introduction

**Recommended visual:** datasets/sample_image_pairs.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\datasets\sample_image_pairs.png`

**Speaker notes:** Each training pair shows a damaged image and its desired clean version. The network learns a correction from many such examples. Weather labels organize sampling and reports, but the model forward function receives only RGB images. Explain that downstream recognition gains were not measured.

**Possible guide questions:**

- Why one model instead of four?
- Do weather labels enter the model?

## 03. Base Research Paper

**One key message:** The paper provides the research concepts, not a fully resolved local specification.

**Slide content:**

- Gao et al.: Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration
- IEEE TCSVT, volume 34, issue 3, March 2024
- Frequency-guided encoder and Supplementary Prior
- Channel-, space- and wavelet-queried decoder

**Verified values and sources:** Frequency-Oriented_Efficient_Transformer_for_All-in-One_Weather-Degraded_Image_Restoration (1).pdf first page; DOI 10.1109/TCSVT.2023.3299324

**Recommended visual:** comparison/figure4_airformer.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\comparison\figure4_airformer.png`

**Speaker notes:** Tao Gao, Yuanbo Wen, Kaihao Zhang, Jing Zhang, Ting Chen, Lidong Liu, and Wenhan Luo. Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration. IEEE Transactions on Circuits and Systems for Video Technology, 34(3), March 2024. DOI: 10.1109/TCSVT.2023.3299324. Explain the three ideas using the authors’ figure. The author model, proposed in the paper, and the trained local model must remain distinct. Online DOI year 2023 in the identifier does not replace the March 2024 volume publication year.

**Possible guide questions:**

- What are the paper’s main contributions?
- Why focus on frequencies?

## 04. Original AIRFormer Architecture

**One key message:** Figure 4 connects a hierarchy, recurrent guidance and frequency refinement.

**Slide content:**

- Four encoder feature positions with increasing abstraction
- Three recurrent Supplementary Priors
- Frequency-refined decoder guided by learned task queries
- Hierarchical image reconstruction tail

**Verified values and sources:** Frequency-Oriented_Efficient_Transformer_for_All-in-One_Weather-Degraded_Image_Restoration (1).pdf Fig.4, printed page 1888; Fig.5 and Eq.10

**Recommended visual:** comparison/figure4_airformer.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\comparison\figure4_airformer.png`

**Speaker notes:** Walk through the unmodified paper diagram from input to output. The prior path supplements the encoder and the decoder refines information through different attention domains. Avoid supplying invented author channel counts. Figure 5 details attention and Table XII supports the preferred Channel, Space, Wavelet order.

**Possible guide questions:**

- What enters the first prior?
- Does the main paper specify every width and depth?

## 05. Implemented AIRFormer-Inspired Architecture

**One key message:** The actual trained graph is a compact four-position interpretation.

**Slide content:**

- Widths 16 / 32 / 64 / 128, one encoder block per position
- Three recurrent Haar-based priors before the next encoder block
- Three skip decoder levels with Channel -> Space -> Wavelet
- RGB correction + input residual + crop/clamp

**Verified values and sources:** Final checkpoint model_config; models/*.py; docs/architecture/trained_model_trace.json

**Recommended visual:** diagrams/actual_airformer_architecture.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\diagrams\actual_airformer_architecture.png`

**Speaker notes:** Read the dimensions for a 128x128 example. Spatial size halves and channels double through the encoder. SP0 uses zero inputs; later priors use the preceding enhanced feature and prior. Decoder stages upsample and fuse matching encoder skips. Show supplementary_prior.png or frequency_refined_decoder.png as optional supporting visuals.

**Possible guide questions:**

- Why does SP0 receive zeros?
- Which operations are assumptions?

## 06. Original vs Implemented Architecture

**One key message:** Matching concepts does not establish mathematical equivalence.

**Slide content:**

- Paper: 58.34M parameters; local checkpoint: 612,435
- Local widths, query composition, prior fusion and tail are chosen
- Spatial/wavelet K/V pooling uses a 256-token cap
- Elementwise Smooth L1 differs from literal Equation 12

**Verified values and sources:** 02_ARCHITECTURE_COMPARISON.md; supplied PDF Table XIII; losses/multidomain.py

**Recommended visual:** comparison/architecture_difference_summary.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\comparison\architecture_difference_summary.png`

**Speaker notes:** State the distinction clearly: the project reconstructs the method family, not certified author code. Four positions and recurrent routing can match while the detailed operations differ. Pooling the keys and values compresses what each query attends to. Smaller parameter count alone does not prove equivalent quality or faster inference.

**Possible guide questions:**

- Is the architecture identical?
- How does pooling change attention?

## 07. Proposed Methodology and Processing Flow

**One key message:** The experiment has a traceable path from local pairs to final metrics.

**Slide content:**

- Acquire local source pairs and verify scene identities
- Build grouped custom train/test manifests
- Train fixed 20 epochs and retain resumable checkpoints
- Evaluate final weights once over the held-out collection

**Verified values and sources:** datasets/splitting.py; airformer/data.py; scripts/train.py; scripts/test_restoration.py

**Recommended visual:** diagrams/model_lifecycle.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\diagrams\model_lifecycle.png`

**Speaker notes:** Follow the lifecycle diagram. Explain that the test set is held out from training and not used to choose an epoch. A failed original post-training test was repaired through the inference backend path, then a separate complete evaluation produced the final records. This evidence extraction did not repeat the full evaluation.

**Possible guide questions:**

- How do you prevent leakage?
- Was test loss used to select weights?

## 08. Dataset Collection and Preparation

**One key message:** 73,461 local pairs support a custom grouped four-weather experiment.

**Slide content:**

- 58,766 training pairs and 14,695 held-out pairs
- Rain14000, Raindrop, Snow100K and local haze mirror
- Scene IDs, identical clean pixels and input bytes grouped
- Haze provenance and official AIR40K membership remain limited

**Verified values and sources:** datasets/dataset_details.csv; data/manifests/four_weather/preflight.json; docs/RESIDE_MIRROR_VERIFICATION.json

**Recommended visual:** datasets/dataset_distribution.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\datasets\dataset_distribution.png`

**Speaker notes:** Give the weather counts from dataset_details.csv: train/test rain 10080/2520, raindrop 688/173, snow 39998/10002, haze 8000/2000. These are pair counts. The split is roughly 80/20 of identity groups, not an official benchmark split. Use sample_image_pairs.png to show genuine local pairs. Balanced sampling affects training visitation, not test weights.

**Possible guide questions:**

- Why is snow dominant?
- Is the haze set SOTS?

## 09. Implementation and Technology Stack

**One key message:** Each research concept maps to an inspectable Python module.

**Slide content:**

- PyTorch modules for encoder, priors, queried attention and decoder
- Differentiable Haar DWT/IDWT and orthonormal DCT-II
- AdamW, CUDA AMP and resumable state checkpoints
- skimage RGB PSNR/SSIM plus per-image error and timing records

**Verified values and sources:** 03_ARCHITECTURE_TO_CODE.md; models/; losses/multidomain.py; utils/metrics.py

**Recommended visual:** code_screenshots/supplementary_prior.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\code_screenshots\supplementary_prior.png`

**Speaker notes:** Open the source mapping rather than claiming module names prove equivalence. The screenshot is real source with line numbers. DWT is used for local multi-scale guidance and attention. DCT is used by the loss. The public airformer model/loss files re-export the canonical implementation, so they are not a second independent architecture.

**Possible guide questions:**

- Where is attention implemented?
- How are transforms differentiable?

## 10. Training Configuration and Progress

**One key message:** The saved compact model completed the planned twenty epochs.

**Slide content:**

- 612,435 parameters; 128x128 crops; batch 8; accumulation 1
- AdamW, LR 2e-4 -> 1e-6 cosine schedule, CUDA AMP
- Logged loss: 0.8643 at epoch 1 -> 0.4941 at epoch 20
- RTX 2000 Ada; resumed four sessions; no validation selection

**Verified values and sources:** 04_TRAINING_CONFIGURATION.md; training/epoch_history.csv; final checkpoint

**Recommended visual:** training/training_loss_curve.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\training\training_loss_curve.png`

**Speaker notes:** Training history is extracted from real TensorBoard scalar events, not estimated from progress bars. The recorded session sum is about 19h38m30s, including overhead and failed post-training testing, so do not label it pure GPU training. Use training_completion_evidence.png if asked why the supervisor reports failure after training completed.

**Possible guide questions:**

- Why 128x128 instead of paper 256x256?
- How did resume preserve optimization state?

## 11. Loss Function and Evaluation Metrics

**One key message:** The implementation optimizes a spatial penalty and a coefficient-difference norm.

**Slide content:**

- Mean elementwise Smooth L1 + 0.04 DCT consistency
- DCT norm: per image over C/H/W, epsilon 0.001
- PSNR and SSIM higher is better; MAE/RMSE lower is better
- Literal Equation 12 reduction differs; unweighted DCT has no high-frequency preference

**Verified values and sources:** losses/multidomain.py; utils/metrics.py; supplied PDF Eq.11-13

**Recommended visual:** loss/loss_equations.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\loss\loss_equations.png`

**Speaker notes:** Explain Smooth L1 as quadratic near zero and linear for larger errors. On this model’s clamped [0,1] RGB outputs, errors stay within the unit threshold. The frequency branch uses a smoothed L2 norm after an orthonormal DCT. Energy preservation means this unweighted norm is equivalent to spatial L2, so do not claim selective high-frequency weighting. Opposite signed errors illustrate the Eq.12 reduction difference.

**Possible guide questions:**

- Why not claim exact Equation 12?
- Does this DCT norm emphasize high frequencies?

## 12. Quantitative Experimental Results

**One key message:** Restoration improves the same held-out image baseline overall.

**Slide content:**

- 14,695 full-resolution test pairs; FP32 evaluation
- PSNR 20.6722 -> 25.6356 dB, gain +4.9635 dB
- SSIM 0.716301 -> 0.805444; mean MAE reduction 49.77%
- 92.40% improve in PSNR; raindrop MAE worsens

**Verified values and sources:** outputs/four_weather_test_epoch20_20261109/summary.json; per_image_metrics.csv; sources/metric_reconciliation.csv

**Recommended visual:** results/final_metrics_table.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\results\final_metrics_table.png`

**Speaker notes:** This is a valid before/after comparison because each output is compared to the same clean scene as its degraded input. The mean is image-weighted and snow has most samples. Raindrop gains are weaker and not all metrics improve. Use performance_dashboard.png for timing details and improvement_distribution.png to show failures.

**Possible guide questions:**

- What fraction improves?
- Why can PSNR improve while mean MAE worsens?

## 13. Qualitative Restoration Results

**One key message:** Best, median and weak cases make successes and failures visible.

**Slide content:**

- Same scene: degraded input / model output / clean target
- Twelve cases: best, median and worst PSNR gain per weather
- Labels use recorded per-image PSNR and SSIM
- All haze cases gain PSNR; worst means weakest gain

**Verified values and sources:** restoration_samples/sample_manifest.csv; final per_image_metrics.csv; verified checkpoint

**Recommended visual:** restoration_samples/four_weather_overview.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\restoration_samples\four_weather_overview.png`

**Speaker notes:** The overview uses the median-gain scene for each weather. For the live presentation, choose two readable weather rows or use separate case graphics rather than compressing all twelve. Discuss a worst case from rain, raindrop or snow. Predictions were reproduced for only twelve existing held-out samples because separate output files were missing, and their float metrics were checked against the final CSV. No synthetic image is shown.

**Possible guide questions:**

- Were cases cherry-picked?
- Is the displayed output from the final checkpoint?

## 14. Paper Comparison and Limitations

**One key message:** Published scores provide context across different protocols.

**Slide content:**

- Contextual comparison only: different datasets and evaluation protocols
- Paper uses AIR40K / official tests and 256x256 training crops
- Local compact model uses a custom split and 128x128 crops
- No matched FLOPs, speed advantage or component-ablation claim

**Verified values and sources:** Supplied PDF Tables XI/XIII, Sec.IV-B; 02_ARCHITECTURE_COMPARISON.md; final provenance

**Recommended visual:** comparison/paper_vs_our_results.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\comparison\paper_vs_our_results.png`

**Speaker notes:** Read the contextual comparison label aloud. The paper reports 58.34M parameters, 5.746G FLOPs and .086 seconds at 256x256 on RTX3090. Our model has fewer parameters, but timing uses a different GPU and variable full-resolution images. Do not compute a quality deficit or a speed ratio and call it a controlled benchmark.

**Possible guide questions:**

- Why are paper PSNR/SSIM not directly comparable?
- What would a fair benchmark require?

## 15. Conclusion and Future Scope

**One key message:** A completed, traceable implementation is the demonstrated contribution.

**Slide content:**

- Built and trained one four-weather restoration pipeline
- Completed fixed-epoch training and full custom held-out testing
- Verified average restoration gain with visible failure cases
- Next: official protocol, matched ablations, stronger raindrop restoration

**Verified values and sources:** Final checkpoint; evaluation artifacts; 06_ENGINEERING_CHALLENGES.md

**Recommended visual:** results/performance_dashboard.png

**Exact asset path:** `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\results\performance_dashboard.png`

**Speaker notes:** Conclude with the work actually delivered: implementation, verified data pipeline, training recovery and completed evaluation. Do not claim a new state-of-the-art architecture. Future changes need controlled experiments: author-specified settings, official datasets, ablations, real-world tests and fixed-condition profiling. None of those experiments is presented as completed.

**Possible guide questions:**

- What is your contribution?
- What is the next controlled experiment?

