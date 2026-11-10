# Research guide questions and answers

Answers refer to the saved compact experiment, not the larger default model.

## 1. What is AIRFormer?

The paper’s frequency-oriented all-in-one restoration Transformer for rain, raindrop, snow and haze. This project is an independent smaller implementation inspired by its encoder, prior and decoder concepts.

Evidence: `Supplied paper first page; models/airformer.py`.

## 2. Why use one model for four weather conditions?

A shared restoration function avoids selecting and storing four separate weather-specific models. The local forward function takes RGB only. This experiment confirms custom-split average restoration gains, not superiority over four separately trained experts.

Evidence: `models/airformer.py; final summary.json`.

## 3. What is a Transformer here?

Attention learns how features influence one another using queries, keys and values, with residual and MLP processing. Our encoder uses pooling-style blocks; decoder refinement performs actual queried attention in channel, spatial and wavelet modes.

Evidence: `models/encoder.py; models/attention.py`.

## 4. Why use PoolFormer-style processing?

Local pooling mixes nearby features using inexpensive non-attention operations. The code uses GroupNorm and AvgPool(z)-z inside a residual, followed by a depthwise/channel MLP. We did not isolate its contribution in a matched ablation.

Evidence: `models/encoder.py::FrequencyGuidedBlock`.

## 5. Why is the encoder hierarchical?

Each reduction halves spatial size and increases channels, moving from fine details toward broader context. The trained widths are 16,32,64,128 at 128,64,32,16 spatial sizes for a 128 crop. Decoder skips reuse fine features.

Evidence: `trained_model_trace.json; models/encoder.py`.

## 6. What is a Supplementary Prior?

A learned guidance branch. This implementation decomposes features into Haar bands, filters details, combines bands with current and previous features, refines, and projects to the next encoder scale. It is a documented approximation of the paper branch.

Evidence: `models/supplementary_prior.py`.

## 7. What is DWT?

A discrete wavelet transform separates each local 2x2 group into one coarse LL coefficient and three directional detail coefficients, while reducing each band’s spatial size. Our transform is fixed and differentiable.

Evidence: `models/supplementary_prior.py::dwt2`.

## 8. Why use Haar wavelets?

Haar has simple orthonormal sums/differences and an explicit inverse, so it is easy to implement and audit without additional dependencies. It is our engineering choice; the available main paper does not uniquely establish Haar.

Evidence: `models/supplementary_prior.py; architecture_assumptions.md`.

## 9. What is the difference between DWT and DCT?

Haar DWT separates local coarse/detail bands and is used in priors and wavelet attention. The orthonormal 2D DCT uses cosine bases over the image dimensions and is used in the training consistency loss.

Evidence: `models/supplementary_prior.py; losses/multidomain.py`.

## 10. How does channel attention work?

After normalization and Q/K/V projections, each head treats channels as the attended elements and spatial positions as embeddings. The affinity matrix is (C/2) by (C/2) for two heads. The result is projected and added through a residual.

Evidence: `models/attention.py::QueriedAttention.forward`.

## 11. How does spatial attention work?

Each pixel feature query compares with spatial keys and gathers weighted values. Full-grid queries are retained, but our trained model pools large K/V grids to 16x16. This changes attention versus the full HW-by-HW paper graph.

Evidence: `models/attention.py; final checkpoint model_config`.

## 12. How does wavelet attention work?

The normalized full-grid features form queries. DWT bands are projected from 4C to C at reduced resolution to form K/V, optionally pooled. A separate projected-band IDWT yields C/4 channels, concatenated with attended C features before projection.

Evidence: `models/attention.py::QueriedAttention.forward`.

## 13. What are Q, K and V?

Q describes what an output position or channel seeks, K supplies descriptors to compare against, and V supplies content to combine. A scaled QK product followed by softmax produces weights used to mix V. Axes differ between the attention modes.

Evidence: `models/attention.py`.

## 14. What is task-queried attention?

The paper uses learned task-biased vectors to guide attention. Our implementation adds a learned broadcast parameter to a feature-derived query. These internal parameters are not explicit weather labels; this composition remains an assumption.

Evidence: `models/attention.py task_query; models/airformer.py::forward`.

## 15. How does the decoder reconstruct the image?

Starting at 128 channels/16x16, it upsamples to encoder stage 2, concatenates that skip and projects to 64 channels. It repeats at 32 and 16 channels, with Channel/Space/Wavelet refinement each time, then predicts an RGB correction added to the input.

Evidence: `models/decoder.py; models/airformer.py`.

## 16. Why use Smooth L1?

It is quadratic near zero and linear for large residuals, giving a smooth small-error penalty. Our code averages elementwise penalties. Because outputs and targets are [0,1], these RGB errors lie within the unit quadratic region, apart from the boundary.

Evidence: `losses/multidomain.py; models/airformer.py clamp`.

## 17. Why use frequency consistency loss?

The paper motivates consistency after DCT. Our implementation computes a smoothed per-image coefficient-difference L2 norm with weight .04. Since the transform is orthonormal and unweighted, its norm equals spatial L2 energy and does not preferentially emphasize high frequencies.

Evidence: `losses/multidomain.py::dct2 and MultiDomainConsistencyLoss`.

## 18. What is backpropagation?

Autograd computes derivatives of the scalar loss through convolutions, attention and fixed differentiable transforms. GradScaler scales loss for mixed precision, backward accumulates gradients, and AdamW updates parameters. No backward update is used in evaluation.

Evidence: `scripts/train.py training step; utils/inference.py inference_mode`.

## 19. Why use AdamW?

It uses adaptive gradient moments and decoupled weight decay. Our settings are LR 2e-4 initially, betas .9/.999 and weight decay 1e-4, with a cosine schedule to 1e-6. These are verified saved settings, not evidence it is the best optimizer for this task.

Evidence: `checkpoint config; scripts/train.py`.

## 20. What is PSNR?

For [0,1] images, PSNR is 10 log10(1/MSE), in decibels. Higher means smaller squared error for that image. The report averages per-image PSNR values; it is not computed from the overall mean MSE.

Evidence: `utils/metrics.py; scripts/test_restoration.py`.

## 21. What is SSIM?

It compares local luminance, contrast and structural patterns. Our evaluation uses skimage on RGB floats with data_range=1, channel_axis=2 and a window up to seven pixels. Higher generally indicates closer local structure, but is not a complete perceptual judgment.

Evidence: `utils/metrics.py`.

## 22. Why is the implemented model smaller?

The trained configuration deliberately uses base width 16 and one block per encoder position. It has 612,435 parameters, compared with the paper’s reported 58.34M. Prior/query/tail details also differ. We do not claim equal capacity or a measured speed advantage.

Evidence: `checkpoint verification; supplied paper Table XIII`.

## 23. Is the architecture identical to the paper?

No. Broad hierarchy, recurrent guidance and three attention domains follow the paper, but SP fusion, additive feature queries, K/V pooling, reconstruction choices, widths/depths and literal Smooth L1 reduction differ or remain unspecified.

Evidence: `02_ARCHITECTURE_COMPARISON.md; canonical source`.

## 24. Why are the paper’s results not directly comparable?

The paper uses AIR40K and official benchmark collections; our results use a different custom split, architecture, crop resolution and image weighting. Their timing also uses RTX3090 and 256x256 images. Scores are contextual, not a controlled leaderboard.

Evidence: `supplied PDF Sec.IV-B and Tables XI/XIII; final provenance`.

## 25. What is the research contribution?

The demonstrated contribution is an independent operational implementation and a traceable four-weather experiment: dataset identity checks, complete training with resume, final held-out evaluation and transparent limits. A new state-of-the-art method or proven component novelty is not established.

Evidence: `final checkpoint; manifests; final evaluation artifacts`.

## 26. What improvements are planned?

Obtain author specifications and official benchmark memberships, establish validation for a future run, perform matched ablations/baselines, improve weak raindrop cases and test real-world generalization. Fixed-condition FLOPs/latency profiling is also needed. These tasks were not run here.

Evidence: `01_PROJECT_AUDIT.md remaining work`.

## 27. Why does the final training status say failed?

Twenty epochs were saved successfully, then the original post-training CUDA full-size test hit a fused attention kernel error. A channel-attention MATH backend path was used for the later separate evaluation, which completed all 14,695 pairs.

Evidence: `retained training.log; models/attention.py; final status.json`.

## 28. How was train/test leakage checked?

Grouping joins the same source scene, identical decoded clean RGB content and identical input bytes, then assigns entire groups to one custom split. Known official-test identities are excluded. Exact grouping does not prove absence of near-duplicate scenes.

Evidence: `datasets/splitting.py; preflight.json`.

## 29. Is the haze dataset official RESIDE SOTS?

No. The local haze content matches a pinned unofficial RESIDE-ITS mirror val directory. Byte identity to that mirror is verified, but original release and benchmark split attribution are not. The 8000/2000 split is a custom experiment.

Evidence: `docs/RESIDE_MIRROR_VERIFICATION.json`.

## 30. How were epochs and learning rates verified?

The final checkpoint says epoch 20; retained TensorBoard scalar events provide losses and post-scheduler learning rates for every epoch from 1 through 20. Resumption loads scheduler/optimizer state.

Evidence: `training/epoch_history.csv; checkpoint state; scripts/train.py`.

## 31. Does a 49.77% MAE reduction apply to every weather?

No. It is the ratio of overall image-mean MAEs before and after restoration. Raindrop MAE actually rises. Report per-weather tables and weak examples to avoid hiding this limitation.

Evidence: `results/verified_metrics.csv; final summary.json`.

## 32. Why can average PSNR improve while mean MAE worsens?

PSNR depends on squared errors and a logarithm per image; MAE depends on absolute errors. Their sample averaging also differs. Reducing large errors can improve PSNR even if many small errors increase. This observation alone does not identify the causal mechanism.

Evidence: `per-image metrics; loss/metric definitions`.

## 33. How were restoration cases selected?

Within each weather, sort all held-out rows by PSNR gain. Use the minimum, the upper middle row and the maximum. The final checkpoint inferred those same twelve pairs; float metrics matched recorded rows within .005 dB and .0001 SSIM.

Evidence: `restoration_samples/sample_manifest.csv`.

## 34. Is the reported inference time deployment latency?

No. It is the mean synchronized timed model call, .380816 seconds across variable full-resolution test images. Decoding, transfer/padding outside the timer, metrics and report generation are excluded. No comparison with paper timing is controlled.

Evidence: `utils/inference.py::_restore_tensor; duration_summary.json`.

## 35. Does every training image appear once each epoch?

No. The inverse-weather-count weighted sampler uses replacement. An epoch contains the configured number of draws, so rare domains are oversampled and some individual records may not appear in that epoch.

Evidence: `scripts/train.py sampler setup; airformer/data.py sample_weights`.

## 36. Does the prior use IDWT?

The supplementary prior filters Haar details and resizes packed coefficients; it does not apply IDWT. IDWT is used in the decoder wavelet-attention branch. Mixing these branches would misdescribe the actual model.

Evidence: `models/supplementary_prior.py::SupplementaryPrior.forward; models/attention.py`.

## 37. What is the literal Equation 12 difference?

The printed paper first averages the signed spatial residual and then applies a piecewise penalty. Code first applies the piecewise penalty to each element and then averages. Errors +.5 and -.5 cancel to zero in the first policy but give .125 in the implemented policy.

Evidence: `supplied PDF Eq.12; losses/multidomain.py`.

