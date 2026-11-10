# Adverse-Weather Image Restoration: Improvement Roadmap

Prepared: 10 November 2026

This roadmap covers methodology, mathematical formulations, architecture,
datasets, training, testing, and estimated effort for improving the current
AIRFormer-inspired model. It separates practical quality improvements from
steps needed to reproduce the reference paper more faithfully.

## 1. Recommended direction

Prioritize **data diversity, larger image context, and validation-controlled
training** before substantially increasing model size. Increasing parameters or
epochs alone is unlikely to fix the weak restoration on unfamiliar web photos.

Maintain two experimental tracks:

- **Quality track:** improve real-world restoration using measured changes to
  data, loss, model capacity, and degradation conditioning.
- **Paper-fidelity track:** match published operations and settings, document
  unresolved specifications, and use the correct benchmark protocol.

Changes such as dynamic prompts and new losses are extensions, not evidence of
an exact AIRFormer reproduction.

## 2. Current model versus the paper

| Item | Trained local model | Reference AIRFormer |
|---|---|---|
| Parameters | 612,435, approximately 0.612M | 58.34M |
| Training crops | 128 x 128 | 256 x 256 |
| Encoder stage depths | [1, 1, 1, 1] | Exact numerical author specification unresolved |
| Spatial/wavelet attention | Pooled K/V, 256-token cap | Paper's depicted attention graph |
| Training pairs | 58,766 | 40,561 AIR40K samples |
| Training duration | 20 epochs | Epoch count not established from supplied main paper |
| Validation | None in completed custom experiment | New development run should have independent validation |
| Haze coverage | Pinned ITS mirror, indoor source | Indoor and outdoor sources |
| Testing | 14,695 custom held-out pairs | Different benchmark collections |

The completed custom test result is **25.6356 dB PSNR / 0.805444 SSIM**.
These scores are not directly comparable to paper scores because the data,
model, and aggregation differ.

The current 20 epochs involved approximately **146,920 optimizer updates**.
Do not assume that the model needs more training solely because 20 sounds small.
Use validation curves to distinguish underfitting, overfitting, and domain mismatch.

See [paper comparison](PAPER_COMPARISON.md),
[results](RESULTS_README.md), and
[architecture audit](architecture/README.md).

## 3. Dataset improvements

### 3.1 Current training distribution

| Weather | Training pairs | Main improvement priority |
|---|---:|---|
| Rain | 10,080 | Real rain; varied streak scales, lighting, cameras |
| Raindrops | 688 | New scenes, droplet shapes, glass/lens conditions |
| Snow | 39,998 | Diversity and severity rather than more similar images |
| Haze | 8,000 | Outdoor haze, varied depth, density, and atmospheric light |
| Total | 58,766 | Coverage and quality before raw count |

Weather-balanced sampling is already enabled. It produces approximately
14,692 draws per weather per epoch. The 688 raindrop pairs are consequently
drawn about **21 times each per epoch on average**, before augmentation.
Repeated sampling does not replace new independent scenes.

### 3.2 Recommended additions

- Add verified **RESIDE OTS** training content for outdoor haze. ITS is the
  Indoor Training Set; the indoor-to-outdoor mismatch is a plausible contributor
  to weak city-photo restoration.
- Add real paired rain examples, such as SPA-Data, to complement synthetic rain.
- Expand raindrop scene diversity using verified paired releases or physically
  realistic synthesis, rather than duplicating the current small collection.
- Include real paired haze collections, such as O-HAZE, while respecting their
  official train/test partitions. Real paired collections can be small but useful.
- Include clean-to-clean examples to measure and train preservation of already
  good images. Treat their sampling proportion as a validation-tuned experiment.
- Introduce mixed conditions, such as rain plus haze, after establishing strong
  results on individual weather conditions.

A practical planning target is **70,000-100,000 verified, diverse training pairs**.
This is a budget target, not a minimum scientifically required for success.
The existing dataset already contains more pairs than AIR40K; diversity matters.

### 3.3 Data quality and partitioning

- Verify pixel alignment, dimensions, RGB conversion, and clean-target quality.
- Detect duplicate and near-duplicate scenes across datasets and releases.
- Keep all weather variants of the same clean scene in the same partition.
- Reserve approximately **10% of current training scenes for validation**.
  Exact pair counts depend on scene grouping.
- Keep the current **14,695-image test partition unchanged** for historical
  comparison. This is a new development protocol, replacing the previous
  no-validation setup for future experiments.
- Keep an additional external evaluation set untouched during development.
- Do not tune on selected favorable demonstration images.

## 4. Methodology and degradation concepts

### 4.1 Haze: depth-dependent degradation

The atmospheric scattering model is:

$$
I(x)=J(x)t(x)+A[1-t(x)],\qquad t(x)=e^{-\beta d(x)}.
$$

Here, I is the hazy image, J is the clean scene, A is atmospheric light,
t is transmission, beta is scattering strength, and d is scene depth.

Implications:

- Train across different transmission levels and atmospheric colors.
- Include outdoor scenes with substantial depth variation.
- Use larger crops to provide surrounding scene context.
- A physics-guided auxiliary branch could predict transmission or atmospheric
  light, but this would be a new extension requiring suitable supervision and
  an ablation; it is not necessary to reproduce AIRFormer.

### 4.2 Rain, snow, and adherent raindrops

A simplified rain-streak model is:

$$
I=J+R.
$$

Raindrops and snow can additionally occlude or refract background information.
They cannot always be treated as additive layers. Surrounding context and
region-sensitive restoration are therefore useful, but fully hidden details
cannot be recovered with certainty from a single image.

Recommended experiments include larger crops, diverse severity levels,
degradation-region attention, and preservation checks on unaffected regions.

### 4.3 Inference context

Use full-image inference when memory permits. The recent city-image test showed
larger corrections with full-image processing than with small tiles, although
substantial haze remained. Larger pixel changes alone do not prove better quality.

Retain tiling as a memory option. Evaluate its impact on quality and boundaries
separately from full-image evaluation.

## 5. Loss formulations and required checks

### 5.1 Current loss

The current implementation approximately computes:

$$
L=L_{\mathrm{SmoothL1}}+
0.04\sqrt{\|\operatorname{DCT}(\hat J)-\operatorname{DCT}(J)\|_F^2+\epsilon^2}.
$$

The implementation uses an orthonormal DCT. For that transform:

$$
\|\operatorname{DCT}(\hat J-J)\|_F=\|\hat J-J\|_F.
$$

Consequently, the unweighted DCT term is equivalent to a spatial Euclidean
error norm. It does **not** specifically emphasize high-frequency components
or selected frequency bands. This identity was checked numerically locally.

The DCT term also uses an unnormalized norm per image. At comparable per-pixel
error, increasing crop dimensions from 128 to 256 approximately doubles that
term, whereas mean Smooth L1 remains comparable. Recheck loss weighting when
changing crop size.

### 5.2 Quality-track alternatives

Start with normalized L1 or Charbonnier reconstruction loss:

$$
L_{\mathrm{char}}=\frac{1}{N}\sum_i
\sqrt{(\hat J_i-J_i)^2+\epsilon^2}.
$$

Then test structural, gradient, or explicitly band-weighted terms separately:

$$
L=L_{\mathrm{char}}+
\lambda_s[1-\operatorname{SSIM}(\hat J,J)]+
\lambda_g L_{\mathrm{gradient}}+
\lambda_w L_{\mathrm{wavelet}}.
$$

For example:

$$
L_{\mathrm{gradient}}=
\operatorname{mean}|\nabla\hat J-\nabla J|,
$$

$$
L_{\mathrm{wavelet}}=
\sum_{s,b}w_{s,b}\operatorname{mean}
|W_{s,b}(\hat J)-W_{s,b}(J)|.
$$

Here, s indexes scale, b indexes subband, and W extracts the corresponding
wavelet coefficients. Explicit weights make frequency emphasis controllable.
The additional loss weights require validation; no universal values are assumed.

Run the following sequence:

1. Preserve the current loss as the baseline.
2. Compare normalized L1 or Charbonnier.
3. Add one structural or frequency term at a time.
4. Inspect component magnitudes, gradient contributions, artifacts, and per-weather metrics.
5. Combine terms only when individual improvements are supported.

### 5.3 Paper-fidelity caveat

Retain a separate published-loss configuration. Equation 12 in the supplied
paper averages signed residuals before applying its piecewise penalty, while
the current implementation averages elementwise penalties. Cancellation makes
these different. Obtain author clarification rather than silently labeling
either interpretation exact.

### 5.4 Training output clamp

The current model clamps output to [0,1] inside its forward pass. Test an
unclamped training output, applying clipping only for evaluation/export.
Clamping can suppress gradients outside the interval. This is a proposed
controlled experiment, not a proven improvement or confirmed paper behavior.

## 6. Bringing the architecture closer to AIRFormer

| Component | Required action |
|---|---|
| Encoder | Preserve four positions and the pooling-based block family |
| Supplementary Prior | Preserve recurrence and pre-block injection; resolve filters, fusion, and alignment |
| Decoder ordering | Preserve Channel -> Space -> Wavelet |
| Wavelet branch | Preserve full-grid queries, reduced-grid K/V, and separate IDWT branch |
| Attention approximation | Disable K/V pooling for the strict reproduction configuration |
| Task queries | Obtain exact tensor shapes, initialization, and feature-combination rules |
| Reconstruction tail | Resolve upsampling, skip fusion, RGB projection, and image residual details |
| Widths and depths | Obtain author settings rather than guessing from parameter count |

The standard attention operation is:

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

K/V pooling changes the information supplied to this operation. Exact query
chunking can reduce temporary memory while preserving all keys, but full spatial
attention still has quadratic token-comparison cost.

Several graph corrections are already present; follow the architecture audit
instead of rebuilding corrected components.

The repository's larger `configs/paper.yaml` model has approximately **2.76M
parameters**, still far below the reported **58.34M**. Matching parameter count
alone would not establish architecture equivalence.

The paper-linked author repository returned 404 during the 10 November 2026
check. Obtain the supplement, original code/configuration, or author clarification
for the remaining specifications. No author contact has been sent.

Architecture changes that alter tensor shapes generally require fresh training
or an explicitly documented partial weight transfer. Do not call partial loading
an exact checkpoint resume.

## 7. Optional extensions for a stronger independent model

- **Input-conditioned prompts:** adapt restoration features to degradation type
  and severity. PromptIR is a relevant reference.
- **Histogram-based attention:** group similarly degraded pixels across spatial
  distances. Histoformer is a relevant adverse-weather reference.
- **Gradual capacity increase:** compare the current small model with one wider
  or deeper configuration under the same dataset and training budget.
- **Progressive crop training:** test a lower-cost 128-pixel stage followed by
  256-pixel fine-tuning with an explicit learning-rate schedule.

These are independent research changes. Add them after establishing a sound
data/training baseline, and test them separately from paper-fidelity claims.

## 8. Proposed training settings

These are starting settings for experiments, not unpublished author settings.

| Parameter | Proposed starting point |
|---|---|
| Crop size | 256 x 256 after memory/throughput pilot |
| Physical batch | 2-4 if memory permits |
| Effective batch | 8-16 through accumulation |
| Optimizer | AdamW |
| Fresh-training learning rate | 2e-4 |
| Fine-tuning learning rate | Approximately 2e-5 to 5e-5 |
| Weight decay | 1e-4 |
| Adam betas | 0.9, 0.999 |
| Schedule | Explicit cosine schedule; minimum 1e-7 for paper-oriented run |
| Precision | Mixed precision with overflow monitoring |
| Gradient clipping | Retain 1.0 initially |
| Initial continuation budget | 20-30 additional epochs, extending only with validation evidence |
| Sampling | Weather-balanced, with monitored scene repetition |
| Checkpoint selection | Equal-weather mean validation PSNR |
| Reproducibility | Repeat finalist with 2-3 seeds if budget allows |

Gradient accumulation increases effective batch size; it does not supply the
spatial context of a larger crop.

The current trainer restores scheduler state during resume. Simply changing
`epochs: 20` to `100` is not a sufficient continuation strategy. Explicitly
initialize a fine-tuning optimizer/schedule or correctly extend the resumed
schedule. The old scheduler state can override the intended new horizon.

Keep paired augmentations synchronized. Compare MixUp against a no-MixUp
baseline if it appears to reduce realistic weather coverage; do not assume it
always improves restoration.

## 9. Testing protocol

- Original-resolution images, batch one, full-image inference where feasible.
- Symmetric padding to multiples of 128, then crop to original dimensions.
- Consistent RGB PSNR/SSIM settings across versions; report MAE/RMSE as useful.
- Per-weather metrics and an equal-weather aggregate, alongside the existing
  image-weighted aggregate, which is heavily influenced by snow.
- Best, median, and failure examples, including clean-image preservation.
- A fixed external real-world set that is not used for checkpoint selection.
- Official benchmark partitions and matching metric conventions for direct
  comparison with published methods.

The downloaded web photos have no paired clean references. They support visual
assessment and discussion of limitations, not reference-based PSNR/SSIM claims.
Input-output pixel difference measures change, not restoration accuracy.

## 10. Approximate compute time

Recorded local hardware: **NVIDIA RTX 2000 Ada Generation, approximately 16 GiB VRAM**.

Measured historical durations:

- 20-epoch training: **19 h 38 min 30 s** summed retained-session wall time.
- Full 14,695-pair test job: **2 h 11 min 5 s**.

The training session sum includes startup, discarded partial work, and the
original failed final-test attempt. It excludes overnight pauses and is not a
measurement of pure GPU training compute.

### 10.1 Provisional planning estimates

| Experiment | Approximate compute budget |
|---|---|
| Current small model, same settings, 20-30 additional epochs | 20-30 hours plus validation |
| Current small model, same settings, 100 epochs from scratch | Approximately 4-5 continuous days |
| Small model, 256 crops, 50 epochs | Roughly 4-12 days, assuming 2-6x current epoch time |
| Larger 2.76M model, pooled attention, 256 crops, 50 epochs | Provisionally 2-6 weeks; pilot required |
| Strict unpooled attention or faithful 58M model | No reliable forecast from existing measurements |

These estimates assume broadly similar data volume and hardware availability.
Validation, increased data, image decoding, model changes, and memory limits can
substantially change them. They are not measured forecasts or accuracy guarantees.

One historical strict batch-one 256-pixel diagnostic training step took about
**100 seconds**. It is not a steady-state throughput benchmark. At that rate,
10,000 steps would take about **11.6 days**, illustrating why strict attention
must be profiled before allocating a long run.

### 10.2 Obtain a reliable estimate

Benchmark 200-500 warmed-up training microbatches under the exact proposed
configuration, including backward computation and accumulation:

$$
T\approx E\left\lceil\frac{N_{\mathrm{train}}}{B_{\mathrm{physical}}}\right\rceil
t_{\mathrm{microbatch}}+T_{\mathrm{validation}}+T_{\mathrm{overhead}}.
$$

Use synchronized CUDA timings. Measure peak memory and exclude initial warmup
when reporting steady-state throughput. A larger GPU does not automatically
resolve inefficient attention kernels; verify kernel compatibility and numerical
agreement before relying on optimized attention.

For the same current model and test suite, budget roughly another 2-3 hours for
final evaluation. Larger models and strict attention require a new test benchmark.

## 11. Execution order and development effort

| Stage | Work | Approximate human development effort |
|---|---|---|
| 1 | Preserve baseline; create scene-disjoint validation; audit data | 1-3 working days |
| 2 | Add verified outdoor haze and raindrop diversity | Several days; acquisition may take longer |
| 3 | Loss normalization, clamp experiment, and 256-crop pilot | 1-3 working days plus compute |
| 4 | Train and evaluate a practical candidate | Compute budgets above; inspect validation during training |
| 5 | Capacity or degradation-conditioning ablations | Several days plus separate training runs |
| 6 | Strict AIRFormer reconstruction | Open-ended until missing specifications are obtained |

Allow approximately **1-2 working weeks** for initial auditing, protocol changes,
implementation, and a practical pilot, excluding long training runs or delayed
dataset access. Do not promise a paper-equivalent model within that time.

Recommended order:

1. Preserve the current checkpoint and metrics as baseline.
2. Establish independent validation and audit scene overlap.
3. Improve outdoor haze and raindrop coverage.
4. Compare normalized losses and 256-pixel fine-tuning.
5. Increase capacity only after those results justify it.
6. Run matched ablations and repeat the best configuration when affordable.
7. Evaluate once on untouched tests and report gains and remaining failures.
8. Maintain a separate strict AIRFormer track with explicit unresolved details.

## 12. References and local evidence

### Local evidence

- [Paper comparison](PAPER_COMPARISON.md)
- [Architecture audit](architecture/README.md)
- [Architecture system validation](architecture/system_validation.json)
- [Completed results](RESULTS_README.md)
- [Trained configuration](../configs/four_weather_fast.yaml)
- [Paper-oriented configuration](../configs/paper.yaml)
- [Loss implementation](../losses/multidomain.py)
- [Training implementation](../scripts/train.py)
- [Model implementation](../models/airformer.py)
- [Recorded duration calculations](../outputs/four_weather_test_epoch20_20261109/duration_summary.json)

### Primary external sources

- [AIRFormer paper DOI](https://doi.org/10.1109/TCSVT.2023.3299324)
- [Paper-linked author repository](https://github.com/chdwyb/AIRFormer)
- [RESIDE dataset: ITS, OTS, and testing collections](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)
- [O-HAZE dataset](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/)
- [O-HAZE paper and scattering formulation](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.pdf)
- [Spatial Attentive Single-Image Deraining With a High Quality Real Rain Dataset](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Spatial_Attentive_Single-Image_Deraining_With_a_High_Quality_Real_Rain_CVPR_2019_paper.html)
- [Attentive Generative Adversarial Network for Raindrop Removal From a Single Image](https://openaccess.thecvf.com/content_cvpr_2018/html/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.html)
- [PromptIR: Prompting for All-in-One Blind Image Restoration](https://arxiv.org/abs/2306.13090)
- [Histoformer: Restoring Images in Adverse Weather Conditions via Histogram Transformer](https://arxiv.org/abs/2407.10172)

External references and repository availability were checked on 10 November 2026.
Proposed settings and time ranges are planning recommendations; improvements
must be demonstrated experimentally rather than assumed.
