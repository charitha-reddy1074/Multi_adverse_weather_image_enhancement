# AIRFormer reproduction decisions

The uploaded AIRFormer paper is the primary source. The original implementation
and pretrained weights are unavailable in this workspace. Accordingly, this
repository is an independent, paper-guided implementation and does not claim
bitwise or parameter-level equivalence with the authors' code.

## Evidence categories

### A — stated in the AIRFormer paper

- A single all-in-one restoration model covers rain streaks, raindrops, snow,
  and haze.
- Figure 4 (paper page 3) confirms the top-level flow: patch embeddings feed a
  hierarchical frequency-guided encoder; Supplementary Prior guides encoder
  learning; the frequency-refined decoder uses encoder and decoder features for
  reconstruction.
- The encoder receives wavelet-based prior information; the decoder uses
  learnable task-specific queries across spatial, channel, and wavelet domains.
- AIR40K combines examples from multiple existing weather datasets.
- Training values currently encoded from the available project notes are:
  256-pixel crops, AdamW (0.9/0.999, weight decay 1e-4), initial LR 2e-4,
  cosine decay to 1e-7, flips/90-degree rotations, MixUp probability 0.5,
  and frequency-loss weight 0.04. Epoch count and batch size need confirmation
  against the paper PDF and remain configurable.
- Evaluation pads to a multiple of 128 and reports PSNR/SSIM.

### B — standard cited operations

- PoolFormer token mixing uses `AvgPool2d(3, stride=1, padding=1) - X` with
  residual addition, followed by normalized pointwise/depthwise MLP mixing.
- The wavelet transform here is a one-level orthonormal Haar DWT. The four
  subbands use the conventional 2x2 Haar basis.
- PSNR and SSIM are computed with `skimage.metrics` on RGB images with data
  range 1.0.

### C — unresolved implementation choices

- Base feature width, block counts, attention head count, exact normalization,
  query tensor shapes/initialization, wavelet boundary handling, and feature
  fusion details are not established by the available source files. The paper
  configuration's 32-channel width and two blocks per level are explicit
  engineering defaults, not paper specifications. The pilot shrinks these.
- The checked-in `.dist/paper.txt` extraction supplies the Figure 4 caption and
  surrounding architecture text. The PDF figure pixels themselves are not
  available for detailed visual inspection in this execution environment, so
  Figure 4 supports the top-level mapping, not unprinted block internals.
- DWT uses replicate padding on odd dimensions, then crops the inverse output
  back to the original shape. The fixed Haar basis is an engineering choice;
  it is not claimed as the paper's exact wavelet family.
- The first SP stage receives zero encoder and prior inputs as stated around
  Equation 10. Later stages receive their corresponding encoder features and
  the previous SP feature; the resulting guidance is added to encoder features.
- Task-biased queries are learned additive embeddings by default. `query_combine`
  makes this combination configurable; the exact projected query operation is
  unresolved without Figure 5 detail.
- Spatial/wavelet attention keeps full-resolution queries but adaptively pools
  keys and values above `max_spatial_tokens`. This is a documented memory-saving
  approximation. Channel, space, and wavelet attention remains ordered as stated.
- The residual RGB skip is enabled as an engineering adaptation; it is not
  claimed as a confirmed paper choice. Set `residual_image: false` to ablate it.
- DCT loss is differentiable and uses the specified Smooth L1 + 0.04 frequency
  term. Since reduction is not fully specified, it computes the per-image
  Frobenius norm across RGB and all 2D coefficients, adds epsilon 0.001 inside
  the square root, and averages across batch. This is an explicit reduction
  decision, not an unverified alternative transform.
- AIR40K's exact sample manifest and split recipe are not available in the
  checked-in files. The loader accepts normalized paired folders; users must
  preserve official splits and dataset terms when assembling them.

Do not use this reproduction to claim the authors' checkpoint results without
training and evaluation on the corresponding official splits.
