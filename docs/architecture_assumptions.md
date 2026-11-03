# AIRFormer implementation assumptions

Updated 2026-11-02. **Paper-guided implementation with explicit assumptions.**
The current graph and evidence are documented in
[the architecture review](architecture/README.md). Earlier three-position
architecture findings are superseded by the corrected implementation.

## Confirmed graph corrections

- Four encoder positions and three recurrent supplementary priors, injected
  after the subsequent embedding and before its block. The first prior receives
  zero features and a zero previous prior; later priors receive preceding
  enhanced features and the previous prior.
- Channel -> Space -> Wavelet decoder order. Strict attention retains all K/V
  tokens; query chunking preserves full-key softmax. Pilot pooling is optional
  and is an explicitly approximate experiment.
- Wavelet attention normalizes the spatial input before DWT, uses full-resolution
  queries and reduced-resolution K/V, and concatenates a separate IDWT branch
  before output projection. Tests cover both branch gradients.

## Specification still required

The available main paper does not establish the following author settings:

- Widths, block counts, embedding kernels/strides/padding and attention heads.
- Query tensor shapes, initialization and composition with features. Current
  broadcast additive/gated queries are assumptions, not a resolved specification.
- Supplementary-prior filtering, fusion, refinement and scale alignment.
- Hierarchical reconstruction operations and their connections. Bilinear skip
  fusion, the RGB projection, input-image residual and output clamp to [0,1]
  remain implementation choices.
- Wavelet family, normalization and boundaries; current code uses orthonormal
  Haar with replicate extension for odd sizes. GN groups, MLP expansion,
  activations, channel L2 normalization and attention scaling are also assumed.
- Equation 12 reduction: SmoothL1(mean residual) as printed differs from the
  implemented mean(elementwise SmoothL1). Loss operations remain unchanged
  pending clarification. DCT normalization and frequency-loss axes are explicit
  interpretations, too.

The configured model has **2,762,691 parameters**, versus **58.34M** in Table XIII.
Matching a count would not prove architectural equivalence. No parameter padding
or invented author hyperparameters are used to close that gap.

## Training and dataset limits

The shared model, PoolFormer-style encoder, wavelet priors, queried attention,
DCT/Smooth-L1 loss family and AdamW/cosine methodology follow the paper's
algorithm families. Tests establish implemented behavior, not author equivalence.
Old three-position diagnostic checkpoints are incompatible; corrected training
starts from fresh initialization.

Local four-weather paired data are ready for the documented custom 80/20
experiment. Haze provenance, exact AIR40K membership, missing official benchmark
releases and author training settings still prevent a research reproduction.
See [dataset status](DATASET_STATUS.md) and [training readiness](TRAINING_READY.md).
