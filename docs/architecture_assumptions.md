# AIRFormer assumptions and confirmed deviations

The supplied main-paper review is complete. The implementation is **NOT EXACT**.
The current authoritative matrix is `PAPER_COMPLIANCE.md`; the illustrated code
walkthrough and extracted original figures are in `architecture/README.md`.
Older assertions that the model fully matches Figure 4/5 are superseded.

Confirmed deviations: Figure 4 draws four encoder block positions while the code
fixes three levels; SP is added after the current block instead of before the
subsequent block; spatial attention caps K/V; wavelet attention has reduced-grid
Q and lacks the separate IDWT concatenation path; its first LayerNorm is after
DWT. The paper-oriented config has 1,146,883 parameters, whereas Table XIII
reports 58.34 million. Literal Equation 12 uses a different reduction order from
the conventional elementwise Smooth L1 in this implementation.

Unresolved reconstruction choices: base widths, block depths N, attention heads,
patch kernel/stride/padding, GN group count, MLP expansion/GELU placement,
attention scaling/L2 normalization, task-query composition and initialization,
Haar family and boundary replication, SP filtering/fusion details, skip-tail
upsampling/fusion and the global RGB residual. The main paper does not supply
all these values. No supplementary architecture specification or original
checkpoint/source code is locally available to resolve them.

Verified specific properties: one shared all-weather model; GN/mixer/residual
and GN/MLP/residual encoder pattern; active frequency priors; initial SP zero
condition; channel/spatial relationships; active learned queries; invertible
wavelet implementation; Channel -> Space -> Wavelet order; second decoder MLP
residuals; reported optimizer/constants/augmentations and evaluation tools.
Verification of these individual properties does not certify whole modules.

The code preserves a valid Haar DWT/IDWT and differentiable DCT-II. Their tests
establish algorithmic behavior, not that the authors used the same filter family,
normalization, query dimensions or reconstructed architecture.

No model/loss math changed in the annotation review. Tests pass for the current
reconstruction, not for a corrected strict-paper graph. Do not present the
current model as the exact architecture in the research paper.
