# AIRFormer paper compliance - completed main-paper re-verification

**Verdict: NOT EXACT. Confirmed architecture deviations remain.**

The user's 14-page publisher PDF has now been read and visually compared with
actual tensor execution. Figures 4/5, Equations 9-13, implementation details and
Tables XII/XIII were inspected. The earlier missing-PDF blocker is resolved.
Completing this review does not mean completing a faithful implementation.
No model or loss operations were changed during the review; added comments
state the limitations explicitly.

The illustrated explanation is [architecture/README.md](architecture/README.md).
Original figure crops and their SHA-256/page/bounds provenance are under
`architecture/figures/`. Actual shapes, attention normalization, gradients,
parameter count, Eq12 counterexample and semantic fingerprints are in
`architecture/verification.json`.

Status definitions: VERIFIED = directly supported specific property, not whole
module certification; DERIVED = standard operation independently checked;
ASSUMED = plausible reconstruction not specified exactly; DEVIATION = differs
from a visible diagram, printed equation or reported configuration; MISSING =
required depicted route absent; BLOCKED = evidence necessary for exact matching
is unavailable. Each row is deliberately scoped to the property it describes.

| Requirement | Paper locator (PDF / printed page) | Code implementation evidence | Test / execution evidence | Status |
|---|---|---|---|---|
| One shared model across weather types | Fig3(d), Eq8: pp3-4 / 1888-1889 | models/airformer.py:16 has one RGB forward, no weather-specific head | model shape/backward and one-checkpoint inference; multi-weather performance not measured | VERIFIED architecture property |
| Encoder diagram topology | Fig4: p3 / 1888 | models/airformer.py:27 fixes three levels; figure draws four patch-embedding/encoder-block positions | verification.json traces 32,64,128 channels at three levels | DEVIATION A01 |
| Feature label meaning | Fig4: p3 / 1888 | x1,x2,x3 are encoder intermediates; x4 labels the DECODER output in the figure | Visual figure review; do not call x4 the fourth encoder feature label | VERIFIED reading of figure |
| Patch embeddings | Fig4: p3 / 1888 | models/patch_embed.py:11 chooses dense 3x3 stride1 conv/GN; later projections in encoder.py use stride2 convs | full model shape trace | ASSUMED kernels/strides; four-position coverage DEVIATION |
| Encoder GN/mixer/residual/GN/MLP/residual pattern | Fig4 block inset: p3 / 1888 | models/encoder.py:35 performs normalization, pool-minus-input residual and normalized MLP residual | mixer tests; source expression and full forward/backward | VERIFIED ordering; pool subtraction/edge handling are reconstruction choices |
| Depthwise channel MLP conv types | Fig4 inset: p3 / 1888 | models/encoder.py has 1x1, depthwise3x3, 1x1 | actual registered groups/kernels and execution | VERIFIED conv types; GELU/ratio2/groups1 ASSUMED |
| Hierarchical downsampling | Fig4: p3 / 1888 | two 3x3 stride2 convolutions, three feature scales | traces H,W -> H/2,W/2 -> H/4,W/4 | ASSUMED operator; DEVIATION number of depicted encoder positions |
| Prior addition at encoder input | Fig4 + Eq10: pp3,6 / 1888,1891 | models/encoder.py:76 runs stage first, then SP and addition; diagram adds after next patch projection before block | hooks show SP receives current post-block feature | DEVIATION A02 |
| First SP receives zero inputs | Eq10: p6 / 1891 | models/encoder.py:86 passes zero,zero | zero-input recorder and execution trace | VERIFIED local initial condition; routing still DEVIATION |
| Previous enhanced feature/prior recurrence | Eq10: p6 / 1891 | models/encoder.py:88 uses current post-block x and previous prior, resizes/projects prior internally | SP hook tensor shapes in verification.json | DEVIATION in stage alignment A02 |
| SP detail/low-frequency treatment | III-A / Fig4 / Eq9: pp3,5-6 | models/supplementary_prior.py:46 preserves LL intermediate and filters three Haar details | LL preservation and SP gradient/output-effect tests | VERIFIED general preservation/filtering behavior; ASSUMED family/filter construction |
| SP fused-filter/refinement graph | Fig4 SP inset: p3 / 1888 | supplementary_prior.py:48/64 uses bilinear packed-band expansion, 6C current/bands/previous fusion and chosen residual refinement | SP gradients demonstrate active operation, not diagram equivalence | ASSUMED A03; spatial equivalence BLOCKED |
| Wavelet definition and boundaries | Eq9: p5 / 1890 | supplementary_prior.py:7 uses fixed orthonormal Haar, replicate odd edges and inverse crop | IDWT(DWT(x)) close within 1e-6 and finite gradients | DERIVED mathematics; Haar/boundaries ASSUMED A08 |
| Decoder attention order | Fig5 / V-A / TableXII: pp5,10 / 1890,1895 | models/decoder.py:13 channel,space,wavelet | order and permutation tests | VERIFIED |
| Two decoder residual branches | Fig5 top row: p5 / 1890 | models/attention.py:58 attention residual then normalized MLP residual | attention backward; checkpointing output/gradient equality | VERIFIED residual pattern |
| First LayerNorm position | Fig5 top row: p5 / 1890 | channel/space normalize x; wavelet does DWT before LayerNorm(4C) | attention.py:64/67; projected shape trace | VERIFIED channel/space; DEVIATION wavelet A06 |
| Channel attention axis | Fig5(a): p5 / 1890 | attention.py:72 reshapes to [B,M,C/M,HW], channel-by-channel logits | fixture logits [1,2,4,4], key-softmax row sums | VERIFIED axis, not query parameterization |
| Channel L2 normalization / scaling | Fig5 and III-B: pp5-6 | attention.py uses F.normalize and d^-0.5 | source expression inspection | ASSUMED; main paper does not specify these details |
| Spatial attention axis | Fig5(b): p5 / 1890 | Q full grid, K/V pooled above cap at attention.py:85 | capped [1,2,96,16] versus figure [1,2,96,96]; nonlocal perturbation test | DEVIATION A05; spatial dependency itself VERIFIED |
| Wavelet Q/K/V grid and channel layout | Fig5(c): p5 / 1890 | attention.py:29/64 uses packed4C and reduced-grid Q/K/V | uncapped [1,2,24,24] versus figure [1,2,96,24] | DEVIATION A06 |
| Separate wavelet IDWT concat branch | Fig5(c): p5 / 1890 | attention.py:100 IDWTs attention output and adds residual; no concat of IDWT branch | execution/source graph; faithful paper crop | MISSING required depicted route A06 |
| Trainable task-biased queries are active | III-B: p6 / 1891 | attention.py:43 registers task_query and adds/gates it into Q | finite nonzero query gradients, AdamW update, disable-query output difference | VERIFIED activity |
| Exact task-query projection/composition | Fig5 + III-B: pp5-6 | q_proj(input)+broadcast bias; no separate projected query-vector route | parameter shapes recorded; main text says task and feature knowledge but gives no composition equation | ASSUMED A04; strict equivalence BLOCKED |
| Hierarchical reconstruction tail | Fig4 caption: p3 / 1888 | decoder.py:41 chooses two bilinear/concat skip stages; airformer.py:48 one RGB conv | model odd-size and inference-dimension tests | ASSUMED A07; diagram's exact tail internals not provided |
| Global image residual | Fig4: p3 / 1888 | airformer.py:76 adds input RGB by default | explicit source/config flag | ASSUMED adaptation A07, not a confirmed paper operation |
| DCT frequency loss / eps / weight | Eq11/13: p6 / 1891 | losses/multidomain.py:53 DCT residual norm, eps.001, lambda.04 | DC/energy/gradient/half/bfloat16 loss tests | VERIFIED form/constants; DCT normalization and reductions ASSUMED |
| Spatial SmoothL1 reduction | Printed Eq12: p6 / 1891 | losses/multidomain.py:51 mean elementwise penalty; printed E is mean signed residual before penalty | cancellation fixture: code0.5 versus printed0.0 | DEVIATION literal formula A09; possible paper notation error unresolved |
| Width/depth/heads/patch numerical settings | Fig4 repetition N; IV-B: pp3,8 | configs/paper.yaml chooses32/[2,2,2]/4; batch8/epochs300 | config/source inspection | ASSUMED; supplementary/original code unavailable locally |
| Model parameter count | TableXIII: p10 / 1895 | paper config 1,146,883 trainable parameters | direct model.parameters sum; table visually reads58.34M | DEVIATION A10 |
| AdamW/lr/cosine/augmentations | IV-B: p8 / 1893 | paper config AdamW(.9,.999), wd1e-4, lr2e-4->1e-7, crop256, flips/quarter-turns/MixUp.5 | paired augmentation tests, CLI training smoke | VERIFIED stated settings; MixUp lambda distribution ASSUMED |
| Evaluation multiple128 and original-size crop | IV-B: p8 / 1893 | utils/inference.py centered pad128/crop; reflect with replicate fallback | geometry and real-image inference tests | VERIFIED multiples/crop; exact boundary convention ASSUMED |
| skimage PSNR/SSIM | IV-C: p8 / 1893 | utils/metrics.py RGB/data_range1/channel_axis2; strict pair sizes | analytic PSNR fixture and held-out Rain12 smoke | VERIFIED tools/execution; no paper-performance reproduction |
| Exact AIR40K membership and published results | TableI: p5 / 1890 | official source manifests differ; no full trained model | actual release integrity reports | BLOCKED membership / MISSING reproduced scores |

## Evidence limits and conclusion

Review covers the supplied main paper, not unavailable supplementary material or
authors' code. All 62 tests pass, but they verify a functioning reconstruction.
They do not certify the missing fourth encoder position, SP routing, full
spatial attention or the wavelet IDWT/concat graph. Some numerical choices
cannot be recovered uniquely from the main paper. A10 confirms that even the
paper-oriented YAML does not instantiate the reported full model.

The transparent claim is **AIRFormer-inspired reconstruction with documented
architecture differences**, not an exact AIRFormer architecture implementation.
See architecture/README.md for extracted figures, code walkthrough, tensor
comparison and a concrete correction sequence. No operation was silently
changed to manufacture a paper-compliance result.
