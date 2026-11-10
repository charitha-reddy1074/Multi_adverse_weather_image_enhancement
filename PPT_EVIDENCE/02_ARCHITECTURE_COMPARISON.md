# Paper versus actually trained implementation

Tao Gao, Yuanbo Wen, Kaihao Zhang, Jing Zhang, Ting Chen, Lidong Liu, and Wenhan Luo. Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration. IEEE Transactions on Circuits and Systems for Video Technology, 34(3), March 2024. DOI: 10.1109/TCSVT.2023.3299324.

Exact is scoped to the specific operation being compared. Shared names are never evidence of whole-block equivalence. Unknown means the author setting cannot be resolved, even when local behavior is clear.

| Component | Original paper | Our implementation | Exact/Partial/Different/Unknown | Evidence |
|---|---|---|---|---|
| Patch embedding | Fig.4 labels embeddings; numerical kernels/strides unresolved | 3x3 stride-1 RGB projection + GroupNorm; learned stride-2 interstage convolutions | **Partial** | models/patch_embed.py; models/encoder.py; Fig.4 |
| Encoder stage count | Fig.4 shows four feature positions | Four encoder stages; three reductions | **Exact** | models/airformer.py; trained_model_trace.json; Fig.4 |
| Encoder widths and depths | Main paper does not supply complete numerical widths, depths or heads | 16/32/64/128; depths [1,1,1,1]; heads 2 | **Unknown** | checkpoint model_config; Fig.4; docs/architecture_assumptions.md |
| Pooling / token mixing | GroupNorm, pooling residual and depth-wise channel MLP in Fig.4 | GroupNorm(1,C), AvgPool3x3(z)-z residual; 2x expansion with GELU/DW conv | **Partial** | models/encoder.py::FrequencyGuidedBlock; Fig.4 |
| Supplementary Prior | Depth-wise wavelet convolution, fused filtering, concatenation, refinement | Haar details filtered; packed bands resized; feature/bands/previous concatenate to 6C | **Different** | models/supplementary_prior.py::SupplementaryPrior; Fig.4 |
| DWT and IDWT | Eq.9 general wavelet; Fig.5(c) has DWT K/V and separate IDWT branch | Fixed Haar, orthonormal scaling, replicate odd edges; IDWT branch in attention | **Partial** | models/supplementary_prior.py::dwt2/idwt2; models/attention.py; Fig.5 |
| Supplementary-prior injection | Eq.10 recurrent prior; first prior initialized with zero feature/prior; addition before next block | Three recurrent modules; downsample then add prior before stage; SP0 inputs zero | **Exact** | models/encoder.py::HierarchicalEncoder.forward; trained_model_trace.json; Eq.10 |
| Channel attention | Fig.5(a): channel-axis Q/K/V attention | Per-head channel affinity, normalized Q/K, scaled SDPA, learned vector added to feature Q | **Partial** | models/attention.py::QueriedAttention.forward; Fig.5(a) |
| Spatial attention | Fig.5(b) depicts HW by HW attention | Full HW query grid, adaptively pooled K/V up to 256 tokens | **Different** | models/attention.py; checkpoint spatial_kv_pooling; Fig.5(b) |
| Wavelet attention | Fig.5(c): full Q; wavelet K/V at HW/4; IDWT concatenation | DWT 4C -> C projection; full-resolution Q; K/V pooling; C/4 IDWT branch | **Partial** | models/attention.py; Fig.5(c); trained_model_trace.json |
| Task-specific queries | Fig.5 and Sec.V-A describe task-biased vectors in place of usual Q | Projected feature query plus broadcast learned parameter initialized to zero | **Different** | models/attention.py task_query and query_combine; Fig.5 |
| Decoder sequence | Table XII supports Channel then Space then Wavelet | Same order at each of three chosen skip levels | **Partial** | models/decoder.py; Table XII |
| Skip connections | Hierarchical encoding/reconstruction shown; exact tail links insufficiently specified | Concatenate encoder stage 2/1/0 after bilinear upsampling | **Unknown** | models/decoder.py::DecoderStage; Fig.4 |
| Reconstruction tail | Hierarchical images reconstruction tail named, full operators unresolved | Three bilinear skip stages, 3x3 RGB head, input residual, crop, clamp | **Partial** | models/airformer.py::forward; models/decoder.py; Fig.4 |
| Smooth L1 loss | Literal Eq.12 applies piecewise penalty to spatial mean signed residual | PyTorch mean of elementwise Smooth L1 penalties | **Different** | losses/multidomain.py; supplied PDF Eq.12 |
| DCT frequency loss | Eq.11 smoothed DCT difference norm; eps .001; lambda .04 in Eq.13 | Orthonormal DCT-II, per-image C/H/W norm, batch mean, eps .001, lambda .04 | **Partial** | losses/multidomain.py; Eq.11-13 |
| Parameter count | 58.34 million, Table XIII | 612,435 trainable parameters from saved configuration | **Different** | checkpoint re-count; Table XIII |
| FLOPs | 5.746 billion at 256x256, Table XIII | No matched final-model measurement | **Unknown** | Table XIII; final summary lacks FLOPs |
| Input resolution | 256x256 training crops; symmetric multiple-of-128 test padding | 128x128 crops; full native test sizes padded to 128 multiples | **Different** | checkpoint config; utils/inference.py; Sec.IV-B |
| Computational optimizations | No evidence for this local pooling cap or backend routing | K/V pooling 256; SDPA; query chunking with all keys; CUDA channel MATH; AMP training | **Different** | models/attention.py; scripts/train.py; final provenance |

## Presentation explanations

- **Patch embedding:** The dense projection follows the idea; it cannot be identified with unspecified author kernels.
- **Encoder stage count:** Exact only for the count of positions, not the whole encoder.
- **Encoder widths and depths:** These are local settings; do not describe them as author hyperparameters.
- **Pooling / token mixing:** Broad processing family matches; normalization/activation/MLP details remain assumed.
- **Supplementary Prior:** Resized coefficient fusion is not demonstrated equivalent to the drawn fused-filter graph.
- **DWT and IDWT:** The prior resizes coefficients without IDWT; attention has an actual inverse branch.
- **Supplementary-prior injection:** Exact applies to this routing relation only; SP internals are different.
- **Channel attention:** Channel axes match; feature-plus-query construction and scaling are assumed.
- **Spatial attention:** Each output position attends to a compressed key grid instead of all native positions.
- **Wavelet attention:** Major branches match, but pooled K/V and chosen query composition differ.
- **Task-specific queries:** Adding a learned bias to image queries is not proof of replacing Q with the paper vector.
- **Decoder sequence:** Order itself is Exact; repetition/topology at each skip level is a local interpretation.
- **Skip connections:** Actual connections are verified; exact author skip fusion cannot be established.
- **Reconstruction tail:** A working multiscale tail, not a literal demonstrated author reconstruction.
- **Smooth L1 loss:** Opposite signed pixel errors can cancel before the paper literal penalty; code penalizes each.
- **DCT frequency loss:** Weight and epsilon match; precise normalization and reduction axes remain interpreted.
- **Parameter count:** Smaller independent model; parameter count does not certify quality or speed.
- **FLOPs:** Do not use earlier pilot complexity or derive a speed claim from parameters.
- **Input resolution:** Training crop differs; test size policy agrees, but boundary padding details are assumed.
- **Computational optimizations:** Pooling approximates attention. Chunking and MATH routing preserve intended local attention computation, not author equivalence.

## Important loss distinction

For errors +0.5 and -0.5, the literal spatial mean residual is zero, so the printed Eq.12 penalty is zero. The implemented elementwise Smooth L1 mean is 0.125. This is a mathematical difference, not an exact implementation claim. For an orthonormal DCT, Parseval energy preservation means the unweighted coefficient L2 norm equals the spatial error L2 norm. A frequency-domain computation alone does not preferentially penalize high frequencies.

- Independent paper-guided reconstruction, not certified author-equivalent AIRFormer. Complete supplementary numerical settings and author implementation are unavailable in the repository evidence.
- Four encoder positions and three prior routes are verified, but widths/depths, additive learned queries, Haar choice, prior fusion and reconstruction tail are local interpretations.
- Trained spatial/wavelet attention pools K/V to 256 tokens, changing the attention graph. No matched ablation attributes quality gains to individual components.
- Custom grouped local split, not AIR40K or the paper's official benchmark tests. Snow dominates the test mean (10,002 of 14,695). No validation split or validation-based checkpoint selection.
- Haze bytes match a pinned unofficial RESIDE-ITS mirror `val` revision, not proof of canonical release attribution or official ITS/SOTS split identity. Near-duplicate leakage is not ruled out by exact hashes.
- Raindrop mean PSNR and SSIM rise slightly, but mean MAE increases. A good aggregate does not guarantee improvement in each sample or metric.
- Local FLOPs are unavailable. Variable-resolution local inference time cannot be compared as a speed ratio with paper 256x256 RTX 3090 timing.
- Unweighted orthonormal DCT L2 preserves error energy. It does not mathematically weight high frequencies more heavily.
- Evaluator source hash differs from final provenance after documented report-routing edits. Original evaluator bytes were not recovered. Checkpoint, both manifests, attention and inference hashes match; recorded metrics reconcile and selected predictions reproduce them.

## Evidence hierarchy and identifiers

1. Saved epoch-20 checkpoint: `checkpoints\four_weather_resume_epoch17\latest.pt`. SHA-256 `523b2f26b123d97e6b139729f378172cb6e46b328c1b9d1d5a6250e69f76403f`.
2. Completed evaluation: `outputs/four_weather_test_epoch20_20261109/status.json`, `summary.json`, `provenance.json`, and `per_image_metrics.csv`.
3. Stored checkpoint `config` and `model_config`, checked against `configs/four_weather_fast.yaml`.
4. Executable source in `models/`, `losses/`, `airformer/`, `datasets/`, `scripts/`, `utils/`.
5. Retained training logs and TensorBoard events, followed by dated documentation. Historical diagnostic runs are separate.

Paths are relative to `D:\Multi_adverse_weather_image_enhancement` unless explicitly absolute. Local source hashes and modification timestamps are in `sources/source_inventory.csv`. Packaged snapshots are in `sources/`. This extraction generated documentation/assets and twelve inference examples only. It did not train, change weights, alter datasets, or repeat full evaluation.
