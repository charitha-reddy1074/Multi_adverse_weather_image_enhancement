# Project working status: AIRFormer paper and current implementation

Snapshot: 5 November 2026, approximately 15:45 IST. Training continues; progress below is a recorded snapshot.

Reference: Gao et al., *Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration*, IEEE TCSVT 34(3), 1886-1899, 2024. DOI: https://doi.org/10.1109/TCSVT.2023.3299324.

## Project statement for review

The project implements an independent, paper-guided all-in-one restoration network for rain, raindrops, snow, and haze. The data pipeline, frequency-guided encoder, supplementary priors, queried decoder, loss, checkpointing, and evaluation pipeline are implemented. A lightweight configuration is currently training on a custom four-weather dataset split. Exact reproduction of the published architecture and benchmark scores has not been established.

Status terminology: **Implemented** means present in executable code; **approximation** means a deliberate resource adaptation; **assumption** means an implementation choice whose equivalence to the authors' operation is unverified; **pending** means no completed experiment/result yet.

## 1. Separate the model configurations

| Item | Published AIRFormer | Repository paper-guided configuration | Active training configuration |
|---|---|---|---|
| Configuration | Authors' original configuration | `configs/paper.yaml` | `configs/four_weather_fast.yaml` |
| Trainable parameters | 58.34M, Table XIII | 2,762,691, recorded architecture verification | **612,435**, counted directly from a model instantiated with the active checkpoint configuration |
| Stage widths | Exact numerical specification unresolved locally | 32 / 64 / 128 / 256 | 16 / 32 / 64 / 128 |
| Encoder depths | Exact numerical specification unresolved locally | 2 / 2 / 2 / 2 | 1 / 1 / 1 / 1 |
| Attention heads | Unresolved locally | 4 | 2 |
| Crop | 256 x 256 | 256 x 256 | **128 x 128** |
| Spatial/wavelet keys and values | Paper attention graph | Full keys/values; exact query chunking available | **Adaptive pooled keys/values**, maximum configured grid budget 256 tokens |
| Epochs | Not specified in the supplied main paper | 300 is an engineering default, not an author setting | **20**, custom experimental schedule |
| Execution status | Published results | Diagnostic optimizer steps recorded; no completed full training demonstrated | **Full dataset training in progress** |
| Permitted conclusion | Published method | Closer graph reconstruction with unresolved details | Lightweight AIRFormer-inspired custom experiment |

The earlier 2.76M figure describes the larger configuration, not the actively training model. Increasing a parameter count alone does not establish equivalence to the paper.

## 2. Architecture comparison

| Component | Paper description / evidence | What is implemented and active | Working status and remaining gap |
|---|---|---|---|
| All-in-one network | One shared network for four weather degradations; Figure 3(d) | One RGB restoration model, shared weights and output head | Implemented at the method level; effectiveness awaits evaluation |
| Hierarchical encoder | Four encoder block positions; Figure 4 | Four positions with progressively increased channels | Graph structure implemented; active widths/depths are chosen lightweight settings |
| Encoder token mixer | PoolFormer-inspired encoder; Section III-A | Pooling mixer, residual blocks and channel MLP | Implemented family; normalization, expansion and numerical details are assumptions |
| Supplementary prior (SP) | DWT, high-frequency fused filtering and refinement; Equation 10 | Three recurrent SP modules; previous enhanced features/prior; zero initial inputs | Connection order implemented and historically checked; exact internal operators remain assumptions |
| SP injection | Additive frequency guidance to encoder | Added after the next embedding and before its encoder block | Implemented connection order; scale-alignment convolution is an assumption |
| Wavelet transform | Low/high-frequency subband separation | Differentiable orthonormal Haar DWT/IDWT | Implemented; exact author family, scaling and boundary convention unresolved |
| Decoder attention domains | Channel, spatial and wavelet queried attention; Figure 5 | All three domains enabled | Implemented at the method level |
| Decoder attention order | Channel -> Space -> Wavelet performs best; Table XII | Same order | Configured to match reported preferred order |
| Channel attention | Attention across channels | Multihead channel attention | Implemented; author query and scaling details unresolved |
| Spatial attention | Spatial token relationships | Active run pools K/V; full-attention path exists separately | Active run is an approximation; not equivalent to full spatial attention |
| Wavelet attention graph | Full spatial Q, reduced wavelet K/V, independent IDWT branch | Separate attention and IDWT branches concatenated before projection | Graph corrections implemented; active K/V pooling adds an approximation |
| Task-biased queries | Learnable task information combined with input-derived features | Learned broadcast task parameters added to projected queries | Implemented concept; tensor shapes, initialization and composition not verified against author code |
| Attention normalization | Attention/refinement blocks in Figure 5 | Pre-attention normalization, attention residual, MLP residual | Implemented; exact normalization/scaling choices include assumptions |
| Reconstruction | Hierarchical reconstruction in Figure 4 | Bilinear upsampling, skip concatenation/fusion, RGB convolution | Functional reconstruction; exact author tail unresolved |
| Global image residual | Not conclusively specified in locally resolved evidence | Input image added to predicted residual | Engineering assumption, enabled |
| Output range | Exact author treatment unresolved | Model clamps output to [0,1] | Engineering assumption; relevant to training gradients and metrics |

## 3. Loss and training settings

| Item | Paper | Active experiment | Assessment |
|---|---|---|---|
| Optimizer | AdamW | AdamW | Implemented match |
| AdamW betas | 0.9 / 0.999 | 0.9 / 0.999 | Match |
| Weight decay | 1e-4 | 1e-4 | Match |
| Initial learning rate | 2e-4 | 2e-4 | Match |
| LR schedule | Cosine annealing | Cosine annealing | Same schedule family |
| Minimum learning rate | **1e-7** | **1e-6** | Active deviation; larger configuration uses 1e-7 |
| Training patches | 256 x 256 random crops | 128 x 128 random crops | Resource adaptation; smaller spatial context |
| Batch size | Not specified in supplied main paper | 8 | Custom setting; author equivalence unknown |
| Epoch count | Not specified in supplied main paper | 20 | Custom schedule; adequacy not established |
| Augmentation | Horizontal/vertical flips, 90-degree rotations, MixUp probability 0.5 | Implemented paired augmentation and MixUp probability 0.5 | Matches listed augmentation types; MixUp coefficient distribution is an assumption |
| Spatial loss | Smooth L1, Equation 12 | Conventional mean of elementwise Smooth L1 | **Known formula discrepancy:** printed equation applies the penalty after a spatial mean residual; current code averages elementwise penalties |
| Frequency loss | DCT consistency, epsilon 0.001; Equation 11 | Orthonormal DCT-II; per-image coefficient norm and batch mean; epsilon 0.001 | Implemented family; reduction/normalization conventions are interpretations |
| Combined loss | Spatial loss + 0.04 x frequency loss | Same combined structure and coefficient | Coefficient matches; loss is not claimed mathematically identical because of the above reductions |
| Weather sampling | Exact training sampling unresolved locally | Inverse-frequency weighted sampling with replacement | Custom balancing; an epoch is 58,766 draws, not a guaranteed visit to every unique pair |
| Mixed precision | Not specified in resolved paper settings | CUDA mixed precision and GradScaler | Engineering optimization |
| Numerical checks | Not specified as an experiment feature | Finite-loss check, gradient clipping, AMP overflow handling | Implemented training safeguards |
| Resume | Author checkpoint format unavailable | Restores model, optimizer, scheduler, scaler, and saved RNG state | Implemented; active run resumed at epoch 4 from completed epoch 3 |
| Hardware | NVIDIA GeForce RTX 3090 | NVIDIA RTX 2000 Ada Generation, approximately 16 GiB | Different platform; timings are not directly comparable |

## 4. Dataset and evaluation comparison

| Item | Paper | Project working status | Implication |
|---|---|---|---|
| Training composition | Curated AIR40K composition, Table I and dataset discussion | Custom mixture of Rain14000, raindrop, Snow100K training and local haze | Four-weather coverage implemented; exact AIR40K membership not reproduced |
| Split | Published benchmark test sets retained | Custom scene/clean-identity grouped approximately 80/20 train/test split | Custom scores cannot be presented as the paper's benchmark scores |
| Train/test counts | Author-specific dataset composition | 58,766 training pairs / 14,695 test pairs | Actual active manifest counts verified |
| Validation | Author validation protocol not established from local main-paper evidence | No separate validation set, as requested | Fixed training schedule; no validation-based checkpoint selection |
| Haze provenance | ITS/OTS/HSTS in paper dataset discussion | Local files match pinned unofficial `dronefreak/RESIDE-ITS` mirror revision; mirror content is labeled `val` | Byte identity verified; original release/split attribution and AIR40K equivalence remain unresolved |
| Leakage checks | Exact authors' procedures unavailable | Cross-split scene IDs, resolved input/target paths and identical clean pixels checked at training startup | Exact overlap checks passed; does not rule out all near-duplicate or related scenes |
| Image integrity | Paired degraded/clean images | Historical archive audit records full decoding and equal-size checks; loaded pairs checked for equal dimensions | Pipeline verified; no claim of all-image geometric alignment |
| Test preprocessing | Symmetric padding to multiples of 128, then crop to original size | Centered padding and crop implemented; reflect/replicate boundaries | Policy implemented; exact boundary convention is an assumption |
| Test image size | Original sizes supported through padding | Final test uses full images with padding; no training crop or augmentation | Implemented evaluator path |
| Metrics | `skimage.metrics` PSNR and SSIM | RGB floating-point metrics, data_range=1; SSIM channel_axis=2 | Same metric library; exact author options not fully established |
| Automatic final test | Published dataset-specific tables | Fixed final model evaluated once after epoch 20; aggregate PSNR, SSIM and loss written to `final_test.json` | Implemented, **not yet completed** |
| Per-weather results | Separate benchmark results by task/dataset | Active trainer outputs aggregate metrics; separate evaluation script can group saved outputs by folder | Per-weather reporting still needs execution and correctly organized predictions |
| Benchmark coverage | Reports evaluation across 15 public datasets | Several canonical releases and author membership details missing | Full published benchmark reproduction pending |
| Visual comparisons | Paper compares restoration outputs with baselines | Inference CLI and output saving implemented | Trained-checkpoint comparisons on representative held-out images pending |
| Ablations | SP, queries, attention domains/order, losses and other experiments | Ablation switches and diagnostic tests present | Full trained ablation experiments not completed |
| Efficiency | Parameter/FLOP/time comparison, Table XIII | Active parameters verified; historical throughput/diagnostic timing available | Matched-resolution final-model inference/FLOP study pending |

Active manifest composition:

| Weather | Training pairs | Held-out test pairs |
|---|---:|---:|
| Rain | 10,080 | 2,520 |
| Raindrop | 688 | 173 |
| Snow | 39,998 | 10,002 |
| Haze | 8,000 | 2,000 |
| **Total** | **58,766** | **14,695** |

Snow contributes about 68% of test pairs. Aggregate scores are therefore strongly influenced by snow; training balancing does not rebalance the final test averages.

## 5. Verified execution status

| Deliverable | Status at snapshot | Evidence / limit |
|---|---|---|
| Data preparation and manifests | Complete for custom experiment | Current four-weather manifests read directly; startup overlap check passed |
| Network and loss implementation | Present and executable | Active model has progressed through multiple full epochs |
| Architecture diagnostic checks | Historical execution evidence available | Shape, prior-routing, wavelet branches and gradient checks recorded in `docs/architecture/`; these do not prove author equivalence |
| Automated tests | Test suite exists; historical log records 70 passed | Not rerun for this report; no claim of a new complete test run |
| Larger strict configuration | Diagnostic optimization only | Historical 256 x 256 batch-one step recorded; not full research training |
| Current training | **Running, epoch 9 approximately 95% complete** | Progress observed during audit; snapshot will become stale |
| Completed checkpoint | **Epoch 8** at verification time | Loaded `checkpoints/four_weather_20261105_resume/latest.pt` |
| Resume | Successful | Log explicitly confirms resuming at epoch 4; subsequent completed epochs recorded |
| Final test results | **Pending** | `final_test.json` absent at verification time |
| Paper PSNR/SSIM reproduction | **Not established** | Different architecture size, attention approximation, data split and training settings |

Completed epoch losses from the resumed run:

| Epoch | Mean training loss |
|---|---:|
| 4 | 0.685517 |
| 5 | 0.654702 |
| 6 | 0.635865 |
| 7 | 0.609784 |
| 8 | 0.592434 |

Loss decreased approximately 13.6% from epoch 4 to epoch 8. This is training-objective improvement, not evidence of a specific held-out PSNR/SSIM or convergence.

## 6. Remaining work and review milestones

| Priority | Work remaining | Reviewable completion evidence |
|---|---|---|
| 1 | Finish the current fixed 20-epoch experiment | Final checkpoint, completed run status and aggregate held-out results |
| 2 | Evaluate each weather type and create visual comparisons | Per-weather PSNR/SSIM, image counts, and representative degraded/restored/clean examples |
| 3 | Establish baseline restoration benefit | Input-versus-clean metrics on the same test pairs, compared with restored-versus-clean metrics |
| 4 | Resolve author architecture/training ambiguities | Supplement/original implementation/config or documented author clarification |
| 5 | Obtain exact dataset membership and missing official tests | Verified source releases and fixed benchmark manifests |
| 6 | Build and benchmark the resolved larger configuration | Audited architecture, parameter count, finite optimizer steps, memory and throughput measurements |
| 7 | Train the revised model and perform controlled ablations | Separate reproducible runs; checkpoint selection independent of final benchmark tests |
| 8 | Compare with published tables under matched protocols | Dataset-specific metrics with disclosed preprocessing, hardware and all remaining deviations |

## Suggested statement to the guide

"We have implemented a working, lightweight AIRFormer-inspired pipeline for joint rain, raindrop, snow and haze restoration. It includes wavelet-based encoder guidance, channel/spatial/wavelet queried decoder attention, and spatial-plus-frequency training loss. The custom four-weather training run is in progress, with completed checkpoints and decreasing training loss. The current experiment uses 612,435 parameters, 128 x 128 crops and pooled decoder keys/values to fit the available hardware. It is a paper-guided reconstruction with documented assumptions, rather than an exact reproduction of the published 58.34M model. Final held-out metrics, per-weather visual analysis, and matched benchmark comparison remain pending."

## Evidence index

- Primary source: supplied publisher PDF in the repository root; Figures 3-5, Equations 9-13, implementation details on PDF page 8, Tables I and XII-XIII.
- Active settings: `configs/four_weather_fast.yaml`.
- Larger independent configuration: `configs/paper.yaml`; its 300 epochs are an engineering default.
- Implementation: `models/airformer.py`, encoder/decoder/attention/prior modules under `models/`, `losses/multidomain.py`.
- Training/evaluation: `scripts/train.py`, `scripts/evaluate.py`, `utils/metrics.py`, `utils/inference.py`.
- Active data: `data/manifests/four_weather/train.csv` and `test.csv`.
- Active evidence: `checkpoints/four_weather_20261105_resume/status.json`, `training.log`, `latest.pt`.
- Architecture evidence and assumptions: `docs/architecture/README.md`, `verification.json`, `system_validation.json`, `docs/architecture_assumptions.md`.
- Haze byte verification: `docs/RESIDE_MIRROR_VERIFICATION.json` (11,000 files, zero mismatches against pinned mirror revision). This newer evidence refines older descriptions that only said provenance was unknown.
- Historical audit/testing: `docs/LOCAL_DATASET_AUDIT.json`, `LOG.md`, tests under `tests/`. Older documents may describe superseded configurations; this report prioritizes the active checkpoint/configuration and newer evidence.
