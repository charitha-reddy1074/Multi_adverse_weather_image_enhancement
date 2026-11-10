"""Called by build_evidence.py with verified extraction globals."""
paper_title='Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration'
paper_pdf='Frequency-Oriented_Efficient_Transformer_for_All-in-One_Weather-Degraded_Image_Restoration (1).pdf'
citation='Tao Gao, Yuanbo Wen, Kaihao Zhang, Jing Zhang, Ting Chen, Lidong Liu, and Wenhan Luo. '+paper_title+'. IEEE Transactions on Circuits and Systems for Video Technology, 34(3), March 2024. DOI: 10.1109/TCSVT.2023.3299324.'
source_block=f'''## Evidence hierarchy and identifiers

1. Saved epoch-20 checkpoint: `{checkpoint.relative_to(ROOT)}`. SHA-256 `{hash_checks['checkpoint']['actual']}`.
2. Completed evaluation: `{BASE}/status.json`, `summary.json`, `provenance.json`, and `per_image_metrics.csv`.
3. Stored checkpoint `config` and `model_config`, checked against `configs/four_weather_fast.yaml`.
4. Executable source in `models/`, `losses/`, `airformer/`, `datasets/`, `scripts/`, `utils/`.
5. Retained training logs and TensorBoard events, followed by dated documentation. Historical diagnostic runs are separate.

Paths are relative to `D:\\Multi_adverse_weather_image_enhancement` unless explicitly absolute. Local source hashes and modification timestamps are in `sources/source_inventory.csv`. Packaged snapshots are in `sources/`. This extraction generated documentation/assets and twelve inference examples only. It did not train, change weights, alter datasets, or repeat full evaluation.
'''
limitations='''- Independent paper-guided reconstruction, not certified author-equivalent AIRFormer. Complete supplementary numerical settings and author implementation are unavailable in the repository evidence.
- Four encoder positions and three prior routes are verified, but widths/depths, additive learned queries, Haar choice, prior fusion and reconstruction tail are local interpretations.
- Trained spatial/wavelet attention pools K/V to 256 tokens, changing the attention graph. No matched ablation attributes quality gains to individual components.
- Custom grouped local split, not AIR40K or the paper's official benchmark tests. Snow dominates the test mean (10,002 of 14,695). No validation split or validation-based checkpoint selection.
- Haze bytes match a pinned unofficial RESIDE-ITS mirror `val` revision, not proof of canonical release attribution or official ITS/SOTS split identity. Near-duplicate leakage is not ruled out by exact hashes.
- Raindrop mean PSNR and SSIM rise slightly, but mean MAE increases. A good aggregate does not guarantee improvement in each sample or metric.
- Local FLOPs are unavailable. Variable-resolution local inference time cannot be compared as a speed ratio with paper 256x256 RTX 3090 timing.
- Unweighted orthonormal DCT L2 preserves error energy. It does not mathematically weight high frequencies more heavily.
- Evaluator source hash differs from final provenance after documented report-routing edits. Original evaluator bytes were not recovered. Checkpoint, both manifests, attention and inference hashes match; recorded metrics reconcile and selected predictions reproduce them.
'''
audit=f'''# Project audit for the research presentation

Audit date: 9 November 2026. Project: **AIRFormer-inspired all-in-one adverse-weather image restoration**.

## Research objective and problem

Train one supervised RGB-to-RGB restoration function for rain streaks, adherent raindrops, snow and haze. These degradations obscure scene structure and detail. One deployed model avoids selecting among four separate weather-specific models. This experiment measures paired restoration quality; it does not demonstrate downstream driving safety, detection accuracy or weather classification.

## Base paper

{citation}

Primary evidence: supplied `{paper_pdf}`, first page, Figures 4/5, Eq.9-13, Section IV-B and Tables I-XIII. The MDPI paper suggested in the earlier layout is not the AIRFormer base paper. Figure provenance is retained in `sources/paper_figure_provenance.json`.

## Completed work

- Implemented a four-position pooling encoder, three recurrent Haar-guided supplementary priors, channel/space/wavelet queried decoder refinement, skip reconstruction and a residual RGB head. Sources: `models/airformer.py`, `encoder.py`, `supplementary_prior.py`, `attention.py`, `decoder.py`.
- Prepared and checked 58,766 training pairs and 14,695 custom held-out pairs from Rain14000, Raindrop, Snow100K and local haze. Sources: `data/manifests/four_weather/train.csv`, `test.csv`, `preflight.json`, `datasets/splitting.py`.
- Completed fixed training through epoch 20 across four retained sessions, 4-7 November. Exact stored configuration: base 16, stage depths [1,1,1,1], two heads, 256-token pooled K/V. Re-counted **{params:,} trainable parameters** by strict loading of final checkpoint.
- Recovered interrupted sessions using saved model, optimizer, cosine scheduler, AMP scaler and random states. Sources: `scripts/train.py`, session training logs, checkpoint state keys.
- Completed independent full-resolution FP32 testing on 9 November, 10:23:20-12:34:25 IST. Sources: `{BASE}/status.json`, `provenance.json`, `duration_summary.json`.
- Produced per-image metrics and visual reports. Existing examples are preserved, with twelve additional package-only predictions matching recorded metrics within documented tolerance.

## Verified final outcomes

| Measure | Degraded input | Restored output |
|---|---:|---:|
| Mean PSNR (dB) | {overall['input_psnr']:.9f} | {overall['restored_psnr']:.9f} |
| Mean SSIM | {overall['input_ssim']:.9f} | {overall['restored_ssim']:.9f} |
| Mean MAE | {overall['input_mae']:.9f} | {overall['restored_mae']:.9f} |
| Mean MSE | {overall['input_mse']:.9f} | {overall['restored_mse']:.9f} |
| Mean per-image RMSE | {overall['input_rmse']:.9f} | {overall['restored_rmse']:.9f} |

PSNR gain **+{overall['mean_psnr_gain_db']:.9f} dB**. Mean MAE reduction **{mae_reduction:.4f}%**. Positive PSNR gains on **{improved:.4%}** of images. Mean timed model call **{overall['mean_inference_seconds']:.9f} s/image**, across variable image sizes on RTX 2000 Ada. Sources: final summary and reconciled per-image CSV (`sources/metric_reconciliation.csv`).

## Outdated claims resolved

- Earlier `docs/dataset_provenance.md` says no local images. Current manifests, file checks and 14,695 completed evaluation records supersede it.
- Older `docs/DATASET_STATUS.md` says no full research training result. Final checkpoint/events and completed evaluation supersede this historical status.
- `docs/architecture_assumptions.md` quotes 2,762,691 parameters for a larger configuration. It is not the 612,435-parameter completed run.
- Historical three-position pilot/audit parameter counts and diagnostic timings in `docs/OPTIMIZATION_LOG.md` and `AUDIT_BENCHMARK.json` are not final-model results.
- `checkpoints/four_weather_resume_epoch17/status.json` reports failure from post-training testing. The epoch-20 checkpoint and events prove training completion; `{BASE}/status.json` proves the later successful full test.
- `docs/MASTER_ARCHITECTURE_AND_MODEL_LIFECYCLE.md` and `docs/PAPER_COMPARISON.md`, dated 9 November, agree with final numeric artifacts. They are explanations, not substitutes for metric files.

## Remaining limitations and work

{limitations}

Future work: obtain complete author specification and official memberships, use a documented validation protocol for a future experiment, run matched component ablations and baselines, address raindrop failure cases, test real-world generalization and profile the trained model under a fixed resolution/hardware protocol. These are proposals, not completed work.

{source_block}
'''
write('01_PROJECT_AUDIT.md',audit)

comparison=[
('Patch embedding','Fig.4 labels embeddings; numerical kernels/strides unresolved','3x3 stride-1 RGB projection + GroupNorm; learned stride-2 interstage convolutions','Partial','models/patch_embed.py; models/encoder.py; Fig.4','The dense projection follows the idea; it cannot be identified with unspecified author kernels.'),
('Encoder stage count','Fig.4 shows four feature positions','Four encoder stages; three reductions','Exact','models/airformer.py; trained_model_trace.json; Fig.4','Exact only for the count of positions, not the whole encoder.'),
('Encoder widths and depths','Main paper does not supply complete numerical widths, depths or heads','16/32/64/128; depths [1,1,1,1]; heads 2','Unknown','checkpoint model_config; Fig.4; docs/architecture_assumptions.md','These are local settings; do not describe them as author hyperparameters.'),
('Pooling / token mixing','GroupNorm, pooling residual and depth-wise channel MLP in Fig.4','GroupNorm(1,C), AvgPool3x3(z)-z residual; 2x expansion with GELU/DW conv','Partial','models/encoder.py::FrequencyGuidedBlock; Fig.4','Broad processing family matches; normalization/activation/MLP details remain assumed.'),
('Supplementary Prior','Depth-wise wavelet convolution, fused filtering, concatenation, refinement','Haar details filtered; packed bands resized; feature/bands/previous concatenate to 6C','Different','models/supplementary_prior.py::SupplementaryPrior; Fig.4','Resized coefficient fusion is not demonstrated equivalent to the drawn fused-filter graph.'),
('DWT and IDWT','Eq.9 general wavelet; Fig.5(c) has DWT K/V and separate IDWT branch','Fixed Haar, orthonormal scaling, replicate odd edges; IDWT branch in attention','Partial','models/supplementary_prior.py::dwt2/idwt2; models/attention.py; Fig.5','The prior resizes coefficients without IDWT; attention has an actual inverse branch.'),
('Supplementary-prior injection','Eq.10 recurrent prior; first prior initialized with zero feature/prior; addition before next block','Three recurrent modules; downsample then add prior before stage; SP0 inputs zero','Exact','models/encoder.py::HierarchicalEncoder.forward; trained_model_trace.json; Eq.10','Exact applies to this routing relation only; SP internals are different.'),
('Channel attention','Fig.5(a): channel-axis Q/K/V attention','Per-head channel affinity, normalized Q/K, scaled SDPA, learned vector added to feature Q','Partial','models/attention.py::QueriedAttention.forward; Fig.5(a)','Channel axes match; feature-plus-query construction and scaling are assumed.'),
('Spatial attention','Fig.5(b) depicts HW by HW attention','Full HW query grid, adaptively pooled K/V up to 256 tokens','Different','models/attention.py; checkpoint spatial_kv_pooling; Fig.5(b)','Each output position attends to a compressed key grid instead of all native positions.'),
('Wavelet attention','Fig.5(c): full Q; wavelet K/V at HW/4; IDWT concatenation','DWT 4C -> C projection; full-resolution Q; K/V pooling; C/4 IDWT branch','Partial','models/attention.py; Fig.5(c); trained_model_trace.json','Major branches match, but pooled K/V and chosen query composition differ.'),
('Task-specific queries','Fig.5 and Sec.V-A describe task-biased vectors in place of usual Q','Projected feature query plus broadcast learned parameter initialized to zero','Different','models/attention.py task_query and query_combine; Fig.5','Adding a learned bias to image queries is not proof of replacing Q with the paper vector.'),
('Decoder sequence','Table XII supports Channel then Space then Wavelet','Same order at each of three chosen skip levels','Partial','models/decoder.py; Table XII','Order itself is Exact; repetition/topology at each skip level is a local interpretation.'),
('Skip connections','Hierarchical encoding/reconstruction shown; exact tail links insufficiently specified','Concatenate encoder stage 2/1/0 after bilinear upsampling','Unknown','models/decoder.py::DecoderStage; Fig.4','Actual connections are verified; exact author skip fusion cannot be established.'),
('Reconstruction tail','Hierarchical images reconstruction tail named, full operators unresolved','Three bilinear skip stages, 3x3 RGB head, input residual, crop, clamp','Partial','models/airformer.py::forward; models/decoder.py; Fig.4','A working multiscale tail, not a literal demonstrated author reconstruction.'),
('Smooth L1 loss','Literal Eq.12 applies piecewise penalty to spatial mean signed residual','PyTorch mean of elementwise Smooth L1 penalties','Different','losses/multidomain.py; supplied PDF Eq.12','Opposite signed pixel errors can cancel before the paper literal penalty; code penalizes each.'),
('DCT frequency loss','Eq.11 smoothed DCT difference norm; eps .001; lambda .04 in Eq.13','Orthonormal DCT-II, per-image C/H/W norm, batch mean, eps .001, lambda .04','Partial','losses/multidomain.py; Eq.11-13','Weight and epsilon match; precise normalization and reduction axes remain interpreted.'),
('Parameter count','58.34 million, Table XIII','612,435 trainable parameters from saved configuration','Different','checkpoint re-count; Table XIII','Smaller independent model; parameter count does not certify quality or speed.'),
('FLOPs','5.746 billion at 256x256, Table XIII','No matched final-model measurement','Unknown','Table XIII; final summary lacks FLOPs','Do not use earlier pilot complexity or derive a speed claim from parameters.'),
('Input resolution','256x256 training crops; symmetric multiple-of-128 test padding','128x128 crops; full native test sizes padded to 128 multiples','Different','checkpoint config; utils/inference.py; Sec.IV-B','Training crop differs; test size policy agrees, but boundary padding details are assumed.'),
('Computational optimizations','No evidence for this local pooling cap or backend routing','K/V pooling 256; SDPA; query chunking with all keys; CUDA channel MATH; AMP training','Different','models/attention.py; scripts/train.py; final provenance','Pooling approximates attention. Chunking and MATH routing preserve intended local attention computation, not author equivalence.')]
arch='# Paper versus actually trained implementation\n\n'+citation+'\n\nExact is scoped to the specific operation being compared. Shared names are never evidence of whole-block equivalence. Unknown means the author setting cannot be resolved, even when local behavior is clear.\n\n| Component | Original paper | Our implementation | Exact/Partial/Different/Unknown | Evidence |\n|---|---|---|---|---|\n'
for name,pap,local,status,ev,plain in comparison:arch+=f'| {name} | {pap} | {local} | **{status}** | {ev} |\n'
arch+='\n## Presentation explanations\n\n'
for name,pap,local,status,ev,plain in comparison:arch+=f'- **{name}:** {plain}\n'
arch+='\n## Important loss distinction\n\nFor errors +0.5 and -0.5, the literal spatial mean residual is zero, so the printed Eq.12 penalty is zero. The implemented elementwise Smooth L1 mean is 0.125. This is a mathematical difference, not an exact implementation claim. For an orthonormal DCT, Parseval energy preservation means the unweighted coefficient L2 norm equals the spatial error L2 norm. A frequency-domain computation alone does not preferentially penalize high frequencies.\n\n'+limitations+'\n'+source_block
write('02_ARCHITECTURE_COMPARISON.md',arch)

dataset_md='''## Dataset composition and preparation

| Weather | Local source | Train | Custom test | Total |
|---|---|---:|---:|---:|
'''
for d in details:dataset_md+=f"| {d['weather']} | {d['dataset']} | {d['train_pairs']:,} | {d['test_pairs']:,} | {d['total_pairs']:,} |\n"
dataset_md+='''
Total 73,461 paired records, 58,766 train and 14,695 test. These count pairs, not unique underlying scenes or unique files. Sources: exact four-weather manifests and `preflight.json` (4 November).

Pairing uses explicit `input_path`/`target_path` mapping from source-specific preparation logic, not sorted filenames. `datasets/splitting.py::write_custom_manifests` groups source scene IDs, SHA-256 of decoded clean RGB pixels including dimensions, and SHA-256 of input file bytes via connected identity groups. Groups are shuffled by seed 42 within source buckets, with 80% of groups assigned to train. This can deviate slightly from exactly 80% of pairs. Known official tests and overlapping identities are excluded from the custom pool. The separate `official_test.csv` has 1,412 rain pairs and is not part of the final 14,695 evaluation.

Fresh historical preflight reports 127,746 unique referenced files, zero missing files, and zero scene overlap, plus one decoded pair per weather. The preparation/integrity records include earlier pixel-hash and decoding checks. This extraction re-counted all manifest rows, rechecked manifest hashes, checked selected sample identities and decoded the twelve chosen pairs. It did not repeat all-image decoding or rehash the full datasets.

Training loader `airformer/data.py` converts RGB to [0,1] tensors, checks paired sizes, pads small images, applies the same random 128x128 crop to each pair, paired horizontal/vertical flips, random quarter-turn rotations and MixUp p=0.5. The MixUp coefficient is uniform in the local code. Weather-balanced sampling uses inverse domain counts with replacement, so an epoch is not a guarantee that every distinct training image is visited exactly once. Test images retain full dimensions, with symmetric padding to 128 multiples and crop-back, without augmentation.

Provenance: owner/project links are catalogued in `datasets/sources.csv`. Local archive identity and preparation are in `docs/LOCAL_DATASET_AUDIT.json` and `scripts/prepare_local_archives.py`. Haze files match `dronefreak/RESIDE-ITS`, revision `39300fc7deb9138a59d5489f91a2f86d14d62eb6`, `val/clear` and `val/haze`: 11,000 files, no byte/size mismatch in `docs/RESIDE_MIRROR_VERIFICATION.json`. That confirms mirror identity, not original release provenance. It is not a canonical SOTS test. Exact hashes do not rule out near-duplicate scenes. Canonical AIR40K membership, CSD and official benchmark releases remain missing.

Genuine local pair paths and scene IDs are in `restoration_samples/sample_manifest.csv`. Dataset graphics are in `datasets/`, including `dataset_details.csv`. No external image or dataset was downloaded.
'''
write('datasets/DATASET_EVIDENCE.md','# Dataset evidence\n\n'+dataset_md)
train=f'''# Final training configuration and recorded history

The configuration below comes from the **saved epoch-20 checkpoint**, not a larger paper-oriented YAML. `sources/checkpoint_verification.json` records the strict-load result and all provenance hashes.

| Setting | Verified final setting |
|---|---|
| Parameters | {params:,} trainable |
| Encoder widths / depths | [16,32,64,128] / [1,1,1,1] |
| Attention | 2 heads, add learned query, Channel -> Space -> Wavelet |
| Resource approximation | Spatial/wavelet K/V pooling enabled; 256-token cap |
| Crop | 128 x 128 RGB |
| Batch / accumulation / effective batch | 8 / 1 / 8; short final batch may be smaller |
| Epochs / seed | 20 completed / 42 |
| Optimizer | AdamW, betas .9/.999, weight decay 1e-4 |
| Learning rate | Initial 2e-4, cosine schedule T_max=20, minimum 1e-6 |
| Loss | Mean elementwise Smooth L1 + .04 smoothed per-image orthonormal DCT-II norm |
| Frequency epsilon | .001 |
| Augmentation | Paired random crop, flips, quarter turns, uniform-lambda MixUp p=.5 |
| Sampling | Weather-balanced inverse-count sampling with replacement |
| AMP / clipping | CUDA autocast + GradScaler; clip gradient norm at 1.0 |
| Gradient checkpointing flag | False; attention query-chunk fallback can independently recompute probabilities |
| Workers | 2 |
| Validation | Disabled; fixed final epoch, no validation model selection |
| Hardware | NVIDIA RTX 2000 Ada Generation; log reports 16.00 GiB VRAM |
| Final checkpoint | `{checkpoint.relative_to(ROOT)}` |

Sources: checkpoint `config`/`model_config`; `scripts/train.py:137-225`; `airformer/data.py`; retained training logs. GPU identity also recorded by completed evaluation provenance. Training uses AMP; completed final evaluation uses **FP32 without autocast**.

## Recorded history

All 20 loss and learning-rate points were read from TensorBoard scalar events in the four retained training sessions. `training/epoch_history.csv` retains exact values, event wall timestamps and source directory. The loss plot is the actual logged mean of training batch losses, not a fitted curve. It may be nonmonotonic because of augmentation, balanced sampling and optimization.

- First recorded epoch loss: **{losses[0]:.9f}**.
- Final recorded epoch loss: **{losses[-1]:.9f}**.
- Completed epoch: **20**.
- Logged final LR: **{history[20]['logged_lr_after_epoch']:.12g}**.
- Learning-rate values are logged **after** `scheduler.step()`. The first epoch uses initial 2e-4 during updates; its post-epoch logged rate is lower. No calculated schedule is presented as logged.

## Pauses, resume and actual time

| Retained run | Epochs retained | Start IST | Finish IST | Session wall seconds |
|---|---|---|---|---:|
'''
for rec,eps in zip(duration['training_sessions'],['1-3','4-10','11-16','17-20']):train+=f"| {rec['run']} | {eps} | {rec['started_at']} | {rec['finished_at']} | {rec['wall_seconds']:.6f} |\n"
train+=f'''
Sum of recorded retained-session wall intervals: **{duration['training_session_wall_seconds']:.6f} s**, approximately 19h 38m 30s. Calendar span: **{duration['training_calendar_seconds']:.6f} s**, 3d 2h 35m 57s. Session wall time includes startup, discarded partial epochs and the original failed final-test attempt. It excludes overnight pauses and abandoned earlier runs. **Pure training compute was not separately measured.** Source: final `duration_summary.json` cross-referencing session `status.json` files.

Resume chain: `four_weather_20261104_fast/epoch_003.pt` -> `four_weather_20261105_resume/epoch_010.pt` -> `four_weather_resume_epoch11/epoch_016.pt` -> `four_weather_resume_epoch17/latest.pt` (20). Resumption restores weights, optimizer, scheduler, scaler and RNG, not just the epoch label. Original checkpoint files remain untouched.

The last session's failure status describes a CUDA attention-kernel crash in testing **after** the epoch-20 save. The final log excerpt contains epoch 17-20 completion lines. The later separate full test completed. Do not describe the failed supervisor as failed training, and do not claim the model was retrained to fix evaluation.

Graphics: `training_loss_curve.png`, `learning_rate_curve.png`, `training_configuration.png`, `training_completion_evidence.png` (accurately labeled rendered excerpt), and `checkpoint_summary.png`.

{dataset_md}

{source_block}
'''
write('04_TRAINING_CONFIGURATION.md',train)

result='# Completed experimental results\n\nCheckpoint: `'+str(checkpoint.relative_to(ROOT))+'`, saved epoch 20. Completed final job: 9 November 2026, 10:23:20-12:34:25 IST.\n\nSources: `'+BASE+'/summary.json`, `per_image_metrics.csv`, `status.json`, `provenance.json`, `duration_summary.json`. All group means for ten input/output metrics were recomputed from 14,695 CSV rows and matched to better than 1e-8.\n\n| Group | N | Input PSNR | Output PSNR | Gain dB | Input SSIM | Output SSIM | Improved PSNR |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for w in ['overall']+WEATHER:
    g=groups[w];result+=f"| {w} | {g['images']:,} | {g['input_psnr']:.6f} | {g['restored_psnr']:.6f} | {g['mean_psnr_gain_db']:+.6f} | {g['input_ssim']:.6f} | {g['restored_ssim']:.6f} | {g['psnr_improved_fraction']:.4%} |\n"
result+='\n## Error metrics\n\nAll are RGB float, range [0,1]. Each image has equal weight; these are means of per-image values.\n\n| Group | Input MAE | Output MAE | Input MSE | Output MSE | Input RMSE | Output RMSE |\n|---|---:|---:|---:|---:|---:|---:|\n'
for w in ['overall']+WEATHER:
    g=groups[w];result+='| '+w+' | '+' | '.join(f'{g[k]:.9f}' for k in ['input_mae','restored_mae','input_mse','restored_mse','input_rmse','restored_rmse'])+' |\n'
result+=f'''
Mean MAE reduction = 100 x (1 - mean restored MAE / mean input MAE) = **{mae_reduction:.6f}%**. It is not the mean of per-image percentage reductions. Mean per-image RMSE differs from sqrt(mean MSE), and the mean PSNR is not PSNR computed from the dataset mean MSE.

## Timing and protocol

- Mean recorded model call: **{overall['mean_inference_seconds']:.9f} s/image**. Sum of calls: **{duration['test_inference_sum_seconds']:.6f} s**.
- Test loop: **{duration['test_loop_seconds']:.6f} s**; complete job **{duration['test_wall_seconds']:.0f} s**, 2h 11m 5s. Preflight **{duration['test_preflight_seconds']:.6f} s**.
- RTX 2000 Ada; PyTorch {prov['torch_version']}; FP32, batch one, native image sizes, symmetric multiple-of-128 padding, crop-back, no tiling.
- `_restore_tensor` synchronizes CUDA around the model call. Decoding, metric computation and work outside the timed call are excluded from inference time. It is not full application latency.
- PSNR/SSIM: skimage RGB floats, data_range=1, SSIM channel_axis=2, window at most 7. Source `utils/metrics.py`.

## Expected cross-checks

| Prompt expectation | Evidence | Resolution |
|---|---|---|
| 20 epochs | checkpoint epoch=20 and scalar history 1-20 | Matches |
| 14,695 pairs | completed status, CSV row count and manifest | Matches |
| PSNR about 25.6356 | {overall['restored_psnr']:.10f} | Matches rounding |
| SSIM about .805444 | {overall['restored_ssim']:.10f} | Matches rounding |
| Gain about +4.9635 | {overall['mean_psnr_gain_db']:.10f} | Matches rounding |
| MAE reduction about 49.77% | {mae_reduction:.8f}% | Matches rounding |

No expected value was substituted for missing measurements.

## Interpretations and failure cases

All weather groups improve in average PSNR/SSIM. Haze has the greatest PSNR gain, though the input baseline is particularly poor. Raindrop is weakest and its mean MAE increases from .065497 to .068148; this must remain visible. {int((gain<=0).sum()):,} images have nonpositive PSNR gain. Snow contributes {groups['snow']['images']/14695:.2%} of the overall image-weighted mean.

The twelve examples are selected by minimum, upper-median and maximum **PSNR gain** within each entire weather group. They are not selected by restored PSNR or aesthetics. Haze's worst-gain sample still improves because all haze samples have positive gain. Best/worst labels describe this recorded criterion, not a universal perceptual ranking.

Existing galleries remain unchanged in the original `outputs/four_weather_test_epoch20_20261109/visualizations/examples_*.png` paths. Large duplicate galleries are excluded from this package. Individual native predictions were not stored separately by the full-evaluation script, so the selected twelve held-out inputs were inferred once with the verified final checkpoint. Float PSNR/SSIM were recomputed before display quantization and agree with original rows within .005 dB and .0001 SSIM. Native aspect ratio is retained. Every identity and numerical delta is in `restoration_samples/sample_manifest.csv`.

No full-dataset evaluation was repeated. Auxiliary accuracy/F1/confusion matrices are correction/error-mask diagnostics, not weather classification or substitutes for PSNR/SSIM. They are not headline presentation results.

{source_block}
'''
write('05_EXPERIMENTAL_RESULTS.md',result)

challenges=[
('Architecture reconstruction','Main paper provides graph and equations but incomplete numerical specification.','Compared Figures 4/5, Eq.9/10 and executable branches; documented uncertain settings.','Independent modular implementation with explicit assumptions, not a claimed exact clone.','Current source, docs/architecture/README.md and architecture_assumptions.md; checkpoint strict load.'),
('Encoder position count','Historical pilot implementation used fewer positions.','Architecture review identified the four positions drawn in Figure 4.','Current model has four stages and three downsamplers; historical counts are superseded.','models/airformer.py; models/encoder.py; trained_model_trace.json shows all four stages.'),
('Supplementary-prior routing','Earlier routing did not reflect recurrent pre-block injection.','Reviewed Eq.10 and tested zero first inputs / preceding-feature recurrence.','Current priors are added after downsampling before subsequent encoder blocks. Internals remain approximate.','models/encoder.py; docs/architecture/trained_model_trace.json; tests/test_architecture_fidelity.py.'),
('Query and wavelet paths','Early diagram interpretation omitted or misrouted wavelet branches.','Inspected Q/K/V axes and separate inverse path; gradient checks recorded.','Current attention normalizes before DWT, retains full-grid queries, projected wavelet K/V and IDWT concatenation. Query composition is still assumed.','models/attention.py; docs/architecture/verification.json and system_validation.json; tests/test_attention.py.'),
('Pairing and leakage','Source variants can share clean scenes, making random pair splits leak.','Mapped dataset-specific filenames; grouped scene, decoded clean pixels and input bytes.','Manifests/preflight report zero scene overlap and no missing referenced files; near duplicates remain unresolved.','datasets/splitting.py; data/manifests/four_weather/preflight.json; docs/LOCAL_DATASET_AUDIT.json.'),
('CUDA memory pressure','Full native-grid spatial attention is costly for local hardware.','Diagnostic workload and memory records preceded the smaller experiment.','Trained small model enables pooled K/V with a 256-token cap. This is an approximation, not a measured no-cost equivalent.','outputs/training_batch_benchmark.json; configs/four_weather_fast.yaml; docs/LOG.md.'),
('AMP overflow','Early FP16 attempts overflowed scaled gradients.','Training log records skipped optimizer updates and reducing scales.','GradScaler skips overflow updates and recovers; final epoch losses and weights are finite.','scripts/train.py; retained training.log; docs/OPTIMIZATION_LOG.md. Diagnostic measurements are not final benchmarks.'),
('Gradient and model checks','Complex transforms/queries need gradient and branch validation.','Historical tests and real-data diagnostic steps inspect finite gradients and task queries.','Saved reports support implemented behavior; not author-equivalence or final ablation evidence. No training tests rerun in this extraction.','docs/architecture/system_validation.json; outputs/paper_review/architecture_tests.xml; tests/test_wavelet.py.'),
('Checkpoint recovery / pauses','Long sessions were interrupted between retained epoch boundaries.','Verified checkpoint epoch/config and resume log entries.','Resume restores model, AdamW, scheduler, scaler and RNG across 3 -> 10 -> 16 -> 20 retained epochs.','scripts/train.py; four retained training logs; epoch_003.pt / epoch_010.pt / epoch_016.pt / final latest.pt.'),
('Final GPU testing crash','First post-training full-size test failed with cutlassF: no kernel found to launch.','Channel attention treats HW as embedding length; CUDA fused backend has shape restrictions.','Current channel attention chooses SDPA MATH on CUDA; a separate final full test completed all 14,695 samples. No retraining.','models/attention.py::attend; checkpoints/four_weather_resume_epoch17/training.log; final status/provenance.'),
('Historical status contradictions','A failed supervisor and older docs imply training/evaluation incomplete.','Prioritized checkpoint contents and completed timestamped outputs.','20 training epochs and final held-out evaluation verified; historical failures retained transparently.','Final checkpoint, TensorBoard scalar history and outputs/four_weather_test_epoch20_20261109/status.json.')]
eng='# Engineering challenges and verified outcomes\n\nHistorical defects below are not descriptions of the current corrected graph. Tests establish local behavior, not paper equivalence.\n\n'
for title,problem,investigation,fix,evidence in challenges:
    eng+=f'## {title}\n\n**Problem:** {problem}\n\n**Investigation:** {investigation}\n\n**Fix:** {fix}\n\n**Verified outcome / evidence:** {evidence}\n\n'
eng+='## Evidence boundaries\n\n`docs/OPTIMIZATION_LOG.md` records older 62-test audit snapshots and random-weight timings. They cannot be relabeled as final-model research scores. Later architecture reports and saved model evidence describe the four-position graph. Current successful testing demonstrates the completed inference path works; it does not isolate which architectural component caused gains.\n\n'+source_block
write('06_ENGINEERING_CHALLENGES.md',eng)

# Exactly fifteen slides with concise body and substantive notes/questions.
slides=[
('Title and Project Overview','One trained model restores four weather categories.',
 ['AIRFormer-inspired multi-weather RGB restoration','Independent implementation guided by Gao et al., IEEE TCSVT 2024','Completed: 20 training epochs and 14,695 held-out tests','Presenter / department / institute / guide: add confirmed details'],
 'diagrams/model_lifecycle.png',f'Checkpoint epoch=20; {BASE}/status.json; supplied paper first page',
 'Introduce the work as an independent AIRFormer-inspired implementation. The completed work includes data preparation, architecture implementation, training, testing and evidence reporting. Personal and institutional details were not supplied, so they must be filled accurately. Use 9 November 2026 as evidence cutoff, not an invented presentation date.',
 ['What work is completed?','Is this an exact AIRFormer reproduction?']),
('Problem Statement and Motivation','One restoration function handles four distinct degradations.',
 ['Rain, raindrops, snow and haze obscure image detail','Learn from degraded/clean images of the same scene','Use one RGB-to-RGB model without a weather-label input','Evaluate reconstruction quality against clean targets'],
 'datasets/sample_image_pairs.png','airformer/data.py; models/airformer.py::forward; supplied paper Introduction',
 'Each training pair shows a damaged image and its desired clean version. The network learns a correction from many such examples. Weather labels organize sampling and reports, but the model forward function receives only RGB images. Explain that downstream recognition gains were not measured.',
 ['Why one model instead of four?','Do weather labels enter the model?']),
('Base Research Paper','The paper provides the research concepts, not a fully resolved local specification.',
 ['Gao et al.: Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration','IEEE TCSVT, volume 34, issue 3, March 2024','Frequency-guided encoder and Supplementary Prior','Channel-, space- and wavelet-queried decoder'],
 'comparison/figure4_airformer.png',paper_pdf+' first page; DOI 10.1109/TCSVT.2023.3299324',
 citation+' Explain the three ideas using the authors’ figure. The author model, proposed in the paper, and the trained local model must remain distinct. Online DOI year 2023 in the identifier does not replace the March 2024 volume publication year.',
 ['What are the paper’s main contributions?','Why focus on frequencies?']),
('Original AIRFormer Architecture','Figure 4 connects a hierarchy, recurrent guidance and frequency refinement.',
 ['Four encoder feature positions with increasing abstraction','Three recurrent Supplementary Priors','Frequency-refined decoder guided by learned task queries','Hierarchical image reconstruction tail'],
 'comparison/figure4_airformer.png',paper_pdf+' Fig.4, printed page 1888; Fig.5 and Eq.10',
 'Walk through the unmodified paper diagram from input to output. The prior path supplements the encoder and the decoder refines information through different attention domains. Avoid supplying invented author channel counts. Figure 5 details attention and Table XII supports the preferred Channel, Space, Wavelet order.',
 ['What enters the first prior?','Does the main paper specify every width and depth?']),
('Implemented AIRFormer-Inspired Architecture','The actual trained graph is a compact four-position interpretation.',
 ['Widths 16 / 32 / 64 / 128, one encoder block per position','Three recurrent Haar-based priors before the next encoder block','Three skip decoder levels with Channel -> Space -> Wavelet','RGB correction + input residual + crop/clamp'],
 'diagrams/actual_airformer_architecture.png','Final checkpoint model_config; models/*.py; docs/architecture/trained_model_trace.json',
 'Read the dimensions for a 128x128 example. Spatial size halves and channels double through the encoder. SP0 uses zero inputs; later priors use the preceding enhanced feature and prior. Decoder stages upsample and fuse matching encoder skips. Show supplementary_prior.png or frequency_refined_decoder.png as optional supporting visuals.',
 ['Why does SP0 receive zeros?','Which operations are assumptions?']),
('Original vs Implemented Architecture','Matching concepts does not establish mathematical equivalence.',
 ['Paper: 58.34M parameters; local checkpoint: 612,435','Local widths, query composition, prior fusion and tail are chosen','Spatial/wavelet K/V pooling uses a 256-token cap','Elementwise Smooth L1 differs from literal Equation 12'],
 'comparison/architecture_difference_summary.png','02_ARCHITECTURE_COMPARISON.md; supplied PDF Table XIII; losses/multidomain.py',
 'State the distinction clearly: the project reconstructs the method family, not certified author code. Four positions and recurrent routing can match while the detailed operations differ. Pooling the keys and values compresses what each query attends to. Smaller parameter count alone does not prove equivalent quality or faster inference.',
 ['Is the architecture identical?','How does pooling change attention?']),
('Proposed Methodology and Processing Flow','The experiment has a traceable path from local pairs to final metrics.',
 ['Acquire local source pairs and verify scene identities','Build grouped custom train/test manifests','Train fixed 20 epochs and retain resumable checkpoints','Evaluate final weights once over the held-out collection'],
 'diagrams/model_lifecycle.png','datasets/splitting.py; airformer/data.py; scripts/train.py; scripts/test_restoration.py',
 'Follow the lifecycle diagram. Explain that the test set is held out from training and not used to choose an epoch. A failed original post-training test was repaired through the inference backend path, then a separate complete evaluation produced the final records. This evidence extraction did not repeat the full evaluation.',
 ['How do you prevent leakage?','Was test loss used to select weights?']),
('Dataset Collection and Preparation','73,461 local pairs support a custom grouped four-weather experiment.',
 ['58,766 training pairs and 14,695 held-out pairs','Rain14000, Raindrop, Snow100K and local haze mirror','Scene IDs, identical clean pixels and input bytes grouped','Haze provenance and official AIR40K membership remain limited'],
 'datasets/dataset_distribution.png','datasets/dataset_details.csv; data/manifests/four_weather/preflight.json; docs/RESIDE_MIRROR_VERIFICATION.json',
 'Give the weather counts from dataset_details.csv: train/test rain 10080/2520, raindrop 688/173, snow 39998/10002, haze 8000/2000. These are pair counts. The split is roughly 80/20 of identity groups, not an official benchmark split. Use sample_image_pairs.png to show genuine local pairs. Balanced sampling affects training visitation, not test weights.',
 ['Why is snow dominant?','Is the haze set SOTS?']),
('Implementation and Technology Stack','Each research concept maps to an inspectable Python module.',
 ['PyTorch modules for encoder, priors, queried attention and decoder','Differentiable Haar DWT/IDWT and orthonormal DCT-II','AdamW, CUDA AMP and resumable state checkpoints','skimage RGB PSNR/SSIM plus per-image error and timing records'],
 'code_screenshots/supplementary_prior.png','03_ARCHITECTURE_TO_CODE.md; models/; losses/multidomain.py; utils/metrics.py',
 'Open the source mapping rather than claiming module names prove equivalence. The screenshot is real source with line numbers. DWT is used for local multi-scale guidance and attention. DCT is used by the loss. The public airformer model/loss files re-export the canonical implementation, so they are not a second independent architecture.',
 ['Where is attention implemented?','How are transforms differentiable?']),
('Training Configuration and Progress','The saved compact model completed the planned twenty epochs.',
 ['612,435 parameters; 128x128 crops; batch 8; accumulation 1','AdamW, LR 2e-4 -> 1e-6 cosine schedule, CUDA AMP',f'Logged loss: {losses[0]:.4f} at epoch 1 -> {losses[-1]:.4f} at epoch 20','RTX 2000 Ada; resumed four sessions; no validation selection'],
 'training/training_loss_curve.png','04_TRAINING_CONFIGURATION.md; training/epoch_history.csv; final checkpoint',
 'Training history is extracted from real TensorBoard scalar events, not estimated from progress bars. The recorded session sum is about 19h38m30s, including overhead and failed post-training testing, so do not label it pure GPU training. Use training_completion_evidence.png if asked why the supervisor reports failure after training completed.',
 ['Why 128x128 instead of paper 256x256?','How did resume preserve optimization state?']),
('Loss Function and Evaluation Metrics','The implementation optimizes a spatial penalty and a coefficient-difference norm.',
 ['Mean elementwise Smooth L1 + 0.04 DCT consistency','DCT norm: per image over C/H/W, epsilon 0.001','PSNR and SSIM higher is better; MAE/RMSE lower is better','Literal Equation 12 reduction differs; unweighted DCT has no high-frequency preference'],
 'loss/loss_equations.png','losses/multidomain.py; utils/metrics.py; supplied PDF Eq.11-13',
 'Explain Smooth L1 as quadratic near zero and linear for larger errors. On this model’s clamped [0,1] RGB outputs, errors stay within the unit threshold. The frequency branch uses a smoothed L2 norm after an orthonormal DCT. Energy preservation means this unweighted norm is equivalent to spatial L2, so do not claim selective high-frequency weighting. Opposite signed errors illustrate the Eq.12 reduction difference.',
 ['Why not claim exact Equation 12?','Does this DCT norm emphasize high frequencies?']),
('Quantitative Experimental Results','Restoration improves the same held-out image baseline overall.',
 ['14,695 full-resolution test pairs; FP32 evaluation','PSNR 20.6722 -> 25.6356 dB, gain +4.9635 dB','SSIM 0.716301 -> 0.805444; mean MAE reduction 49.77%','92.40% improve in PSNR; raindrop MAE worsens'],
 'results/final_metrics_table.png',BASE+'/summary.json; per_image_metrics.csv; sources/metric_reconciliation.csv',
 'This is a valid before/after comparison because each output is compared to the same clean scene as its degraded input. The mean is image-weighted and snow has most samples. Raindrop gains are weaker and not all metrics improve. Use performance_dashboard.png for timing details and improvement_distribution.png to show failures.',
 ['What fraction improves?','Why can PSNR improve while mean MAE worsens?']),
('Qualitative Restoration Results','Best, median and weak cases make successes and failures visible.',
 ['Same scene: degraded input / model output / clean target','Twelve cases: best, median and worst PSNR gain per weather','Labels use recorded per-image PSNR and SSIM','All haze cases gain PSNR; worst means weakest gain'],
 'restoration_samples/four_weather_overview.png','restoration_samples/sample_manifest.csv; final per_image_metrics.csv; verified checkpoint',
 'The overview uses the median-gain scene for each weather. For the live presentation, choose two readable weather rows or use separate case graphics rather than compressing all twelve. Discuss a worst case from rain, raindrop or snow. Predictions were reproduced for only twelve existing held-out samples because separate output files were missing, and their float metrics were checked against the final CSV. No synthetic image is shown.',
 ['Were cases cherry-picked?','Is the displayed output from the final checkpoint?']),
('Paper Comparison and Limitations','Published scores provide context across different protocols.',
 ['Contextual comparison only: different datasets and evaluation protocols','Paper uses AIR40K / official tests and 256x256 training crops','Local compact model uses a custom split and 128x128 crops','No matched FLOPs, speed advantage or component-ablation claim'],
 'comparison/paper_vs_our_results.png','Supplied PDF Tables XI/XIII, Sec.IV-B; 02_ARCHITECTURE_COMPARISON.md; final provenance',
 'Read the contextual comparison label aloud. The paper reports 58.34M parameters, 5.746G FLOPs and .086 seconds at 256x256 on RTX3090. Our model has fewer parameters, but timing uses a different GPU and variable full-resolution images. Do not compute a quality deficit or a speed ratio and call it a controlled benchmark.',
 ['Why are paper PSNR/SSIM not directly comparable?','What would a fair benchmark require?']),
('Conclusion and Future Scope','A completed, traceable implementation is the demonstrated contribution.',
 ['Built and trained one four-weather restoration pipeline','Completed fixed-epoch training and full custom held-out testing','Verified average restoration gain with visible failure cases','Next: official protocol, matched ablations, stronger raindrop restoration'],
 'results/performance_dashboard.png','Final checkpoint; evaluation artifacts; 06_ENGINEERING_CHALLENGES.md',
 'Conclude with the work actually delivered: implementation, verified data pipeline, training recovery and completed evaluation. Do not claim a new state-of-the-art architecture. Future changes need controlled experiments: author-specified settings, official datasets, ablations, real-world tests and fixed-condition profiling. None of those experiments is presented as completed.',
 ['What is your contribution?','What is the next controlled experiment?'])]
slide_md='# Fifteen-slide research presentation content\n\nTarget: 12-15 minutes. Use one main visual per slide, white background, navy headings and teal highlights. Personal/guide/institute names and actual presentation date are not supplied. Add them only when confirmed. Sources and details belong in speaker notes, not dense slide bodies.\n\n'
for i,(title,key,bullets,visual,sources_,notes,questions) in enumerate(slides,1):
    assert (OUT/visual).exists(),visual
    slide_md+=f'## {i:02d}. {title}\n\n**One key message:** {key}\n\n**Slide content:**\n\n'+''.join('- '+b+'\n' for b in bullets)+f'\n**Verified values and sources:** {sources_}\n\n**Recommended visual:** {visual}\n\n**Exact asset path:** `{OUT/visual}`\n\n**Speaker notes:** {notes}\n\n**Possible guide questions:**\n\n'+''.join('- '+q+'\n' for q in questions)+'\n'
write('07_SLIDE_CONTENT.md',slide_md)

qa=[
('What is AIRFormer?','The paper’s frequency-oriented all-in-one restoration Transformer for rain, raindrop, snow and haze. This project is an independent smaller implementation inspired by its encoder, prior and decoder concepts.','Supplied paper first page; models/airformer.py'),
('Why use one model for four weather conditions?','A shared restoration function avoids selecting and storing four separate weather-specific models. The local forward function takes RGB only. This experiment confirms custom-split average restoration gains, not superiority over four separately trained experts.','models/airformer.py; final summary.json'),
('What is a Transformer here?','Attention learns how features influence one another using queries, keys and values, with residual and MLP processing. Our encoder uses pooling-style blocks; decoder refinement performs actual queried attention in channel, spatial and wavelet modes.','models/encoder.py; models/attention.py'),
('Why use PoolFormer-style processing?','Local pooling mixes nearby features using inexpensive non-attention operations. The code uses GroupNorm and AvgPool(z)-z inside a residual, followed by a depthwise/channel MLP. We did not isolate its contribution in a matched ablation.','models/encoder.py::FrequencyGuidedBlock'),
('Why is the encoder hierarchical?','Each reduction halves spatial size and increases channels, moving from fine details toward broader context. The trained widths are 16,32,64,128 at 128,64,32,16 spatial sizes for a 128 crop. Decoder skips reuse fine features.','trained_model_trace.json; models/encoder.py'),
('What is a Supplementary Prior?','A learned guidance branch. This implementation decomposes features into Haar bands, filters details, combines bands with current and previous features, refines, and projects to the next encoder scale. It is a documented approximation of the paper branch.','models/supplementary_prior.py'),
('What is DWT?','A discrete wavelet transform separates each local 2x2 group into one coarse LL coefficient and three directional detail coefficients, while reducing each band’s spatial size. Our transform is fixed and differentiable.','models/supplementary_prior.py::dwt2'),
('Why use Haar wavelets?','Haar has simple orthonormal sums/differences and an explicit inverse, so it is easy to implement and audit without additional dependencies. It is our engineering choice; the available main paper does not uniquely establish Haar.','models/supplementary_prior.py; architecture_assumptions.md'),
('What is the difference between DWT and DCT?','Haar DWT separates local coarse/detail bands and is used in priors and wavelet attention. The orthonormal 2D DCT uses cosine bases over the image dimensions and is used in the training consistency loss.','models/supplementary_prior.py; losses/multidomain.py'),
('How does channel attention work?','After normalization and Q/K/V projections, each head treats channels as the attended elements and spatial positions as embeddings. The affinity matrix is (C/2) by (C/2) for two heads. The result is projected and added through a residual.','models/attention.py::QueriedAttention.forward'),
('How does spatial attention work?','Each pixel feature query compares with spatial keys and gathers weighted values. Full-grid queries are retained, but our trained model pools large K/V grids to 16x16. This changes attention versus the full HW-by-HW paper graph.','models/attention.py; final checkpoint model_config'),
('How does wavelet attention work?','The normalized full-grid features form queries. DWT bands are projected from 4C to C at reduced resolution to form K/V, optionally pooled. A separate projected-band IDWT yields C/4 channels, concatenated with attended C features before projection.','models/attention.py::QueriedAttention.forward'),
('What are Q, K and V?','Q describes what an output position or channel seeks, K supplies descriptors to compare against, and V supplies content to combine. A scaled QK product followed by softmax produces weights used to mix V. Axes differ between the attention modes.','models/attention.py'),
('What is task-queried attention?','The paper uses learned task-biased vectors to guide attention. Our implementation adds a learned broadcast parameter to a feature-derived query. These internal parameters are not explicit weather labels; this composition remains an assumption.','models/attention.py task_query; models/airformer.py::forward'),
('How does the decoder reconstruct the image?','Starting at 128 channels/16x16, it upsamples to encoder stage 2, concatenates that skip and projects to 64 channels. It repeats at 32 and 16 channels, with Channel/Space/Wavelet refinement each time, then predicts an RGB correction added to the input.','models/decoder.py; models/airformer.py'),
('Why use Smooth L1?','It is quadratic near zero and linear for large residuals, giving a smooth small-error penalty. Our code averages elementwise penalties. Because outputs and targets are [0,1], these RGB errors lie within the unit quadratic region, apart from the boundary.','losses/multidomain.py; models/airformer.py clamp'),
('Why use frequency consistency loss?','The paper motivates consistency after DCT. Our implementation computes a smoothed per-image coefficient-difference L2 norm with weight .04. Since the transform is orthonormal and unweighted, its norm equals spatial L2 energy and does not preferentially emphasize high frequencies.','losses/multidomain.py::dct2 and MultiDomainConsistencyLoss'),
('What is backpropagation?','Autograd computes derivatives of the scalar loss through convolutions, attention and fixed differentiable transforms. GradScaler scales loss for mixed precision, backward accumulates gradients, and AdamW updates parameters. No backward update is used in evaluation.','scripts/train.py training step; utils/inference.py inference_mode'),
('Why use AdamW?','It uses adaptive gradient moments and decoupled weight decay. Our settings are LR 2e-4 initially, betas .9/.999 and weight decay 1e-4, with a cosine schedule to 1e-6. These are verified saved settings, not evidence it is the best optimizer for this task.','checkpoint config; scripts/train.py'),
('What is PSNR?','For [0,1] images, PSNR is 10 log10(1/MSE), in decibels. Higher means smaller squared error for that image. The report averages per-image PSNR values; it is not computed from the overall mean MSE.','utils/metrics.py; scripts/test_restoration.py'),
('What is SSIM?','It compares local luminance, contrast and structural patterns. Our evaluation uses skimage on RGB floats with data_range=1, channel_axis=2 and a window up to seven pixels. Higher generally indicates closer local structure, but is not a complete perceptual judgment.','utils/metrics.py'),
('Why is the implemented model smaller?','The trained configuration deliberately uses base width 16 and one block per encoder position. It has 612,435 parameters, compared with the paper’s reported 58.34M. Prior/query/tail details also differ. We do not claim equal capacity or a measured speed advantage.','checkpoint verification; supplied paper Table XIII'),
('Is the architecture identical to the paper?','No. Broad hierarchy, recurrent guidance and three attention domains follow the paper, but SP fusion, additive feature queries, K/V pooling, reconstruction choices, widths/depths and literal Smooth L1 reduction differ or remain unspecified.','02_ARCHITECTURE_COMPARISON.md; canonical source'),
('Why are the paper’s results not directly comparable?','The paper uses AIR40K and official benchmark collections; our results use a different custom split, architecture, crop resolution and image weighting. Their timing also uses RTX3090 and 256x256 images. Scores are contextual, not a controlled leaderboard.','supplied PDF Sec.IV-B and Tables XI/XIII; final provenance'),
('What is the research contribution?','The demonstrated contribution is an independent operational implementation and a traceable four-weather experiment: dataset identity checks, complete training with resume, final held-out evaluation and transparent limits. A new state-of-the-art method or proven component novelty is not established.','final checkpoint; manifests; final evaluation artifacts'),
('What improvements are planned?','Obtain author specifications and official benchmark memberships, establish validation for a future run, perform matched ablations/baselines, improve weak raindrop cases and test real-world generalization. Fixed-condition FLOPs/latency profiling is also needed. These tasks were not run here.','01_PROJECT_AUDIT.md remaining work'),
('Why does the final training status say failed?','Twenty epochs were saved successfully, then the original post-training CUDA full-size test hit a fused attention kernel error. A channel-attention MATH backend path was used for the later separate evaluation, which completed all 14,695 pairs.','retained training.log; models/attention.py; final status.json'),
('How was train/test leakage checked?','Grouping joins the same source scene, identical decoded clean RGB content and identical input bytes, then assigns entire groups to one custom split. Known official-test identities are excluded. Exact grouping does not prove absence of near-duplicate scenes.','datasets/splitting.py; preflight.json'),
('Is the haze dataset official RESIDE SOTS?','No. The local haze content matches a pinned unofficial RESIDE-ITS mirror val directory. Byte identity to that mirror is verified, but original release and benchmark split attribution are not. The 8000/2000 split is a custom experiment.','docs/RESIDE_MIRROR_VERIFICATION.json'),
('How were epochs and learning rates verified?','The final checkpoint says epoch 20; retained TensorBoard scalar events provide losses and post-scheduler learning rates for every epoch from 1 through 20. Resumption loads scheduler/optimizer state.','training/epoch_history.csv; checkpoint state; scripts/train.py'),
('Does a 49.77% MAE reduction apply to every weather?','No. It is the ratio of overall image-mean MAEs before and after restoration. Raindrop MAE actually rises. Report per-weather tables and weak examples to avoid hiding this limitation.','results/verified_metrics.csv; final summary.json'),
('Why can average PSNR improve while mean MAE worsens?','PSNR depends on squared errors and a logarithm per image; MAE depends on absolute errors. Their sample averaging also differs. Reducing large errors can improve PSNR even if many small errors increase. This observation alone does not identify the causal mechanism.','per-image metrics; loss/metric definitions'),
('How were restoration cases selected?','Within each weather, sort all held-out rows by PSNR gain. Use the minimum, the upper middle row and the maximum. The final checkpoint inferred those same twelve pairs; float metrics matched recorded rows within .005 dB and .0001 SSIM.','restoration_samples/sample_manifest.csv'),
('Is the reported inference time deployment latency?','No. It is the mean synchronized timed model call, .380816 seconds across variable full-resolution test images. Decoding, transfer/padding outside the timer, metrics and report generation are excluded. No comparison with paper timing is controlled.','utils/inference.py::_restore_tensor; duration_summary.json'),
('Does every training image appear once each epoch?','No. The inverse-weather-count weighted sampler uses replacement. An epoch contains the configured number of draws, so rare domains are oversampled and some individual records may not appear in that epoch.','scripts/train.py sampler setup; airformer/data.py sample_weights'),
('Does the prior use IDWT?','The supplementary prior filters Haar details and resizes packed coefficients; it does not apply IDWT. IDWT is used in the decoder wavelet-attention branch. Mixing these branches would misdescribe the actual model.','models/supplementary_prior.py::SupplementaryPrior.forward; models/attention.py'),
('What is the literal Equation 12 difference?','The printed paper first averages the signed spatial residual and then applies a piecewise penalty. Code first applies the piecewise penalty to each element and then averages. Errors +.5 and -.5 cancel to zero in the first policy but give .125 in the implemented policy.','supplied PDF Eq.12; losses/multidomain.py')]
viva='# Research guide questions and answers\n\nAnswers refer to the saved compact experiment, not the larger default model.\n\n'
for i,(q,ans,ev) in enumerate(qa,1):viva+=f'## {i}. {q}\n\n{ans}\n\nEvidence: `{ev}`.\n\n'
write('08_VIVA_QA.md',viva)

comparison_md=f'''# Published results and protocol context

{citation}

**{context}**

Paper weather scores verified visually from Table XI final row (PDF page 10): rain 30.25/.949, raindrop 28.26/.911, haze 32.33/.989, snow 31.75/.954. These are averages of involved benchmark datasets, not local per-image group means. Separate single-weather model scores in Tables VI-IX were not substituted.

Table XIII reports 58.34M parameters, 5.746G FLOPs and .086 seconds/image. Section V-B specifies 256x256 efficiency inputs; Section IV-B specifies RTX3090. Local counts are {params:,}; local FLOPs are unavailable; .380816 s/image uses variable full-size tests on RTX2000 Ada. No speed ratio or FLOPs advantage is supported.

Paper training: AIR40K, reported 40,561 pairs; 256 crops; AdamW .9/.999, weight decay 1e-4, initial LR 2e-4, cosine minimum 1e-7; flips/90-degree rotations/MixUp .5. Main paper points to supplementary material for additional details; batch size and epochs are unresolved. The individual Table I train counts have a historical five-pair discrepancy (sum 40,556 versus reported 40,561) retained in dataset audit records. Do not invent samples to reconcile it.

Local training: custom 58,766 train / 14,695 test, 128 crops, 20 epochs, batch 8, cosine minimum 1e-6. No official AIR40K membership or matched baseline/ablation. See architecture comparison for the twenty component findings.

Paper image excerpts in this directory are byte-for-byte copies of existing extracted figures. Attribution: {citation} Full source PDF hash and crop coordinates are in `sources/paper_figure_provenance.json`. The supplied PDF carries a licensed-use notice. These excerpts are supplied for the requested internal research review with attribution. Permission for public redistribution is not established; do not describe them as openly licensed. The full PDF is not placed in the ZIP.
'''
write('comparison/PAPER_CONTEXT.md',comparison_md)

# Verify file integrity, source claims, image decoding and all expected outputs.
expected=['01_PROJECT_AUDIT.md','02_ARCHITECTURE_COMPARISON.md','03_ARCHITECTURE_TO_CODE.md','04_TRAINING_CONFIGURATION.md','05_EXPERIMENTAL_RESULTS.md','06_ENGINEERING_CHALLENGES.md','07_SLIDE_CONTENT.md','08_VIVA_QA.md',
 'diagrams/actual_airformer_architecture.png','diagrams/actual_airformer_architecture.svg','diagrams/supplementary_prior.png','diagrams/frequency_refined_decoder.png','diagrams/model_lifecycle.png',
 'datasets/dataset_distribution.png','datasets/train_test_split.png','datasets/sample_image_pairs.png','datasets/dataset_pipeline.png','datasets/dataset_details.csv',
 'training/training_loss_curve.png','training/learning_rate_curve.png','training/training_configuration.png','training/training_completion_evidence.png','training/checkpoint_summary.png',
 'results/final_metrics_table.png','results/psnr_before_after.png','results/psnr_improvement_by_weather.png','results/ssim_before_after.png','results/mae_rmse_comparison.png','results/performance_dashboard.png','results/improvement_distribution.png',
 'comparison/paper_vs_implementation_table.png','comparison/parameter_comparison.png','comparison/paper_vs_our_results.png','comparison/architecture_difference_summary.png',
 'loss/multidomain_loss_diagram.png','loss/loss_equations.png','loss/loss_code_evidence.png','restoration_samples/four_weather_overview.png']+[f'restoration_samples/{w}_{case}.png' for w in WEATHER for case in ['best','median','worst']]
assert all((OUT/p).is_file() for p in expected)
image_checks=[]
for p in OUT.rglob('*.png'):
    with Image.open(p) as im:im.verify()
    with Image.open(p) as im:image_checks.append(dict(file=str(p.relative_to(OUT)).replace('\\','/'),width=im.width,height=im.height,mode=im.mode))
csvwrite('sources/image_verification.csv',image_checks)
from xml.etree import ElementTree as ET
for p in OUT.rglob('*.svg'):ET.parse(p)
# Trace validation: source checkpoint hash and stage dimensions.
trace=js('docs/architecture/trained_model_trace.json');assert trace['checkpoint_sha256']==hash_checks['checkpoint']['actual'] and trace['total_trainable_parameters']==params
shape_expected={'patch_embed':[1,16,128,128],'encoder.stages.0':[1,16,128,128],'encoder.stages.1':[1,32,64,64],'encoder.stages.2':[1,64,32,32],'encoder.stages.3':[1,128,16,16],'decoder.stages.0':[1,64,32,32],'decoder.stages.1':[1,32,64,64],'decoder.stages.2':[1,16,128,128],'reconstruct':[1,3,128,128]}
assert all(trace['trace'][k]['output_shape']==v for k,v in shape_expected.items())
doc_sources={1:'checkpoint; final artifacts; docs/MASTER_ARCHITECTURE_AND_MODEL_LIFECYCLE.md',2:'canonical model/loss source; supplied PDF',3:'AST of canonical Python source',4:'checkpoint config; TensorBoard; retained status/logs',5:BASE+'/summary.json; per_image_metrics.csv',6:'docs/LOG.md; verification records; source; final artifacts',7:'verified evidence documents and generated assets',8:'canonical source; supplied PDF; verified final artifacts'}
for p in OUT.rglob('*'):
    if p.is_file() and p.suffix in ['.md','.csv','.json','.py'] and not any(x['asset_filename']==str(p.relative_to(OUT)).replace('\\','/') for x in ASSETS):
        rel=str(p.relative_to(OUT)).replace('\\','/');n=int(p.name[:2]) if p.name[:2].isdigit() else 0
        asset(rel,'1-15' if n else 'support',doc_sources.get(n,'Read-only extraction from repository; sources/source_inventory.csv'),'Documentation/extracted data, not an independent new experiment.' if p.suffix!='.py' else 'Package-only generation script; no source model mutations.')
csvwrite('PPT_ASSET_MANIFEST.csv',ASSETS)
verification=f'''# Evidence verification and known limitations

## Completed automated checks

- Every requested image/document asset exists, including four implemented-model diagram designs, twelve case comparisons and fifteen-slide Markdown content.
- All {len(image_checks)} PNGs decode with Pillow; SVG XML parses. Dimensions are recorded in `sources/image_verification.csv`.
- Five of six final provenance SHA-256 signatures match: checkpoint, train manifest, test manifest, attention and inference. The evaluator hash differs. `docs/LOG.md` records later report-routing changes to docs/reports; the original exact evaluator snapshot was not recovered. Both hashes and this limitation are retained in `sources/checkpoint_verification.json`. CSV reconciliation and twelve matching predictions independently support recorded metrics; they do not reconstruct original evaluator bytes.
- Final checkpoint strict-loaded with {params:,} trainable parameters and saved epoch 20. Existing checkpoint-hashed shape trace matches every main encoder/decoder stage dimension; original trace is packaged.
- Exact train/test manifest counts: 58,766/14,695. The final metrics CSV contains 14,695 rows and the per-weather counts match the summary.
- Fifty group/metric means (five groups x ten metrics) were recomputed and reconciled to <1e-8 absolute difference in `sources/metric_reconciliation.csv`.
- Twenty loss/LR epochs were read from scalar events; learning rates are logged post-scheduler, not reconstructed. First loss {losses[0]:.9f}; last {losses[-1]:.9f}.
- All twelve case selections reference the exact test-manifest row and input/target paths used in the evaluation CSV. Small-sample FP32 inference only; no training or repeated full evaluation. Largest PSNR difference from stored rows: {max(float(r['psnr_delta_abs']) for r in sample_manifest):.9g} dB; largest SSIM difference: {max(float(r['ssim_delta_abs']) for r in sample_manifest):.9g}.
- Original paper figures copied unchanged and attributed; no fabricated architecture/code/log screenshots. Code images contain literal repository lines and line numbers. Completion image is explicitly a rendered log excerpt. Original composite prediction galleries remain outside the package to avoid large duplicates.
- Generated documentation distinguishes published results, local results, historical diagnostics and planned work. Paper score tables carry the contextual-only warning.

## Visual review

Reviewed all presentation-image assets on seven contact sheets, with full-size review of the main architecture, supplementary prior, decoder, loss diagram/equations, comparison table, completion log, source excerpt and a weak restoration case. Corrected flow routing, separated parallel loss branches and adjusted text fit. Verified native sample aspect ratios. The four-weather overview is tall; slide notes recommend using larger individual cases during presentation. All generated PNGs decode. Rehashed all 176 original inventoried evidence files after generation: zero changed. Model/checkpoint, dataset and existing result files were preserved.

## Missing evidence and limits

{limitations}

- Presenter, guide, institute and actual presentation date are not supplied. Slide 1 names those fields explicitly; no identity was invented.
- Original separate restoration files were not retained by the final evaluation reporter; existing composite galleries were inspected in place, and twelve final-weight predictions were produced with numeric verification. The float tensors are not full-dataset cached predictions.
- New code images and completion-log images are accurate rendered excerpts, not screenshots captured from an IDE/terminal.
- No dependency installation was needed. Used existing Python packages: torch, torchvision, numpy, matplotlib, Pillow, pygments, PyYAML, tensorboard and skimage; paper text/figures already extracted locally.
- No public redistribution permission for the licensed paper excerpts has been established. Keep attribution and verify permission before public posting.
- Full source datasets/checkpoints are excluded from the ZIP. Source paths are machine-specific for traceability; images/tables/documentation inside the package are usable without them. Rebuilding requires the original repository and environment.

## Asset manifest

`PPT_ASSET_MANIFEST.csv` records asset filename, slide number, asset type, original source path, verification status and limitations. `sources/source_inventory.csv` records relevant original file hashes/sizes/timestamps. `sources/image_verification.csv` records decoded asset dimensions. The following table maps presentation images to their sources:

| Asset | Slide | Type | Source | Verified | Limitations |
|---|---|---|---|---|---|
'''
for r in ASSETS:
    if r['asset_type'] in ['png','svg']:verification+='| '+' | '.join(str(r[k]).replace('|','/') for k in ['asset_filename','slide_number','asset_type','original_source_path','verified','known_limitations'])+' |\n'
write('09_EVIDENCE_VERIFICATION.md',verification)
summary_md=f'''# AIRFormer research PPT evidence package

Completed evidence package for a fifteen-slide M.Tech guide review. Start with `07_SLIDE_CONTENT.md`, then use the matching image assets. `08_VIVA_QA.md` contains {len(qa)} implementation-grounded questions and answers. This is the requested evidence package; no PPTX is included.

Verified final state: **20 completed epochs**, **{params:,} trainable parameters**, **14,695 held-out pairs**, **PSNR {overall['restored_psnr']:.6f} dB**, **SSIM {overall['restored_ssim']:.6f}**, **gain +{overall['mean_psnr_gain_db']:.6f} dB**, **mean MAE reduction {mae_reduction:.2f}%**.

Assets: four model-related diagram designs (PNG plus main SVG), one dataset pipeline, five training graphics, seven results graphics, twelve individual triptych comparisons plus four-weather overview, eight presentation code excerpts plus the loss source excerpt, four contextual comparison graphics, four unchanged paper excerpts and three loss graphics. Native prediction display copies are included. Large duplicate composite galleries remain in the original output directory.

Traceability: `PPT_ASSET_MANIFEST.csv`, `09_EVIDENCE_VERIFICATION.md`, sources hashes and metric reconciliation, exact epoch history and qualitative sample manifest. No dataset archive, checkpoint, virtual environment or full paper PDF is packaged.

## Statements to use

- Independent AIRFormer-inspired reconstruction with explicitly documented assumptions.
- Completed fixed-epoch training and a custom four-weather held-out experiment.
- Measured improvement over the same degraded-image baseline on this split.

## Statements to avoid

- Exact paper reproduction or state-of-the-art benchmark parity.
- Smaller model is faster or uses fewer FLOPs without matched measurements.
- High-frequency emphasis follows automatically from an unweighted orthonormal DCT norm.
- Haze is official SOTS, all images improve, or all weather metrics improve.
- Session wall duration is pure training compute.

{limitations}

Absolute package path: `{OUT}`. ZIP: `{OUT/'AIRFORMER_PPT_EVIDENCE.zip'}`.
'''
write('PPT_SUMMARY.md',summary_md)
for p in ['09_EVIDENCE_VERIFICATION.md','PPT_SUMMARY.md','PPT_ASSET_MANIFEST.csv']:asset(p,'1-15','Verified extraction package','Manifest/verification documentation.')
csvwrite('PPT_ASSET_MANIFEST.csv',ASSETS)
zip_path=OUT/'AIRFORMER_PPT_EVIDENCE.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p!=zip_path and '__pycache__' not in p.parts:z.write(p,arcname='PPT_EVIDENCE/'+str(p.relative_to(OUT)).replace('\\','/'))
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
print(json.dumps(dict(package=str(OUT),zip_bytes=zip_path.stat().st_size,assets=len(ASSETS),pngs=len(image_checks),viva_questions=len(qa),first_loss=losses[0],last_loss=losses[-1]),indent=2),flush=True)
