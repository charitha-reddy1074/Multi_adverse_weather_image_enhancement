# AIRFormer reproduction activity log

This file is the durable handoff log for implementation decisions, checks, and
remaining work. It records verified actions only; dataset downloads, training,
and benchmark results are not claimed unless a later entry says so.

## 2026-10-30 — Repository and implementation

- Implemented the independent PyTorch AIRFormer project: dense patch embedding,
  hierarchical encoder, recurrent Haar Supplementary Prior, queried
  channel/space/wavelet decoder, reconstruction, DCT multi-domain loss,
  datasets, manifests, training, inference, and evaluation entrypoints.
- Added pilot and paper-oriented configurations, architecture assumptions,
  dataset provenance, and reproduction status documentation.
- Adjusted the initial Supplementary Prior to start its first stage from zero
  features per Equation 10 and to preserve the approximation band while
  filtering detail bands.
- Documented the Figure 5 queried-attention interpretation and default
  Channel -> Space -> Wavelet order.
- Added Table XI/XII ablation controls and a Table XIII parameter/FLOPs/timing
  profiler. These diagnostics are references; they do not establish parity.
- Audited additional dataset sources and added Rain200H/L, Outdoor-Rain,
  RainCityscapes, CDD-11, SPA-Data, and RESIDE-S metadata to the source catalog;
  RainDrop and Snow100K were already represented. Added manifest weather labels
  and Rain200H/L expected counts.
- Added source and pairing caveats: CDD-11 contains non-weather degradations;
  RainDrop's Test-A is a subset of Test-B and AIR40K's count needs audit;
  RainCityscapes requires scene-level split control; ACDC20 is not aligned
  restoration ground truth; and no canonical RESIDE-S archive was verified.
- Added README installation and execution steps, official/source links for all
  ten requested datasets, and corrected the validation split instructions.
- Corrected gradient-accumulation scaling for the final partial accumulation
  window so it is normalized by the actual number of microbatches.
- Updated this log for future handoff.

## Repository commits pushed to `main`

These commits are visible in the requested GitHub repository:

- `11a0052` — Implement modular AIRFormer weather restoration reproduction
- `3a50126` — Align initial supplementary prior with Equation 10
- `faed06a` — Document Figure 5 queried attention interpretation
- `8df06a3` — Preserve low-pass band in supplementary prior
- `3f435c5` — Implement Equation 11-13 multi-domain loss
- `c13e7a7` — Add Table XI and XII ablation controls
- `cafad89` — Add Table XIII complexity profiler
- `409e895` — Add verified weather restoration dataset catalog
- `50196b7` — Document fresh machine setup and execution workflow

The final audit changes in this log, README, and training accumulation step are
being committed/pushed after checks for this update.

## Verification status

- Python syntax compilation (`compileall`) passed for all source and test Python
  files after the final audit edit.
- The source catalog parsed as CSV (22 rows) and contains all requested
  datasets. `git diff --check` passed after the final audit edits.
- YAML parsing could not be run because PyYAML is not installed locally; verify
  config loading after installing the declared requirements.
- Runtime tests have **not** passed or failed: this environment lacks pytest,
  PyTorch/torchvision, Pillow, scikit-image, and PyYAML. The attempted pytest
  command stopped with `No module named pytest`.
- No dataset archive is locally verified or included in Git. `scripts/download_datasets.py`
  only checks project-page reachability; it does not download archives.
- No training run, checkpoint, PSNR/SSIM result, or visual comparison exists.

## Checklist for the next execution machine

1. Install Python 3.10/3.11 and a matching PyTorch/torchvision build, then
   `python -m pip install -r requirements.txt`.
2. Download permitted datasets from links in README / `datasets/sources.csv`;
   retain archive provenance and checksums.
3. Inspect pair identity and dimensions. Create a scene-disjoint validation
   split only from training scenes. Do not use ACDC20 as pixel-aligned targets.
4. Resolve whether RESIDE-S is available as a reproducible archive before
   considering it; do not treat the regular RESIDE haze benchmark as RESIDE-S.
5. Put pairs into the documented layout and run:

   ```bash
   python scripts/build_manifests.py --root datasets/AIR40K/raw --output data/manifests
   python -m pytest tests/ -q
   python train.py --config configs/pilot.yaml
   ```

6. Inspect the first batch and checkpoint, then run inference/evaluation using
   the README commands. Record dataset versions, split counts, hardware, and
   actual metrics in a new log entry.

## 2026-11-02T15:00:44+05:30 - Architecture correction and dataset preparation started

- Inspected supplied paper Figures 4/5, Equations 10-13 and implementation settings.
- Authors repository https://github.com/chdwyb/AIRFormer returned 404; IEEE article and candidate supplementary endpoints inaccessible to automated access. Missing numerical specification remains unresolved.
- Corrected four encoder positions and three previous-feature priors injected between subsequent embeddings and blocks; first SP receives two zeros.
- Corrected full-HW wavelet Q, reduced-grid C-width K/V, independent IDWT C/4 branch, concatenation before output projection and pre-attention LayerNorm.
- Strict attention defaults to unpooled K/V using full SDPA; pilot pooling is an explicit resource approximation. Kept assumed query composition, SP internals, reconstruction and loss rather than inventing author settings.
- Updated configurations and test fixtures for structurally incompatible four-position model. Began extraction and integrity checks of every local archive, including newly present haze, clean, raindrop and complete-looking Snow100K training files.
- Initial regression run: 58 passed, four old-graph fixtures failed. Updating axis/route tests to validate corrected graph, including manual softmax equivalence.

## 2026-11-02T15:06:17+05:30 - Corrected architecture validation

- Regression suite: 65 passed, zero failed/skipped, including CUDA AMP, optimizer recovery, fresh CLI training and resume. Ruff F/E9 and git diff --check passed.
- Re-ran scripts/verify_paper_architecture.py: four encoder positions, three decoder skip stages, 2,762,691 parameters versus published 58.34M. Saved actual shape/attention/query-gradient traces in docs/architecture/verification.json. Full SDPA agrees numerically with explicit softmax on diagnostic inputs.
- Added regression checks for exact preceding feature/prior identities, pre-block addition, strict/pilot distinction, full-resolution wavelet Q and gradients through attention plus independent IDWT concat branches.
- Rewrote architecture README around current corrections/assumptions and marked old audit/training/compliance reports historical. Loss operations retained; model semantic AST changed as expected.
- All eight archives extracted without detected extraction errors. Full pair decoding, split grouping and leakage checks remain in progress.
- Started fresh real-image diagnostic optimization across rain/raindrop/snow, plus batch-one 256-patch feasibility on local RTX 2000 Ada. No previous checkpoint is loaded; this is not a full research-training run.

## 2026-11-02T15:10:40+05:30 - Windows attention memory adaptation

- A real batch-one 256x256 strict forward/backward initially failed: installed Windows PyTorch has no Flash Attention kernel and attempted a 64 GiB attention allocation.
- Added exact query chunking with per-chunk gradient checkpointing. All keys remain available to every query; no pooling or changed softmax domain is introduced. Added output/input-gradient/parameter-gradient equivalence tests for spatial and wavelet modes.
- Updated complete regression result: 66 passed, zero failed/skipped. Recomputed architecture evidence; Ruff F/E9 and diff whitespace checks pass. Retrying the 256-patch system check with the exact chunked path.
- Fresh real-data diagnostic: 12 successful AdamW/AMP updates from scratch over four rain, four raindrop and four snow pairs with finite gradients; no diagnostic checkpoint or temporary training dataset was saved.
- Added configs/local_available.yaml for an explicitly resource-adapted source-release experiment once its validated manifests finish. This is not an exact-paper training config.

## 2026-11-02T15:13:35+05:30 - Strict 256-patch feasibility and methodology review

- Fresh strict 256x256, batch-one real-image forward/backward/AdamW update passed using exact query chunking: 99.774 seconds; 0.9466 GiB peak allocated CUDA memory. Updated docs/architecture/system_validation.json, retaining the initial OOM observation. This is not converged research training.
- Confirmed unchanged loss semantic fingerprints against the prior baseline. Query gradients are finite and nonzero in channel/spatial/wavelet modes, including optional pooled variants.
- Rechecked published methodology against current AdamW betas/decay, cosine LR endpoints, paired crops/flips/90-degree rotations/MixUp, DCT/Smooth-L1 implementation, centered multiple-of-128 test padding, and skimage PSNR/SSIM. Added a methodology table with remaining assumptions and pilot deviations.
- Visual spot check of one raindrop, snow and quarantined haze pair found corresponding scene structure; this is a spot check, not proof of registration for every pair. Temporary contact-sheet image scheduled for deletion after review.
- Requested source/release/split identity for the locally added haze/clean/transmission archives; without it they stay outside training.

## 2026-11-02T15:23:27+05:30 - User-directed custom 80/20 experiment

- User requested training/testing only with an 80/20 split, no separate validation dataset, and offered to obtain additional exclusively-training releases.
- Added per-source identity-group splitting using scene IDs, clean-pixel hashes and input-byte hashes. Known official test releases stay separate; user-authorized local haze pairs enter a clearly labeled custom experiment despite unresolved original release/split identity. Clean-only ITS images and transmission maps are not RGB training pairs.
- Added a trainer mode with no validation/model selection. It saves the final training checkpoint and runs held-out testing only after the configured training schedule. Test metrics never choose a best checkpoint. Updated configs/local_available.yaml accordingly.
- Added integration coverage for train/test-only training and a regression test for grouping renamed clean-image duplicates. Complete extraction/decode audit is finishing; applying the 80/20 repartition after its initial manifest write.


## 2026-11-02T15:47:13+05:30 - Architecture, dataset and cleanup recheck

- Rechecked Figures 4/5 and Equations 9-13 against the corrected four-position encoder, pre-block recurrent priors, strict full-key attention, full-resolution wavelet queries, separate IDWT branch, concatenation and pre-attention normalization. No new author query, SP, tail or loss specification was recovered; loss/model math retained. Replaced stale architecture-assumptions text and documented the output-clamp assumption.
- Revisited author homepage, linked repository and IEEE article: author email confirmed, repository still 404, publisher automated access challenged; no supplementary specification recovered and no author message sent. Updated source-status evidence.
- Verified current train/test/official-test manifests: 58,766 / 14,695 / 1,412 pairs, all 127,746 referenced files present, zero cross-split scene IDs. Earlier full extraction/decode audit remains preserved; no redundant extraction or all-image decoding claimed. ITS clean-only data remain unpaired; transmission maps excluded from RGB targets.
- Fixed preparation to stop before rewriting manifests when required archives are missing or fail verification, preventing stale extracted data from entering a new manifest. Added missing/corrupt-archive regression coverage.
- Fresh CUDA AdamW optimization passed on real rain, raindrop, snow and haze pairs using the 2,762,691-parameter strict model and 64x64 crops; all gradients finite and all query gradients nonzero. Appended evidence to system_validation.json. No checkpoint saved; this is diagnostic optimization, not full research training.
- Validation: 70 tests passed with zero failures/skips; Ruff F/E9 and git diff whitespace checks passed. Parameter count remains 2,762,691 versus published 58.34M. Exact reproduction training awaits missing specification/membership; the custom experiment remains prepared.
- Removed temporary contact sheet, four redundant intermediate page-5 renders, pytest/Ruff caches and source bytecode caches. Preserved source tests, datasets, archives, canonical figure crops, audit evidence and existing checkpoints.


## 2026-11-02T15:51:34+05:30 - Training job

- Started fresh training with configs/rain_raindrop.yaml; PID 28156; output D:\Multi_adverse_weather_image_enhancement\checkpoints\rain_raindrop. Training and test membership are fixed by the config.


## 2026-11-04T12:49:18+05:30 - Training job

- Started fresh training with configs/four_weather.yaml; PID 22644; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261104. Training and test membership are fixed by the config.


## 2026-11-04T13:04:00+05:30 - Training job

- Training failed: configs/four_weather.yaml; exit code 1073807364; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261104. See training.log and status.json for evidence.


## 2026-11-04T13:06:24+05:30 - Training job

- Started fresh training with configs/four_weather.yaml; PID 22728; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261104_restart. Training and test membership are fixed by the config.


## 2026-11-04T13:10:47+05:30 - Training job

- Training failed: configs/four_weather.yaml; exit code 1; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261104_restart. See training.log and status.json for evidence.


## 2026-11-04T13:10:47+05:30 - Training job

- Started fresh training with configs/four_weather_fast.yaml; PID 20980; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261104_fast. Training and test membership are fixed by the config.


## 2026-11-04 - Four-weather throughput optimization

- User requested reducing training time. Benchmarked identical model/crops/AMP at effective batch size 8: batch 2/accumulation 4 = 10.08 images/s; batch 4/accumulation 2 = 14.77 images/s; batch 8/accumulation 1 = 17.38 images/s. Synthetic GPU-only timings exclude data loading and final evaluation; evidence in outputs/training_batch_benchmark.json.
- Deliberately stopped the restart job during startup (no checkpoint available), then launched configs/four_weather_fast.yaml into checkpoints/four_weather_20261104_fast. All datasets, split membership, 20 epochs and model/loss settings retained. Projected GPU computation is 18.8 hours; actual end-to-end time remains unmeasured.


## 2026-11-04T16:35:54+05:30 - Training job

- Training failed: configs/four_weather_fast.yaml; exit code 1; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261104_fast. See training.log and status.json for evidence.

## 2026-11-04 - User-requested pause after epoch 3

- Verified latest.pt contains completed epoch 3 and full model/optimizer/scheduler/scaler/RNG state; preserved identical SHA-256 copy as checkpoints/four_weather_20261104_fast/epoch_003.pt.
- Stopped training process 20980 and its children. Epoch 4 was already in progress; that partial epoch was intentionally discarded to resume from epoch 4 as requested. No dataset changes.
- Updated run status to paused with completed_epochs=3, next_epoch=4 and the resume command. The launcher's failed exit entry reflects deliberate process termination, not a training error.



## 2026-11-05T10:06:51+05:30 - Training job

- Started training (resume from D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261104_fast\epoch_003.pt) with configs/four_weather_fast.yaml; PID 26532; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261105_resume. Training and test membership are fixed by the config.


## 2026-11-05T16:45:11+05:30 - Preserved epoch 10 checkpoint

- Saved checkpoints/four_weather_20261105_resume/epoch_010.pt; verified completed epoch 10 and identical SHA-256 013b1fb8a5ab2595ff3f58abf5bdd6794c2793fdfff26ddba8dff37cc3ccd8dd. Full training state retained. Training continues without interruption.


## 2026-11-05T16:46:03+05:30 - Training job

- Training failed: configs/four_weather_fast.yaml; exit code 1; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261105_resume. See training.log and status.json for evidence.


## 2026-11-05T16:46:16.656213+05:30 - User-requested pause after epoch 10

- Stopped training PID 26532 and its workers. Preserved verified epoch_010.pt with full training state. Partial epoch 11 discarded; resume starts at epoch 11. Updated paused status and resume command.


## 2026-11-06T10:36:40+05:30 - Training job

- Started training (resume from D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_20261105_resume\epoch_010.pt) with configs/four_weather_fast.yaml; PID 36048; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_resume_epoch11. Training and test membership are fixed by the config.


## 2026-11-06T16:21:51+05:30 - Training job

- Training failed: configs/four_weather_fast.yaml; exit code 1; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_resume_epoch11. See training.log and status.json for evidence.


## 2026-11-06T16:22:09.816821+05:30 - User-requested pause after epoch 16

- Preserved checkpoints/four_weather_resume_epoch11/epoch_016.pt; verified completed epoch 16, full model/optimizer/scheduler/scaler/RNG state and SHA-256 31103c8918601bccd0a192ec2f0d5fab3a805f2682ede9dfeb898c2facdb74e5.
- Stopped training PID 36048 and workers; partial epoch 17 discarded. Resume starts at epoch 17 with the same 20-epoch schedule. Supervisor exit code 1 reflects deliberate termination. Resume command recorded in run status.json.


## 2026-11-07T11:57:44+05:30 - Training job

- Started training (resume from D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_resume_epoch11\epoch_016.pt) with configs/four_weather_fast.yaml; PID 18536; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_resume_epoch17. Training and test membership are fixed by the config.


## 2026-11-07T15:46:44+05:30 - Training job

- Training failed: configs/four_weather_fast.yaml; exit code 1; output D:\Multi_adverse_weather_image_enhancement\checkpoints\four_weather_resume_epoch17. See training.log and status.json for evidence.
- Clarification verified on 2026-11-09: the supervisor labeled the combined training/testing job as failed. All 20/20 training epochs completed and `latest.pt` contains epoch 20 with the configured 20-epoch schedule. The subsequent final-test phase failed in CUDA attention with `RuntimeError: cutlassF: no kernel found to launch!`. This exit code does not mean the saved training was lost or needs to be repeated.


## 2026-11-09 - Final-test error diagnosis and repair

- Reproduced the CUDA failure with the saved epoch-20 model on a full-resolution 512 x 384 test image. Channel attention uses the spatial area as its embedding dimension; the selected fused CUDA kernel could not handle this shape.
- Updated `models/attention.py` to use PyTorch's math SDPA backend for CUDA channel attention, preserving attention scale and model weights. Verified the previously failing image succeeds with the original epoch-20 checkpoint.
- Added regression coverage for full-resolution CUDA inference and CPU/CUDA channel-attention output and gradient agreement. Verified all 11 attention, model, and training-integration tests pass.
- Ran a validation smoke check with one real pair from each weather group. This was a diagnostic check, not the full test result. No retraining or test-based model selection was performed.


## 2026-11-09 - Full testing and automatic report preparation

- Added `scripts/test_restoration.py` for full-resolution FP32 evaluation of the fixed final checkpoint, with per-image progress saved to disk, resume checks, checkpoint/data/code SHA-256 provenance, and automatic reports. Added matplotlib to `requirements.txt` and installed it in the project virtual environment.
- Prepared PSNR, SSIM, MAE, MSE, RMSE, loss, inference timing, input-baseline comparisons, and PSNR/SSIM gains overall and by weather. The experiment remains the custom four-weather split; haze is from the pinned unofficial ITS mirror, not an official SOTS benchmark.
- User explicitly requested auxiliary pixel-error mask scores after clarification that the restoration model does not predict weather classes. Prepared pixel-pooled accuracy, precision, recall, F1, and confusion matrices at fixed thresholds 0.05, 0.10, and 0.20; 0.10 is the primary diagnostic. True mask: mean absolute RGB(input - target) exceeds the threshold. Predicted mask: mean absolute RGB(input - restored) exceeds the threshold. These assess correction locations, not weather classification or restoration success.
- Prepared PNG/SVG quality-by-weather charts, improvement distributions, baseline comparisons, auxiliary confusion matrices, and worst/median/best examples with output-error heatmaps. These are generated automatically after the complete test set finishes, alongside `summary.json`, `final_test.json`, `report.md`, and `report.html`.
- Preflight fully decoded all 14,695 input/target pairs, verified matching sizes and test split labels, rejected duplicate test inputs, and checked train/test path and scene separation. The largest padded image shape in every weather group passed inference, metric, and loss checks. Training had previously checked clean-pixel hashes for split leakage.
- Added two evaluator tests covering known auxiliary-mask counts, undefined scores, complete report generation, resume behavior, and rejection of checkpoint changes. These passed; together with the 11 model/training checks, 13 tests passed. Inspected generated fixture charts to verify rendering; fixture previews were removed and are not real model results.
- Started the full evaluation at 2026-11-09T10:23:20+05:30 with evaluation PID 14080, using `checkpoints/four_weather_resume_epoch17/latest.pt`. Results are written to `outputs/four_weather_test_epoch20_20261109`; the original failed job history is retained.
- Test composition: rain 2,520; raindrop 173; snow 10,002; haze 2,000. At 2026-11-09T10:28:10+05:30, 775/14,695 images had completed without a recorded error, at approximately 3.13 images/second. Estimated remaining time at that point was 74 minutes and can vary with image sizes. Full-test scores are pending; see the evaluation's `status.json` and log files for live progress.

- Started scripts/log_evaluation_completion.py to append the evaluation's actual completed/failed outcome to LOG.md automatically when testing ends. This watcher reads status only and does not interrupt evaluation.


## 2026-11-09T12:34:55+05:30 - Full model testing completed

- Evaluation output: `outputs/four_weather_test_epoch20_20261109`. Status evidence: `status.json`; logs: `evaluation.stdout.log` and `evaluation.stderr.log`.
- All 14,695 test images evaluated. Mean restored PSNR: 25.6356 dB; SSIM: 0.805444; mean PSNR gain: +4.9635 dB.
- Final metrics, auxiliary pixel-mask scores (not weather classification), and visualizations are available in `summary.json`, `final_test.json`, `report.md`, `report.html`, and `visualizations/` within the evaluation output.


## 2026-11-09T14:01:40+05:30 - Results README, paper comparison, and measured durations

- Created RESULTS_README.md with verified full-test metrics, per-weather results, auxiliary pixel-mask definitions/scores/confusion matrix, checkpoint/configuration details, visualizations, and measured training/testing durations.
- Created PAPER_COMPARISON.md comparing the authors' datasets, architecture, settings, published all-in-one scores, hardware, and timing with the completed local experiment. Visually checked supplied PDF pages 5, 7, 8, and 10; did not claim a controlled benchmark or fabricate unreported paper training durations.
- Verified the actual trained architecture has 612,435 trainable parameters. Final metrics: PSNR 25.635622 dB, SSIM 0.805444, and PSNR gain +4.963452 dB over degraded inputs on 14,695 pairs.
- Calculated retained training-session wall time as 70,710.027 seconds (19 h 38 min 30 s), including startup/discarded partial epochs/failed final-test attempt; pure training compute is not separately measured. Final testing job took 7,865 seconds (2 h 11 min 5 s). Saved exact calculations in outputs/four_weather_test_epoch20_20261109/duration_summary.json.
- Generated training-loss PNG/SVG from all 20 completed epoch summaries. Updated README.md and historical training/reproduction-document banners to point to completed results. Verified both new documents' local links, the complete per-image record count, successful evaluation status, chart rendering, and git diff whitespace checks. No model retraining or evaluation rerun was performed for this documentation request.


## 2026-11-09T14:09:02+05:30 - Consolidated project Markdown documentation

- Moved root LOG.md, TRAINING_README.md, RESULTS_README.md, and PAPER_COMPARISON.md into docs/. Moved the generated test report to docs/reports/four_weather_test_epoch20_20261109/report.md. Files named README.md remain in their existing locations; existing docs/ organization is preserved.
- Updated affected Markdown/image links and the browser report's detailed-report link. Updated training/evaluation logging to write docs/LOG.md and report generation to place future project Markdown reports under docs/reports/.
- Verified all local link targets across 17 main/docs Markdown files, the browser report link, and future report routing. Both evaluator regression tests passed. Original model weights, metrics, and recorded durations were retained.


## 2026-11-09T14:25:23+05:30 - Master architecture-to-code and model-lifecycle learning guide

- Created docs/MASTER_ARCHITECTURE_AND_MODEL_LIFECYCLE.md with an 18-section learning path: paper-to-code mapping, directory responsibilities/import aliases, actual trained configuration, encoder/decoder tensor shapes, recurrent priors, Haar transforms, queried attention, DCT/Smooth-L1 objective, data engineering, training/AMP, checkpoint/resume behavior, full evaluation, inference, verification, lifecycle history, measured results/durations, and guide questions.
- Preserved the distinction between paper-supported topology and implementation assumptions, pooled attention and exact query chunking, model restoration and auxiliary mask metrics, and measured wall time versus pure compute. Included details often missed in explanations: first-prior zero inputs, separate wavelet IDWT branch, replacement sampling, output clamping, and the unweighted orthonormal DCT norm interpretation.
- Loaded the saved epoch-20 weights for a CPU evaluation-only synthetic 128 x 128 trace; recorded module input/output shapes, attention shapes, three prior routes, and the 612,435-parameter breakdown in docs/architecture/trained_model_trace.json. No weights were updated and no training/full-test run was repeated.
- Linked the master guide from README.md. Verified every master-guide local link and section anchor, shape/parameter evidence, final result counts/PSNR, a numerical check of the DCT norm explanation, and git diff whitespace checks.
