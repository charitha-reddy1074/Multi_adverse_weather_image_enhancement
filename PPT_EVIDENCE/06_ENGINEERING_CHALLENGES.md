# Engineering challenges and verified outcomes

Historical defects below are not descriptions of the current corrected graph. Tests establish local behavior, not paper equivalence.

## Architecture reconstruction

**Problem:** Main paper provides graph and equations but incomplete numerical specification.

**Investigation:** Compared Figures 4/5, Eq.9/10 and executable branches; documented uncertain settings.

**Fix:** Independent modular implementation with explicit assumptions, not a claimed exact clone.

**Verified outcome / evidence:** Current source, docs/architecture/README.md and architecture_assumptions.md; checkpoint strict load.

## Encoder position count

**Problem:** Historical pilot implementation used fewer positions.

**Investigation:** Architecture review identified the four positions drawn in Figure 4.

**Fix:** Current model has four stages and three downsamplers; historical counts are superseded.

**Verified outcome / evidence:** models/airformer.py; models/encoder.py; trained_model_trace.json shows all four stages.

## Supplementary-prior routing

**Problem:** Earlier routing did not reflect recurrent pre-block injection.

**Investigation:** Reviewed Eq.10 and tested zero first inputs / preceding-feature recurrence.

**Fix:** Current priors are added after downsampling before subsequent encoder blocks. Internals remain approximate.

**Verified outcome / evidence:** models/encoder.py; docs/architecture/trained_model_trace.json; tests/test_architecture_fidelity.py.

## Query and wavelet paths

**Problem:** Early diagram interpretation omitted or misrouted wavelet branches.

**Investigation:** Inspected Q/K/V axes and separate inverse path; gradient checks recorded.

**Fix:** Current attention normalizes before DWT, retains full-grid queries, projected wavelet K/V and IDWT concatenation. Query composition is still assumed.

**Verified outcome / evidence:** models/attention.py; docs/architecture/verification.json and system_validation.json; tests/test_attention.py.

## Pairing and leakage

**Problem:** Source variants can share clean scenes, making random pair splits leak.

**Investigation:** Mapped dataset-specific filenames; grouped scene, decoded clean pixels and input bytes.

**Fix:** Manifests/preflight report zero scene overlap and no missing referenced files; near duplicates remain unresolved.

**Verified outcome / evidence:** datasets/splitting.py; data/manifests/four_weather/preflight.json; docs/LOCAL_DATASET_AUDIT.json.

## CUDA memory pressure

**Problem:** Full native-grid spatial attention is costly for local hardware.

**Investigation:** Diagnostic workload and memory records preceded the smaller experiment.

**Fix:** Trained small model enables pooled K/V with a 256-token cap. This is an approximation, not a measured no-cost equivalent.

**Verified outcome / evidence:** outputs/training_batch_benchmark.json; configs/four_weather_fast.yaml; docs/LOG.md.

## AMP overflow

**Problem:** Early FP16 attempts overflowed scaled gradients.

**Investigation:** Training log records skipped optimizer updates and reducing scales.

**Fix:** GradScaler skips overflow updates and recovers; final epoch losses and weights are finite.

**Verified outcome / evidence:** scripts/train.py; retained training.log; docs/OPTIMIZATION_LOG.md. Diagnostic measurements are not final benchmarks.

## Gradient and model checks

**Problem:** Complex transforms/queries need gradient and branch validation.

**Investigation:** Historical tests and real-data diagnostic steps inspect finite gradients and task queries.

**Fix:** Saved reports support implemented behavior; not author-equivalence or final ablation evidence. No training tests rerun in this extraction.

**Verified outcome / evidence:** docs/architecture/system_validation.json; outputs/paper_review/architecture_tests.xml; tests/test_wavelet.py.

## Checkpoint recovery / pauses

**Problem:** Long sessions were interrupted between retained epoch boundaries.

**Investigation:** Verified checkpoint epoch/config and resume log entries.

**Fix:** Resume restores model, AdamW, scheduler, scaler and RNG across 3 -> 10 -> 16 -> 20 retained epochs.

**Verified outcome / evidence:** scripts/train.py; four retained training logs; epoch_003.pt / epoch_010.pt / epoch_016.pt / final latest.pt.

## Final GPU testing crash

**Problem:** First post-training full-size test failed with cutlassF: no kernel found to launch.

**Investigation:** Channel attention treats HW as embedding length; CUDA fused backend has shape restrictions.

**Fix:** Current channel attention chooses SDPA MATH on CUDA; a separate final full test completed all 14,695 samples. No retraining.

**Verified outcome / evidence:** models/attention.py::attend; checkpoints/four_weather_resume_epoch17/training.log; final status/provenance.

## Historical status contradictions

**Problem:** A failed supervisor and older docs imply training/evaluation incomplete.

**Investigation:** Prioritized checkpoint contents and completed timestamped outputs.

**Fix:** 20 training epochs and final held-out evaluation verified; historical failures retained transparently.

**Verified outcome / evidence:** Final checkpoint, TensorBoard scalar history and outputs/four_weather_test_epoch20_20261109/status.json.

## Evidence boundaries

`docs/OPTIMIZATION_LOG.md` records older 62-test audit snapshots and random-weight timings. They cannot be relabeled as final-model research scores. Later architecture reports and saved model evidence describe the four-position graph. Current successful testing demonstrates the completed inference path works; it does not isolate which architectural component caused gains.

## Evidence hierarchy and identifiers

1. Saved epoch-20 checkpoint: `checkpoints\four_weather_resume_epoch17\latest.pt`. SHA-256 `523b2f26b123d97e6b139729f378172cb6e46b328c1b9d1d5a6250e69f76403f`.
2. Completed evaluation: `outputs/four_weather_test_epoch20_20261109/status.json`, `summary.json`, `provenance.json`, and `per_image_metrics.csv`.
3. Stored checkpoint `config` and `model_config`, checked against `configs/four_weather_fast.yaml`.
4. Executable source in `models/`, `losses/`, `airformer/`, `datasets/`, `scripts/`, `utils/`.
5. Retained training logs and TensorBoard events, followed by dated documentation. Historical diagnostic runs are separate.

Paths are relative to `D:\Multi_adverse_weather_image_enhancement` unless explicitly absolute. Local source hashes and modification timestamps are in `sources/source_inventory.csv`. Packaged snapshots are in `sources/`. This extraction generated documentation/assets and twelve inference examples only. It did not train, change weights, alter datasets, or repeat full evaluation.
