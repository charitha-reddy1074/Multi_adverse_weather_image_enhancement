# AIRFormer audit - 2026-10-31

**NOT FULLY VERIFIED.** This is an independent reconstruction, not a demonstrated
reproduction of the authors' model. No full training has been launched.

Current follow-up verdict: **architecture comparison complete; implementation
NOT EXACT**. User supplied the PDF after the initial five-iteration audit.
The missing-PDF blockers and no-actionable-high-finding conclusions below are
historical observations, superseded by the new confirmed architecture findings
in PAPER_COMPLIANCE and the illustrated architecture/README.md.

## Repository inventory

All tracked source/config/test/document files were inspected before editing.
`models/` holds patch embedding, three encoder levels, SP, queried attention,
two decoder stages and RGB reconstruction. `losses/multidomain.py` is the
canonical loss; `airformer/model.py`, `airformer/losses.py` and
`datasets/weather_dataset.py` are compatibility imports, not duplicate models.
`airformer/data.py` supplies data loading/augmentation. `scripts/` provides
training, inference, evaluation, profiling, visualization and dataset tools.
Root train/evaluate/inference files are entrypoint wrappers. Existing tests
numbered 31 on CPU. No original checkpoints, experiment results, image datasets,
PDF or `.dist/paper.txt` were present at audit start. Git was clean.

The runtime initially had Python 3.14.6, CPU torch 2.13.0, torchvision 0.28.0,
and no TensorBoard/security scanners. NVIDIA RTX 2000 Ada Generation has 16 GiB;
the driver supports CUDA 13.2. `.venv/` now isolates project packages and has
torch 2.13.0+cu130 / torchvision 0.28.0+cu130. System Python was not modified.

## Iteration 1 - baseline and executable pipeline defects

### A. Research compliance

Tensor inspection confirms all three attention modes execute distinct operations
in CHANNEL -> SPACE -> WAVELET order (`models/attention.py`, `models/decoder.py`).
Actual paper comparison is BLOCKED: no PDF was supplied. DOI resolution was
attempted; IEEE direct PDF endpoints returned HTTP 418. Existing documentation
claiming an available extraction is stale. See `PAPER_COMPLIANCE.md`.

### B. Dataset status

At baseline all required datasets were MISSING: zero image files/archives.
`datasets/sources.csv` contained project pages rather than usable archives.
Official source pages were consulted; acquisition results are recorded separately.

### C. Security and correctness findings

| Severity | File / cause | Correction | Evidence |
|---|---|---|---|
| HIGH | `airformer/data.py` local TF imports shadow the module import; normal crops fail with UnboundLocalError | Removed two local imports | Baseline pytest: 1 failed / 30 passed; rerun 31 passed |
| HIGH | `run_train.bat`, `run_train.sh` use datasets/test for checkpoint selection | Launch the pilot config and its separate val manifest | Source inspection and train split guards |
| HIGH (unsafe default on older supported torch) | `utils/inference.py`, `scripts/train.py`, `scripts/profile_model.py` leave checkpoint unpickling implicit | Explicit weights_only=True, deserialize to CPU | Roundtrip test and restricted loader inference test |

### D. Optimization

No architecture/tensor operations changed. Baseline profiler later measured
271,155 pilot parameters / 1,146,883 paper-config parameters at 32x32.
This scale is not evidence of parameter parity with the original AIRFormer.

### E. Tests

31 passed, 0 failed, 0 skipped after the loader fix. Evidence:
`outputs/audit_iteration1_tests.xml` (local ignored output).

### F. Remaining work

Acquire paper/data, validate real pairs, enable CUDA, extend behavioral tests.

## Iteration 2 - data integrity and evaluation safeguards

### A. Research compliance

CUDA enabled; existing suite: 32 passed, 0 failed, 0 skipped. Encoder/SP/query
composition and reduced K/V attention remain documented reconstruction choices.
No speculative architecture rewrite was performed.

### B. Dataset status

Rain12 (7,351,249 bytes) and official Rain14000 (898,292,342 bytes) downloaded.
Rain14000 archive README specifies 900 training / 100 testing clean scenes,
each with 14 rainy variants: 12,600 / 1,400 pairs, not AIR40K's 11,200 / 2,800.
Snow100K training transfer started. Other source failures are in DATASET_STATUS.

### C. Security and correctness

`airformer/data.py` now rejects unpaired folder records.
`scripts/evaluate.py` now rejects incomplete identity sets and mismatched sizes
instead of reporting metrics on a silent subset; profiling inputs use the
model device. Shared `datasets/pairing.py:image_map` rejects duplicate image
identities across extensions. `scripts/train.py` rejects missing configured
manifests rather than silently using another dataset.

### D. Optimization

Consolidated three image-map implementations into `datasets/pairing.py`.
No model/loss tensor changes; no claimed inference speedup or VRAM reduction.

### E. Tests

`outputs/audit_iteration2_tests.xml`: 32 passed / 0 failed / 0 skipped.

### F. Remaining work

Add full pixel decoding, archive safety, query/SP behavioral tests and leakage
guard across separate train/val manifests.

## Iteration 3 - behavioral coverage and isolated dependencies

### A. Research compliance

Extended tests verify SP output contribution and finite nonzero stage gradients;
all three task query parameters update with AdamW and change output when disabled.
CPU and CUDA tiny-batch loss reduction passes. Checkpointed decoder output and
gradients match the ordinary decoder. See `tests/test_audit_pipeline.py`.

### B. Dataset status

`scripts/prepare_available_data.py` safely extracts available archives, decodes
pairs, groups all variants of a clean scene and reserves deterministic 5% of
official training scenes for validation. Official tests remain held out.
Full validation is pending completion at this report entry.

### C. Security

`datasets/archive.py` rejects traversal, Windows drives/ADS, links and differing
existing files; verifies matching existing members for restart safety.
`datasets/pairing.py:assert_disjoint_datasets` rejects scene/path/pixel-identical
clean overlap even across renamed sources. Training uses this guard and rejects
test-labelled records. Generic manifest builder now fails on invalid/unpaired
data before publishing manifests. Validation uses inference padding to 128.

Initial dependency scan of inherited Anaconda found 202 advisories in 28
packages, most unrelated to this project. Project was isolated; relevant image,
network and packaging dependencies were patched. Final isolated scan:
`outputs/pip_audit_isolated.json`: no known vulnerabilities; CUDA-tagged torch
cannot be queried on PyPI and is an explicit scan limitation.

### D. Optimization

Removed unused imports and duplicated mapping code. Test-only thread cap avoids
oversubscription on tiny fixtures; no training thread policy or model math changed.

### E. Tests

`outputs/audit_iteration3_tests.xml`: 56 passed, 0 failed, 0 skipped.
Ruff initially identified three unused test imports; removed without behavior change.

### F. Remaining work

Finish real-data validation, CLI training/resume/inference/evaluation smoke tests,
machine-readable benchmarks and final static scans. Paper remains unavailable.

## Iteration 4 - real-data integration and AMP correction

### A. Research compliance

No `models/` or `losses/` tensor code changed. Real image restoration preserves
input dimensions, and RGB reference metrics use scikit-image without target
resizing. Query/SP/attention order/gradient tests remain passing. Architecture
verification against actual Figure 4/5 remains BLOCKED by unavailable PDF.

### B. Dataset status

`data/manifests/integrity.json` and copied `docs/DATASET_INTEGRITY.json` record
14,012 decoded/aligned-size pairs, no split leakage, 11,970 train / 630 val /
1,412 test. Archive SHA-256 values are recorded. The 5% validation reservation
uses whole clean scenes, keeping all 14 variants together. Source release
counts remain distinct from AIR40K claims.

### C. Security and numerical findings

| Severity | File/line | Root cause | Fix and verification |
|---|---|---|---|
| HIGH | `scripts/train.py:183` | FP16 initial GradScaler overflow hit unconditional error_if_nonfinite=True before scaler could recover | FP32 still fails on nonfinite gradients; AMP lets scaler skip overflowed updates and reduce scale; CUDA regression forces overflow and verifies weights remain finite and later update succeeds |
| MEDIUM | `scripts/train.py:140` and RNG save block | Resume restored weights/optimizer but not random states | Save/restore Python, NumPy (primitive list serialization), torch, CUDA and DataLoader generator states; CLI resume test reads weights_only checkpoint and reaches next epoch |
| MEDIUM | `scripts/train.py:27` | Validation bypassed inference multiple-of-128 padding | Reuse inference tensor helper, then crop before loss/metrics; real-image CLI validation succeeds |
| MEDIUM | `scripts/download_datasets.py:18` | Catalog could supply unsupported URL schemes; redirects were unchecked | HTTPS-only source check, requests HEAD without following redirects; final Bandit medium count zero |

Main pipeline defects fixed earlier are now enforced by startup guards at
`scripts/train.py:92`, `scripts/train.py:98`, `scripts/train.py:101` and strict
evaluation checks `scripts/evaluate.py:21` / `scripts/evaluate.py:29`.
Restricted checkpoint calls are at `utils/inference.py:72`,
`scripts/profile_model.py:46`, `scripts/train.py:140`.

### D. Optimization and benchmark

`docs/AUDIT_BENCHMARK.json`: fixed-seed random pilot model, CUDA FP32 inference,
input/output [1,3,128,128], 271,155 parameters, 0.023502 seconds/inference over
3 timed runs after 2 warmups; peak allocated tensor memory 89,984,000 bytes.
AMP training mean 1.110758 seconds/attempt including scale adaptation, peak
allocated 219,691,520 bytes. Six attempts produced three successful finite steps;
losses 3.789946 -> 3.104602 -> 2.434709; scale settled at 8192.
The measured training time includes cold/backward/scaler overhead and is not
a steady-state throughput claim. No before/after speedup is claimed.

### E. Tests and real-image execution

56 tests passed, 0 failed, 0 skipped (`outputs/audit_iteration4_tests.xml`).
`outputs/audit_smoke/commands.log` records two real training pairs / one separate
validation pair, one tiny epoch, resume for another tiny epoch, inference on a
held-out Rain12 image and strict RGB evaluation. Diagnostic measurement:
PSNR 16.754061 dB, SSIM 0.263462 for one image. This is a tiny nearly untrained
19,743-parameter fixture, **not a benchmark score or useful pretrained model**.
Its poor output was visually inspected (`outputs/audit_smoke/visuals/sample.png`).
The resume smoke changes the epoch horizon and tests restoration/wiring; it does
not establish equality to an uninterrupted two-epoch schedule.

### F. Remaining work

Final broader tests, source-duplicate audit, documentation correction and bounded
recheck. No full training or additional blocked downloads will be looped blindly.

## Iteration 5 - final recheck and explicit stop

### A. Research compliance

Added actual attention probability-shape/normalization checks:
channel [1,2,4,4], spatial [1,2,63,16], wavelet [1,2,20,16] for the test fixture.
A remote spatial perturbation changes another location's output. Test queries
have nonzero finite gradients, update with AdamW and affect output. Wavelet
reconstruction and SP effects remain tested. `git diff -- models losses` is
empty: architecture and loss equations were preserved.

Paper config parameter count is 1,146,883, vastly different from the historical
58.34M reference in `scripts/profile_model.py`. That reference cannot itself be
verified without the paper. Do not change widths/depths to force a match.
The exact architecture remains BLOCKED; pooled K/V remains a DEVIATION.

### B. Dataset status

Rain14000/Rain12 are READY as their actual source releases. A complete input
byte-hash audit found 14,010 unique byte hashes among 14,012 paired samples:
Rain14000 `240_7.jpg` duplicates `240_6.jpg`, and `700_5.jpg` duplicates
`700_2.jpg`, both under `data/raw/extracted/Rain14000/rainy_image_dataset/training/rainy_image/`.
These are original-release within-training duplicates, preserved and documented
to avoid silently changing sampling. `docs/DATASET_DUPLICATES.json` records exact
paths. Cross-split duplicate clean pixels are rejected; no such overlap found.
All 15,024 source RGB images and 12 grayscale rain-layer images were inventoried;
the 12 grayscale layers are not restoration targets or model inputs.

Snow training transfer stalled at 587,726,848 bytes; it was stopped while
preserving the partial file. One changed-timeout resume attempt then failed
public-link retrieval. No failed download was restarted from zero. All other
required blocked endpoints and exact next actions are in DATASET_STATUS.

### C. Security

Final Bandit: no HIGH or MEDIUM findings. Remaining LOW findings are intentional
augmentation PRNG use (not cryptography), token_mixer names misidentified as
credentials, and audit-only assertions/subprocess calls with argument arrays,
no shell=True, trusted local entrypoints. Nothing was suppressed to pass checks.
Final isolated dependency audit: 67 records, zero known vulnerabilities among
auditable packages; CUDA torch and torchvision unavailable on PyPI are BLOCKED
for advisory-query coverage. `pip check` passes. No credentials were identified
in source. Download catalog paths/identifiers are checked before log/file writes.

### D. Optimization

Paper-config single-image 256x256 CUDA FP32 inference measured 0.205302 seconds,
2 timed runs after 1 warmup. No training memory claim for paper batch 8.
No unsupported FLOP estimate reported. Outputs/gradients of decoder checkpointing
are tested equivalent; shared image-map refactor preserves path identities and
adds explicit collision rejection. No memory/runtime gain is invented.

### E. Final tests

**62 total; 62 passed; 0 failed; 0 skipped** with CUDA available.
`outputs/audit_iteration5_tests.xml`; CPU/CUDA overfit, forced AMP overflow
recovery, real checkpoint loader and CLI training/resume tests included.
Ruff F/E9: all checks passed. `git diff --check` passed. PowerShell downloader
syntax and final dataset/provenance summaries are checked before handoff.
Syntax-only compilation is not substituted for runtime testing.

### F. Remaining work and stop condition

Stopped at the **five full audit iteration limit**. **NOT FULLY VERIFIED.**
Highest-priority blocker: obtain authorized paper PDF and inspect Figure 4/5,
Eq 9-13 and implementation settings. Exact next action: place the DOI-linked PDF
under `docs/papers/` and repeat only the blocked paper comparison; do not assume
historical notes are primary evidence. Separately obtain unavailable original
datasets through owner-supported access, validate dataset-specific scene/pair
mappings, then extend manifests with disjoint train/val/test scenes. Exact
AIR40K membership and five-image training-count discrepancy remain unresolved.

No actionable confirmed high/critical code defect remains in the tested paths.
Unknown paper mismatches, unavailable multi-weather data, unmeasured paper
training memory and absence of a fully trained checkpoint remain explicit
limitations. Full all-weather training/performance validation is not achieved.

Final dataset recheck reran preparation with byte-duplicate reporting enabled;
14,012 validated pairs, zero fatal issues, two preserved source-release
within-training duplicate inputs. `docs/DATASET_INTEGRITY.json` is the final
durable summary; `docs/AUDIT_RESULTS.json` records test/scanner totals.

## User-requested full main-paper architecture re-verification

This is a focused follow-up to the earlier bounded audit, not a claim that its
previously blocked architecture requirements have now been implemented.
The supplied PDF was inspected visually and textually: Figure 4 (PDF3/1888),
Figure 5 and Eq9 (PDF5/1890), Eq10-13 (PDF6/1891), training details (PDF8/1893),
Tables XII/XIII (PDF10/1895). Faithful figure crops, provenance, tensor traces
and the current verdict are in `docs/architecture/README.md` and
`docs/architecture/verification.json`. The updated requirement matrix is
`docs/PAPER_COMPLIANCE.md`.

### Research findings, prioritized

| Severity | Finding | Actual implementation evidence | Status/action |
|---|---|---|---|
| HIGH | A01: encoder topology differs from four depicted encoder positions | `models/airformer.py:27` fixes three levels; hooked output shapes 32/64/128 channels | DEVIATION; structural correction required, not performed during comparison |
| HIGH | A02: supplementary-prior stage/addition placement differs | `models/encoder.py:76` runs block before SP; diagram adds prior before subsequent block | DEVIATION; align graph and Equation10 recurrence |
| HIGH | A05: spatial attention differs from full token attention | `models/attention.py:85` caps K/V; observed logits [1,2,96,16] versus figure [1,2,96,96] | DEVIATION; strict path must exclude pooling adaptation |
| HIGH | A06: wavelet attention query, normalization and fusion differ | `models/attention.py:64` DWT before normalization; reduced Q; line100 IDWTs attention result, no separate concat branch | DEVIATION/MISSING figure route; rebuild graph before exact claim |
| HIGH | A10: configuration differs from reported model size | direct count1,146,883 versus visually verified TableXIII58.34M | DEVIATION; do not choose widths merely to force a number |
| MEDIUM | A04/A07/A08: query projection, SP internals, reconstruction and numerical settings not fully specified | assumed feature Q+bias, 6C SP fusion, bilinear skips and image residual | ASSUMED/BLOCKED exact equivalence; supplementary/original evidence needed |
| MEDIUM | A09: literal loss reduction differs | `losses/multidomain.py:51` mean elementwise SmoothL1 vs printed mean-residual then penalty | Code0.5 vs printed0 on +/-1 cancellation fixture; notation/intent unresolved |

Figure-reading precision: Figure 4 has four encoder block positions, but x4
labels the decoder output. Its x1/x2/x3 labels refer to encoder intermediates.
The README does not misidentify x4 as an encoder-output label.

Verified matches include shared weights, encoder normalization/mixer/MLP/residual
pattern, initial zero SP inputs, channel attention axis, active queries/priors,
wavelet invertibility, decoder order Channel/Space/Wavelet, and stated training
constants. Verified properties do not certify whole mismatching components.

### Changes and checks

- Added honest inline comments/docstrings to model and loss files; all nine
  model/loss semantic AST fingerprints match the pre-comment snapshot.
- Created `docs/architecture/README.md` with extracted Figures4/5, loss/table
  evidence, file walkthrough, tensor dimensions and required correction sequence.
- Added repeatable extraction and execution-diagnostic scripts. The source PDF,
  weights, model operations, dataset files/manifests and sampling are unchanged.
- 62 tests passed, zero failures/skips; Ruff F/E9 passed. The tests establish a
  functioning reconstruction, not conformance to the differing paper graph.
- No new optimization/security-sensitive production path or full training run
  was introduced. No dataset download/repartition was performed.

**Final re-verification result: complete review of the available main paper;
NOT an exact architecture implementation.** Confirmed HIGH research deviations
remain uncorrected because this request is for transparent comparison/comments,
and the user previously required preserving model logic. Their correction would
change the architecture and checkpoint layout. Unknown author-specific settings
cannot be filled in honestly from the main paper alone.
