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
