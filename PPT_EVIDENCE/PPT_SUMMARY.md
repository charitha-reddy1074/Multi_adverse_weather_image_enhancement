# AIRFormer research PPT evidence package

Completed evidence package for a fifteen-slide M.Tech guide review. Start with `07_SLIDE_CONTENT.md`, then use the matching image assets. `08_VIVA_QA.md` contains 37 implementation-grounded questions and answers. This is the requested evidence package; no PPTX is included.

Verified final state: **20 completed epochs**, **612,435 trainable parameters**, **14,695 held-out pairs**, **PSNR 25.635622 dB**, **SSIM 0.805444**, **gain +4.963452 dB**, **mean MAE reduction 49.77%**.

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

- Independent paper-guided reconstruction, not certified author-equivalent AIRFormer. Complete supplementary numerical settings and author implementation are unavailable in the repository evidence.
- Four encoder positions and three prior routes are verified, but widths/depths, additive learned queries, Haar choice, prior fusion and reconstruction tail are local interpretations.
- Trained spatial/wavelet attention pools K/V to 256 tokens, changing the attention graph. No matched ablation attributes quality gains to individual components.
- Custom grouped local split, not AIR40K or the paper's official benchmark tests. Snow dominates the test mean (10,002 of 14,695). No validation split or validation-based checkpoint selection.
- Haze bytes match a pinned unofficial RESIDE-ITS mirror `val` revision, not proof of canonical release attribution or official ITS/SOTS split identity. Near-duplicate leakage is not ruled out by exact hashes.
- Raindrop mean PSNR and SSIM rise slightly, but mean MAE increases. A good aggregate does not guarantee improvement in each sample or metric.
- Local FLOPs are unavailable. Variable-resolution local inference time cannot be compared as a speed ratio with paper 256x256 RTX 3090 timing.
- Unweighted orthonormal DCT L2 preserves error energy. It does not mathematically weight high frequencies more heavily.
- Evaluator source hash differs from final provenance after documented report-routing edits. Original evaluator bytes were not recovered. Checkpoint, both manifests, attention and inference hashes match; recorded metrics reconcile and selected predictions reproduce them.


Absolute package path: `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE`. ZIP: `D:\Multi_adverse_weather_image_enhancement\PPT_EVIDENCE\AIRFORMER_PPT_EVIDENCE.zip`.
