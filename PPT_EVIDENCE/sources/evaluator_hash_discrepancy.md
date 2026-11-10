# Evaluator hash discrepancy

Recorded final-evaluation evaluator SHA-256: `ddca22e553a4a8171a032e4905404dc7a6fd122fbe71e6521fb953d703c7ff27`.

Current `scripts/test_restoration.py` SHA-256: `7226149536714bd8257f392901619967565dad0aae65e42141657932a8450fce`.

`docs/LOG.md`, 9 November, records moving Markdown reports to `docs/reports/` after completion. Current evaluator includes `markdown_report_path()` and report-link routing consistent with that record. This explains a plausible cause but does not prove byte-for-byte scope of all edits because the exact old evaluator snapshot was not recovered.

Checkpoint, train manifest, test manifest, attention and inference SHA-256 all match. Fifty summary means reconcile with the stored CSV; selected outputs reproduce all twelve recorded PSNR/SSIM values. No numeric artifact was replaced, and no full evaluation was rerun.
