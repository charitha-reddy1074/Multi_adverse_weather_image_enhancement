# Reproduction status - main-paper architecture re-verification

**NOT EXACT. Full main-paper comparison is complete, but architectural
corrections and full research reproduction are not complete.**

Read `architecture/README.md` for the original extracted Figures 4/5, per-module
logic, exact tensor-shape comparison, parameter evidence and a prioritized
correction sequence. `PAPER_COMPLIANCE.md` maps requirements to primary evidence,
code and behavior tests using explicit status labels.

Confirmed differences include the three-level encoder versus four depicted
encoder positions; SP stage placement; capped spatial K/V; a different wavelet
query/IDWT graph and normalization position; unresolved query/tail internals;
1.15M versus 58.34M parameters; and literal Equation 12 loss reduction.

Comments now explain these differences where the operations execute. No model
or loss operations changed: semantic AST fingerprints before/after match.
All 62 existing CPU/CUDA and integration tests pass. Passing tests verify that
the reconstruction works, not that it equals the published architecture.

The supplied 14-page publisher PDF is now available. Main-paper unknowns remain
explicit assumptions; unavailable supplementary/original-code details cannot be
certified. Original checkpoint parity and full trained restoration performance
have not been established.

CUDA setup, Rain14000/Rain12 acquisition/integrity reports and training pipeline
remain as previously documented. Datasets/manifests/sampling were not changed by
this comparison. A working pipeline can train the current reconstruction; the
strict-paper goal requires resolving the documented graph differences first.

No full training run or useful pretrained restoration model is claimed.
A trained saved checkpoint can support repeated demonstrations without retraining.
