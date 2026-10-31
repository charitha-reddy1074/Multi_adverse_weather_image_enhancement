# Minimal optimization log

Main-paper re-verification follow-up: explanatory comments were added to model
and loss files, but semantic AST fingerprints match their pre-comment versions.
No model optimization or architectural correction was applied. The comparison
now confirms substantial paper deviations; see `architecture/README.md`.
Existing 62 tests pass after annotation; this is execution evidence only.

The audit preserves the existing model and loss mathematics. No attention,
encoder, SP, wavelet, learned query or reconstruction component was removed.

| Iteration | Change | Behavioral evidence | Runtime / memory claim |
|---|---|---|---|
| 1 | Remove local import shadowing in data loader | Existing failing pairing test now passes | Correctness fix; no speed claim |
| 2 | Consolidate image_map from data/evaluation into datasets/pairing.py | Exact relative identities preserved; duplicate-extension rejection | Fewer duplicate helpers; no tensor-memory change |
| 3 | Remove unused imports; test-only torch thread cap | Full suite passes; checkpointing outputs/gradients equivalent | Faster tiny-test execution only; no inference gain claimed |
| 4 | AMP overflow recovery and real-data pipeline measurement | Six AMP attempts recover at scale8192; three finite updates | Diagnostic pilot128 inference 23.502ms; peak allocated 85.82MiB inference / 209.51MiB training |
| 5 | Final fixed workload and regression recheck; no further model refactor | 62 tests pass; actual attention axes differ and softmax normalizes keys | Paper-config256 inference 205.302ms; no before/after speedup claim |

Baseline measurements (random weights, not research scores): pilot 271,155
parameters, CPU 32x32 mean inference 0.507584 s over 3 iterations after 1 warmup;
paper config 1,146,883 parameters, CUDA 32x32 mean inference 0.026663 s.
These runs used default CPU threads and shared desktop GPU activity. They are
diagnostics, not fair RTX3090 comparisons or published performance reproduction.
No important model-code optimization was attempted without a primary paper and
a representative workload. GPU peak memory and training timing are measured
separately in the final benchmark, if available; unavailable fields remain null.

Final measured tensors/times/numerical output are saved in
`docs/AUDIT_BENCHMARK.json` and local `outputs/audit_smoke/benchmark.json`.
Pilot training time is 1.110758s per attempted step including initial scale
adaptation, not warmed-up throughput. Three successful losses decreased from
3.789946 to 2.434709. Measurements use random weights and shared desktop GPU
activity; do not compare them directly to RTX3090 paper timings.
`git diff -- models losses` is empty, so model parameter count/forward math
cannot have changed during this audit. No full-model numerical equivalence to
the unavailable authors' implementation is implied.
