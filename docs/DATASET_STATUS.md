# Local dataset status - 2026-11-02

**Extracted and ready for the user-requested custom 80/20 training/testing
experiment across rain, raindrop, snow and haze. No validation set.**

All eight local archives extracted successfully. All **74,873 candidate pairs**
passed complete image decoding and equal-dimension checks (zero invalid pairs).
Archive sizes and SHA-256 are recorded in [the machine-readable audit](LOCAL_DATASET_AUDIT.json).
These are local provenance hashes, not publisher-supplied checksums.

## Custom train/test manifests

| Source | Training pairs | Testing pairs |
|---|---:|---:|
| Rain14000 | 10,080 | 2,520 |
| Raindrop | 688 | 173 |
| Snow100K training | 39,998 | 10,002 |
| Local haze (source unverified) | 8,000 | 2,000 |
| **Total** | **58,766** | **14,695** |

Manifests: `data/manifests/available/train.csv` and `test.csv`.
The actual fraction is 79.9962% training. Small rounding
differences from exactly 80/20 keep entire clean-identity groups together.
Seed: 42. Split per source by scene/clean-image identity, with pixel-identical
clean images and byte-identical degraded inputs grouped before assigning splits.
Checks found no crossing scene IDs, identical clean pixels or identical input
bytes. Near-duplicates or related views still require source metadata review.

There is **no `val.csv` in the available-data experiment**. A separate
`official_test.csv` preserves all **1,412** known official tests: Rain14000 1,400
and Rain12 12. They were excluded from the custom train/test pool.
The historical Rain-only manifests directly under `data/manifests/` retain the
previous source-release setup; the current local config uses `available/` only.

Haze uses 10,000 input images paired by scene ID with 1,000 clean images.
The user authorized a custom split, but the original source/release/split remains
unverified. These pairs are suitable for a clearly disclosed custom experiment;
this does **not** establish canonical RESIDE or AIR40K benchmark membership.

## Extracted but not usable as RGB training pairs

| Files | Local content | Missing requirement |
|---|---|---|
| `ITS_clear.zip` | 10,000 clean images, fully extracted | Matching ITS degraded/hazy images and release identity |
| `trans.zip` | 10,000 transmission maps | These are not clean RGB restoration targets; exclude from input/target manifests |
| Snow100K auxiliary files | Snow masks alongside 50,000 input/clean pairs | Masks are auxiliary, not additional RGB targets |
| Rain12 rain layers | 12 rain layers | Auxiliary layers, not clean targets; 12 paired tests remain held out |

Visual spot checks of one new raindrop, snow and haze pair showed matching scene
structure. This is limited evidence, not an all-image geometric alignment proof.
The temporary contact sheet was removed after inspection.

## Sufficiency and remaining datasets

The available paired data are sufficient to start a **custom four-weather
experiment**. Additional exclusively-training datasets are optional for that
experiment. The most useful missing component is the degraded counterpart of
`ITS_clear.zip`, with its source/release documented. Verified RESIDE ITS/OTS
paired training releases would improve haze provenance and indoor/outdoor coverage;
CSD training would add another snow source.

**Exact paper reproduction is still not ready.** The authors' AIR40K membership
and Snow100K-L training selection remain unavailable. Canonical benchmark releases
still missing locally: Rain800, RainTrainH/Rain100H, Rain100L, Rain1200, raindrop
Test-A/Test-B, Snow100K-L/M/S tests, CSD, confirmed RESIDE ITS/OTS experiment
membership, and HSTS. Do not substitute the custom test split for those results.
Rain14000's original release split differs from the paper's AIR40K sampling.
The paper table's count ambiguity also remains unresolved.

```powershell
# Rebuild the base Rain manifests if needed, then audit all archives and split.
./.venv/Scripts/python scripts/prepare_available_data.py
./.venv/Scripts/python scripts/prepare_local_archives.py
```

Original archives are retained. Extraction refuses differing existing files and
unsafe archive paths/links. No full research-training result is claimed.

Recheck on 2026-11-02: every one of the **127,746 unique files** referenced by
the three current manifests exists; scene IDs do not cross their splits.
This recheck does not repeat the earlier complete decoding/pixel-hash audit.
Preparation now stops before rewriting manifests if any required paired-data
archive is missing or fails verification, preventing reuse of stale extractions.
