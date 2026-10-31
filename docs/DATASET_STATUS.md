# Dataset availability - audit 2026-10-31

Architecture-only follow-up: full main-paper architecture comparison is complete
and the current model is NOT EXACT (`architecture/README.md`). No dataset files,
splits, manifests, or sampling were changed during that comparison.

Follow-up: the user supplied the paper PDF. Table I was visually verified on PDF
page 5 (printed 1890). It reports Raindrop Test-B=249, alongside Test-A=58;
the original DeRaindrop README says 239 and describes overlap. Keep these
release differences explicit. Current Snow100K partial file is 6,831,996,928
bytes (about 6.36 GiB), increased since the original audit snapshot below;
it is still not a completed/validated archive. No download was restarted during
this follow-up.

Initial-phase download priority (engineering recommendation, not paper sampling):
use ready Rain14000 first; then obtain Raindrop training plus Test-A, RESIDE ITS
paired training, and Snow100K training (including paired clean images, preferably
the L subset for the paper-oriented experiment). These cover the four weather
domains. CSD and OTS add snow/outdoor-haze coverage next. Rain800, RainTrainH,
Rain100L, Rain1200, Rain12, Snow100K M/S tests and HSTS complete the benchmark
later; Rain12 is already present. Keep every official test release held out.

Statuses reflect actual downloads, not the existence of folder names. Required
sources were checked through owner pages and downloads attempted where possible.
Verified endpoint catalog and commands: `datasets/downloads.csv`,
`scripts/download_required.ps1`. No published checksum was found for acquired
archives; local SHA-256 provides provenance, not an owner-signed integrity proof.

| Dataset | Source | Expected Count (AIR40K train/test) | Actual Count | Paired Count | Status | Required Action |
|---|---|---:|---:|---:|---|---|
| Rain14000 | [owner](https://xueyangfu.github.io/projects/cvpr2017.html) | 11200/2800 | release 12600/1400 inputs; 900/100 clean scenes | 14000 | READY | Source-release setup ready; exact AIR40K membership remains unresolved |
| Rain800 | [owner](https://github.com/hezhangsprinter/ID-CGAN) | 700/100 | 0 | 0 | DOWNLOAD BLOCKED | Owner Drive link cannot retrieve public file; README reports deletion; obtain restored release |
| Rain100H/RainTrainH | [survey](https://github.com/hongwang01/Video-and-Single-Image-Deraining), [research mirror](https://github.com/csdwren/PReNet) | 1800/100 | 0 | 0 | DOWNLOAD BLOCKED | Original project 404, documented OneDrive mirror 403; Baidu manual access; check source-scene overlap |
| Rain100L | same sources as Rain100H | 0/100 | 0 | 0 | DOWNLOAD BLOCKED | Obtain original 100-test release; do not substitute 200-test release |
| Rain1200 | [DID-MDN owner](https://github.com/hezhangsprinter/DID-MDN), [original paper](https://arxiv.org/abs/1802.07412) | 0/1200 | 0 | 0 | DOWNLOAD BLOCKED | Owner Drive dataset link not publicly retrievable; resolve TestA/TestB membership |
| Rain12 | [original author](https://yu-li.github.io/paper/li_cvpr16_rain.zip) | 0/12 | 12 inputs / 12 GT / 12 rain layers | 12 | READY | Test only; actual held-out metric smoke passed |
| Raindrop / RainDA / RainDB | [owner](https://github.com/rui1996/DeRaindrop) | 861/307 | 0 | 0 | DOWNLOAD BLOCKED | Drive cannot retrieve public file; upstream describes 239 TestB with TestA a subset; record actual release |
| Snow100K-L | [owner](https://sites.google.com/view/yunfuliu/desnownet) | 9000/16801 | 587726848 bytes partial training archive | 0 | PARTIAL | Transfer stalled; preserved partial; one resume attempt now blocked by Drive access; test Drive also blocked |
| Snow100K-M | same Snow100K owner | 0/16588 | 0 | 0 | DOWNLOAD BLOCKED | Official shared test archive unavailable via gdown |
| Snow100K-S | same Snow100K owner | 0/16611 | 0 | 0 | DOWNLOAD BLOCKED | Official shared test archive unavailable via gdown |
| CSD | [owner](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) | 8000/2000 | 0 | 0 | DOWNLOAD BLOCKED | Google Drive reports download quota exceeded; wait for quota reset or owner alternative |
| RESIDE ITS | [owner standard](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) | 4500/500 | 0 | 0 | DOWNLOAD BLOCKED | bit.ly/3iwHmh0 resolves to Box 404; owner Baidu access alternative |
| RESIDE OTS | [owner beta](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) | 4495/500 | 0 | 0 | DOWNLOAD BLOCKED | bit.ly/3k8a0Gf resolves to Box 404; owner Baidu access alternative |
| RESIDE HSTS | [owner standard](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) | 0/10 | 0 | 0 | DOWNLOAD BLOCKED | bit.ly/394pcAm resolves to Box 404; exclude unpaired real subset |

Table I training rows sum to **40,556**, not reported **40,561**. Test rows
sum to 57,629 using the stated 307 raindrop count. No pairs are invented to
close gaps. Official Rain14000 release explicitly uses 900/100 scenes and 14
variants per scene, unlike the requested AIR40K split. Exact AIR40K sampling
and re-partitioning remain BLOCKED without authors' membership lists.

Safe setup uses `scripts/prepare_available_data.py`: a deterministic seed-42
5% scene holdout from official Rain14000 training (45 scenes / 630 variants),
leaving 11,970 training pairs. All 1,400 official Rain14000 and 12 Rain12 test
pairs remain test-only. This is a documented source-release experiment, not
the exact AIR40K experiment. Validation checks decoding and equal dimensions,
exact input identity, clean pixel hashes across splits. Same dimensions alone
do not prove geometric alignment; original release provenance plus visual
spot checks are needed. RESIDE scene mapping is not claimed tested without data.

Acquisition metadata, observed sizes and unknown licenses are in
`datasets/downloads.csv`. Code licenses do not imply dataset redistribution
rights. Do not bypass login, quotas or access conditions. Download only through
documented sources. Failed archives are not normalized or used for training.

Example repeatable commands (PowerShell, repo root):

```powershell
./scripts/download_required.ps1 -Datasets Rain14000,Rain12
./.venv/Scripts/python scripts/prepare_available_data.py
./.venv/Scripts/python train.py --config configs/pilot.yaml
```

The preparation script validates existing extracted files on rerun; it never
overwrites differing files. ZIP/TAR paths, links, Windows drive names and ADS
are rejected. RAR extraction is deliberately unavailable until a validated
extractor is configured; no CSD file was acquired. Preserve original archives.

Audit history: iteration1 all missing; iteration2 Rain14000/Rain12 archives
acquired and Snow transfer started; iteration3 safe extraction and split guards
implemented; iteration4 all 14,012 available pairs validated and real metrics
worked; iteration5 full input-file byte hashes found two original-release
duplicates within training, preserved to keep sampling explicit. Exact paths are
in `docs/DATASET_DUPLICATES.json`. There are 14,010 unique input byte hashes;
this does not rule out re-encoded or perceptually similar copies. Cross-split
clean-pixel hashes found no overlap. All paired source images are RGB; the 12
Rain12 grayscale rain layers are excluded from model inputs/targets.

Local archive SHA-256 (not published owner checksums):

- Rain14000: `c967e0bc33e2ab70828aa614c000024a695b532684ddced9b83a75b941e91674`
- Rain12: `3ba5f100149158a8f6378b48203646565fd21a7b394b4ef33124ff5c4fae0581`

Available-source training is ready. Full all-weather training remains blocked.
