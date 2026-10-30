# AIR40K dataset provenance and audit

The paper reports AIR40K as a curated sample of existing public datasets. This
workspace has no dataset image files and no verified direct archive URLs.
`datasets/sources.csv` records owner/research project pages, not guaranteed
download endpoints. `python scripts/download_datasets.py` performs only a
best-effort HTTP HEAD check on those pages. It does not download data.

The paper's Table I composition supplied for this reproduction specifies:

| Dataset | Train | Test |
|---|---:|---:|
| Rain14000 | 11,200 | 2,800 |
| Rain800 | 700 | 100 |
| Rain100H | 1,800 | 100 |
| Rain100L | 0 | 100 |
| Rain1200 | 0 | 1,200 |
| Rain12 | 0 | 12 |
| Raindrop | 861 | 307 total |
| Snow100K-L | 9,000 | 16,801 |
| Snow100K-M | 0 | 16,588 |
| Snow100K-S | 0 | 16,611 |
| CSD | 8,000 | 2,000 |
| RESIDE ITS | 4,500 | 500 |
| RESIDE OTS | 4,495 | 500 |
| RESIDE HSTS | 0 | 10 |

The listed training counts total **40,556**, while the paper reports **40,561**.
The five-pair difference is preserved as an audit discrepancy; no examples are
invented. Verify Rain100H against RainTrainH/Rain1800 original naming before
pairing. For RESIDE, multiple hazy inputs may reference one clean scene; record
each input-target pair and use the clean scene identity for leakage checks.

Once files have been obtained under their owners' terms, organize them by split
and source as `raw/<split>/<source>/{degraded,clean}/...` and run:

```bash
python scripts/build_manifests.py --root datasets/AIR40K/raw --output data/manifests
python scripts/verify_dataset.py --root datasets/AIR40K/train
```

The paper's table supplies train and test splits but not a separate validation
split. For model selection, reserve validation scenes from the training sources
and record that deviation in the manifest. Never use AIR40K test samples for
epoch-by-epoch checkpoint selection. The trainer expects `data/manifests/val.csv`
or a sibling `val/` folder.

## Additional requested datasets: availability and correct use

These entries are catalogued in `datasets/sources.csv` and use the same
manifest format. No dataset archive has been fetched in this workspace. The
catalog links to documented project/archive instructions; it does not claim
that archives are locally present. Put paired data in
`raw/<split>/<source>/{degraded,clean}/` with matching relative filenames,
then run the manifest builder. Do not use a benchmark's test split for training.

| Dataset | Finding | Pipeline status |
|---|---|---|
| Rain200H / Rain200L | Publicly documented paired sets, 1,800 train and 200 test each | Included in source catalog and count audit |
| Snow100K-L / Snow100K-S | Official Snow100K owner page documents paired synthesized data; already supported | Existing AIR40K entries; official subset counts kept |
| Outdoor-Rain | Paired synthetic joint rain/haze data; 9,000 train. Test protocol varies (Test1 often 750, while some reports state 1,500) | Catalogued as `rain_haze`; preserve exact archive split metadata |
| Raindrop | Official repository documents 861 train pairs; Test-A (58) is a subset of Test-B (239) | The AIR40K table's 307 test count conflicts with the upstream split; audit actual archive identities and do not add the subsets |
| SPA-Data | Real rain paired benchmark; standard held-out test has 1,000 pairs. Published training count varies because patches and source images are counted differently | Catalogued; derive train count from archive and record unit |
| RainCityscapes | Synthetic paired rain variants; commonly reported 9,432 train and 1,188 test from 262/33 clear scenes | Catalogued as `rain_haze`; preserve scene IDs and enforce scene-level split |
| CDD-11 | Public OneRestore project links the 11-degradation archive through Hugging Face. Labels include low-light and composite classes, not only the four AIRFormer weather classes | Catalogued as `mixed_weather`; select weather-only or explicitly named combinations in curated manifests |
| RESIDE-S | Sand-dust benchmark described by a 2025 paper using RESIDE-derived synthesis; separate from RESIDE haze ITS/OTS/SOTS/HSTS. No canonical owner archive confirmed here | Catalogued as `sand_dust`; confirm archive provenance before use |
| ACDC20 | Official 4,006 adverse-condition images have semantic annotations and corresponding normal-condition images of the same scene | Weak correspondence for restoration: do not use as pixel-aligned target pairs or calculate restoration PSNR/SSIM |

The project trains the unified AIRFormer four-weather model (rain, raindrop,
snow, haze). `rain_haze` and `mixed_weather` records can be loaded as paired
weather-domain examples, but include them only in experiments that report this
choice. `sand_dust` and `mixed_weather` extend beyond the AIR40K taxonomy. The
model has no per-dataset heads; `weather_type` is metadata for sampling and
reporting only.

RainCityscapes derives multiple variants from a small set of Cityscapes clear
scenes. Keep all variants from a source scene in a single split. RESIDE-S also
needs source-scene provenance to prevent base scenes crossing train/test.
ACDC normal-condition correspondences can support qualitative or weakly
supervised work, but are not registered restoration ground truth.

The manifest builder only pairs exact relative paths. Dataset-specific mappings
for differing input/target file names must be reviewed and supplied in a CSV
with `input_path,target_path,weather_type,source_dataset,split,scene_id` columns.
Do not pair files by sorted order. Keep raw archives and checksums outside the
normalized manifests and record their source URL, license/terms, archive name,
download date, and SHA-256 before use.
