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

The manifest builder only pairs exact relative paths. Dataset-specific mappings
for differing input/target file names must be reviewed and supplied in a CSV
with `input_path,target_path,weather_type,source_dataset,split,scene_id` columns.
Do not pair files by sorted order. Keep raw archives and checksums outside the
normalized manifests and record their source URL, license/terms, archive name,
download date, and SHA-256 before use.
