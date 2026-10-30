# Dataset acquisition and local layout

## What the paper says
AIR40K is a curated training benchmark assembled by sampling existing datasets. The main paper names Rain14000, Rain800, Rain100H, Rain100L, Rain1200, Rain12, Raindrop, Snow100K, CSD, ITS, OTS and HSTS. It does not provide one consolidated archive URL in the paper text itself. AIR40K is a sampling/mixture, not an independently standardized public dataset with a guaranteed single download endpoint.

## Dataset sources / download pages
See `sources.csv`. Useful verified project/source pages include:
- Rain14000 / Rain800 / Rain100H / Rain100L / Rain1200 references are commonly hosted by the original authors' project pages or related restoration repositories.
- Raindrop: https://github.com/rui1996/DeRaindrop
- Haze ITS/OTS: https://github.com/Boyiliee/BPPNet (dataset directions may vary by project); original RESIDE resources are commonly linked by the RESIDE project page: https://sites.google.com/view/reside-dehaze-datasets/reside-standard
- Snow100K / CSD: consult original paper/project and the authors' dataset links; links may require cloud drive access.
- Rain14000 original project: https://xueyangfu.github.io/projects/cvpr2017.html
- Rain800 / Rain12 dataset references: https://github.com/hezhangsprinter/ID-CGAN and https://github.com/csdwren/RecDerain (README contains dataset mirrors).

## Important
This project intentionally does not silently scrape unknown cloud-drive URLs or bundle copyrighted dataset archives. Download from the cited dataset owner pages and extract locally. Some sources require manual download, account sign-in, or acceptance of terms. The paper's exact AIR40K file list/sampling manifest is not present in the uploaded main paper; if you obtain the official AIR40K archive, preserve its official split and use it directly.

## Expected directory structure
For paired training, use a distinct relative path for each source dataset to
avoid filename collisions:
```
datasets/AIR40K/raw/train/<source>/degraded/<same-relative-name>.png
datasets/AIR40K/raw/train/<source>/clean/<same-relative-name>.png
datasets/AIR40K/raw/test/<source>/degraded/<same-relative-name>.png
datasets/AIR40K/raw/test/<source>/clean/<same-relative-name>.png
```
Different source datasets may have their own naming and folder conventions. Convert/copy into this normalized layout only after checking pair alignment; never pair files by arbitrary sorted order.

The training script defaults to `configs/pilot.yaml`, which uses the train
directory. It expects the validation split at the sibling `test/` directory.
Pass `--config configs/paper.yaml` to select the paper-oriented settings.
Dataset downloads remain at their owners' published locations because AIR40K's
consolidated package and exact file manifest are unavailable here; preserve
each dataset's terms and official train/test separation.
