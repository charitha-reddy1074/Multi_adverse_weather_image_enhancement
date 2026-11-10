# Dataset evidence

## Dataset composition and preparation

| Weather | Local source | Train | Custom test | Total |
|---|---|---:|---:|---:|
| rain | Rain14000 | 10,080 | 2,520 | 12,600 |
| raindrop | Raindrop | 688 | 173 | 861 |
| snow | Snow100K local training archive | 39,998 | 10,002 | 50,000 |
| haze | Unofficial RESIDE-ITS mirror val | 8,000 | 2,000 | 10,000 |

Total 73,461 paired records, 58,766 train and 14,695 test. These count pairs, not unique underlying scenes or unique files. Sources: exact four-weather manifests and `preflight.json` (4 November).

Pairing uses explicit `input_path`/`target_path` mapping from source-specific preparation logic, not sorted filenames. `datasets/splitting.py::write_custom_manifests` groups source scene IDs, SHA-256 of decoded clean RGB pixels including dimensions, and SHA-256 of input file bytes via connected identity groups. Groups are shuffled by seed 42 within source buckets, with 80% of groups assigned to train. This can deviate slightly from exactly 80% of pairs. Known official tests and overlapping identities are excluded from the custom pool. The separate `official_test.csv` has 1,412 rain pairs and is not part of the final 14,695 evaluation.

Fresh historical preflight reports 127,746 unique referenced files, zero missing files, and zero scene overlap, plus one decoded pair per weather. The preparation/integrity records include earlier pixel-hash and decoding checks. This extraction re-counted all manifest rows, rechecked manifest hashes, checked selected sample identities and decoded the twelve chosen pairs. It did not repeat all-image decoding or rehash the full datasets.

Training loader `airformer/data.py` converts RGB to [0,1] tensors, checks paired sizes, pads small images, applies the same random 128x128 crop to each pair, paired horizontal/vertical flips, random quarter-turn rotations and MixUp p=0.5. The MixUp coefficient is uniform in the local code. Weather-balanced sampling uses inverse domain counts with replacement, so an epoch is not a guarantee that every distinct training image is visited exactly once. Test images retain full dimensions, with symmetric padding to 128 multiples and crop-back, without augmentation.

Provenance: owner/project links are catalogued in `datasets/sources.csv`. Local archive identity and preparation are in `docs/LOCAL_DATASET_AUDIT.json` and `scripts/prepare_local_archives.py`. Haze files match `dronefreak/RESIDE-ITS`, revision `39300fc7deb9138a59d5489f91a2f86d14d62eb6`, `val/clear` and `val/haze`: 11,000 files, no byte/size mismatch in `docs/RESIDE_MIRROR_VERIFICATION.json`. That confirms mirror identity, not original release provenance. It is not a canonical SOTS test. Exact hashes do not rule out near-duplicate scenes. Canonical AIR40K membership, CSD and official benchmark releases remain missing.

Genuine local pair paths and scene IDs are in `restoration_samples/sample_manifest.csv`. Dataset graphics are in `datasets/`, including `dataset_details.csv`. No external image or dataset was downloaded.
