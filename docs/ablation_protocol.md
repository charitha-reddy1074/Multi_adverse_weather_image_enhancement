# AIRFormer ablation controls for Tables XI–XII

The paper text states that Table XI starts from a U-shaped PoolFormer, changes
encoder pooling to self-attention, adds the Supplementary Prior, replaces the
decoder's spatial attention with frequency refinement, and replaces usual Q
with task-queried vectors. Table XII compares permutations of the three decoder
attention axes and reports Channel → Space → Wavelet as best. The checked-in
paper text extraction includes table captions but not the rasterized row labels
or numeric cells, so the commands below expose those factors without claiming
unreadable table values or exact row-by-row reproduction.

## Controls

- `encoder_token_mixer`: `pooling` or `self_attention`.
- `use_supplementary_prior`: on/off (`--disable-sp` turns it off).
- `decoder_mode`: `spatial`, `frequency`, or `none`.
- Task-query switch: on/off (`--disable-task-queries` turns it off).
- Frequency loss: `--lambda-freq 0` removes the DCT term; the paper setting is
  0.04.

For Table XI-style component runs, start each experiment from the same pilot
config, seed, data manifest, and training schedule. Example baseline and
incremental commands:

```bash
# U-shaped PoolFormer-style baseline with ordinary spatial decoder attention
python scripts/train.py --config configs/pilot.yaml --encoder-token-mixer pooling --decoder-mode spatial --disable-sp --disable-task-queries --lambda-freq 0
# Encoder self-attention / supplementary-prior variants
python scripts/train.py --config configs/pilot.yaml --encoder-token-mixer self_attention --decoder-mode spatial --disable-sp --disable-task-queries --lambda-freq 0
python scripts/train.py --config configs/pilot.yaml --encoder-token-mixer pooling --decoder-mode spatial --disable-task-queries --lambda-freq 0
python scripts/train.py --config configs/pilot.yaml --encoder-token-mixer self_attention --decoder-mode spatial --disable-task-queries --lambda-freq 0
# Frequency refinement, then task queries, then the full multi-domain loss
python scripts/train.py --config configs/pilot.yaml --decoder-mode frequency --disable-task-queries --lambda-freq 0
python scripts/train.py --config configs/pilot.yaml --decoder-mode frequency
```

These are controlled implementation variants. Verify Table XI's exact row
combinations against the original table image before comparing reported scores.

## Table XII attention permutations

`--attention-order` accepts any permutation of `channel`, `space`, and
`wavelet`. Run all six with all other settings held fixed:

```bash
python scripts/train.py --config configs/pilot.yaml --attention-order channel,space,wavelet
python scripts/train.py --config configs/pilot.yaml --attention-order channel,wavelet,space
python scripts/train.py --config configs/pilot.yaml --attention-order space,channel,wavelet
python scripts/train.py --config configs/pilot.yaml --attention-order space,wavelet,channel
python scripts/train.py --config configs/pilot.yaml --attention-order wavelet,channel,space
python scripts/train.py --config configs/pilot.yaml --attention-order wavelet,space,channel
```

The primary configuration remains Channel → Space → Wavelet.
