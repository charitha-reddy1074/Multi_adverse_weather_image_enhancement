# Train the four-weather model

**Current status, 9 November 2026:** all 20 epochs and the full 14,695-image
test evaluation are complete. Read [final results and durations](RESULTS_README.md)
and [the paper comparison](PAPER_COMPARISON.md). The pause/active-run sections
below are historical instructions; no training job needs to be resumed for the
completed experiment.

## Paused after epoch 3 — 4 November 2026

Training was stopped at the user's request. `epoch_003.pt` is a verified copy of
the completed epoch-3 checkpoint, including optimizer, scheduler, AMP and RNG
state. Partial epoch-4 progress was discarded. The datasets remain unchanged.
Resume from epoch 4 using:

```powershell
.venv/Scripts/python -u train.py --config configs/four_weather_fast.yaml --resume checkpoints/four_weather_20261104_fast/epoch_003.pt --output checkpoints/four_weather_20261104_fast
```

The command above runs in the foreground; the old launcher's status/log are not
updated by a direct resume. The paused status records the intended resume command.

## Active optimized run — 4 November 2026

The active background job uses `configs/four_weather_fast.yaml` and writes to
`checkpoints/four_weather_20261104_fast/`. It retains all prepared training pairs,
the held-out test split, model, crop size and 20 epochs. Batch size is 8 with one
accumulation step, retaining the original effective batch size of 8.

The short synthetic GPU benchmark measured 17.38 images/second versus 10.08 for
batch size 2 / accumulation 4. This projects approximately 18.8 hours of GPU
training computation; data loading, startup checks and final evaluation add time.
Actual completion time is not yet measured. Evidence:
`outputs/training_batch_benchmark.json`.

```powershell
Get-Content checkpoints/four_weather_20261104_fast/status.json
Get-Content checkpoints/four_weather_20261104_fast/training.log -Tail 8
```

The earlier `four_weather_20261104_restart` process was deliberately stopped
during startup to apply this change. Use the active output directory above for
monitoring and the fast config for any resume. The original walkthrough below
describes the earlier batch-size-2 setup.

Run these commands in PowerShell from `D:\Multi_adverse_weather_image_enhancement`.
Training runs locally and does not need an open Codex conversation or API credits.
Keep the computer powered on, connected to power, and awake while it runs.

## 1. Know which data the run uses

| Weather | Local source | Train pairs | Test pairs |
|---|---|---:|---:|
| Rain | Rain14000 | 10,080 | 2,520 |
| Raindrop | DeRaindrop training archive | 688 | 173 |
| Snow | Snow100K training archive | 39,998 | 10,002 |
| Haze | RESIDE ITS unofficial Hugging Face mirror, `val` content | 8,000 | 2,000 |
| **Total** | | **58,766** | **14,695** |

This run preserves the existing custom 80/20 split by clean-scene identity.
There is no validation split and no best-checkpoint selection. The final model
is tested once after the fixed training schedule. Another 1,412 known official
rain test pairs remain separate and are not used by this run.

All 10,000 local haze images and their 1,000 clean targets match the file sizes
and SHA-256 hashes in [the Hugging Face mirror](https://huggingface.co/datasets/dronefreak/RESIDE-ITS)
at revision `39300fc7deb9138a59d5489f91a2f86d14d62eb6`. They correspond to
`val/haze` and `val/clear`. This identifies the mirror content; the mirror is
unofficial and lists its license as unknown. The original local archive download
URL is unknown. See [verification evidence](RESIDE_MIRROR_VERIFICATION.json).

This is a custom experiment using all usable local weather sources. It is not an
exact AIR40K reproduction or a standard RESIDE SOTS evaluation. Additional
catalogued datasets such as CSD, OTS and HSTS are not installed. `ITS_clear.zip`
alone has no matching local training haze images. Transmission maps and snow
masks are auxiliary files and are not RGB restoration targets. The interrupted
NH-HAZE download is not used.

Source references: [dataset catalog](../datasets/sources.csv),
[download details](../datasets/downloads.csv), and
[local archive hashes](LOCAL_DATASET_AUDIT.json).

## 2. Check Python and the GPU

```powershell
Set-Location D:\Multi_adverse_weather_image_enhancement
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The checked machine has an NVIDIA RTX 2000 Ada Generation GPU with 16 GiB VRAM.
Use the existing virtual environment. Reinstall dependencies only if the
environment has been removed; its pinned Windows/CUDA packages are listed in
`requirements-lock-windows-cu130.txt`.

## 3. Prepare and check the manifests

The manifests and config have already been prepared. To repeat the checks:

```powershell
# Online: compare all existing haze/clean images with pinned HF metadata.
.\.venv\Scripts\python.exe scripts/verify_reside_mirror.py
# Offline: check file existence, scenes, and one decoded pair per weather;
# preserve the audited split and write the four-weather config/manifests.
.\.venv\Scripts\python.exe scripts/prepare_training_run.py
Get-Content data/manifests/four_weather/preflight.json
```

The HF comparison fetches metadata only. It does not download the full dataset.
Preparation must report zero missing files and zero crossing scene IDs.
The earlier full decoding and duplicate-pixel audit is recorded in
`docs/LOCAL_DATASET_AUDIT.json`; the trainer also checks clean-pixel identity
across train/test before optimization. Startup can therefore take several minutes.
Do not change or rebuild the manifests during an active run.

If the extracted local dataset folders have been removed, use the retained
archives to rebuild them, then repeat the two commands above:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_available_data.py
.\.venv\Scripts\python.exe scripts/prepare_local_archives.py
```

These rebuild commands fully inspect the archives and can take a long time.
They are not needed for normal starts or resumes.

## 4. Check whether training is already running

The prepared output folder is `checkpoints/four_weather_20261103`.
The automatic launch was interrupted. No training status file or Python
training process was present when checked. Start training yourself in step 5.
Before the first launch, the status and log commands below will report missing
files; that is expected.

```powershell
Get-Content checkpoints/four_weather_20261103/status.json
Get-Content checkpoints/four_weather_20261103/training.log -Tail 8
nvidia-smi
```

Do not start a second run while this one is active. `status.json` records the
training PID and whether the wrapper is running, completed or failed. After
a reboot or forced termination it can be stale; compare its PID with
`Get-Process -Id <PID>` and the latest log output.

## 5. Start a new run only if needed

Use a new output folder for every fresh experiment. This command occupies the
terminal until training and final evaluation finish:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_training_job.py --config configs/four_weather.yaml --output checkpoints/four_weather_20261103
```

The supplied configuration uses 20 epochs, 128x128 random crops, batch size 2,
gradient accumulation 4, mixed precision and balanced weather sampling.
Balanced sampling draws 58,766 examples per epoch with replacement, so every
weather has similar expected weight; it does not visit every image each epoch.
The model is the existing resource-adapted AIRFormer reconstruction with width
16, four encoder positions and pooled attention keys/values. Its settings differ
from the paper. Runtime depends on the measured batch speed; no completion
time is guaranteed.

For another detached run from your own PowerShell terminal:

```powershell
$trainingPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$trainingRoot = (Get-Location).Path
Start-Process -FilePath $trainingPython -ArgumentList '-u','scripts/run_training_job.py','--config','configs/four_weather.yaml','--output','checkpoints/four_weather_20261103' -WorkingDirectory $trainingRoot -WindowStyle Hidden
```

## 6. Follow progress

```powershell
Get-Content checkpoints/four_weather_20261103/training.log -Tail 8 -Wait
```

Press Ctrl+C to stop following the log; the detached training process continues.
For training curves, run this in another terminal and open the printed local URL:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir checkpoints/four_weather_20261103/runs
```

Expected files:

- `training.log`: training progress, losses and errors.
- `status.json`: wrapper state and exit code.
- `latest.pt`: full model, optimizer, scheduler and RNG state after each epoch.
- `runs/`: TensorBoard event files.
- `final_test.json`: aggregate PSNR, SSIM and loss after all epochs complete.

No `best.pt` is expected because this run has no validation set. Test metrics
are aggregate across the custom four-weather test set, not per-weather results.

## 7. Resume after an interruption

First confirm that the old process has stopped. Resume from the last completed
epoch using the same config, manifests and output folder:

```powershell
.\.venv\Scripts\python.exe -u train.py --config configs/four_weather.yaml --resume checkpoints/four_weather_20261103/latest.pt --output checkpoints/four_weather_20261103
```

The foreground resume command prints directly to the terminal; it does not
update the original wrapper's `status.json` or `training.log`. Check its terminal
output and checkpoint modification time. Work since the most recent epoch
checkpoint is lost on interruption. If no `latest.pt` exists, start fresh in a
new output folder. Do not resume the old `rain_raindrop` diagnostic checkpoint.

## 8. Check the finished result

```powershell
Get-Content checkpoints/four_weather_20261103/status.json
Get-Content checkpoints/four_weather_20261103/final_test.json
```

For the wrapped run, success means `state` is `completed`, `exit_code` is `0`,
and `final_test.json` exists. A direct resume is complete when the command exits
successfully and prints `final_test=...`. Keep the test set held out and avoid
tuning settings against repeated test results.

If CUDA runs out of memory, stop the failed run, make a new config with
`batch_size: 1` and `gradient_accumulation_steps: 8`, and use a new output folder.
If files are missing, fix the local data and rerun step 3. If the GPU becomes
unavailable, restore CUDA before starting a long run.
