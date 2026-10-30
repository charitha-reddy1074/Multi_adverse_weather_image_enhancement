@echo off
python scripts\verify_dataset.py --root datasets\train
python scripts\train.py --train-root datasets\train --val-root datasets\test --epochs 300 --batch-size 8 --crop-size 256
pause
