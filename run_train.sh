#!/usr/bin/env bash
set -e
python train.py --config configs/pilot.yaml "$@"
