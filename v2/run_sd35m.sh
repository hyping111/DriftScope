#!/usr/bin/env bash

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dexter

echo "Running on $(hostname)"
mkdir -p ./logs/sd35m

CUDA_VISIBLE_DEVICES=0 nohup python -u run_sd35.py \
    --out_subfolder sd35m \
    --prompt "A photo of a" \
    > ./logs/sd35m/db_sd35m.log 2>&1 &