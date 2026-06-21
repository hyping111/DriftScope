#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dexter

mkdir -p ./logs/sd15
CUDA_VISIBLE_DEVICES=0 nohup python -u run_several.py \
    --gradient_checkpointing \
    --out_subfolder sd15 \
    --method db15 \
    --prompt "A photo of a" \
    > ./logs/sd15/prompt_1.log 2>&1 &