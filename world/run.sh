#!/bin/bash

python build_validation_data.py \
    --env wa \
    --output_dir data \
    --train_ratio 0.8 \
    --max_samples 800 \
    --random_seed 42

