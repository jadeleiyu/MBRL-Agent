#!/bin/bash
# 构建train/val数据分割的示例脚本

# 构建数据
python build_validation_data.py \
    --env wa \
    --output_dir data \
    --train_ratio 0.8 \
    --max_samples 800 \
    --random_seed 42

echo "数据构建完成！"
echo "训练数据: data/train.parquet"
echo "验证数据: data/test.parquet"