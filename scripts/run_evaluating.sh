#!/bin/bash

# 설정 변수
TYPE="augmented_ocr"

NORMAL_DATA_DIR="processed_dataset_${TYPE}_neg32_normal"
FRAUD_DATA_DIR="processed_dataset_${TYPE}_neg32_fraud"
MODEL_DIR="checkpoints/0113_${TYPE}"
OUTPUT_DIR="./evaluation_results_${TYPE}"
WANDB_PROJECT_NAME="0113_norispace_project_eval_${TYPE}"

# evaluate.py 스크립트 실행
python src/evaluate.py \
    --normal_data_dir "$NORMAL_DATA_DIR" \
    --fraud_data_dir "$FRAUD_DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --num_workers 4 \
    --device "cuda:1" \
    --wandb_project "${WANDB_PROJECT_NAME}"
