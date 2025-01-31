#!/bin/bash

# 설정 변수
TYPE="augmented_ocr_no_val_test"

NORMAL_DATA_DIR="processed_dataset_augmented_ocr_neg32_normal_no_val_0130"
FRAUD_DATA_DIR="processed_dataset_augmented_ocr_neg32_fraud_no_val_0130"
MODEL_DIR="checkpoints_0131_test"
WANDB_PROJECT_NAME="0131_norispace_project_eval_${TYPE}"
GPU_NUM=3

# evaluate.py 스크립트 실행
python src/evaluate.py \
    --normal_data_dir "$NORMAL_DATA_DIR" \
    --fraud_data_dir "$FRAUD_DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --batch_size 16 \
    --num_workers 4 \
    --device "cuda:${GPU_NUM}" \
    --wandb_project "${WANDB_PROJECT_NAME}"
