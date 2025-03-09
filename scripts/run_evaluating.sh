#!/bin/bash

# 설정 변수
TYPE="augmented_ocr_no_val_one_base_model"

NORMAL_DATA_DIR="processed_dataset_augmented_ocr_neg32_normal_no_val_modified_augmentation_0131"
FRAUD_DATA_DIR="processed_dataset_augmented_ocr_neg32_fraud_no_val_0130"
MODEL_DIR="checkpoints_0203_no_val_one_model_base_model"
WANDB_PROJECT_NAME="0203_norispace_project_eval_no_val_${TYPE}"
GPU_NUM=1

# evaluate.py 스크립트 실행
python src/evaluate.py \
    --normal_data_dir "$NORMAL_DATA_DIR" \
    --fraud_data_dir "$FRAUD_DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --batch_size 16 \
    --num_workers 4 \
    --device "cuda:${GPU_NUM}" \
    --wandb_project "${WANDB_PROJECT_NAME}"
