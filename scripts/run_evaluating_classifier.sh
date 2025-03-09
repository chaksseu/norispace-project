#!/bin/bash

###############################################################################
# Configuration
###############################################################################
TYPE="balanced_pos_eval_no_val_0202"

NORMAL_DATA_DIR="processed_dataset_augmented_ocr_neg32_normal_no_val_0130"
FRAUD_DATA_DIR="processed_dataset_augmented_ocr_neg32_fraud_no_val_0130"
MODEL_DIR="checkpoints_0202_no_val_classifier"

WANDB_PROJECT_NAME="norispace_eval_project_no_val_classifier_"
WANDB_ENTITY="norispace-project"
WANDB_RUN_NAME="eval_classifier_${TYPE}"

DEVICE="cuda:3"

BATCH_SIZE=16
NUM_WORKERS=4

###############################################################################
# Execute evaluation.py
###############################################################################
python src/evaluate_classifier.py \
    --normal_data_dir "${NORMAL_DATA_DIR}" \
    --fraud_data_dir "${FRAUD_DATA_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --wandb_project "${WANDB_PROJECT_NAME}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "${WANDB_RUN_NAME}"
