#!/bin/bash

###############################################################################
# Parameter Configuration
###############################################################################

# Directory paths for datasets
NORMAL_DATA_DIR="processed_dataset_no_ocr_neg32_normal"   # Root directory for Normal dataset (train/val/test)
FRAUD_DATA_DIR="processed_dataset_no_ocr_neg32_fraud"     # Root directory for Fraud dataset (val/test)
OUTPUT_DIR="./checkpoints"                   # Directory to save model checkpoints

# Weights & Biases (W&B) project configuration
WANDB_PROJECT_NAME="0112_norispace_project"

# Training hyperparameters
BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-5
EPOCHS=1024
EVAL_EPOCH=4
SAVE_EPOCH=1024
NUM_WORKERS=4
MIXED_PRECISION="fp16"

# Flag to use pretrained model
PRETRAINED="--pretrained"

###############################################################################
# Execute Training Script
###############################################################################

accelerate launch src/train_contrastive.py \
  --normal_data_dir "${NORMAL_DATA_DIR}" \
  --fraud_data_dir "${FRAUD_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --eval_epoch "${EVAL_EPOCH}" \
  --save_epoch "${SAVE_EPOCH}" \
  --num_workers "${NUM_WORKERS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --wandb_project "${WANDB_PROJECT_NAME}" \
  ${PRETRAINED}
