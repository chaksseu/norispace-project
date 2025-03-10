#!/bin/bash

###############################################################################
# Parameter Configuration
###############################################################################

# Directory paths for datasets
NORMAL_DATA_DIR="processed_dataset_augmented_ocr_neg32_normal_no_val_0307"   # Root directory for Normal dataset (train/val/test)

FRAUD_DATA_DIR="processed_dataset_augmented_ocr_neg32_fraud_no_val_0130"     # Root directory for Fraud dataset (val/test)
CHEKCPOINTS_DIR="./checkpoints_0308_no_val_32_one_small_model"                   # Directory to save model checkpoints

# Weights & Biases (W&B) project configuration
WANDB_PROJECT_NAME="0308_norispace_project"
WANDB_RUN_NAME="0308_augmented_ocr_no_val_32_one_small_model"

# Training hyperparameters
BATCH_SIZE=64 # 64
GRADIENT_ACCUMULATION_STEPS=4 # 4
LR=1e-5
EPOCHS=400
SAVE_EPOCH=20
NUM_WORKERS=4
MIXED_PRECISION="fp16"

# Flag to use pretrained ConvNeXt-Small model
PRETRAINED="--pretrained"

# gpu number is in accelerate config

###############################################################################
# Execute Training Script
###############################################################################

accelerate launch src/train.py \
  --normal_data_dir "${NORMAL_DATA_DIR}" \
  --fraud_data_dir "${FRAUD_DATA_DIR}" \
  --output_dir "${CHEKCPOINTS_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --save_epoch "${SAVE_EPOCH}" \
  --num_workers "${NUM_WORKERS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --wandb_project "${WANDB_PROJECT_NAME}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  ${PRETRAINED}
