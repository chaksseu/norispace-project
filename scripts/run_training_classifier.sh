#!/bin/bash

###############################################################################
# Parameter Configuration
###############################################################################

# Directory paths for dataset
# DATA_DIR should contain train/pos, train/neg, test/pos, test/neg directories
DATA_DIR="processed_dataset_augmented_ocr_neg32_normal_no_val_0130"
CHECKPOINTS_DIR="./checkpoints_0202_no_val_classifier"

# Weights & Biases (W&B) project configuration
WANDB_PROJECT_NAME="0202_norispace_project_classifier"
WANDB_ENTITY="norispace-project"
WANDB_RUN_NAME="0202_augmented_ocr_no_val_classifier"

# Training hyperparameters
BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-5
EPOCHS=400
SAVE_EPOCH=20
NUM_WORKERS=4
MIXED_PRECISION="fp16"

# Flag to use pretrained ConvNeXt-Small model
PRETRAINED="--pretrained"

###############################################################################
# Execute Training Script
###############################################################################

accelerate launch src/train_classifier.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${CHECKPOINTS_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --save_epoch "${SAVE_EPOCH}" \
  --num_workers "${NUM_WORKERS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --wandb_project "${WANDB_PROJECT_NAME}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  ${PRETRAINED}
