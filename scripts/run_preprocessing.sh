#!/bin/bash

################################################################################
# Parameter Configuration
################################################################################

# Path to the YOLO model file
YOLO_MODEL_PATH="Noris_YOLO.pt"  # Example YOLO model file path

# Specify the label type: either "normal" or "fraud"
LABEL="normal"

# Input and output directories
INPUT_DIR="dataset/$LABEL"  
PROCESSED_DIR="processed_dataset_augmented_ocr_neg32_$LABEL"  

# Preprocessing hyperparameters
BATCH_SIZE=16
MARGIN=50
NEGATIVE_PER_IMAGE=32
NUM_WORKERS=4
CUDA_DEVICE=0

# (Optional) Directory containing text images for augmentation
# Only need for augmented_ocr
TEXT_IMAGE_DIR="text_images_pool"  

# Train, validation, and test ratio configuration based on LABEL
if [ "$LABEL" == "normal" ]; then
  TRAIN_RATIO=0.8
  VAL_RATIO=0.1
  TEST_RATIO=0.1
elif [ "$LABEL" == "fraud" ]; then
  TRAIN_RATIO=0.0
  VAL_RATIO=0.5
  TEST_RATIO=0.5
else
  echo "Invalid LABEL value. Please use 'normal' or 'fraud'."
  exit 1
fi

# Seed for reproducibility
SPLIT_SEED=42

# Make sure this directory exists and contains your prepared text images

################################################################################
# Execute Preprocessing Script
################################################################################

python src/data_preprocess.py \
  --yolo_model_path "${YOLO_MODEL_PATH}" \
  --input_dir "${INPUT_DIR}" \
  --processed_dir "${PROCESSED_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --margin "${MARGIN}" \
  --num_color_jitter "${NEGATIVE_PER_IMAGE}" \
  --num_workers "${NUM_WORKERS}" \
  --split_seed "${SPLIT_SEED}" \
  --train_ratio "${TRAIN_RATIO}" \
  --val_ratio "${VAL_RATIO}" \
  --test_ratio "${TEST_RATIO}" \
  --generate_negatives \
  --label "${LABEL}" \
  --cuda_device "${CUDA_DEVICE}" \
  --text_image_dir "${TEXT_IMAGE_DIR}"
