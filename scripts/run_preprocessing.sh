#!/bin/bash

################################################################################
# Parameter Configuration
################################################################################

# Path to the YOLO model file
YOLO_MODEL_PATH="Noris_YOLO.pt"       # Path to the YOLO model (.pt file)

# Specify the label type: either "normal" or "fraud"
LABEL="fraud"

# Input and output directories based on the LABEL
INPUT_DIR="dataset/$LABEL"             # Original data folder (e.g., "dataset/normal" or "dataset/fraud")
PROCESSED_DIR="processed_dataset_no_ocr_neg32_$LABEL"  # Directory to save processed data (anchor, pos, neg images + CSV)

# Preprocessing hyperparameters
BATCH_SIZE=32
MARGIN=50
NEGATIVE_PER_IMAGE=32
NUM_WORKERS=8
CUDA_DEVICE=1


# Set train, validation, and test ratios based on the LABEL
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
  --cuda_device "${CUDA_DEVICE}"  # 추가된 부분
