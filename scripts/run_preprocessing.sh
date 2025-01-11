#!/bin/bash

################################################################################
# 파라미터 설정
################################################################################
YOLO_MODEL_PATH="Noris_YOLO.pt"       # YOLO 모델(.pt) 경로

# normal 또는 fraud
LABEL="fraud"
# 원본 데이터 폴더 (예: "dataset/normal" 또는 "dataset/fraud")
INPUT_DIR="dataset/$LABEL"
# 결과물이 저장될 폴더 (anchor, pos, neg 이미지 + CSV)
PROCESSED_DIR="processed_dataset_$LABEL"

# LABEL에 따라 train, val, test 비율 설정
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

BATCH_SIZE=32
MARGIN=50
NEGATIVE_PER_IMAGE=8
NUM_WORKERS=4


SPLIT_SEED=42

################################################################################
# 파이썬 스크립트 실행
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
  --label "${LABEL}"
