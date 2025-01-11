#!/bin/bash
################################################################################
# run_preprocessing.sh
# 사용 예시:
#   chmod +x run_preprocessing.sh
#   ./run_preprocessing.sh
################################################################################

# 가상환경 활성화 예시 (필요 시 주석 해제)
# source /path/to/venv/bin/activate

################################################################################
# 파라미터 설정
################################################################################
YOLO_MODEL_PATH="Noris_YOLO.pt"       # YOLO 모델(.pt) 경로
INPUT_DIR="dataset/normal"            # 원본 normal/fruad 데이터 폴더
PROCESSED_DIR="processed_dataset_test"  # 최종 YOLO/전처리 결과가 쌓일 폴더

BATCH_SIZE=64
MARGIN=50
NEGATIVE_PER_IMAGE=20
NUM_WORKERS=8
GENERATE_NEG="--generate_negatives"   # neg 이미지를 생성하려면 플래그 추가
# GENERATE_NEG=""                     # neg를 생성하지 않으려면 주석 처리

SPLIT_SEED=42
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

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
  ${GENERATE_NEG}
