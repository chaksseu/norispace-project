#!/bin/bash
################################################################################
# run_training.sh
# 사용 예:
#   chmod +x run_training.sh
#   ./run_training.sh
################################################################################

# (선택) 가상환경 활성화
# source /path/to/venv/bin/activate

###############################################################################
# 파라미터 설정
###############################################################################
NORMAL_DATA_DIR="processed_dataset_normal"   # Normal 데이터셋 루트 (train/val/test)
FRAUD_DATA_DIR="processed_dataset_fraud"     # Fraud 데이터셋 루트 (val/test)
OUTPUT_DIR="./checkpoints"                   # 모델 체크포인트 저장 위치
WANDB_PROJECT_NAME="0111_norispace_project"

BATCH_SIZE=32
LR=1e-4
EPOCHS=2048
EVAL_EPOCH=10
SAVE_EPOCH=100
NUM_WORKERS=4

# pretrained 사용 여부
PRETRAINED="--pretrained"

###############################################################################
# 학습 스크립트 실행
###############################################################################
python src/train_contrastive.py \
  --normal_data_dir "${NORMAL_DATA_DIR}" \
  --fraud_data_dir "${FRAUD_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --eval_epoch "${EVAL_EPOCH}" \
  --save_epoch "${SAVE_EPOCH}" \
  --num_workers "${NUM_WORKERS}" \
  --wandb_project "${WANDB_PROJECT_NAME}" \
  ${PRETRAINED}
