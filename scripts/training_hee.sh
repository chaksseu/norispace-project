#!/bin/bash

# ------------------------------------------------------------------------------------
# Contrastive Learning Training Script with Updated Features
# ------------------------------------------------------------------------------------

# 에러 발생 시 스크립트 중단
set -e

# W&B API Key 설정
export WANDB_API_KEY="be0eca18374643fcb4fc922e243f869ae65de5b1"  # 본인의 W&B API Key를 입력하세요

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set. Please configure it as an environment variable."
    exit 1
fi

# GPU 확인 및 설정
if ! command -v nvidia-smi &> /dev/null; then
    echo "GPU not found. Running on CPU."
    export CUDA_VISIBLE_DEVICES=""
else
    echo "Detected GPUs: $(nvidia-smi --list-gpus)"
fi

# 기본 파라미터 설정
AUGMENTED_FILE_PATH="./data/augmented_ocr/"  # 처리된 데이터셋 경로
OUTPUT_DIR="./results_ocr"  # 모델 체크포인트 저장 경로
BATCH_SIZE=128  # 배치 크기
GRAD_ACC_STEPS=1  # Gradient Accumulation Steps
LR=1e-5  # 학습률
EPOCHS=1000  # 총 학습 에폭 수
EVAL_EPOCH=1  # 검증 주기 (에폭 단위)
SAVE_EPOCH=50  # 체크포인트 저장 주기 (에폭 단위)
NUM_WORKERS=16  # DataLoader 워커 수
PRETRAINED=true  # ConvNeXt 모델에 사전 학습 가중치 사용 여부
WANDB_PROJECT="fake_detection"  # W&B 프로젝트 이름
WANDB_ENTITY="noah_"  # W&B 엔터티(팀) 이름
WANDB_RUN_NAME="Noris_ocr"  # W&B 실행 이름
CUDA_DEVICES=0  # 사용할 GPU ID
MIXED_PRECISION="fp16"  # 혼합 정밀도 모드 (fp16 사용)
THRESHOLD=0.11

# 실행 시작 시간 기록
START_TIME=$(date +%s)

# main.py 파일 확인
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in the current directory."
    exit 1
fi

echo "Starting training at $(date)"
echo "Using GPU: $CUDA_DEVICES"
echo "W&B Project: $WANDB_PROJECT, Entity: $WANDB_ENTITY, Run Name: $WANDB_RUN_NAME"

# Accelerate 설정 확인
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "Initializing Accelerate configuration for distributed training."
    accelerate config
fi

# Python 스크립트 실행 및 로그 저장
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --mixed_precision="$MIXED_PRECISION" main_hee.py \
  --augmented_file_path "$AUGMENTED_FILE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --lr $LR \
  --epochs $EPOCHS \
  --eval_epoch $EVAL_EPOCH \
  --save_epoch $SAVE_EPOCH \
  --num_workers $NUM_WORKERS \
  $(if [ "$PRETRAINED" = true ]; then echo "--pretrained"; fi) \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --threshold "$THRESHOLD"
  > training.log 2>&1

# 에러 발생 시 로그 출력
if [ $? -ne 0 ]; then
    echo "Error occurred during training. Check training.log for details."
    tail -n 20 training.log
    exit 1
fi

# 실행 종료 시간 기록
END_TIME=$(date +%s)

# 실행 시간 출력
ELAPSED_TIME=$(($END_TIME - $START_TIME))
echo "** Training completed at $(date)"
echo "** Training took $ELAPSED_TIME seconds."
