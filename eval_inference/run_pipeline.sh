#!/bin/bash

# 필요시 실행 전에 실행 권한 추가
# chmod +x run_inference_pipeline.sh

# 사용자 정의 변수 설정
CHECKPOINT_PATH="./checkpoint_epoch_200.pt"
INPUT_DIR="./eval_data/data"
PROCESSED_DIR="./eval_data/processed"
OUTPUT_DIR="./eval_data/eval_results"
YOLO_MODEL_PATH="./YOLO_best.pt"
THRESHOLD=0.2
DEVICE="cuda"
BATCH_SIZE=32
MODE="last_processed"

# Python 스크립트 실행
{
    python inference_pipeline.py \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --input_dir "$INPUT_DIR" \
      --processed_dir "$PROCESSED_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --yolo_model_path "$YOLO_MODEL_PATH" \
      --threshold "$THRESHOLD" \
      --device "$DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --mode "$MODE" 
} || {
    echo "Error: Pipeline 실행 중 문제가 발생했습니다."
    exit 1
}

echo "Pipeline 실행이 완료되었습니다. 결과는 $OUTPUT_DIR에 저장되었습니다."
