# 인퍼런스 파이프라인 README

## 개요
이 프로젝트는 YOLO 기반 전처리, Anchor/이미지 생성, 모델 추론을 포함한 이미지 처리 파이프라인을 제공합니다.

---

## 파일 설명

### 1. **run_pipeline.sh**
파이프라인을 실행하는 bash 스크립트입니다.

- **사용법:**
  1. 실행 권한 추가 (필요시):
     ```
     chmod +x run_pipeline.sh
     ```
  2. 스크립트 실행:
     ```
     ./run_pipeline.sh
     ```

- **주요 변수:**
  - `CHECKPOINT_PATH`: 모델 체크포인트 경로
  - `INPUT_DIR`: 원본 이미지 디렉토리 (평가를 하기 위한 처방전)
  - `PROCESSED_DIR`: 전처리된 데이터 저장 디렉토리
  - `OUTPUT_DIR`: 추론 결과 저장 디렉토리
  - `YOLO_MODEL_PATH`: YOLO 모델 경로
  - `THRESHOLD`: 분류 임계값
  - `DEVICE`: 사용 디바이스 (`cuda` 또는 `cpu`)
  - `BATCH_SIZE`: 배치 크기
  - `MODE`: 실행 모드

---

### 2. **inference_pipeline.py**
YOLO 전처리, Anchor/이미지 생성, 모델 추론을 수행하는 Python 스크립트입니다.

- **주요 기능:**
  - **전처리:** YOLO를 이용해 이미지에서 바운딩 박스를 검출하고, Excluded 및 Cropped 이미지를 생성합니다.
  - **Anchor/이미지 생성:** 검출된 바운딩 박스를 기반으로 Anchor와 이미지를 생성합니다.
  - **추론:** 사전 학습된 모델을 사용해 Anchor와 이미지를 임베딩하고, 거리 기반으로 "Normal" 또는 "Fraud"로 분류합니다.
  - **결과 저장:** 분류된 이미지를 각각의 디렉토리에 저장하고, 결과를 파일로 기록합니다.

---

## 파이프라인 워크플로우

1. **YOLO 전처리**
   - 이미지를 분석하여 바운딩 박스를 검출하고, Excluded 및 Cropped 이미지를 생성합니다.
2. **Anchor 및 이미지 생성**
   - 검출된 바운딩 박스를 기반으로 Anchor를 크롭하고, 이미지를 생성합니다.
3. **추론**
   - 생성된 Anchor와 이미지를 모델에 입력하여 "Normal" 또는 "Fraud"로 분류합니다.

---

## 출력 결과

- **전처리 데이터:** `PROCESSED_DIR`에 저장됩니다.
  - Excluded 이미지
  - Cropped 이미지
  - `results.csv`: YOLO 검출 결과

- **추론 결과:** `OUTPUT_DIR`에 저장됩니다.
  - `predicted_normal`: 정상으로 분류된 이미지
  - `predicted_fraud`: 사기로 분류된 이미지
  - `results.txt`: 분류 결과 요약

---

## 참고 사항

- YOLO와 PyTorch가 설치되어 있어야 합니다.
- 입력 및 출력 디렉토리 구조를 준수해야 합니다.
- 필요에 따라 `run_pipeline.sh`에서 변수 값을 조정할 수 있습니다.
