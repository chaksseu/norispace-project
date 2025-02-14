#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_preprocess.py

Workflow:
1) YOLO를 이용해 bounding box를 검출
   - 결과를 processed_dir/results.csv 에 저장 (original_file, label, x1, y1, x2, y2, ...)
   - bbox 영역만 잘라낸 이미지(cropped_파일명) -> Positive set 원본
   - bbox 부분을 검정색으로 마스킹한 이미지(excluded_파일명) -> Anchor set 원본

2) results.csv를 바탕으로 margin을 고려한 anchor/pos/neg 이미지를 생성
   - anchor: excluded 이미지를 margin 포함해 잘라낸 것
   - pos   : anchor 영역 안에 bbox 크롭 이미지를 paste 후 margin 포함해 잘라낸 것
   - neg   : pos에 사용할 bbox 이미지를 OCR 기반 random_jitter 후 붙여 margin 포함해 잘라낸 여러 장

3) 생성된 모든 샘플(각 파일별 디렉토리)을 정확히 train:val:test 비율로 분할
   - 그리고 각 (anchor, pos, neg) 파일 경로 + 라벨 + subset 정보를 CSV로 최종 기록
"""

import os
import argparse
import logging
import shutil
import random
import multiprocessing
from functools import partial

import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO

import cv2
import easyocr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline with YOLO detection, anchor/pos/neg creation, and train/val/test split")

    # YOLO + 기본 전처리 관련
    parser.add_argument("--yolo_model_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--margin", type=int, default=50)

    # 증강(neg) 관련
    parser.add_argument("--num_color_jitter", type=int, default=5)
    parser.add_argument("--generate_negatives", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    # train/val/test split 관련
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # 추가: 정상/부정 라벨
    parser.add_argument("--label", type=str, default="normal",
                        help="Label for the input dataset (normal/fraud)")

    args = parser.parse_args()
    return args


# ----------------------------------------------------------------------
# [1] YOLO Inference -> results.csv, excluded_*, cropped_*
# ----------------------------------------------------------------------
def preprocess_with_yolo(yolo_model_path, input_dir, processed_dir, batch_size=64, label="normal"):
    """
    YOLO 모델로 input_dir 하위 모든 이미지를 탐지한다.
    - processed_dir/results.csv 에 bbox 정보와 label을 함께 저장
    - excluded_XXX, cropped_XXX 이미지(중간 결과) 저장
    """
    logging.info("[Step1] YOLO 감지 -> excluded/cropped 생성 -> results.csv 저장")

    tmp_dir = os.path.join(processed_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    model = YOLO(yolo_model_path)
    csv_data = []

    def process_batch(image_paths):
        results = model(image_paths)
        for image_path, result in zip(image_paths, results):
            image_file = os.path.basename(image_path)
            try:
                img = Image.open(image_path).convert("RGB")
                w, h = img.size
            except Exception as e:
                logging.warning(f"Image load failed {image_path}: {e}")
                continue

            exclude_save_path = os.path.join(tmp_dir, f"excluded_{os.path.splitext(image_file)[0]}.png")
            crop_save_path    = os.path.join(tmp_dir, f"cropped_{os.path.splitext(image_file)[0]}.png")

            if len(result.boxes) > 0:
                # 가장 confidence 높은 bbox만 사용
                best_box = max(result.boxes, key=lambda box: box.conf[0])
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                confidence = float(best_box.conf[0].cpu().numpy())
                class_id   = int(best_box.cls[0].cpu().numpy())

                csv_data.append({
                    "original_file": image_file,
                    "label": label,  # ★ normal/fraud
                    "class_id": class_id,
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                # cropped
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_img.save(crop_save_path)

                # excluded
                excluded_img = img.copy()
                mask = Image.new("L", excluded_img.size, 0)
                mask.paste(255, (int(x1), int(y1), int(x2), int(y2)))
                black_img = Image.new("RGB", excluded_img.size, (0, 0, 0))
                excluded_final = Image.composite(black_img, excluded_img, mask)
                excluded_final.save(exclude_save_path)

            else:
                # bbox 검출 없음 -> 저장하지 않음
                continue

    # input_dir 하위 모든 이미지 수집
    all_images = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg','.jpeg','.png','.tif','.bmp')):
                all_images.append(os.path.join(root, fname))

    # 배치 추론
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i : i + batch_size]
        process_batch(batch)

    # CSV 저장
    df = pd.DataFrame(csv_data)
    df_path = os.path.join(processed_dir, "results.csv")
    df.to_csv(df_path, index=False)
    logging.info(f"YOLO 전처리 완료: {df_path}")


# ----------------------------------------------------------------------
# [2] OCR+증강 -> anchor/pos/neg 생성
# ----------------------------------------------------------------------
def random_jitter(image: Image.Image) -> Image.Image:
    """
    EasyOCR로 텍스트를 찾아서 color/brightness/contrast/회전/이동/스케일 등을 무작위 적용.
    """
    try:
        import numpy as np
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        reader = easyocr.Reader(['en'], gpu=True)
        results = reader.readtext(cv_img, detail=1)

        augmented = image.copy()

        for bbox, text, conf in results:
            if conf < 0.6:
                continue
            # bbox 좌표
            (topleft, topright, botright, botleft) = bbox
            x_min = int(min(topleft[0], botleft[0]))
            y_min = int(min(topleft[1], topright[1]))
            x_max = int(max(botright[0], topright[0]))
            y_max = int(max(botright[1], botleft[1]))

            crop_region = augmented.crop((x_min, y_min, x_max, y_max))

            # 부분 색상/밝기/대비 등
            if random.random() < 0.7:
                factor = random.uniform(0.85, 1.15)
                crop_region = ImageEnhance.Color(crop_region).enhance(factor)
            if random.random() < 0.7:
                factor = random.uniform(0.85, 1.15)
                crop_region = ImageEnhance.Brightness(crop_region).enhance(factor)
            if random.random() < 0.7:
                factor = random.uniform(0.85, 1.15)
                crop_region = ImageEnhance.Contrast(crop_region).enhance(factor)
            if random.random() < 0.7:
                factor = random.uniform(0.85, 1.15)
                crop_region = ImageEnhance.Sharpness(crop_region).enhance(factor)

            # 회전
            if random.random() < 0.7:
                angle = random.uniform(-2, 2)
                crop_region = crop_region.rotate(
                    angle, expand=True, fillcolor=(255,255,255)
                )

            # 스케일
            if random.random() < 0.7:
                scale = random.uniform(0.9, 1.1)
                new_w = max(1, int(crop_region.width * scale))
                new_h = max(1, int(crop_region.height * scale))
                crop_region = crop_region.resize((new_w, new_h), Image.BICUBIC)

            # 이동
            dx, dy = 0, 0
            if random.random() < 0.7:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)

            # 원본에 다시 붙이기
            paste_x = max(0, min(augmented.width - crop_region.width, x_min + dx))
            paste_y = max(0, min(augmented.height - crop_region.height, y_min + dy))
            augmented.paste(crop_region, (paste_x, paste_y))

        # 전체 이미지에도 color/brightness 등
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            augmented = ImageEnhance.Color(augmented).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            augmented = ImageEnhance.Brightness(augmented).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            augmented = ImageEnhance.Contrast(augmented).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            augmented = ImageEnhance.Sharpness(augmented).enhance(factor)

        # Affine
        angle = 0
        translate_x, translate_y = 0, 0
        scale_factor = 1.0

        if random.random() < 0.5:
            angle = random.uniform(-2, 2)
        if random.random() < 0.5:
            translate_x = random.uniform(-2, 2)
            translate_y = random.uniform(-2, 2)
        if random.random() < 0.5:
            scale_factor = random.uniform(0.96, 1.04)

        angle_rad = np.deg2rad(angle)
        cos_t = np.cos(angle_rad) * scale_factor
        sin_t = np.sin(angle_rad) * scale_factor

        w, h = augmented.size
        cx, cy = w/2, h/2

        a = cos_t
        b = sin_t
        c = (1 - cos_t)*cx - sin_t*cy + translate_x
        d = -sin_t
        e = cos_t
        f = sin_t*cx + (1 - cos_t)*cy + translate_y

        affine_mat = (a, b, c, d, e, f)

        final_img = augmented.transform(
            augmented.size,
            Image.AFFINE,
            affine_mat,
            resample=Image.BICUBIC,
            fillcolor=(255,255,255)
        )
        return final_img

    except Exception as e:
        logging.error(f"random_jitter error: {e}")
        return image


def create_samples_for_row(row, tmp_dir, processed_dir, subset,
                           num_color_jitter=5, generate_neg=False, margin=50):
    """
    row: results.csv의 각 행 (original_file, label, x1, y1, x2, y2, ...)
    subset: train/val/test
    """
    file_name = row["original_file"]
    label     = row["label"]  # "normal" or "fraud"
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

    # 폴더
    anchor_dir = os.path.join(processed_dir, subset, "anchor")
    pos_dir    = os.path.join(processed_dir, subset, "pos")
    neg_dir    = os.path.join(processed_dir, subset, "neg")

    excluded_path = os.path.join(tmp_dir, f"excluded_{os.path.splitext(file_name)[0]}.png")
    cropped_path  = os.path.join(tmp_dir, f"cropped_{os.path.splitext(file_name)[0]}.png")

    if pd.isnull(x1) or pd.isnull(y1) or pd.isnull(x2) or pd.isnull(y2):
        # bbox 없음
        return

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return

    if not os.path.exists(excluded_path):
        return

    # margin 적용
    with Image.open(excluded_path).convert("RGB") as exc_img:
        w, h = exc_img.size
        crop_x1 = max(x1 - margin, 0)
        crop_y1 = max(y1 - margin, 0)
        crop_x2 = min(x2 + margin, w)
        crop_y2 = min(y2 + margin, h)

        # Anchor
        anchor_crop = exc_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        anchor_filename = f"{os.path.splitext(file_name)[0]}_anchor.png"
        anchor_save_path = os.path.join(anchor_dir, anchor_filename)
        anchor_crop.save(anchor_save_path)

        # Positive
        if not os.path.exists(cropped_path):
            return
        with Image.open(cropped_path).convert("RGB") as cr_img:
            bw, bh = (x2 - x1), (y2 - y1)
            if bw <= 0 or bh <= 0:
                return
            resized_crop = cr_img.resize((bw, bh), Image.BILINEAR)

            combined_pos = exc_img.copy()
            combined_pos.paste(resized_crop, (x1, y1))
            pos_crop = combined_pos.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            pos_filename = f"{os.path.splitext(file_name)[0]}_pos.png"
            pos_save_path = os.path.join(pos_dir, pos_filename)
            pos_crop.save(pos_save_path)

            # Negative
            if generate_neg:
                for i in range(num_color_jitter):
                    jittered_img = random_jitter(resized_crop)
                    combined_neg = exc_img.copy()
                    combined_neg.paste(jittered_img, (x1, y1))
                    neg_crop = combined_neg.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    neg_filename = f"{os.path.splitext(file_name)[0]}_neg_{i}.png"
                    neg_save_path = os.path.join(neg_dir, neg_filename)
                    neg_crop.save(neg_save_path)

    # 끝.


def worker_fn(row, tmp_dir, processed_dir, subset,
              num_color_jitter=5, generate_neg=False, margin=50):
    create_samples_for_row(
        row=row,
        tmp_dir=tmp_dir,
        processed_dir=processed_dir,
        subset=subset,
        num_color_jitter=num_color_jitter,
        generate_neg=generate_neg,
        margin=margin
    )


# ----------------------------------------------------------------------
# [3] train/val/test split
#     => anchor/pos/neg 생성 후, 각 파일 경로와 label, subset 정보 CSV 기록
# ----------------------------------------------------------------------
def create_anchor_pos_neg_and_split(processed_dir,
                                    num_color_jitter=5,
                                    generate_neg=False,
                                    margin=50,
                                    train_ratio=0.8,
                                    val_ratio=0.1,
                                    test_ratio=0.1,
                                    num_workers=4,
                                    seed=42):
    """
    Step2: results.csv를 읽어, anchor/pos/neg 파일을 생성한 뒤,
    확정적으로 train/val/test 비율대로 스플릿.
    최종적으로 (anchor_file, pos_file, neg_file, label, subset) CSV를 작성해둘 수 있음.
    """
    # CSV 로드
    csv_path = os.path.join(processed_dir, "results.csv")
    if not os.path.exists(csv_path):
        logging.error(f"results.csv가 {processed_dir}에 없습니다.")
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        logging.warning("results.csv가 비어있습니다.")
        return

    total_samples = len(df)
    logging.info(f"총 샘플 수: {total_samples}")

    # 셔플
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 각 subset 개수
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count

    df['subset'] = ['train'] * train_count + ['val'] * val_count + ['test'] * test_count

    tmp_dir = os.path.join(processed_dir, "tmp")
    if not os.path.exists(tmp_dir):
        logging.error(f"임시 폴더가 존재하지 않습니다: {tmp_dir}")
        return

    # 폴더 생성
    for sp in ["train", "val", "test"]:
        for sub_dir in ["anchor", "pos", "neg"]:
            sp_dir = os.path.join(processed_dir, sp, sub_dir)
            os.makedirs(sp_dir, exist_ok=True)

    # 병렬 처리
    rows = df.to_dict("records")
    logging.info("[Step2] anchor/pos/neg 생성 중...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for row in rows:
            subset = row['subset']
            futures.append(executor.submit(
                worker_fn, row, tmp_dir, processed_dir, subset,
                num_color_jitter, generate_neg, margin
            ))
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Creating anchor/pos/neg"):
            pass

    logging.info("anchor/pos/neg + train/val/test 분할 완료.")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info(f"임시 폴더 삭제: {tmp_dir}")

    # 여기서 추가적으로, (anchor_file, pos_file, neg_file, label, subset) 형태로 정리한 CSV를 만들려면
    # 파일 탐색하여 매핑 로직을 추가하면 됩니다.
    # 본 예시에서는 별도의 "results_splitted.csv" 생략.

def main():
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_arguments()

    # Step1) YOLO로 bbox 검출
    preprocess_with_yolo(
        yolo_model_path=args.yolo_model_path,
        input_dir=args.input_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        label=args.label
    )

    # Step2) anchor/pos/neg 생성 + train/val/test 분할
    create_anchor_pos_neg_and_split(
        processed_dir=args.processed_dir,
        num_color_jitter=args.num_color_jitter,
        generate_neg=args.generate_negatives,
        margin=args.margin,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        seed=args.split_seed
    )

    logging.info("=== 전처리( YOLO + anchor/pos/neg + split ) 완료 ===")


if __name__ == "__main__":
    main()
