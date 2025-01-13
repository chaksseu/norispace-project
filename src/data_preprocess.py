#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_preprocess.py

Overall workflow:
1) Use YOLO to detect bounding boxes in images:
   - Saves 'results.csv' with bounding box info (filename, label, x1, y1, x2, y2, etc.)
   - Saves cropped images (bounding box regions)
   - Saves excluded images (bounding box areas masked in black)
2) Based on 'results.csv', create anchor/pos/neg samples:
   - anchor: Crop the excluded image (with margin)
   - pos: Paste the cropped bounding box onto the excluded image, then crop with margin
   - neg: Apply OCR-based augmentation (text replacement, distortion) to the bounding box, then paste and crop with margin
3) Split all generated samples into train/val/test subsets
   - Each sample (anchor, pos, neg) will be stored in corresponding directories
   - Potentially record file paths in a final CSV if needed
"""

import os
import argparse
import logging
import shutil
import random
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import cv2
import easyocr
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Global list to hold prepared text images (e.g., from OCR or various fonts/styles)
# ------------------------------------------------------------------------------
text_images_pool = []

def load_text_image_pool(folder_path: str):
    """
    Loads all images from the specified folder_path into 'text_images_pool'.
    Each image is assumed to be a single character or text snippet.
    This can be extended to store any type of text-based images for augmentation.
    """
    text_images_pool = []
    if not os.path.isdir(folder_path):
        logging.warning(f"Text image directory '{folder_path}' not found.")
        return

    for fn in os.listdir(folder_path):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, fn)
            try:
                timg = Image.open(img_path).convert("RGB")
                text_images_pool.append(timg)
            except Exception as e:
                logging.warning(f"Failed to load image '{fn}': {e}")
    return text_images_pool
# Configure logging to display info messages
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline with YOLO detection and OCR-based augmentation"
    )

    # YOLO and basic preprocessing parameters
    parser.add_argument("--yolo_model_path", type=str, required=True,
                        help="Path to the YOLO model (.pt file)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the original images")
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Directory to save processed outputs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for YOLO inference")
    parser.add_argument("--margin", type=int, default=50,
                        help="Margin size to include around bounding boxes")

    # Augmentation (negatives) parameters
    parser.add_argument("--num_color_jitter", type=int, default=5,
                        help="Number of negative samples to generate per image")
    parser.add_argument("--generate_negatives", action="store_true",
                        help="Flag to generate negative samples with augmentations")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for processing")

    # Train/Val/Test split parameters
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for data splitting")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of data for the training set")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion of data for the validation set")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Proportion of data for the test set")
    # Label parameter
    parser.add_argument("--label", type=str, default="normal",
                        help="Label for the input dataset (e.g., normal/fraud)")

    # CUDA device selection
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA device index (default=0)")

    # (New) Directory containing text images for augmentation
    parser.add_argument("--text_image_dir", type=str, default=None,
                        help="Directory where text images for augmentation are stored")

    args = parser.parse_args()
    return args

# ------------------------------------------------------------------------------
# [1] YOLO Inference -> results.csv, excluded_*, cropped_*
# ------------------------------------------------------------------------------
def preprocess_with_yolo(yolo_model_path, input_dir, processed_dir,
                         batch_size=64, label="normal", device="cuda:0"):
    """
    Runs YOLO model on all images in 'input_dir' to detect bounding boxes.
    Saves bounding box info to 'results.csv', and also creates:
      - excluded_<filename>.png (masked bounding box area)
      - cropped_<filename>.png  (the bounding box region itself)
    """
    logging.info("[Step1] Running YOLO detection and generating excluded/cropped images.")

    tmp_dir = os.path.join(processed_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    model = YOLO(yolo_model_path).to(device)
    csv_data = []

    def process_batch(image_paths):
        results = model(image_paths)
        for image_path, result in zip(image_paths, results):
            image_file = os.path.basename(image_path)
            try:
                img = Image.open(image_path).convert("RGB")
                w, h = img.size
            except Exception as e:
                logging.warning(f"Failed to load image {image_path}: {e}")
                continue

            exclude_save_path = os.path.join(tmp_dir, f"excluded_{os.path.splitext(image_file)[0]}.png")
            crop_save_path = os.path.join(tmp_dir, f"cropped_{os.path.splitext(image_file)[0]}.png")

            if len(result.boxes) > 0:
                # Take the bounding box with the highest confidence
                best_box = max(result.boxes, key=lambda box: box.conf[0])
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                confidence = float(best_box.conf[0].cpu().numpy())
                class_id = int(best_box.cls[0].cpu().numpy())

                # Record bounding box info
                csv_data.append({
                    "original_file": image_file,
                    "label": label,
                    "class_id": class_id,
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                # Save cropped region
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_img.save(crop_save_path)

                # Save excluded image (bounding box area masked)
                excluded_img = img.copy()
                mask = Image.new("L", excluded_img.size, 0)
                mask.paste(255, (int(x1), int(y1), int(x2), int(y2)))
                black_img = Image.new("RGB", excluded_img.size, (0, 0, 0))
                excluded_final = Image.composite(black_img, excluded_img, mask)
                excluded_final.save(exclude_save_path)

    # Collect all images
    all_images = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                all_images.append(os.path.join(root, fname))

    # Process in batches
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i: i + batch_size]
        process_batch(batch)

    # Save bounding box info to CSV
    df = pd.DataFrame(csv_data)
    df_path = os.path.join(processed_dir, "results.csv")
    df.to_csv(df_path, index=False)
    logging.info(f"YOLO preprocessing completed. Results saved to: {df_path}")


# ------------------------------------------------------------------------------
# [2] OCR-based Augmentation -> anchor/pos/neg
# ------------------------------------------------------------------------------
def random_jitter(image: Image.Image, use_ocr=True) -> Image.Image:
    """
    Applies random augmentation to 'image':
    1) OCR detection on text regions
    2) For each text region:
       - (A) Distort original text region, OR
       - (B) Replace with a randomly selected text image from 'text_images_pool'
         (with random resizing / padding)
    3) Apply global color/brightness/contrast/sharpness changes and affine transform
    """
    try:
        augmented = image.copy()
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if use_ocr:
            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            results = reader.readtext(cv_img, detail=1)  # returns [(bbox, text, conf), ...]

            for detection in results:
                bbox, text, conf = detection
                if conf < 0.6:
                    # Skip low-confidence text
                    continue

                (topleft, topright, botright, botleft) = bbox
                x_min = int(min(topleft[0], botleft[0]))
                y_min = int(min(topleft[1], topright[1]))
                x_max = int(max(botright[0], topright[0]))
                y_max = int(max(botright[1], botleft[1]))

                # Ensure valid region
                if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
                    continue

                region_w = x_max - x_min
                region_h = y_max - y_min

                text_images_pool = load_text_image_pool("text_images_pool")

                # Decide whether to use a text image from the pool or to distort the original
                
                if text_images_pool and random.random() < 0.3:
                    # (B) Replace with a random text image
                    text_img = random.choice(text_images_pool)
                    # Random padding
                    pad_x = random.randint(-5, 5)
                    pad_y = random.randint(-5, 5)
                    new_w = max(1, region_w + pad_x)
                    new_h = max(1, region_h + pad_y)
                    text_img_resized = text_img.resize((new_w, new_h), Image.BILINEAR)

                    # Center the replaced image in the region
                    paste_x = x_min + (region_w - new_w) // 2
                    paste_y = y_min + (region_h - new_h) // 2

                    augmented.paste(text_img_resized, (paste_x, paste_y))

                else:
                    # (A) Distort the original text region
                    crop_region = augmented.crop((x_min, y_min, x_max, y_max))

                    # Random color/brightness/contrast/sharpness
                    if random.random() < 0.7:
                        factor = random.uniform(0.8, 1.2)
                        crop_region = ImageEnhance.Color(crop_region).enhance(factor)
                    if random.random() < 0.7:
                        factor = random.uniform(0.8, 1.2)
                        crop_region = ImageEnhance.Brightness(crop_region).enhance(factor)
                    if random.random() < 0.7:
                        factor = random.uniform(0.8, 1.2)
                        crop_region = ImageEnhance.Contrast(crop_region).enhance(factor)
                    if random.random() < 0.7:
                        factor = random.uniform(0.8, 1.2)
                        crop_region = ImageEnhance.Sharpness(crop_region).enhance(factor)

                    # Random rotation
                    if random.random() < 0.7:
                        angle = random.uniform(-5, 5)
                        crop_region = crop_region.rotate(angle, expand=True, fillcolor=(255, 255, 255))

                    # Random scaling
                    if random.random() < 0.7:
                        scale = random.uniform(0.9, 1.3)
                        new_crop_w = max(1, int(crop_region.width * scale))
                        new_crop_h = max(1, int(crop_region.height * scale))
                        crop_region = crop_region.resize((new_crop_w, new_crop_h), Image.BICUBIC)

                    # Random translation
                    dx, dy = 0, 0
                    if random.random() < 0.7:
                        dx = random.randint(-2, 2)
                        dy = random.randint(-2, 2)

                    # Place the augmented region back
                    paste_x = max(0, min(augmented.width - crop_region.width, x_min + dx))
                    paste_y = max(0, min(augmented.height - crop_region.height, y_min + dy))
                    augmented.paste(crop_region, (paste_x, paste_y))

        # Global augmentations: color/brightness/contrast/sharpness
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

        # Global affine transform
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
        cx, cy = w / 2, h / 2

        a = cos_t
        b = sin_t
        c = (1 - cos_t) * cx - sin_t * cy + translate_x
        d = -sin_t
        e = cos_t
        f = sin_t * cx + (1 - cos_t) * cy + translate_y

        affine_mat = (a, b, c, d, e, f)
        final_img = augmented.transform(
            augmented.size,
            Image.AFFINE,
            affine_mat,
            resample=Image.BICUBIC,
            fillcolor=(255, 255, 255)
        )

        return final_img

    except Exception as e:
        logging.error(f"random_jitter error: {e}")
        return image


def create_samples_for_row(row, tmp_dir, processed_dir, subset,
                           num_color_jitter=5, generate_neg=False, margin=50):
    """
    Creates anchor, pos, and neg samples for a single row from results.csv.
    - anchor: excluded image cropped with margin
    - pos: bounding box is pasted on anchor and cropped with margin
    - neg: uses random_jitter (OCR-based augmentation) to generate multiple negative samples
    """
    file_name = row["original_file"]
    label = row["label"]
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

    anchor_dir = os.path.join(processed_dir, subset, "anchor")
    pos_dir = os.path.join(processed_dir, subset, "pos")
    neg_dir = os.path.join(processed_dir, subset, "neg")

    excluded_path = os.path.join(tmp_dir, f"excluded_{os.path.splitext(file_name)[0]}.png")
    cropped_path = os.path.join(tmp_dir, f"cropped_{os.path.splitext(file_name)[0]}.png")

    # Check bounding box validity
    if pd.isnull(x1) or pd.isnull(y1) or pd.isnull(x2) or pd.isnull(y2):
        return

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return

    if not os.path.exists(excluded_path):
        return

    with Image.open(excluded_path).convert("RGB") as exc_img:
        w, h = exc_img.size
        # Apply margin
        crop_x1 = max(x1 - margin, 0)
        crop_y1 = max(y1 - margin, 0)
        crop_x2 = min(x2 + margin, w)
        crop_y2 = min(y2 + margin, h)

        # anchor
        anchor_crop = exc_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        anchor_filename = f"{os.path.splitext(file_name)[0]}_anchor.png"
        anchor_save_path = os.path.join(anchor_dir, anchor_filename)
        anchor_crop.save(anchor_save_path)

        # pos
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

            # neg
            if generate_neg:
                for i in range(num_color_jitter):
                    jittered_img = random_jitter(resized_crop, use_ocr=True)
                    combined_neg = exc_img.copy()
                    combined_neg.paste(jittered_img, (x1, y1))
                    neg_crop = combined_neg.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    neg_filename = f"{os.path.splitext(file_name)[0]}_neg_{i}.png"
                    neg_save_path = os.path.join(neg_dir, neg_filename)
                    neg_crop.save(neg_save_path)


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


# ------------------------------------------------------------------------------
# [3] Split into train/val/test
# ------------------------------------------------------------------------------
def create_anchor_pos_neg_and_split(processed_dir,
                                    num_color_jitter=5,
                                    generate_neg=False,
                                    margin=50,
                                    train_ratio=0.8,
                                    val_ratio=0.1,
                                    test_ratio=0.1,
                                    num_workers=4,
                                    seed=42):
    csv_path = os.path.join(processed_dir, "results.csv")
    if not os.path.exists(csv_path):
        logging.error(f"results.csv not found in {processed_dir}.")
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        logging.warning("results.csv is empty.")
        return

    total_samples = len(df)
    logging.info(f"Total samples: {total_samples}")

    # Shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Determine the number of samples for each subset
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count

    # Assign subset labels
    df['subset'] = ['train'] * train_count + ['val'] * val_count + ['test'] * test_count

    tmp_dir = os.path.join(processed_dir, "tmp")
    if not os.path.exists(tmp_dir):
        logging.error(f"Temporary directory does not exist: {tmp_dir}")
        return

    # Create output directories
    for sp in ["train", "val", "test"]:
        for sub_dir in ["anchor", "pos", "neg"]:
            sp_dir = os.path.join(processed_dir, sp, sub_dir)
            os.makedirs(sp_dir, exist_ok=True)

    rows = df.to_dict("records")
    logging.info("[Step2] Creating anchor/pos/neg samples...")

    # Parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for row in rows:
            subset = row['subset']
            futures.append(executor.submit(
                worker_fn, row, tmp_dir, processed_dir, subset,
                num_color_jitter, generate_neg, margin
            ))
        # Show progress bar
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Creating anchor/pos/neg"):
            pass

    logging.info("Anchor/Pos/Neg creation and train/val/test splitting completed.")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info(f"Temporary directory removed: {tmp_dir}")


def main():
    try:
        # Ensure the multiprocessing start method is 'spawn'
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_arguments()

    # (New) Load text images if directory is provided
    if args.text_image_dir is not None:
        load_text_image_pool(args.text_image_dir)
        logging.info(f"Loaded {len(text_images_pool)} text images for augmentation.")

    # Set CUDA device if available
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda_device}"
        torch.cuda.set_device(args.cuda_device)
        logging.info(f"Using CUDA device: {device}")
    else:
        device = "cpu"
        logging.info("CUDA not available. Using CPU.")

    # Step1: YOLO detection -> results.csv, excluded/cropped images
    preprocess_with_yolo(
        yolo_model_path=args.yolo_model_path,
        input_dir=args.input_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        label=args.label,
        device=device
    )

    # Step2: Create anchor/pos/neg samples and split
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

    logging.info("=== Preprocessing Complete (YOLO + Anchor/Pos/Neg + Split) ===")


if __name__ == "__main__":
    main()
