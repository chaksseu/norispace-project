#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_preprocess.py

Workflow:
1) Detect bounding boxes using YOLO
   - Save results to processed_dir/results.csv (original_file, label, x1, y1, x2, y2, ...)
   - Save cropped images of the bounding box region (cropped_filename) -> Positive set originals
   - Save images with the bounding box area masked in black (excluded_filename) -> Anchor set originals

2) Based on results.csv, create anchor/pos/neg images considering the margin
   - anchor: Crop the excluded image including the margin
   - pos: Paste the cropped bounding box image within the anchor region and crop with margin
   - neg: Apply OCR-based random jitter to the bounding box image and paste, then crop with margin to create multiple negatives

3) Split all generated samples (organized by file directories) into train:val:test ratios accurately
   - Record each (anchor, pos, neg) file path + label + subset information in a final CSV
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

import torch  # 추가된 부분

# Configure logging to display informational messages
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    """
    Parses command-line arguments for configuring the preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(description="Preprocessing pipeline with YOLO detection, anchor/pos/neg creation, and train/val/test split")

    # YOLO and basic preprocessing parameters
    parser.add_argument("--yolo_model_path", type=str, required=True,
                        help="Path to the YOLO model (.pt file)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the original images")
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Directory to save processed outputs (cropped, excluded images, results.csv)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for YOLO inference")
    parser.add_argument("--margin", type=int, default=50,
                        help="Margin size to include around bounding boxes")

    # Augmentation (negatives) parameters
    parser.add_argument("--num_color_jitter", type=int, default=5,
                        help="Number of color jitter augmentations to apply for negatives")
    parser.add_argument("--generate_negatives", action="store_true",
                        help="Flag to generate negative samples with augmentations")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for processing")

    # Train/Val/Test split parameters
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for data splitting")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of data to include in the training set")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion of data to include in the validation set")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Proportion of data to include in the test set")

    # Additional parameter: label type (normal or fraud)
    parser.add_argument("--label", type=str, default="normal",
                        help="Label for the input dataset (normal/fraud)")

    # 추가된 부분: CUDA 장치 번호 지정
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA device number to use for GPU processing (default: 0)")

    args = parser.parse_args()
    return args


# ----------------------------------------------------------------------
# [1] YOLO Inference -> results.csv, excluded_*, cropped_*
# ----------------------------------------------------------------------
def preprocess_with_yolo(yolo_model_path, input_dir, processed_dir, batch_size=64, label="normal", device="cuda:0"):
    """
    Runs YOLO model on all images in input_dir to detect bounding boxes.
    - Saves bounding box information and labels to processed_dir/results.csv
    - Saves cropped images (cropped_*) containing only the bounding box region
    - Saves excluded images (excluded_*) with the bounding box area masked in black

    Args:
        yolo_model_path (str): Path to the YOLO model file.
        input_dir (str): Directory containing the original images.
        processed_dir (str): Directory to save processed results.
        batch_size (int): Number of images to process in each batch.
        label (str): Label for the dataset ("normal" or "fraud").
        device (str): Device to run the YOLO model on (e.g., "cuda:0", "cpu").
    """
    logging.info("[Step1] Running YOLO detection -> Creating excluded/cropped images and saving results.csv")

    # Temporary directory to store intermediate excluded and cropped images
    tmp_dir = os.path.join(processed_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load YOLO model on specified device
    model = YOLO(yolo_model_path).to(device)
    csv_data = []

    def process_batch(image_paths):
        """
        Processes a batch of images with YOLO, saves cropped and excluded images,
        and records bounding box details.

        Args:
            image_paths (list): List of image file paths to process.
        """
        results = model(image_paths)
        for image_path, result in zip(image_paths, results):
            image_file = os.path.basename(image_path)
            try:
                img = Image.open(image_path).convert("RGB")
                w, h = img.size
            except Exception as e:
                logging.warning(f"Failed to load image {image_path}: {e}")
                continue

            # Define paths to save excluded and cropped images
            exclude_save_path = os.path.join(tmp_dir, f"excluded_{os.path.splitext(image_file)[0]}.png")
            crop_save_path = os.path.join(tmp_dir, f"cropped_{os.path.splitext(image_file)[0]}.png")

            if len(result.boxes) > 0:
                # Use the bounding box with the highest confidence
                best_box = max(result.boxes, key=lambda box: box.conf[0])
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, x1), max(0, y1)  # Ensure coordinates are within image bounds
                x2, y2 = min(w, x2), min(h, y2)
                confidence = float(best_box.conf[0].cpu().numpy())
                class_id = int(best_box.cls[0].cpu().numpy())

                # Record bounding box details in CSV data
                csv_data.append({
                    "original_file": image_file,
                    "label": label,  # "normal" or "fraud"
                    "class_id": class_id,
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                # Save cropped image containing only the bounding box region
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_img.save(crop_save_path)

                # Save excluded image with the bounding box area masked in black
                excluded_img = img.copy()
                mask = Image.new("L", excluded_img.size, 0)
                mask.paste(255, (int(x1), int(y1), int(x2), int(y2)))
                black_img = Image.new("RGB", excluded_img.size, (0, 0, 0))
                excluded_final = Image.composite(black_img, excluded_img, mask)
                excluded_final.save(exclude_save_path)

            else:
                # If no bounding box is detected, do not save excluded/cropped images
                continue

    # Collect all image file paths from the input directory and its subdirectories
    all_images = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                all_images.append(os.path.join(root, fname))

    # Process images in batches
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i: i + batch_size]
        process_batch(batch)

    # Save bounding box information to CSV
    df = pd.DataFrame(csv_data)
    df_path = os.path.join(processed_dir, "results.csv")
    df.to_csv(df_path, index=False)
    logging.info(f"YOLO preprocessing completed: {df_path}")


# ----------------------------------------------------------------------
# [2] OCR+Augmentation -> Create anchor/pos/neg
# ----------------------------------------------------------------------
def random_jitter(image: Image.Image) -> Image.Image:
    """
    Applies random augmentations to the image based on OCR-detected text regions.
    Enhances color, brightness, contrast, sharpness, and applies affine transformations.

    Args:
        image (PIL.Image.Image): The image to augment.

    Returns:
        PIL.Image.Image: The augmented image.
    """
    try:
        
        
        import numpy as np
        # Convert PIL Image to OpenCV format
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Initialize EasyOCR reader
        #reader = easyocr.Reader(['en'], gpu=True)
        # Perform OCR to detect text regions
        #results = reader.readtext(cv_img, detail=1)

        augmented = image.copy()
        '''
        for detection in results:
            bbox, text, conf = detection
            if conf < 0.6:
                continue  # Skip low-confidence detections
            # Extract bounding box coordinates
            (topleft, topright, botright, botleft) = bbox
            x_min = int(min(topleft[0], botleft[0]))
            y_min = int(min(topleft[1], topright[1]))
            x_max = int(max(botright[0], topright[0]))
            y_max = int(max(botright[1], botleft[1]))

            crop_region = augmented.crop((x_min, y_min, x_max, y_max))

            # Apply random color, brightness, contrast, and sharpness enhancements
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

            # Apply random rotation
            if random.random() < 0.7:
                angle = random.uniform(-2, 2)
                crop_region = crop_region.rotate(
                    angle, expand=True, fillcolor=(255, 255, 255)
                )

            # Apply random scaling
            if random.random() < 0.7:
                scale = random.uniform(0.9, 1.1)
                new_w = max(1, int(crop_region.width * scale))
                new_h = max(1, int(crop_region.height * scale))
                crop_region = crop_region.resize((new_w, new_h), Image.BICUBIC)

            # Apply random translation
            dx, dy = 0, 0
            if random.random() < 0.7:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)

            # Paste the augmented crop back into the image with translation
            paste_x = max(0, min(augmented.width - crop_region.width, x_min + dx))
            paste_y = max(0, min(augmented.height - crop_region.height, y_min + dy))
            augmented.paste(crop_region, (paste_x, paste_y))
        '''


        # Apply random global enhancements to the entire image
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

        # Apply random affine transformations (rotation, translation, scaling)
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

        # Define the affine transformation matrix
        a = cos_t
        b = sin_t
        c = (1 - cos_t) * cx - sin_t * cy + translate_x
        d = -sin_t
        e = cos_t
        f = sin_t * cx + (1 - cos_t) * cy + translate_y

        affine_mat = (a, b, c, d, e, f)

        # Apply affine transformation
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
    Creates anchor, positive, and negative samples for a single row in results.csv
    and saves them to the appropriate subset directories.

    Args:
        row (dict): A dictionary representing a row from results.csv containing bounding box details.
        tmp_dir (str): Temporary directory containing excluded and cropped images.
        processed_dir (str): Directory to save the final anchor/pos/neg images.
        subset (str): The subset to which the sample belongs ('train', 'val', or 'test').
        num_color_jitter (int): Number of color jitter augmentations for negative samples.
        generate_neg (bool): Flag to indicate whether to generate negative samples.
        margin (int): Margin to include around bounding boxes when cropping.
    """
    file_name = row["original_file"]
    label = row["label"]  # "normal" or "fraud"
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

    # Define directories for anchor, positive, and negative samples within the subset
    anchor_dir = os.path.join(processed_dir, subset, "anchor")
    pos_dir = os.path.join(processed_dir, subset, "pos")
    neg_dir = os.path.join(processed_dir, subset, "neg")

    # Paths to the excluded and cropped images
    excluded_path = os.path.join(tmp_dir, f"excluded_{os.path.splitext(file_name)[0]}.png")
    cropped_path = os.path.join(tmp_dir, f"cropped_{os.path.splitext(file_name)[0]}.png")

    # Skip processing if bounding box coordinates are invalid or missing
    if pd.isnull(x1) or pd.isnull(y1) or pd.isnull(x2) or pd.isnull(y2):
        return

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return

    if not os.path.exists(excluded_path):
        return

    # Apply margin to bounding box coordinates
    with Image.open(excluded_path).convert("RGB") as exc_img:
        w, h = exc_img.size
        crop_x1 = max(x1 - margin, 0)
        crop_y1 = max(y1 - margin, 0)
        crop_x2 = min(x2 + margin, w)
        crop_y2 = min(y2 + margin, h)

        # Create and save the anchor image by cropping the excluded image with margin
        anchor_crop = exc_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        anchor_filename = f"{os.path.splitext(file_name)[0]}_anchor.png"
        anchor_save_path = os.path.join(anchor_dir, anchor_filename)
        anchor_crop.save(anchor_save_path)

        # Create and save the positive image by pasting the cropped bounding box
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

            # Create and save negative samples with augmentations if enabled
            if generate_neg:
                for i in range(num_color_jitter):
                    jittered_img = random_jitter(resized_crop)
                    combined_neg = exc_img.copy()
                    combined_neg.paste(jittered_img, (x1, y1))
                    neg_crop = combined_neg.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    neg_filename = f"{os.path.splitext(file_name)[0]}_neg_{i}.png"
                    neg_save_path = os.path.join(neg_dir, neg_filename)
                    neg_crop.save(neg_save_path)

    # End of processing for the current row


def worker_fn(row, tmp_dir, processed_dir, subset,
              num_color_jitter=5, generate_neg=False, margin=50):
    """
    Worker function for multiprocessing that creates samples for a single row.

    Args:
        row (dict): A dictionary representing a row from results.csv.
        tmp_dir (str): Temporary directory for intermediate files.
        processed_dir (str): Directory to save processed samples.
        subset (str): The subset to which the sample belongs ('train', 'val', or 'test').
        num_color_jitter (int): Number of color jitter augmentations for negatives.
        generate_neg (bool): Flag to generate negative samples.
        margin (int): Margin to include around bounding boxes when cropping.
    """
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
# [3] Train/Val/Test Split
#     => Create anchor/pos/neg and record file paths with labels and subsets in CSV
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
    Reads results.csv to generate anchor, positive, and negative samples,
    then splits the data into train, validation, and test sets based on specified ratios.
    Finally, records the file paths along with labels and subset information in a CSV.

    Args:
        processed_dir (str): Directory containing results.csv and to save processed samples.
        num_color_jitter (int): Number of color jitter augmentations for negatives.
        generate_neg (bool): Flag to generate negative samples.
        margin (int): Margin to include around bounding boxes when cropping.
        train_ratio (float): Proportion of data to include in the training set.
        val_ratio (float): Proportion of data to include in the validation set.
        test_ratio (float): Proportion of data to include in the test set.
        num_workers (int): Number of parallel workers for processing.
        seed (int): Random seed for reproducibility.
    """
    # Load results.csv containing bounding box information
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

    # Shuffle the DataFrame to ensure random distribution
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate the number of samples for each subset
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count

    # Assign subset labels to each sample
    df['subset'] = ['train'] * train_count + ['val'] * val_count + ['test'] * test_count

    tmp_dir = os.path.join(processed_dir, "tmp")
    if not os.path.exists(tmp_dir):
        logging.error(f"Temporary directory does not exist: {tmp_dir}")
        return

    # Create directories for anchor, pos, and neg samples within each subset
    for sp in ["train", "val", "test"]:
        for sub_dir in ["anchor", "pos", "neg"]:
            sp_dir = os.path.join(processed_dir, sp, sub_dir)
            os.makedirs(sp_dir, exist_ok=True)

    # Convert DataFrame rows to dictionaries for processing
    rows = df.to_dict("records")
    logging.info("[Step2] Creating anchor/pos/neg samples...")
    
    # Use multiprocessing to parallelize sample creation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for row in rows:
            subset = row['subset']
            futures.append(executor.submit(
                worker_fn, row, tmp_dir, processed_dir, subset,
                num_color_jitter, generate_neg, margin
            ))
        # Display progress bar for sample creation
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Creating anchor/pos/neg"):
            pass

    logging.info("Anchor/Pos/Neg creation and train/val/test splitting completed.")
    # Remove the temporary directory after processing
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info(f"Temporary directory removed: {tmp_dir}")

    # Note: To create a final CSV mapping anchor, pos, neg files with labels and subsets,
    # additional file path scanning and mapping logic would be required.
    # This example does not include the creation of 'results_splitted.csv'.
    # You can implement this as needed based on your specific requirements.


def main():
    """
    The main function orchestrates the entire preprocessing pipeline:
    1) Detect bounding boxes with YOLO and create excluded/cropped images
    2) Generate anchor/pos/neg samples and split data into train/val/test sets
    """
    try:
        # Set the multiprocessing start method to 'spawn' for compatibility
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # If the start method has already been set, ignore the error

    args = parse_arguments()

    # CUDA 장치 설정
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda_device}"
        torch.cuda.set_device(args.cuda_device)
        logging.info(f"Using CUDA device: {device}")
    else:
        device = "cpu"
        logging.info("CUDA not available. Using CPU.")

    # Step1) Run YOLO to detect bounding boxes and create excluded/cropped images
    preprocess_with_yolo(
        yolo_model_path=args.yolo_model_path,
        input_dir=args.input_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        label=args.label,
        device=device  # 추가된 부분
    )

    # Step2) Create anchor/pos/neg samples and split data into train/val/test sets
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
