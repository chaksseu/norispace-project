#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_text_pool.py

This script performs two primary functions based on the selected mode:

1. --mode generate
   - Generates synthetic text images for English letters (A-Z, a-z) and digits (0-9) using specified fonts.
   - Saves each character as an individual image in the designated output directory.

2. --mode ocr
   - Utilizes YOLO to detect bounding boxes in input images.
   - Extracts each detected text region using OCR.
   - Saves each extracted text region as an individual image in the designated output directory.

Usage Examples:

  # Generate synthetic text images
  python create_text_pool.py --mode generate \
      --output_dir text_images_pool \
      --font_path fonts/Roboto-Bold.ttf \
      --font_size 32 \
      --img_width 64 \
      --img_height 64 \
      --no_digits

  # Extract text images via OCR
  python create_text_pool.py --mode ocr \
      --input_dir dataset/docs \
      --output_dir text_images_pool \
      --yolo_model_path /path/to/Noris_YOLO.pt \
      --batch_size 16 \
      --conf_threshold 0.25 \
      --cuda_device 0
"""

import os
import argparse
import logging
import random
import string

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import torch

from ultralytics import YOLO
from tqdm import tqdm

# Initialize global text images pool
text_images_pool = []

def load_text_image_pool(folder_path: str):
    """
    Loads all images from the specified folder_path into 'text_images_pool'.
    Each image is assumed to be a single character or text snippet.
    """
    global text_images_pool
    if not os.path.isdir(folder_path):
        logging.warning(f"Text image directory '{folder_path}' not found.")
        return

    for fn in os.listdir(folder_path):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, fn)
            try:
                timg = Image.open(img_path).convert("RGB")
                text_images_pool.append(timg)
                logging.debug(f"Loaded text image: {fn}")
            except Exception as e:
                logging.warning(f"Failed to load image '{fn}': {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create or collect text images for augmentation."
    )

    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate", "ocr"],
                        help="Operation mode: 'generate' or 'ocr'")

    # Arguments for 'generate' mode
    parser.add_argument("--output_dir", type=str, default="text_images_pool",
                        help="Directory to save generated or extracted text images")
    parser.add_argument("--font_path", type=str, default="Arial.ttf",
                        help="Path to a .ttf font for text generation")
    parser.add_argument("--font_size", type=int, default=32,
                        help="Font size for generated text images")
    parser.add_argument("--img_width", type=int, default=64,
                        help="Width of generated text image")
    parser.add_argument("--img_height", type=int, default=64,
                        help="Height of generated text image")
    parser.add_argument("--no_digits", action="store_true",
                        help="If set, do not include digits in text image generation")

    # Arguments for 'ocr' mode
    parser.add_argument("--input_dir", type=str, default="dataset/docs",
                        help="Folder of images to run YOLO->OCR on")
    parser.add_argument("--yolo_model_path", type=str, default="model.pt",
                        help="Path to YOLO model (.pt file)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for YOLO inference")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold for YOLO detection")
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA device index (default=0)")

    args = parser.parse_args()
    return args

# ------------------------------------------------------------------------------
# Function to Generate Synthetic Text Images
# ------------------------------------------------------------------------------
def generate_text_images(output_dir, font_path, font_size=32,
                         image_size=(64, 64),
                         include_digits=True):
    """
    Generates text images for:
      - Uppercase letters: A-Z
      - Lowercase letters: a-z
      - Digits 0-9 (optional)
    Saves each character as an image in 'output_dir'.

    :param output_dir: Directory to save generated text images
    :param font_path: Path to a .ttf or .otf font
    :param font_size: Font size to use for rendering
    :param image_size: Tuple (width, height) for the output image
    :param include_digits: Whether to include digits (0-9)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

    # Characters to generate
    uppercase = string.ascii_uppercase  # A-Z
    lowercase = string.ascii_lowercase  # a-z
    digits = string.digits if include_digits else ""

    all_chars = uppercase + lowercase + digits

    # Load the specified font
    try:
        font = ImageFont.truetype(font_path, font_size)
        logging.info(f"Loaded font: {font_path} with size {font_size}")
    except Exception as e:
        logging.error(f"Failed to load font: {font_path} - {e}")
        return

    for char in tqdm(all_chars, desc="Generating text images"):
        # Create blank image (white background)
        img = Image.new("RGB", image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Compute text size to center the character
        w, h = draw.textsize(char, font=font)
        x_pos = (img.width - w) // 2
        y_pos = (img.height - h) // 2

        # Draw the character in black
        draw.text((x_pos, y_pos), char, font=font, fill=(0, 0, 0))

        # Save to file, e.g., "A_Roboto-Bold.png"
        safe_font_name = os.path.splitext(os.path.basename(font_path))[0]
        filename = f"{char}_{safe_font_name}.png"
        out_path = os.path.join(output_dir, filename)
        try:
            img.save(out_path)
            logging.debug(f"Saved text image: {out_path}")
        except Exception as e:
            logging.warning(f"Failed to save text image '{out_path}': {e}")

    logging.info(f"Completed generating text images in '{output_dir}'.")

# ------------------------------------------------------------------------------
# Function to Extract Text Images via YOLO + OCR
# ------------------------------------------------------------------------------
def yolo_and_ocr_crops(input_dir, output_dir,
                       yolo_model_path, batch_size=16,
                       conf_threshold=0.25,
                       ocr_lang_list=['en'],
                       cuda_device=0):
    """
    1) Use YOLO model to detect bounding boxes on images in 'input_dir'.
    2) For each detected bounding box, crop the region.
    3) Apply EasyOCR on that cropped region. For each text region found by OCR,
       save it as an individual image in 'output_dir'.

    :param input_dir: Directory with input images
    :param output_dir: Directory to save text crops
    :param yolo_model_path: Path to a YOLO model (e.g., .pt file)
    :param batch_size: YOLO batch size
    :param conf_threshold: Confidence threshold for YOLO detections
    :param ocr_lang_list: Languages for EasyOCR (e.g., ['en', 'ko'])
    :param cuda_device: CUDA device index for YOLO and OCR
    """
    # Set CUDA device
    if torch.cuda.is_available():
        device = f"cuda:{cuda_device}"
        torch.cuda.set_device(cuda_device)
        logging.info(f"Using CUDA device: {device}")
    else:
        device = "cpu"
        logging.info("CUDA not available. Using CPU.")

    # Load YOLO model
    try:
        model = YOLO(yolo_model_path).to(device)
        logging.info(f"Loaded YOLO model from '{yolo_model_path}' on {device}")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return

    # Prepare EasyOCR
    try:
        reader = easyocr.Reader(ocr_lang_list, gpu=(device.startswith("cuda")))
        logging.info(f"Initialized EasyOCR with languages: {ocr_lang_list}")
    except Exception as e:
        logging.error(f"Failed to initialize EasyOCR: {e}")
        return

    # Collect all images from input_dir
    all_images = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                all_images.append(os.path.join(root, fname))

    if not all_images:
        logging.warning(f"No images found in '{input_dir}'.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

    logging.info(f"Found {len(all_images)} images in '{input_dir}'")

    # Process in mini-batches for YOLO
    for i in tqdm(range(0, len(all_images), batch_size), desc="Processing images"):
        batch_paths = all_images[i: i + batch_size]
        try:
            results = model(batch_paths, conf=conf_threshold)
        except Exception as e:
            logging.error(f"YOLO inference failed for batch starting at index {i}: {e}")
            continue

        for image_path, result in zip(batch_paths, results):
            image_file = os.path.basename(image_path)
            base_name = os.path.splitext(image_file)[0]

            try:
                # Load original image in OpenCV
                cv_img = cv2.imread(image_path)
                if cv_img is None:
                    logging.warning(f"Failed to load image: {image_path}")
                    continue
                h, w, c = cv_img.shape
            except Exception as e:
                logging.warning(f"Failed to read image {image_path}: {e}")
                continue

            # Initialize count for OCR crops
            count = 0

            # Iterate through each detected bounding box
            for box in result.boxes:
                # box.xyxy[0] => (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(int(x2), w), min(int(y2), h)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop the YOLO bounding box region
                crop_region = cv_img[y1:y2, x1:x2]
                if crop_region.size == 0:
                    continue

                # Convert to PIL for EasyOCR
                crop_pil = Image.fromarray(cv2.cvtColor(crop_region, cv2.COLOR_BGR2RGB))

                # Apply OCR on the cropped region
                ocr_results = reader.readtext(np.array(crop_region), detail=1)
                # For each text region found by OCR
                for ocr_idx, detection in enumerate(ocr_results):
                    bbox, text, conf = detection
                    if conf < 0.6:
                        # Skip low-confidence text
                        continue

                    # Bbox in local coordinates (relative to 'crop_region')
                    (tl, tr, br, bl) = bbox
                    lx_min = int(min(tl[0], bl[0]))
                    ly_min = int(min(tl[1], tr[1]))
                    lx_max = int(max(br[0], tr[0]))
                    ly_max = int(max(br[1], bl[1]))

                    # Ensure valid sub-region
                    if lx_max <= lx_min or ly_max <= ly_min:
                        continue

                    # Crop again for the OCR sub-region
                    sub_crop = crop_region[ly_min: ly_max, lx_min: lx_max]
                    if sub_crop.size == 0:
                        continue

                    sub_crop_pil = Image.fromarray(cv2.cvtColor(sub_crop, cv2.COLOR_BGR2RGB))

                    # Save the OCR text image
                    save_name = f"{base_name}_box_{count}_ocr_{ocr_idx}.png"
                    out_path = os.path.join(output_dir, save_name)
                    try:
                        sub_crop_pil.save(out_path)
                        logging.debug(f"Saved OCR text image: {out_path}")
                    except Exception as e:
                        logging.warning(f"Failed to save OCR text image '{out_path}': {e}")

                count += 1

    logging.info("YOLO + OCR text extraction completed.")

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main():
    args = parse_arguments()

    if args.mode == "generate":
        # Generate synthetic text images
        generate_text_images(
            output_dir=args.output_dir,
            font_path=args.font_path,
            font_size=args.font_size,
            image_size=(args.img_width, args.img_height),
            include_digits=(not args.no_digits)
        )
        logging.info("Text image generation completed.")

    elif args.mode == "ocr":
        # Load text images pool for potential use in augmentation (if needed)
        load_text_image_pool(args.output_dir)
        logging.info(f"Loaded {len(text_images_pool)} text images from '{args.output_dir}' for augmentation.")

        # Extract text images via YOLO + OCR
        yolo_and_ocr_crops(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            yolo_model_path=args.yolo_model_path,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold,
            ocr_lang_list=["en"],  # Add more languages if needed
            cuda_device=args.cuda_device
        )
        logging.info("YOLO + OCR text extraction completed.")

if __name__ == "__main__":
    main()
