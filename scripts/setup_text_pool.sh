#!/bin/bash

################################################################################
# Script to Setup Text Images Pool for Data Augmentation
################################################################################

# Exit immediately if a command exits with a non-zero status
set -e

# Paths and directories
DOWNLOAD_FONT_SCRIPT="scripts/download_fonts.sh"
CREATE_TEXT_POOL_SCRIPT="src/create_text_pool.py"
FONTS_DIR="fonts"
TEXT_IMAGES_POOL_DIR="text_images_pool_0130"
INPUT_IMAGES_DIR="dataset/normal"  # Update as per your dataset
YOLO_MODEL_PATH="Noris_YOLO.pt"   # Update with your YOLO model path
CUDA_DEVICE=1

# Step 1: Download Fonts
echo "=== Step 1: Downloading Fonts ==="
if [ -f "$DOWNLOAD_FONT_SCRIPT" ]; then
    bash "$DOWNLOAD_FONT_SCRIPT"
else
    echo "Error: '$DOWNLOAD_FONT_SCRIPT' not found!"
    exit 1
fi

# Step 2: Generate Synthetic Text Images for Each Font
echo "=== Step 2: Generating Synthetic Text Images ==="
for font_file in "$FONTS_DIR"/*.ttf; do
    if [ -f "$font_file" ]; then
        echo "Generating text images using font: $(basename "$font_file")"
        python "$CREATE_TEXT_POOL_SCRIPT" \
            --mode generate \
            --output_dir "$TEXT_IMAGES_POOL_DIR" \
            --font_path "$font_file" \
            --font_size 32 \
            --img_width 64 \
            --img_height 64 \
            --no_digits
    else
        echo "No .ttf fonts found in '$FONTS_DIR/'"
    fi
done

# Step 3: Extract Text Images via OCR
echo "=== Step 3: Extracting Text Images via YOLO + OCR ==="
if [ -f "$CREATE_TEXT_POOL_SCRIPT" ]; then
    python "$CREATE_TEXT_POOL_SCRIPT" \
        --mode ocr \
        --input_dir "$INPUT_IMAGES_DIR" \
        --output_dir "$TEXT_IMAGES_POOL_DIR" \
        --yolo_model_path "$YOLO_MODEL_PATH" \
        --batch_size 16 \
        --conf_threshold 0.25 \
        --cuda_device "$CUDA_DEVICE"
else
    echo "Error: '$CREATE_TEXT_POOL_SCRIPT' not found!"
    exit 1
fi

echo "=== Text Images Pool Setup Completed ==="
