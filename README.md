# Norispace Project

## Overview

The **Norispace Project** is a pipeline designed for detecting and classifying normal and fraudulent data through advanced image processing and machine learning techniques. The workflow encompasses font downloading, text image pool creation, data preprocessing using YOLO and OCR, and training a ConvNeXt-based model with triplet loss for effective anomaly detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Prepare the Dataset](#1-prepare-the-dataset)
  - [2. Setup Text Image Pool](#2-setup-text-image-pool)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Training the Model](#4-training-the-model)
- [Scripts and Components](#scripts-and-components)
  - [Preprocessing Scripts](#preprocessing-scripts)
  - [Training Scripts](#training-scripts)
- [Configuration](#configuration)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Font Management**: Automatically downloads a set of free fonts from Google Fonts. (아직 사용 X)
- **Text Image Pool Creation**: Generates synthetic text images and extracts text regions using OCR. (OCR로 자른 데이터만 사용)
- **Data Preprocessing**: Utilizes YOLO for object detection and processes data into anchor, positive, and negative samples.
- **Model Training**: Trains a ConvNeXt-Small model using triplet loss for anomaly detection.
- **Experiment Tracking**: Integrates with Weights & Biases (W&B) for comprehensive experiment logging.
- **Scalability**: Leverages `accelerate` for distributed training and GPU optimization.

## Repository Structure

```
├── dataset
│   ├── fraud
│   │   └── ... # Original fraudulent data images
│   └── normal
│       └── ... # Original normal data images
├── fonts
│   └── ... # Downloaded font files (.ttf) (사용 X)
├── processed_dataset_neg32_normal
│   └── ... # Processed normal dataset (train / val / test) / (anchor / pos / neg)
├── processed_dataset_neg32_fraud
│   └── ... # Processed fraud dataset (train / val / test) / (anchor / pos / neg)
├── scripts
│   ├── download_fonts.sh
│   ├── setup_text_pool.sh
│   ├── run_preprocessing.sh
│   └── run_training.sh
├── src
│   ├── create_text_pool.py
│   ├── data_preprocess.py
│   └── train.py
├── checkpoints
│   └── ... # Model checkpoints
├── README.md
└── requirements.txt (아직 없음)
```

## Prerequisites

- 추후 작성

## Installation

- 추후 작성

## Usage

### 1. Prepare the Dataset

Organize your dataset by placing normal and fraud data into their respective directories.

- **Normal Data**: Place all normal images in `dataset/normal/`
- **Fraud Data**: Place all fraudulent images in `dataset/fraud/`

```
dataset/
├── normal/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
└── fraud/
    ├── imageA.png
    ├── imageB.jpg
    └── ...
```

### 2. Setup Text Image Pool

This step involves downloading fonts and generating a pool of text images using OCR.

**Note**: Currently, only the OCR-based text pool creation is well implemented. The PIL-based text image generation is pending.

```bash
bash scripts/setup_text_pool.sh
```

**Steps Performed**:

1. **Download Fonts**: Executes `scripts/download_fonts.sh` to download specified fonts into the `fonts/` directory.
2. **Extract Text Images via OCR**: Runs `src/create_text_pool.py` in `ocr` mode to detect and extract text regions from input images using YOLO and OCR, saving them to `text_images_pool/`.

### 3. Data Preprocessing

Preprocess the dataset by detecting bounding boxes, generating anchor/positive/negative samples, and splitting the data into training, validation, and testing sets.

```bash
bash scripts/run_preprocessing.sh
```

### 4. Training the Model

Train the ConvNeXt-Small model using the preprocessed data.

```bash
bash scripts/run_training.sh
```