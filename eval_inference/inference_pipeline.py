import argparse
import os
import logging
import shutil
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights
from torch.nn.functional import normalize
from PIL import Image
from natsort import natsorted
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline: YOLO preprocess + anchor/img creation + Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--processed_dir", type=str, required=True, help="Directory for processed data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for inference results")
    parser.add_argument("--yolo_model_path", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for classification")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for YOLO and inference")
    parser.add_argument("--margin", type=int, default=50, help="Margin for anchor/img cropping")
    parser.add_argument("--mode", type=str, default="eval", help="Mode name (e.g. eval)")
    args = parser.parse_args()
    return args

def get_model(pretrained=True):
    model = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[2] = nn.Identity()
    return model

def preprocess_with_yolo(yolo_model_path, input_dir, processed_dir, batch_size=64):
    """
    YOLO를 이용해 input_dir 내 이미지에 대해 bounding box 검출.
    결과를 processed_dir/results.csv 에 저장.
    각 category(또는 input_dir basename)별로 excluded/cropped 이미지 생성.
    """
    model = YOLO(yolo_model_path)
    os.makedirs(processed_dir, exist_ok=True)
    csv_data = []

    def process_images_in_batch(image_paths, folder_name):
        results = model(image_paths)
        for image_path, result in zip(image_paths, results):
            image_file = os.path.basename(image_path)
            try:
                img = Image.open(image_path)
                img.verify()
                img = Image.open(image_path).convert("RGB")
                img_width, img_height = img.size
            except Exception as e:
                logging.warning(f"Image load failed {image_path}: {e}")
                csv_data.append({
                    "folder": folder_name,
                    "file": image_file,
                    "class_id": None,
                    "confidence": None,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None
                })
                continue

            anchor_save_path = os.path.join(processed_dir, folder_name, "excluded")
            image_save_path = os.path.join(processed_dir, folder_name, "cropped")
            os.makedirs(anchor_save_path, exist_ok=True)
            os.makedirs(image_save_path, exist_ok=True)

            if len(result.boxes) > 0:
                best_box = max(result.boxes, key=lambda box: box.conf[0])
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = max(0,x1), max(0,y1), min(x2,img_width), min(y2,img_height)
                confidence = best_box.conf[0].cpu().numpy()
                class_id = int(best_box.cls[0].cpu().numpy())

                csv_data.append({
                    "folder": folder_name,
                    "file": image_file,
                    "class_id": class_id,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

                # cropped
                cropped = img.crop((x1, y1, x2, y2))
                cropped.save(os.path.join(image_save_path, f"cropped_{os.path.splitext(image_file)[0]}.png"))

                # excluded: bbox 영역을 검은색으로 마스킹한 전체 이미지
                excluded = img.copy()
                mask = Image.new("L", excluded.size, 0)
                mask.paste(255, (int(x1), int(y1), int(x2), int(y2)))
                excluded_final = Image.composite(Image.new("RGB", excluded.size, (0,0,0)), excluded, mask)
                excluded_final.save(os.path.join(anchor_save_path, f"excluded_{os.path.splitext(image_file)[0]}.png"))
            else:
                # YOLO 결과 없음
                csv_data.append({
                    "folder": folder_name,
                    "file": image_file,
                    "class_id": None,
                    "confidence": None,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None
                })
                # excluded에 원본 저장 (cropped 없음)
                img.save(os.path.join(anchor_save_path, f"excluded_{os.path.splitext(image_file)[0]}.png"))

    # 상위 디렉토리에 바로 이미지가 있는 경우
    top_level_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))]
    if top_level_images:
        cat_name = os.path.basename(input_dir)
        for i in range(0, len(top_level_images), batch_size):
            batch = top_level_images[i:i+batch_size]
            batch_paths = [os.path.join(input_dir, b) for b in batch]
            process_images_in_batch(batch_paths, cat_name)
    else:
        # 하위 폴더(예: normal, fraud)
        for folder_name in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder_name)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))]
                image_paths = [os.path.join(folder_path, f) for f in natsorted(image_files)]
                for i in range(0, len(image_paths), batch_size):
                    batch_image_paths = image_paths[i:i+batch_size]
                    process_images_in_batch(batch_image_paths, folder_name)

    df = pd.DataFrame(csv_data)
    df = df.sort_values(by=["folder", "file"])
    csv_output_path = os.path.join(processed_dir, "results.csv")
    df.to_csv(csv_output_path, index=False)
    logging.info(f"Preprocessing 완료! 결과는 {csv_output_path}에 저장되었습니다.")

def process_row(row, data_path, data_type, mode, output_path, margin=50):
    """
    YOLO 결과 바탕으로 anchor/img 생성.
    anchor: excluded 이미지에서 margin 포함한 bbox 영역 크롭
    img: excluded에 bbox 부위에 원본 cropped 이미지를 paste한 뒤 margin 크롭
    """
    try:
        file_name = row['file']
        x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]

        if pd.isnull(x1) or pd.isnull(y1) or pd.isnull(x2) or pd.isnull(y2):
            logging.warning(f"Skipping {file_name}: Missing bbox coords")
            return

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width <= 0 or bbox_height <= 0:
            logging.warning(f"Skipping {file_name}: Invalid bbox size")
            return

        excluded_path = os.path.join(data_path, data_type, "excluded", f"excluded_{os.path.splitext(file_name)[0]}.png")
        if not os.path.exists(excluded_path):
            logging.warning(f"Excluded image not found: {excluded_path}")
            return

        with Image.open(excluded_path).convert("RGB") as excluded_img:
            excluded_img = excluded_img.copy()
            img_width, img_height = excluded_img.size

            # margin 적용
            crop_x1 = max(x1 - margin, 0)
            crop_y1 = max(y1 - margin, 0)
            crop_x2 = min(x2 + margin, img_width)
            crop_y2 = min(y2 + margin, img_height)
            crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

            base_output_dir = os.path.join(output_path, mode, data_type)
            sample_dir = os.path.join(base_output_dir, os.path.splitext(file_name)[0])
            anchor_dir = os.path.join(sample_dir, "anchor")
            img_dir = os.path.join(sample_dir, "img")

            os.makedirs(anchor_dir, exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)

            # Anchor
            anchor_cropped = excluded_img.crop(crop_box)
            anchor_path = os.path.join(anchor_dir, "anchor.png")
            anchor_cropped.save(anchor_path)

            # img
            cropped_file_name = f"cropped_{os.path.splitext(file_name)[0]}.png"
            cropped_path = os.path.join(data_path, data_type, "cropped", cropped_file_name)

            if not os.path.exists(cropped_path):
                logging.warning(f"Image not found: {cropped_path}. Skipping img for {file_name}.")
                return

            with Image.open(cropped_path).convert("RGB") as cropped_img:
                # bbox크기로 resize
                resized_cropped = cropped_img.resize((bbox_width, bbox_height), Image.BILINEAR)
                combined_pos_img = excluded_img.copy()
                combined_pos_img.paste(resized_cropped, (x1, y1))
                img_cropped = combined_pos_img.crop(crop_box)
                img_path = os.path.join(img_dir, "img.png")
                img_cropped.save(img_path)

    except Exception as e:
        logging.error(f"Unexpected error processing row {row.get('file', 'Unknown')}: {e}")

def run_inference(checkpoint_path, processed_dir, output_dir, threshold, device_str, batch_size, mode):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    anchor_model = get_model(pretrained=False)
    posneg_model = get_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    anchor_model.load_state_dict(checkpoint['anchor_model_state'])
    posneg_model.load_state_dict(checkpoint['posneg_model_state'])

    anchor_model = anchor_model.to(device)
    posneg_model = posneg_model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    normal_dir = os.path.join(output_dir, "predicted_normal")
    fraud_dir = os.path.join(output_dir, "predicted_fraud")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(fraud_dir, exist_ok=True)

    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    results_path = os.path.join(output_dir, "results.txt")

    # Open the file once using 'with open'
    with open(results_path, "w") as fw:
        fw.write("Category,Anchor,Image,Result\n")

        mode_dir = os.path.join(processed_dir, mode)  
        if not os.path.exists(mode_dir):
            logging.warning(f"No eval directory at {mode_dir}")
            return

        categories = [d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))]
        for category in categories:
            cat_path = os.path.join(mode_dir, category)
            sample_dirs = [sd for sd in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, sd))]
            for sample_name in sample_dirs:
                sample_path = os.path.join(cat_path, sample_name)
                anchor_path = os.path.join(sample_path, "anchor")
                img_path_ = os.path.join(sample_path, "img")

                if not (os.path.exists(anchor_path) and os.path.exists(img_path_)):
                    continue

                a_files = sorted([f for f in os.listdir(anchor_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                i_files = sorted([f for f in os.listdir(img_path_) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

                length = min(len(a_files), len(i_files))
                for i in range(0, length, batch_size):
                    a_batch = a_files[i:i + batch_size]
                    i_batch = i_files[i:i + batch_size]

                    anchor_tensors = []
                    image_tensors = []

                    # Load batch
                    for af in a_batch:
                        a_full = os.path.join(anchor_path, af)
                        try:
                            a_img = Image.open(a_full).convert("RGB")
                            anchor_tensors.append(transform(a_img).unsqueeze(0))
                        except:
                            logging.warning(f"Skipping anchor {af}")
                    for imgf in i_batch:
                        i_full = os.path.join(img_path_, imgf)
                        try:
                            i_img = Image.open(i_full).convert("RGB")
                            image_tensors.append(transform(i_img).unsqueeze(0))
                        except:
                            logging.warning(f"Skipping img {imgf}")

                    if not anchor_tensors or not image_tensors:
                        continue

                    anchor_batch = torch.cat(anchor_tensors, dim=0).to(device)
                    image_batch = torch.cat(image_tensors, dim=0).to(device)

                    anchor_model.eval()
                    posneg_model.eval()
                    with torch.no_grad():
                        anchor_emb = anchor_model(anchor_batch)
                        image_emb = posneg_model(image_batch)
                        anchor_emb = normalize(anchor_emb, p=2, dim=1)
                        image_emb = normalize(image_emb, p=2, dim=1)
                        dist = torch.nn.functional.pairwise_distance(anchor_emb, image_emb)

                    for idx_, d_ in enumerate(dist):
                        is_normal = d_.item() < threshold
                        result = "Normal" if is_normal else "Fraud"
                        a_file = a_batch[idx_]
                        img_file = i_batch[idx_]

                        if is_normal:
                            shutil.copy(
                                os.path.join(img_path_, img_file),
                                os.path.join(normal_dir, f"{category}_{sample_name}_img_{img_file}")
                            )
                        else:
                            shutil.copy(
                                os.path.join(img_path_, img_file),
                                os.path.join(fraud_dir, f"{category}_{sample_name}_img_{img_file}")
                            )

                        # Write result
                        fw.write(f"{category},{a_file},{img_file},{result}\n")

    logging.info(f"Inference 완료! 결과는 {results_path}에 저장되었습니다.")




def main():
    args = parse_args()

    # 1. YOLO 전처리
    preprocess_with_yolo(args.yolo_model_path, args.input_dir, args.processed_dir, batch_size=args.batch_size)

    # 2. anchor/img 생성
    results_csv = os.path.join(args.processed_dir, "results.csv")
    if not os.path.exists(results_csv):
        logging.error("results.csv not found. Stopping.")
        return

    df = pd.read_csv(results_csv)
    for idx, row in df.iterrows():
        folder = row.get("folder")
        if folder is None:
            logging.warning("No folder info, skipping row")
            continue
        # anchor/img 생성
        process_row(row, data_path=args.processed_dir, data_type=folder, mode=args.mode, output_path=args.processed_dir, margin=args.margin)

    # 3. Inference
    run_inference(
        checkpoint_path=args.checkpoint_path,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device_str=args.device,
        batch_size=args.batch_size,
        mode=args.mode
    )

if __name__ == "__main__":
    main()