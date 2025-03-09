#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score  # average precision 추가
)
from sklearn.exceptions import UndefinedMetricWarning
from PIL import Image
import wandb
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Classifier Evaluation (Balanced Normal vs Fraud)")

    parser.add_argument("--normal_data_dir", type=str, required=True,
                        help="Path to normal dataset root (should have test/pos)")
    parser.add_argument("--fraud_data_dir", type=str, required=True,
                        help="Path to fraud dataset root (should have test/pos)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing saved model checkpoints (.pt or .pth)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on (e.g., cuda:0 or cpu)")

    # Weights & Biases 기본설정 (프로젝트와 엔터티는 동일하게 사용)
    parser.add_argument("--wandb_project", type=str, default="norispace_eval_project")
    parser.add_argument("--wandb_entity", type=str, default="norispace-project")
    # 기본 런 이름은 사용하지 않고, 각 체크포인트 이름을 런 이름으로 사용
    parser.add_argument("--wandb_run_name", type=str, default="eval-run")

    return parser.parse_args()

class BalancedEvalDataset(Dataset):
    """
    - normal_data_dir/test/pos => label=0
    - fraud_data_dir/test/pos => label=1
    - 각 클래스는 shuffle된 후, 최소 개수로 슬라이스 되어 균형 잡힌 샘플 구성을 함.
    """
    def __init__(self, normal_data_dir, fraud_data_dir, transform=None):
        super().__init__()
        self.transform = transform

        normal_pos_dir = os.path.join(normal_data_dir, "test", "pos")
        fraud_pos_dir = os.path.join(fraud_data_dir, "test", "pos")

        if not os.path.isdir(normal_pos_dir):
            raise ValueError(f"Not found: {normal_pos_dir}")
        if not os.path.isdir(fraud_pos_dir):
            raise ValueError(f"Not found: {fraud_pos_dir}")

        # 1) Normal pos 파일 수집 (label=0)
        normal_files = [
            os.path.join(normal_pos_dir, f) for f in os.listdir(normal_pos_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # 2) Fraud pos 파일 수집 (label=1)
        fraud_files = [
            os.path.join(fraud_pos_dir, f) for f in os.listdir(fraud_pos_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # 각각 shuffle
        np.random.shuffle(normal_files)
        np.random.shuffle(fraud_files)

        # Normal과 Fraud의 개수를 균형 맞추기
        min_count = min(len(normal_files), len(fraud_files))
        normal_files = normal_files[:min_count]
        fraud_files = fraud_files[:min_count]

        self.samples = [(fp, 0) for fp in normal_files] + [(fp, 1) for fp in fraud_files]
        # 전체 샘플 shuffle (이후 DataLoader가 순서에 따른 편향을 피할 수 있도록)
        np.random.shuffle(self.samples)

        print(f"[BalancedEvalDataset] Balanced samples => normal={min_count}, fraud={min_count}, total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

def get_model(pretrained=False):
    """
    ConvNeXt-Small 모델을 불러오고, 분류기(classifier) 레이어를 2-class MLP로 교체.
    (학습 시 사용한 구조와 동일해야 함.)
    """
    model = models.convnext_small(
        weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 2)
    )
    return model

def load_checkpoint(model, checkpoint_path, device="cpu"):
    """체크포인트에서 모델 state dict 로드."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        # model.state_dict()를 직접 저장한 경우
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model

def evaluate_model(model, loader, device):
    """추론 후, accuracy, precision, recall, f1, roc_auc, average precision 계산."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # [B, 2]
            preds = torch.argmax(outputs, dim=1)  # 예측 클래스
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 클래스 1에 대한 확률

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuracy, Precision, Recall, F1 계산
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0

    # Average Precision 계산
    try:
        avg_precision = average_precision_score(all_labels, all_probs)
    except ValueError:
        avg_precision = 0.0

    metrics_dict = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision
    }
    return metrics_dict

def main():
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    args = parse_args()

    device = torch.device(args.device)

    # Transform (학습 시와 동일)
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    # BalancedEvalDataset 생성
    test_dataset = BalancedEvalDataset(
        normal_data_dir=args.normal_data_dir,
        fraud_data_dir=args.fraud_data_dir,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # model_dir 내 모든 체크포인트 찾기
    checkpoints = [
        os.path.join(args.model_dir, f)
        for f in os.listdir(args.model_dir)
        if f.endswith(".pt") or f.endswith(".pth")
    ]
    if not checkpoints:
        print(f"No checkpoints found in: {args.model_dir}")
        return

    # 각 체크포인트별로 별도의 W&B 런 생성 후 평가
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"Evaluating checkpoint: {ckpt_name}")

        # 각 체크포인트에 대해 W&B 런 생성 (런 이름을 체크포인트 이름으로 설정)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=ckpt_name,
            config=vars(args),
            reinit=True  # 매 루프마다 새 런 생성
        )

        clf_model = get_model(pretrained=False)
        clf_model = load_checkpoint(clf_model, ckpt_path, device=device)
        clf_model.to(device)

        metrics = evaluate_model(clf_model, test_loader, device)

        run.log({
            "checkpoint": ckpt_name,
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "test_avg_precision": metrics["avg_precision"]
        })

        run.finish()

if __name__ == "__main__":
    main()
