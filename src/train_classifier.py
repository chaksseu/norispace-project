#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights
from accelerate import Accelerator
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.exceptions import UndefinedMetricWarning

import wandb
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Normal(0) vs Fraud(1) Classification (pos->0, neg->1)")

    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (ex: data_dir/train/pos, data_dir/train/neg, ...)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")

    # Training params
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_epoch", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")

    # Model config
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained ConvNeXt-Small model")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="normal_fraud_classification")
    parser.add_argument("--wandb_entity", type=str, default="norispace-project")
    parser.add_argument("--wandb_run_name", type=str, default="pos-neg-classifier-run")

    return parser.parse_args()

class PosNegDataset(Dataset):
    """
    pos -> normal(0), neg -> fraud(1).
    One prefix => 1 'pos' image, multiple 'neg' images => pick neg randomly.
    """
    def __init__(self, data_dir, mode="train", transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.pos_dir = os.path.join(self.data_dir, mode, "pos")
        self.neg_dir = os.path.join(self.data_dir, mode, "neg")

        if not os.path.isdir(self.pos_dir) or not os.path.isdir(self.neg_dir):
            raise ValueError(f"Directories not found: {self.pos_dir}, {self.neg_dir}")

        pos_files = [
            f for f in os.listdir(self.pos_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_pos" in f
        ]

        self.prefixes = []
        for pf in pos_files:
            prefix = pf.rsplit("_pos", 1)[0]
            self.prefixes.append(prefix)

        self.prefixes = sorted(list(set(self.prefixes)))

        # Map prefix -> list of neg files
        self.neg_dict = {}
        for prefix in self.prefixes:
            all_neg = [
                f for f in os.listdir(self.neg_dir)
                if f.startswith(prefix + "_neg_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.neg_dict[prefix] = all_neg

        self.dataset_len = 2 * len(self.prefixes)  # 2 samples (pos, neg) per prefix
        print(f"[PosNegDataset-{mode}] Found {len(self.prefixes)} prefixes => total {self.dataset_len} samples")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        prefix_index = idx // 2
        prefix = self.prefixes[prefix_index]

        # Even -> pos(0), Odd -> neg(1)
        is_pos = (idx % 2 == 0)
        if is_pos:
            filename = f"{prefix}_pos.png"
            path = os.path.join(self.pos_dir, filename)
            label = 0
        else:
            neg_candidates = self.neg_dict[prefix]
            if not neg_candidates:
                raise ValueError(f"No neg files for prefix={prefix} in {self.neg_dir}")
            neg_filename = np.random.choice(neg_candidates)
            path = os.path.join(self.neg_dir, neg_filename)
            label = 1

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

def main():
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    accelerator.init_trackers("PosNegClassification", config=vars(args))

    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name
        )
        wandb.config.update(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    # Pretrained ConvNeXt-Small transform
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    # Only train / test
    train_dataset = PosNegDataset(
        data_dir=args.data_dir,
        mode="train",
        transform=transform
    )
    test_dataset = PosNegDataset(
        data_dir=args.data_dir,
        mode="test",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    def get_model(pretrained=True):
        # Use a small MLP head (hidden layer + dropout) instead of a single Linear
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

    model = get_model(pretrained=args.pretrained)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    (
        model,
        optimizer,
        train_loader,
        test_loader
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        test_loader
    )

    def train_one_epoch(epoch):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(accelerator.device), labels.to(accelerator.device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            bs = images.size(0)
            total_loss += (loss.item() * args.gradient_accumulation_steps * bs)
            total_samples += bs

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / total_samples if total_samples else 0.0
        if accelerator.is_main_process:
            wandb.log({"train_loss": avg_loss, "epoch": epoch})
        return avg_loss

    def evaluate(loader, epoch=0):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(accelerator.device), labels.to(accelerator.device)
                outputs = model(images)  # [B, 2]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", pos_label=1
        )
        # If you need ROC-AUC on probabilities, do:
        # probs = torch.softmax(outputs, dim=1)[:, 1] (accumulate in a loop, then roc_auc_score)
        roc_auc = 0.0

        if accelerator.is_main_process:
            wandb.log({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_roc_auc": roc_auc,
                "epoch": epoch
            })

        print(f"[Test][Epoch {epoch}] acc={accuracy:.4f}, prec={precision:.4f}, rec={recall:.4f}, f1={f1:.4f}")

    pbar = tqdm(total=args.epochs, desc="Epochs", disable=not accelerator.is_main_process)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(epoch)

        # Save checkpoints
        if (epoch % args.save_epoch) == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f"{args.wandb_run_name}_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

        pbar.update(1)

    # Final test
    if accelerator.is_main_process:
        pbar.close()
        print("Evaluating on Test set...")
    evaluate(test_loader, epoch=args.epochs)

    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
