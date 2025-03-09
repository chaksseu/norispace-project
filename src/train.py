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
from torchvision.models import ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
from accelerate import Accelerator
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    auc,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.exceptions import UndefinedMetricWarning

import matplotlib.pyplot as plt
import wandb
from PIL import Image

def parse_args():
    """
    Parses command-line arguments for configuring the training and evaluation process.
    """
    parser = argparse.ArgumentParser(description="Contrastive Learning with Normal-only training & Mixed Normal+Fraud evaluation")

    # Directory paths
    parser.add_argument("--normal_data_dir", type=str, required=True,
                        help="Path to the Normal dataset root (train/anchor|pos|neg, val/..., test/...)")
    parser.add_argument("--fraud_data_dir", type=str, required=True,
                        help="Path to the Fraud dataset root (same structure as normal_data_dir)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_epoch", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")

    # Model configuration
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained ConvNeXt-Small model")

    # Weights & Biases configuration
    parser.add_argument("--wandb_project", type=str, default="contrastive_learning")
    parser.add_argument("--wandb_entity", type=str, default="norispace-project")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()

###############################################################################
# 1) ANPTrainDataset: (train) Normal-only, Triplet (anchor, pos, neg)
###############################################################################
class ANPTrainDataset(Dataset):
    """
    Training Dataset for Normal-only data using Triplet Loss.
    Loads triplets of (anchor, positive, negative) images from the specified directories.
    """

    def __init__(self, normal_data_dir, transform=None):
        super().__init__()
        self.root_dir = normal_data_dir
        self.transform = transform

        # Define directories for anchor, positive, and negative samples
        self.anchor_dir = os.path.join(self.root_dir, "train", "anchor")
        self.pos_dir = os.path.join(self.root_dir, "train", "pos")
        self.neg_dir = os.path.join(self.root_dir, "train", "neg")

        # Verify that all necessary directories exist
        if not all([os.path.isdir(self.anchor_dir),
                    os.path.isdir(self.pos_dir),
                    os.path.isdir(self.neg_dir)]):
            raise ValueError(f"Train directories not found under: {self.root_dir}")

        # List all anchor files with the "_anchor" suffix
        anchor_files = [
            f for f in os.listdir(self.anchor_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_anchor" in f
        ]

        # Extract unique prefixes to identify corresponding pos and neg samples
        self.samples = []
        for af in anchor_files:
            prefix = af.rsplit("_anchor", 1)[0]
            self.samples.append(prefix)

        self.samples = list(set(self.samples))
        print(f"[ANPTrainDataset] Found {len(self.samples)} training samples in {self.root_dir}/train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix = self.samples[idx]

        # Load anchor image
        anchor_filename = f"{prefix}_anchor.png"
        anchor_path = os.path.join(self.anchor_dir, anchor_filename)
        anchor_img = self._load_img(anchor_path)

        # Load positive image
        pos_filename = f"{prefix}_pos.png"
        pos_path = os.path.join(self.pos_dir, pos_filename)
        pos_img = self._load_img(pos_path)

        # Load a random negative image corresponding to the prefix
        neg_candidates = [
            f for f in os.listdir(self.neg_dir)
            if f.startswith(prefix + "_neg_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not neg_candidates:
            raise ValueError(f"No neg files found for prefix={prefix} in {self.neg_dir}")
        neg_filename = np.random.choice(neg_candidates)
        neg_path = os.path.join(self.neg_dir, neg_filename)
        neg_img = self._load_img(neg_path)

        return {
            "anchor": anchor_img,
            "positive": pos_img,
            "negative": neg_img
        }

    def _load_img(self, path):
        """
        Loads an image from the specified path and applies transformations if any.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

###############################################################################
# 2) MixedValTestDataset: (val, test) => Normal + Fraud
#    => anchor-pos pairs, labels: normal=0, fraud=1
###############################################################################
class MixedValTestDataset(Dataset):
    """
    Validation/Test Dataset combining Normal and Fraud data.
    Loads anchor-pos pairs with labels indicating Normal (0) or Fraud (1).
    """

    def __init__(self, normal_data_dir, fraud_data_dir, mode="val", transform=None):
        super().__init__()
        self.transform = transform
        self.mode = mode

        # Define directories for Normal data
        self.normal_anchor_dir = os.path.join(normal_data_dir, mode, "anchor")
        self.normal_pos_dir = os.path.join(normal_data_dir, mode, "pos")

        # Define directories for Fraud data
        self.fraud_anchor_dir = os.path.join(fraud_data_dir, mode, "anchor")
        self.fraud_pos_dir = os.path.join(fraud_data_dir, mode, "pos")

        # List all anchor files for Normal and Fraud
        normal_anchor_files = [
            f for f in os.listdir(self.normal_anchor_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_anchor" in f
        ]
        fraud_anchor_files = [
            f for f in os.listdir(self.fraud_anchor_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_anchor" in f
        ]

        # Shuffle and balance the number of Normal and Fraud samples
        np.random.shuffle(normal_anchor_files)
        np.random.shuffle(fraud_anchor_files)
        min_count = min(len(normal_anchor_files), len(fraud_anchor_files))
        self.normal_anchor_files = normal_anchor_files[:min_count]
        self.fraud_anchor_files = fraud_anchor_files[:min_count]

        # Create samples with corresponding labels
        # label: normal=0, fraud=1
        self.samples = []
        for naf in self.normal_anchor_files:
            prefix = naf.rsplit("_anchor", 1)[0]
            self.samples.append((prefix, 0))  # normal=0
        for faf in self.fraud_anchor_files:
            prefix = faf.rsplit("_anchor", 1)[0]
            self.samples.append((prefix, 1))  # fraud=1

        # Shuffle the combined samples
        np.random.shuffle(self.samples)
        print(f"[MixedValTestDataset-{mode}] Normal+Fraud => {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, label = self.samples[idx]

        # Select appropriate directories based on the label
        if label == 0:  # normal
            anchor_dir = self.normal_anchor_dir
            pos_dir = self.normal_pos_dir
        else:           # fraud
            anchor_dir = self.fraud_anchor_dir
            pos_dir = self.fraud_pos_dir

        # Load anchor and positive images
        anchor_path = os.path.join(anchor_dir, f"{prefix}_anchor.png")
        pos_path = os.path.join(pos_dir, f"{prefix}_pos.png")

        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img = Image.open(pos_path).convert("RGB")
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)

        return {
            "anchor": anchor_img,
            "pos": pos_img,
            "label": label  # normal=0, fraud=1
        }

###############################################################################
# Triplet 
###############################################################################
class TripletLoss(nn.Module):
    """
    Implements the Triplet Loss.
    Encourages the distance between anchor and positive to be smaller than
    the distance between anchor and negative by a margin.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_emb, pos_emb, neg_emb):
        """
        Computes the contrastive loss based on anchor, positive, and negative embeddings.
        """
        pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, neg_emb)
        pos_loss = 0.5 * (pos_dist ** 2)
        neg_loss = 0.5 * torch.clamp(self.margin - neg_dist, min=0.0) ** 2
        return pos_loss.mean() + neg_loss.mean()

def main():
    """
    The main function orchestrates the training and evaluation pipeline.
    """
    # Suppress warnings related to undefined metrics
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    args = parse_args()

    # Initialize the Accelerator for distributed training if applicable
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    accelerator.init_trackers("Triplelet training", config=vars(args))

    # Initialize Weights & Biases (W&B) for experiment tracking
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name
        )
        wandb.config.update(vars(args))

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define transformations using pretrained ConvNeXt-Small weights
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    # weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    #---------------------------------------------------------------------------
    # (1) Train Dataset: Normal-only (triplet)
    #---------------------------------------------------------------------------
    train_dataset = ANPTrainDataset(args.normal_data_dir, transform=transform)

    def collate_fn_train(batch):
        """
        Custom collate function for triplet training.
        Stacks anchors, positives, and negatives separately.
        """
        anchors, positives, negatives = [], [], []
        for b in batch:
            anchors.append(b["anchor"])
            positives.append(b["positive"])
            negatives.append(b["negative"])
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train
    )

    #---------------------------------------------------------------------------
    # (2) Val / Test Dataset: Normal + Fraud => anchor-pos
    #---------------------------------------------------------------------------
    test_dataset = MixedValTestDataset(
        args.normal_data_dir, args.fraud_data_dir, mode="test", transform=transform
    )

    def collate_fn_eval(batch):
        """
        Custom collate function for evaluation datasets.
        Stacks anchors, positives, and labels separately.
        """
        anchors, poss, labels = [], [], []
        for b in batch:
            anchors.append(b["anchor"])
            poss.append(b["pos"])
            labels.append(b["label"])
        return torch.stack(anchors), torch.stack(poss), torch.tensor(labels, dtype=torch.long)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_eval
    )

    #---------------------------------------------------------------------------
    # Model Setup (anchor_model, posneg_model)
    #---------------------------------------------------------------------------
    def get_model(pretrained=True):
        """
        Initializes the ConvNeXt-Small model, removing the final classification layer.
        """
        model = models.convnext_small(
            weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # model = models.convnext_base(
        #     weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        # )
        # Remove the final classification layer to obtain feature embeddings
        model.classifier[2] = nn.Identity()
        return model

    anchor_model = get_model(pretrained=args.pretrained)
    # posneg_model = get_model(pretrained=args.pretrained)

    # Initialize the optimizer with parameters from both models
    optimizer = optim.AdamW(
        list(anchor_model.parameters()),# + list(posneg_model.parameters()),
        lr=args.lr
    )
    criterion = TripletLoss(margin=1.0)

    # Prepare models, optimizer, and data loaders with Accelerator
    (
        anchor_model,
        # posneg_model,
        optimizer,
        train_loader,
        test_loader
    ) = accelerator.prepare(
        anchor_model,
        # posneg_model,
        optimizer,
        train_loader,
        test_loader
    )

    global_step = 0

    #---------------------------------------------------------------------------
    # Triplet Forward Pass
    #---------------------------------------------------------------------------
    def get_triplet_emb(anc, pos, neg):
        """
        Computes normalized embeddings for anchor, positive, and negative samples.
        """
        anc_emb = anchor_model(anc)
        # pos_emb = posneg_model(pos)
        # neg_emb = posneg_model(neg)
        pos_emb = anchor_model(pos)
        neg_emb = anchor_model(neg)

        anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
        pos_emb = nn.functional.normalize(pos_emb, p=2, dim=1)
        neg_emb = nn.functional.normalize(neg_emb, p=2, dim=1)
        return anc_emb, pos_emb, neg_emb

    #---------------------------------------------------------------------------
    # Training Loop
    #---------------------------------------------------------------------------
    def train_one_epoch(loader, epoch):
        """
        Trains the model for one epoch using triplet loss.
        """
        nonlocal global_step
        anchor_model.train()
        # posneg_model.train()

        total_loss = 0.0
        total_samples = 0

        for step, (anc, pos, neg) in enumerate(loader):
            anc = anc.to(accelerator.device)
            pos = pos.to(accelerator.device)
            neg = neg.to(accelerator.device)

            # Get normalized embeddings
            anc_emb, pos_emb, neg_emb = get_triplet_emb(anc, pos, neg)

            # Compute contrastive loss
            loss = criterion(anc_emb, pos_emb, neg_emb)
            loss = loss / args.gradient_accumulation_steps

            # Backpropagate the loss
            accelerator.backward(loss)

            bs = anc.size(0)
            total_loss += (loss.item() * args.gradient_accumulation_steps * bs)
            total_samples += bs

            # Perform optimizer step if accumulation steps are met
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    global_step += 1
                    wandb.log({"train_step_loss": loss.item() * args.gradient_accumulation_steps})

        avg_loss = total_loss / total_samples if total_samples else 0.0
        return avg_loss


    #---------------------------------------------------------------------------
    # 에폭 단위로 tqdm 초기화
    #---------------------------------------------------------------------------
    pbar = tqdm(total=args.epochs, desc="Epochs", disable=not accelerator.is_main_process)

    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_one_epoch(train_loader, epoch)

        if accelerator.is_main_process:
            # Log epoch-level metrics even if not evaluating
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
            })

        # Save model checkpoints at specified epochs
        if (epoch % args.save_epoch) == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f"{args.wandb_run_name}_checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'anchor_model_state': anchor_model.state_dict(),
                # 'posneg_model_state': posneg_model.state_dict(),
                'posneg_model_state': anchor_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # 에폭이 끝날 때마다 pbar 업데이트
        pbar.update(1)

    if accelerator.is_main_process:
        # Close the progress bar and finish the W&B run
        pbar.close()
        wandb.finish()

if __name__ == "__main__":
    main()
