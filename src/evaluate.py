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
    precision_recall_fscore_support,
    average_precision_score,
    roc_curve,
    auc,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score
)
from sklearn.exceptions import UndefinedMetricWarning
from PIL import Image
import wandb
from tqdm import tqdm

def parse_args():
    """
    Parses command-line arguments for configuring the evaluation process.
    """
    parser = argparse.ArgumentParser(description="Evaluate Saved Models on Validation and Test Sets")

    # Directory paths
    parser.add_argument("--normal_data_dir", type=str, required=True,
                        help="Path to the Normal dataset root (train/anchor|pos|neg, val/..., test/...)")
    parser.add_argument("--fraud_data_dir", type=str, required=True,
                        help="Path to the Fraud dataset root (same structure as normal_data_dir)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing saved model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")

    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on (cuda or cpu)")

    # Weights & Biases configuration
    parser.add_argument("--wandb_project", type=str, default="contrastive_learning_evaluation")
    parser.add_argument("--wandb_entity", type=str, default="norispace-project")

    return parser.parse_args()

###############################################################################
# 1) MixedValTestDataset: (val, test) => Normal + Fraud
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
# Evaluation Function
###############################################################################
def evaluate_model(model_checkpoint, args, transform):
    """
    Loads the saved model checkpoint and evaluates it on validation and test datasets.
    Logs the metrics to W&B and saves them locally.
    """
    # Initialize W&B run for this evaluation
    run_name = os.path.splitext(os.path.basename(model_checkpoint))[0]
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_checkpoint": model_checkpoint,
            "normal_data_dir": args.normal_data_dir,
            "fraud_data_dir": args.fraud_data_dir,
            "batch_size": args.batch_size
        }
    )

    # Define datasets and loaders
    #for phase in ["val", "test"]:
    for phase in ["test"]:
        dataset = MixedValTestDataset(
            args.normal_data_dir, args.fraud_data_dir, mode=phase, transform=transform
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn_eval(batch)
        )

        # Perform evaluation
        distances, labels = perform_evaluation(model_checkpoint, loader, args.device)
        metrics = compute_metrics(labels, distances)

        # Log metrics to W&B
        for key, value in metrics.items():
            wandb.log({f"{phase}/{key}": value})

    # Finish the W&B run
    wandb.finish()

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

def perform_evaluation(model_checkpoint, loader, device):
    """
    Loads the model from the checkpoint and computes distances between anchor and positive embeddings.
    Returns all distances and corresponding labels.
    """
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    
    # Initialize models
    anchor_model = get_model(pretrained=False).to(device)
    posneg_model = get_model(pretrained=False).to(device)
    
    # Load state dicts
    anchor_model.load_state_dict(checkpoint['anchor_model_state'])
    posneg_model.load_state_dict(checkpoint['posneg_model_state'])
    
    # Set models to evaluation mode
    anchor_model.eval()
    posneg_model.eval()
    
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for anc, pos, labels in tqdm(loader, desc=f"Evaluating {os.path.basename(model_checkpoint)}"):
            anc = anc.to(device)
            pos = pos.to(device)
            labels = labels.cpu().numpy()

            # Compute embeddings
            anc_emb = anchor_model(anc)
            pos_emb = posneg_model(pos)
            anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
            pos_emb = nn.functional.normalize(pos_emb, p=2, dim=1)

            # Compute pairwise distances
            dist = torch.nn.functional.pairwise_distance(anc_emb, pos_emb)
            dist = dist.cpu().numpy()

            all_distances.extend(dist)
            all_labels.extend(labels)

    return np.array(all_distances), np.array(all_labels)

def compute_metrics(labels, distances):
    """
    Computes various evaluation metrics based on distances and true labels.
    """
    metrics = {}
    
    # ROC AUC
    #roc_auc = roc_auc_score(labels, distances)
    #metrics['roc_auc_1'] = roc_auc
    fpr, tpr, thresholds = roc_curve(labels, distances)
    metrics['roc_auc'] = auc(fpr, tpr)

    # PR AUC
    precision, recall, _ = precision_recall_curve(labels, distances)
    pr_auc = auc(recall, precision)
    metrics['pr_auc'] = pr_auc

    # Average Precision
    avg_precision = average_precision_score(labels, distances)
    metrics['avg_precision'] = avg_precision

    # Calculate average distances for each class
    if len(np.unique(labels)) >= 2:
        avg_distance_normal = np.mean(distances[labels == 0])
        avg_distance_fraud = np.mean(distances[labels == 1])
        metrics['avg_distance_normal'] = avg_distance_normal
        metrics['avg_distance_fraud'] = avg_distance_fraud

    # Accuracy at optimal threshold
    #fpr, tpr, thresholds = roc_curve(labels, distances)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds = (distances > optimal_threshold).astype(int)
    acc = accuracy_score(labels, preds)
    metrics['accuracy'] = acc
    f1 = f1_score(labels, preds)
    metrics['f1_score'] = f1
    metrics['optimal_threshold'] = optimal_threshold

    return metrics

def get_model(pretrained=True):
    """
    Initializes the ConvNeXt-Small model, removing the final classification layer.
    """
    model = models.convnext_small(
        weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    # Remove the final classification layer to obtain feature embeddings
    model.classifier[2] = nn.Identity()
    return model

def main():
    """
    The main function orchestrates the evaluation pipeline.
    """
    # Suppress warnings related to undefined metrics
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    args = parse_args()

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="evaluation"
    )

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define transformations using pretrained ConvNeXt-Small weights
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    # Find all model checkpoints in the model directory
    model_checkpoints = [
        os.path.join(args.model_dir, f) for f in os.listdir(args.model_dir)
        if f.endswith('.pt') or f.endswith('.pth')
    ]

    if not model_checkpoints:
        print(f"No model checkpoints found in {args.model_dir}")
        return

    # Evaluate each model checkpoint
    for checkpoint in model_checkpoints:
        evaluate_model(checkpoint, args, transform)

if __name__ == "__main__":
    main()
