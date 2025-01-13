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
    precision_recall_fscore_support,
    average_precision_score,
    auc,
    accuracy_score
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
    parser.add_argument("--eval_epoch", type=int, default=1)
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
    val_dataset = MixedValTestDataset(
        args.normal_data_dir, args.fraud_data_dir, mode="val", transform=transform
    )
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_eval
    )
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
        # Remove the final classification layer to obtain feature embeddings
        model.classifier[2] = nn.Identity()
        return model

    anchor_model = get_model(pretrained=args.pretrained)
    posneg_model = get_model(pretrained=args.pretrained)

    # Initialize the optimizer with parameters from both models
    optimizer = optim.AdamW(
        list(anchor_model.parameters()) + list(posneg_model.parameters()),
        lr=args.lr
    )
    criterion = TripletLoss(margin=1.0)

    # Prepare models, optimizer, and data loaders with Accelerator
    (
        anchor_model,
        posneg_model,
        optimizer,
        train_loader,
        val_loader,
        test_loader
    ) = accelerator.prepare(
        anchor_model,
        posneg_model,
        optimizer,
        train_loader,
        val_loader,
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
        pos_emb = posneg_model(pos)
        neg_emb = posneg_model(neg)
        anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
        pos_emb = nn.functional.normalize(pos_emb, p=2, dim=1)
        neg_emb = nn.functional.normalize(neg_emb, p=2, dim=1)
        return anc_emb, pos_emb, neg_emb

    #---------------------------------------------------------------------------
    # Training Loop
    #---------------------------------------------------------------------------
    def train_one_epoch(loader, epoch, pbar):
        """
        Trains the model for one epoch using triplet loss.
        """
        nonlocal global_step
        anchor_model.train()
        posneg_model.train()

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
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)

        avg_loss = total_loss / total_samples if total_samples else 0.0
        return avg_loss

    #---------------------------------------------------------------------------
    # Evaluation: Normal (0) vs Fraud (1) Classification
    #         Uses discrete predictions based on distance thresholds
    #---------------------------------------------------------------------------
    def evaluate_balanced(loader, phase="val"):
        """
        Evaluates the model by computing distances between anchor and positive embeddings.
        Logs key metrics directly to W&B without using tables.
        """
        anchor_model.eval()
        posneg_model.eval()

        all_distances = []
        all_labels = []
        

        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for anc, imgs, labels in loader:
                anc = anc.to(accelerator.device)
                imgs = imgs.to(accelerator.device)
                labels = labels.cpu().numpy()  # normal=0, fraud=1

                # Compute embeddings
                anc_emb = anchor_model(anc)
                img_emb = posneg_model(imgs)
                anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
                img_emb = nn.functional.normalize(img_emb, p=2, dim=1)




                #val_loss = criterion(anc_emb, pos_emb, neg_emb)
                #bs = anc.size(0)
                #total_val_loss += (val_loss.item() * bs)
                #total_val_samples += bs



                # Compute pairwise distances
                dist = torch.nn.functional.pairwise_distance(anc_emb, img_emb)
                dist = dist.cpu().numpy()

                all_distances.extend(dist)
                all_labels.extend(labels)

        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)


        #avg_val_loss = total_val_loss / total_val_samples if total_val_samples else 0.0

        # Skip evaluation if only one class is present
        if len(np.unique(all_labels)) < 2:
            if accelerator.is_main_process:
                wandb.log({f"{phase}/info": "Only one class present, metrics undefined."})
            return


        # Calculate average distances for each class
        avg_distance_normal = np.mean(all_distances[all_labels == 0])
        avg_distance_fraud = np.mean(all_distances[all_labels == 1])

        if accelerator.is_main_process:
            wandb.log({
                f"{phase}/avg_distance_normal": avg_distance_normal,
                f"{phase}/avg_distance_fraud": avg_distance_fraud,
            })

        # Initialize lists to store metrics for each threshold
        thresholds = np.linspace(0.0, 0.3, 30)  # 50 thresholds ranging from 0.0 to 0.5
        thr_metrics = []
        precisions = []
        recalls = []

        
        for th in thresholds:
            # Generate predictions based on the current threshold
            # distance <= th => normal (0), distance > th => fraud (1)
            preds = np.where(all_distances <= th, 0, 1)

            # Compute precision, recall, F1-score
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, preds, average='binary', zero_division=0
            )

            # Compute accuracy
            acc = accuracy_score(all_labels, preds)

            # Store the metrics
            thr_metrics.append({
                "threshold": th,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": acc
            })
            precisions.append(precision)
            recalls.append(recall)

            # Log metrics for each threshold directly to W&B
            if accelerator.is_main_process:
                wandb.log({
                    #f"{phase}/precision_threshold_{th:.2f}": precision,
                    #f"{phase}/recall_threshold_{th:.2f}": recall,
                    f"{phase}/f1_score_threshold_{th:.2f}": f1,
                    f"{phase}/accuracy_threshold_{th:.2f}": acc
                })

        # Sort the (recall, precision) pairs based on recall for AP calculation
        paired = sorted(zip(recalls, precisions), key=lambda x: x[0])
        sorted_recalls = [p[0] for p in paired]
        sorted_precisions = [p[1] for p in paired]

        # Compute Area Under the PR Curve as Average Precision
        pr_auc_value = auc(sorted_recalls, sorted_precisions)
        avg_precision = average_precision_score(all_labels, -all_distances)  # Using -distance as score

        if accelerator.is_main_process:
            # Log Average Precision
            wandb.log({f"{phase}/pr_auc_value": pr_auc_value})
            wandb.log({f"{phase}/avg_precision": avg_precision})

            # Plot and log the PR Curve
            plt.figure(figsize=(8, 6))
            plt.step(sorted_recalls, sorted_precisions, where='post',
                     label=f'AP={avg_precision}, PR AUC={pr_auc_value:.3f}')
            plt.fill_between(sorted_recalls, sorted_precisions, step='post', alpha=0.2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{phase.capitalize()} PR Curve')
            plt.grid(True)
            plt.legend(loc='upper right')
            pr_curve_path = os.path.join(args.output_dir, f"{phase}_precision_recall_curve.png")
            plt.savefig(pr_curve_path)
            plt.close()
            wandb.log({f"{phase}/precision_recall_curve": wandb.Image(pr_curve_path)})

            # Log the number of Normal and Fraud samples
            num_normal = np.sum(all_labels == 0)
            num_fraud = np.sum(all_labels == 1)
            wandb.log({
                f"{phase}/num_normal": int(num_normal),
                f"{phase}/num_fraud": int(num_fraud)
            })
        
        #return avg_val_loss

    # Calculate total training steps for progress bar
    total_steps = len(train_loader) * args.epochs
    pbar = tqdm(total=total_steps, desc="Training Steps", disable=not accelerator.is_main_process)

    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_one_epoch(train_loader, epoch, pbar)

        # Placeholder for validation loss (if triplet loss on validation set is needed)

        if (epoch % args.eval_epoch) == 0:
            # Perform evaluation on the validation set
            evaluate_balanced(val_loader, phase="val")
            # Optionally, evaluate on the test set each epoch
            # evaluate_balanced(test_loader, phase="test")

            if accelerator.is_main_process:
                # Log epoch-level metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    #"val_loss": val_loss,
                })
        else:
            if accelerator.is_main_process:
                # Log epoch-level metrics even if not evaluating
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    #"val_loss": val_loss,
                })

        # Save model checkpoints at specified epochs
        if (epoch % args.save_epoch) == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f"{args.wandb_run_name}_checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'anchor_model_state': anchor_model.state_dict(),
                'posneg_model_state': posneg_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    if accelerator.is_main_process:
        # Close the progress bar and finish the W&B run
        pbar.close()
        wandb.finish()


if __name__ == "__main__":
    main()
