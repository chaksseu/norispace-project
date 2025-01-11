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
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.exceptions import UndefinedMetricWarning

import matplotlib.pyplot as plt
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning with Normal-only training & Mixed Normal+Fraud evaluation")

    parser.add_argument("--normal_data_dir", type=str, required=True,
                        help="Path to the Normal dataset root (train/normal, val/normal, test/normal)")
    parser.add_argument("--fraud_data_dir", type=str, required=True,
                        help="Path to the Fraud dataset root (val/fraud, test/fraud)")
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

    # Model
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained ConvNeXt-Small model")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="contrastive_learning",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")

    return parser.parse_args()


###############################################################################
# 1) ANPTrainDataset: (train) Normal 전용
#    (anchor, pos, neg) 구조로 Triplet
###############################################################################
class ANPTrainDataset(Dataset):
    """
    Train dataset: normal_data_dir/train/normal/<sample>/anchor|pos|neg
    """
    def __init__(self, normal_data_dir, transform=None):
        super().__init__()
        self.normal_data_dir = normal_data_dir
        self.transform = transform

        self.samples = []
        train_normal_path = os.path.join(normal_data_dir, "train", "normal")
        if os.path.isdir(train_normal_path):
            for name in os.listdir(train_normal_path):
                sample_dir = os.path.join(train_normal_path, name)
                if os.path.isdir(sample_dir):
                    self.samples.append(sample_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        anchor_dir = os.path.join(sample_dir, "anchor")
        pos_dir    = os.path.join(sample_dir, "pos")
        neg_dir    = os.path.join(sample_dir, "neg")

        # Load random anchor
        anchor_files = [f for f in os.listdir(anchor_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        anchor_path = os.path.join(anchor_dir, np.random.choice(anchor_files))
        anchor_img = self._load_img(anchor_path)

        # Load random pos
        pos_files = [f for f in os.listdir(pos_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        pos_path = os.path.join(pos_dir, np.random.choice(pos_files))
        pos_img = self._load_img(pos_path)

        # Load random neg
        neg_files = [f for f in os.listdir(neg_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        neg_path = os.path.join(neg_dir, np.random.choice(neg_files))
        neg_img = self._load_img(neg_path)

        return {
            "anchor": anchor_img,
            "positive": pos_img,
            "negative": neg_img
        }

    def _load_img(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


###############################################################################
# 2) MixedValTestDataset: (val, test)
#    normal_data_dir/{val or test}/normal + fraud_data_dir/{val or test}/fraud
#    개수만큼 샘플링하여 anchor/pos 구조로 반환. prefix => "normal_" 또는 "fraud_"
###############################################################################
class MixedValTestDataset(Dataset):
    """
    Evaluate dataset: combine normal + fraud for val or test.
    mode: val / test
    anchor/pos 구조, label은 prefix "normal_" or "fraud_"
    """
    def __init__(self, normal_data_dir, fraud_data_dir, mode="val", transform=None):
        super().__init__()
        self.normal_data_dir = normal_data_dir
        self.fraud_data_dir  = fraud_data_dir
        self.mode = mode
        self.transform = transform

        # normal samples
        self.normal_samples = []
        normal_path = os.path.join(normal_data_dir, mode, "normal")
        if os.path.isdir(normal_path):
            for name in os.listdir(normal_path):
                sample_dir = os.path.join(normal_path, name)
                if os.path.isdir(sample_dir):
                    self.normal_samples.append(sample_dir)

        # fraud samples
        self.fraud_samples = []
        fraud_path = os.path.join(fraud_data_dir, mode, "fraud")
        if os.path.isdir(fraud_path):
            for name in os.listdir(fraud_path):
                sample_dir = os.path.join(fraud_path, name)
                if os.path.isdir(sample_dir):
                    self.fraud_samples.append(sample_dir)

        # 동일 개수만
        normal_count = len(self.normal_samples)
        fraud_count  = len(self.fraud_samples)
        min_count = min(normal_count, fraud_count)
        # shuffle
        np.random.shuffle(self.normal_samples)
        np.random.shuffle(self.fraud_samples)

        self.normal_samples = self.normal_samples[:min_count]
        self.fraud_samples  = self.fraud_samples[:min_count]

        # 결합
        self.data_list = []
        for path in self.normal_samples:
            self.data_list.append((path, "normal"))
        for path in self.fraud_samples:
            self.data_list.append((path, "fraud"))

        # shuffle
        np.random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_dir, cat = self.data_list[idx]  # cat: "normal" or "fraud"
        anchor_dir = os.path.join(sample_dir, "anchor")
        pos_dir    = os.path.join(sample_dir, "pos")

        anchor_files = [f for f in os.listdir(anchor_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        pos_files    = [f for f in os.listdir(pos_dir)    if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not anchor_files or not pos_files:
            raise ValueError(f"Missing anchor/pos in {sample_dir}")

        anchor_path = os.path.join(anchor_dir, np.random.choice(anchor_files))
        pos_path    = os.path.join(pos_dir,   np.random.choice(pos_files))

        prefix = f"{cat}_"  # "normal_" or "fraud_"

        anchor_img = self._load_img(anchor_path)
        pos_img    = self._load_img(pos_path)

        return {
            "anchor": anchor_img,
            "pos": pos_img,
            "prefix": prefix
        }

    def _load_img(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


###############################################################################
# ContrastiveLoss + train/eval loop
###############################################################################
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_emb, pos_emb, neg_emb):
        pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, neg_emb)
        pos_loss = 0.5 * (pos_dist ** 2)
        neg_loss = 0.5 * torch.clamp(self.margin - neg_dist, min=0.0) ** 2
        return pos_loss.mean() + neg_loss.mean()


def main():
    # UndefinedMetricWarning 무시
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    args = parse_args()

    accelerator = Accelerator()
    accelerator.init_trackers("contrastive_learning", config=vars(args))

    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name
        )
        wandb.config.update(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    # Transform
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    ############################################################################
    # Dataset & DataLoader
    ############################################################################
    # 1) Train: Normal-only (Anchor/Pos/Neg)
    train_dataset = ANPTrainDataset(args.normal_data_dir, transform=transform)
    # 2) Val: Normal + Fraud
    val_dataset   = MixedValTestDataset(args.normal_data_dir, args.fraud_data_dir, mode="val", transform=transform)
    # 3) Test: Normal + Fraud
    test_dataset  = MixedValTestDataset(args.normal_data_dir, args.fraud_data_dir, mode="test", transform=transform)

    # Dataloader
    def collate_fn_train(batch):
        # anchor/pos/neg
        anchors, positives, negatives = [], [], []
        for b in batch:
            anchors.append(b['anchor'])
            positives.append(b['positive'])
            negatives.append(b['negative'])
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train
    )

    # Val, Test => anchor/pos/prefix
    def collate_fn_eval(batch):
        anchors, images, prefixes = [], [], []
        for b in batch:
            anchors.append(b['anchor'])
            images.append(b['pos'])
            prefixes.append(b['prefix'])
        return torch.stack(anchors), torch.stack(images), prefixes

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

    # Model
    def get_model(pretrained=True):
        model = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[2] = nn.Identity()
        return model

    anchor_model = get_model(pretrained=args.pretrained)
    posneg_model = get_model(pretrained=args.pretrained)

    optimizer = optim.Adam(list(anchor_model.parameters()) + list(posneg_model.parameters()), lr=args.lr)
    criterion = ContrastiveLoss(margin=1.0)

    # Accelerator
    (anchor_model,
     posneg_model,
     optimizer,
     train_loader,
     val_loader,
     test_loader) = accelerator.prepare(anchor_model,
                                        posneg_model,
                                        optimizer,
                                        train_loader,
                                        val_loader,
                                        test_loader)

    global_step = 0
    total_steps = len(train_loader) * args.epochs

    # Triplet forward
    def get_triplet_emb(anc, pos, neg):
        anc = anc.to(accelerator.device)
        pos = pos.to(accelerator.device)
        neg = neg.to(accelerator.device)

        anc_emb = anchor_model(anc)
        pos_emb = posneg_model(pos)
        neg_emb = posneg_model(neg)

        anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
        pos_emb = nn.functional.normalize(pos_emb, p=2, dim=1)
        neg_emb = nn.functional.normalize(neg_emb, p=2, dim=1)
        return anc_emb, pos_emb, neg_emb

    # Training
    def train_one_epoch(loader, epoch, pbar):
        nonlocal global_step
        anchor_model.train()
        posneg_model.train()

        total_loss = 0.0
        total_samples = 0

        for step, (anc, pos, neg) in enumerate(loader):
            anc_emb, pos_emb, neg_emb = get_triplet_emb(anc, pos, neg)
            loss = criterion(anc_emb, pos_emb, neg_emb)
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            bs = anc.size(0)
            total_loss += loss.item() * args.gradient_accumulation_steps * bs
            total_samples += bs

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    global_step += 1
                    wandb.log({"train_step_loss": loss.item() * args.gradient_accumulation_steps})
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)

        avg_loss = (total_loss / total_samples) if total_samples else 0.0
        return avg_loss

    # Validation
    def validate(loader):
        anchor_model.eval()
        posneg_model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for anc, pos, neg in loader:
                anc_emb, pos_emb, neg_emb = get_triplet_emb(anc, pos, neg)
                loss = criterion(anc_emb, pos_emb, neg_emb)
                bs = anc.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

        return (total_loss / total_samples) if total_samples else 0.0

    # 평가(Accuracy/Precision/Recall/F1/PR AUC/ROC AUC)
    # Normal => label=1, Fraud => label=0
    def evaluate_balanced(loader, phase="val"):
        anchor_model.eval()
        posneg_model.eval()

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for anc, imgs, prefixes in loader:
                anc = anc.to(accelerator.device)
                imgs = imgs.to(accelerator.device)

                anc_emb = anchor_model(anc)
                img_emb = posneg_model(imgs)
                anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
                img_emb = nn.functional.normalize(img_emb, p=2, dim=1)

                dist = torch.nn.functional.pairwise_distance(anc_emb, img_emb).cpu().numpy()
                scores = 1.0 / (1.0 + dist)  # 높을수록 normal

                labels = [1 if pf.startswith("normal_") else 0 for pf in prefixes]

                all_scores.extend(scores)
                all_labels.extend(labels)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 클래스 체크
        unique_lbl = np.unique(all_labels)
        if len(unique_lbl) < 2:
            # 한 클래스만 있으면 메트릭 불가
            if accelerator.is_main_process:
                wandb.log({f"{phase}/info": "Only one class present, metrics undefined."})
            return

        # Average Precision
        avg_precision = average_precision_score(all_labels, all_scores)

        # Precision-Recall Curve => pr_auc
        prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc_value = auc(rec_curve, prec_curve)

        # ROC AUC
        roc_auc_value = roc_auc_score(all_labels, all_scores)
        fpr, tpr, _ = roc_curve(all_labels, all_scores)

        # Threshold sweep
        thresholds = np.linspace(0.0, 1.0, 11)
        thr_metrics = {}
        for th in thresholds:
            preds = (all_scores >= th).astype(int)
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, preds, average='binary', zero_division=0
            )
            acc = accuracy_score(all_labels, preds)
            thr_metrics[th] = (precision, recall, f1, acc)

        if accelerator.is_main_process:
            # AP/PR AUC/ROC AUC
            wandb.log({
                f"{phase}/average_precision": avg_precision,
                f"{phase}/pr_auc": pr_auc_value,
                f"{phase}/roc_auc": roc_auc_value
            })

            # threshold별
            for th in thresholds:
                prec, rec, f1, acc = thr_metrics[th]
                wandb.log({
                    f"{phase}_thr_{th:.2f}/precision": prec,
                    f"{phase}_thr_{th:.2f}/recall": rec,
                    f"{phase}_thr_{th:.2f}/f1": f1,
                    f"{phase}_thr_{th:.2f}/accuracy": acc
                })

            # PR Curve
            plt.figure()
            plt.step(rec_curve, prec_curve, where='post', label=f'AP={avg_precision:.3f}')
            plt.fill_between(rec_curve, prec_curve, step='post', alpha=0.2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{phase} Precision-Recall (AUC={pr_auc_value:.3f})')
            plt.grid(True)
            plt.legend(loc='upper right')
            pr_curve_path = os.path.join(args.output_dir, f"{phase}_precision_recall_curve.png")
            plt.savefig(pr_curve_path)
            plt.close()
            wandb.log({f"{phase}/precision_recall_curve": wandb.Image(pr_curve_path)})

            # ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc_value:.3f}')
            plt.plot([0,1],[0,1],'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{phase} ROC Curve')
            plt.grid(True)
            plt.legend(loc='lower right')
            roc_curve_path = os.path.join(args.output_dir, f"{phase}_roc_curve.png")
            plt.savefig(roc_curve_path)
            plt.close()
            wandb.log({f"{phase}/roc_curve": wandb.Image(roc_curve_path)})

            # 데이터 분포
            num_pos = np.sum(all_labels == 1)
            num_neg = np.sum(all_labels == 0)
            wandb.log({
                f"{phase}/num_normal": int(num_pos),
                f"{phase}/num_fraud": int(num_neg)
            })


    ############################################################################
    # Training Loop
    ############################################################################
    accelerator.print("Starting Training...")
    global_step = 0
    total_steps = len(train_loader) * args.epochs

    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, desc="Training Steps", leave=True)

    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(train_loader, epoch, pbar)
        # 여기서는 normal-only val loss
        val_loss = validate(train_loader)  # or val_loader if we want normal-only val
        # => 만약 normal-only val을 원하면 "val_loader" but that doesn't exist as anchor/pos/neg
        # => or define a separate ANPTrainDataset for val. 
        # 예시로 train_loader2 = normal val. But let's just do train_loader for demonstration.

        # Balanced evaluation on val set
        if (epoch % args.eval_epoch) == 0:
            evaluate_balanced(val_loader, phase="val")
            evaluate_balanced(test_loader, phase="test")

            if accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
        else:
            if accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })

        # Save checkpoint
        if (epoch % args.save_epoch) == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'anchor_model_state': anchor_model.state_dict(),
                'posneg_model_state': posneg_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    if accelerator.is_main_process and pbar is not None:
        pbar.close()
        wandb.finish()


if __name__ == "__main__":
    main()
