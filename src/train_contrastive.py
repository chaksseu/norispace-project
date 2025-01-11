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
    auc,
    accuracy_score
)
from sklearn.exceptions import UndefinedMetricWarning

import matplotlib.pyplot as plt
import wandb
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning with Normal-only training & Mixed Normal+Fraud evaluation")

    # 경로 설정
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

    # Model
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained ConvNeXt-Small model")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="contrastive_learning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


###############################################################################
# 1) ANPTrainDataset: (train) Normal-only, Triplet (anchor, pos, neg)
###############################################################################
class ANPTrainDataset(Dataset):
    """
    Training용 Dataset (Normal-only):
    normal_data_dir/train/anchor, normal_data_dir/train/pos, normal_data_dir/train/neg
    에서 prefix를 추출하여 (anchor, pos, neg) 로드합니다.
    """
    def __init__(self, normal_data_dir, transform=None):
        super().__init__()
        self.root_dir = normal_data_dir
        self.transform = transform

        self.anchor_dir = os.path.join(self.root_dir, "train", "anchor")
        self.pos_dir    = os.path.join(self.root_dir, "train", "pos")
        self.neg_dir    = os.path.join(self.root_dir, "train", "neg")

        if not all([os.path.isdir(self.anchor_dir),
                    os.path.isdir(self.pos_dir),
                    os.path.isdir(self.neg_dir)]):
            raise ValueError(f"Train directories not found under: {self.root_dir}")

        anchor_files = [f for f in os.listdir(self.anchor_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_anchor" in f]
        # prefix: 예) "xxxxx_anchor.png" -> "xxxxx"
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

        # anchor
        anchor_filename = f"{prefix}_anchor.png"
        anchor_path = os.path.join(self.anchor_dir, anchor_filename)
        anchor_img = self._load_img(anchor_path)

        # pos
        pos_filename = f"{prefix}_pos.png"
        pos_path = os.path.join(self.pos_dir, pos_filename)
        pos_img = self._load_img(pos_path)

        # neg
        neg_candidates = [
            f for f in os.listdir(self.neg_dir)
            if f.startswith(prefix + "_neg_") and f.lower().endswith(('.png','.jpg','.jpeg'))
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


###############################################################################
# 2) MixedValTestDataset: (val, test) => Normal + Fraud
#    => anchor-pos 쌍, 라벨은 normal=1, fraud=0
###############################################################################
class MixedValTestDataset(Dataset):
    """
    Val/Test용 Dataset:
    - normal_data_dir/{val|test}/anchor + pos
    - fraud_data_dir/{val|test}/anchor + pos
    를 동등하게 섞는다.
    """
    def __init__(self, normal_data_dir, fraud_data_dir, mode="val", transform=None):
        super().__init__()
        self.transform = transform
        self.mode = mode

        # 예: normal_data_dir/val/anchor, normal_data_dir/val/pos
        self.normal_anchor_dir = os.path.join(normal_data_dir, mode, "anchor")
        self.normal_pos_dir    = os.path.join(normal_data_dir, mode, "pos")

        # 예: fraud_data_dir/val/anchor, ...
        self.fraud_anchor_dir = os.path.join(fraud_data_dir, mode, "anchor")
        self.fraud_pos_dir    = os.path.join(fraud_data_dir, mode, "pos")

        normal_anchor_files = [
            f for f in os.listdir(self.normal_anchor_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_anchor" in f
        ]
        fraud_anchor_files = [
            f for f in os.listdir(self.fraud_anchor_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_anchor" in f
        ]

        # 개수를 맞춤(최소 개수만큼)
        np.random.shuffle(normal_anchor_files)
        np.random.shuffle(fraud_anchor_files)
        min_count = min(len(normal_anchor_files), len(fraud_anchor_files))
        self.normal_anchor_files = normal_anchor_files[:min_count]
        self.fraud_anchor_files  = fraud_anchor_files[:min_count]

        # (prefix, label) = ( "xxx", 1 ) for normal / ( "yyy", 0 ) for fraud
        # anchor 파일명: e.g., "somefile_anchor.png" -> prefix = "somefile"
        self.samples = []
        for naf in self.normal_anchor_files:
            prefix = naf.rsplit("_anchor", 1)[0]
            self.samples.append((prefix, 0))  # normal=0
        for faf in self.fraud_anchor_files:
            prefix = faf.rsplit("_anchor", 1)[0]
            self.samples.append((prefix, 1))  # fraud=1

        # 섞기
        np.random.shuffle(self.samples)
        print(f"[MixedValTestDataset-{mode}] Normal+Fraud => {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, label = self.samples[idx]

        if label == 1:  # normal
            anchor_dir = self.normal_anchor_dir
            pos_dir    = self.normal_pos_dir
        else:           # fraud
            anchor_dir = self.fraud_anchor_dir
            pos_dir    = self.fraud_pos_dir

        anchor_path = os.path.join(anchor_dir, f"{prefix}_anchor.png")
        pos_path    = os.path.join(pos_dir, f"{prefix}_pos.png")

        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img    = Image.open(pos_path).convert("RGB")
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img    = self.transform(pos_img)

        # label=1(normal), 0(fraud)
        return {
            "anchor": anchor_img,
            "pos": pos_img,
            "label": label
        }


###############################################################################
# ContrastiveLoss (Triplet)
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

    # W&B
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name
        )
        wandb.config.update(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    # Pretrained transform
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    #---------------------------------------------------------------------------
    # (1) Train Dataset: Normal-only (triplet)
    #---------------------------------------------------------------------------
    train_dataset = ANPTrainDataset(args.normal_data_dir, transform=transform)

    def collate_fn_train(batch):
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
    # 모델 (anchor_model, posneg_model)
    #---------------------------------------------------------------------------
    def get_model(pretrained=True):
        model = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[2] = nn.Identity()
        return model

    anchor_model = get_model(pretrained=args.pretrained)
    posneg_model = get_model(pretrained=args.pretrained)

    optimizer = optim.Adam(
        list(anchor_model.parameters()) + list(posneg_model.parameters()),
        lr=args.lr
    )
    criterion = ContrastiveLoss(margin=1.0)

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
    # Triplet forward
    #---------------------------------------------------------------------------
    def get_triplet_emb(anc, pos, neg):
        anc_emb = anchor_model(anc)
        pos_emb = posneg_model(pos)
        neg_emb = posneg_model(neg)
        anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
        pos_emb = nn.functional.normalize(pos_emb, p=2, dim=1)
        neg_emb = nn.functional.normalize(neg_emb, p=2, dim=1)
        return anc_emb, pos_emb, neg_emb

    #---------------------------------------------------------------------------
    # Train Loop
    #---------------------------------------------------------------------------
    def train_one_epoch(loader, epoch, pbar):
        nonlocal global_step
        anchor_model.train()
        posneg_model.train()

        total_loss = 0.0
        total_samples = 0

        for step, (anc, pos, neg) in enumerate(loader):
            anc = anc.to(accelerator.device)
            pos = pos.to(accelerator.device)
            neg = neg.to(accelerator.device)

            anc_emb, pos_emb, neg_emb = get_triplet_emb(anc, pos, neg)
            loss = criterion(anc_emb, pos_emb, neg_emb)
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            bs = anc.size(0)
            total_loss += (loss.item() * args.gradient_accumulation_steps * bs)
            total_samples += bs

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
    # Evaluation: Normal vs Fraud 분류
    # distance <= threshold => Normal(1), distance > threshold => Fraud(0)
    #---------------------------------------------------------------------------
    def evaluate_balanced(loader, phase="val"):
        anchor_model.eval()
        posneg_model.eval()

        all_distances = []
        all_labels = []

        with torch.no_grad():
            for anc, imgs, labels in loader:
                anc = anc.to(accelerator.device)
                imgs = imgs.to(accelerator.device)
                # label = 0(normal), 1(fraud)
                # => all_labels 에 확장
                labels = labels.cpu().numpy()

                anc_emb = anchor_model(anc)
                img_emb = posneg_model(imgs)
                anc_emb = nn.functional.normalize(anc_emb, p=2, dim=1)
                img_emb = nn.functional.normalize(img_emb, p=2, dim=1)

                dist = torch.nn.functional.pairwise_distance(anc_emb, img_emb)
                dist = dist.cpu().numpy()

                all_distances.extend(dist)
                all_labels.extend(labels)

        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)

        if len(np.unique(all_labels)) < 2:
            if accelerator.is_main_process:
                wandb.log({f"{phase}/info": "Only one class present, metrics undefined."})
            return

        # "distance가 작으면 normal" 이므로, 점수(score) = -distance
        # => score가 클수록 normal
        scores = -all_distances

        # 1) Average Precision
        avg_precision = average_precision_score(all_labels, scores)

        # 2) Precision-Recall Curve
        prec_curve, rec_curve, _ = precision_recall_curve(all_labels, scores)
        pr_auc_value = auc(rec_curve, prec_curve)

        # 3) ROC AUC
        # roc_auc_value = roc_auc_score(all_labels, scores)
        # fpr, tpr, _ = roc_curve(all_labels, scores)

        # 4) Threshold sweep => acc, precision, recall, f1
        thresholds = np.arange(0.0, 0.7, 0.01)
        thr_metrics = []
        for th in thresholds:
            # distance <= th => normal(1)
            # distance > th  => fraud(0)
            # => distance <= th <=> -distance >= -th => scores >= -th
            preds = np.where(all_distances <= th, 1, 0)
            # 또는 preds = (scores >= -th).astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, preds, average='binary', zero_division=0
            )
            acc = accuracy_score(all_labels, preds)
            thr_metrics.append({
                "threshold": th,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": acc
            })
            if accelerator.is_main_process:
                wandb.log({
                    f"{phase}_th_{th:.2f}/precision": precision,
                    f"{phase}_th_{th:.2f}/recall": recall,
                    f"{phase}_th_{th:.2f}/f1": f1,
                    f"{phase}_th_{th:.2f}/accuracy": acc
                })

        if accelerator.is_main_process:
            # 5) Average Precision 로그
            wandb.log({f"{phase}/average_precision": avg_precision})

            # 테이블 형태로 로깅
            table = wandb.Table(columns=["Threshold", "Precision", "Recall", "F1 Score", "Accuracy"])
            for metric in thr_metrics:
                table.add_data(
                    round(metric["threshold"], 2),
                    round(metric["precision"], 4),
                    round(metric["recall"], 4),
                    round(metric["f1_score"], 4),
                    round(metric["accuracy"], 4)
                )
            wandb.log({f"{phase}/Threshold Sweep Metrics": table})

            # PR Curve
            plt.figure(figsize=(8, 6))
            plt.step(rec_curve, prec_curve, where='post', label=f'AP={avg_precision:.3f}, PR AUC={pr_auc_value:.3f}')
            plt.fill_between(rec_curve, prec_curve, step='post', alpha=0.2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{phase.capitalize()} PR Curve')
            plt.grid(True)
            plt.legend(loc='upper right')
            pr_curve_path = os.path.join(args.output_dir, f"{phase}_precision_recall_curve.png")
            plt.savefig(pr_curve_path)
            plt.close()
            wandb.log({f"{phase}/precision_recall_curve": wandb.Image(pr_curve_path)})

            # ROC Curve
            '''
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc_value:.3f}')
            plt.plot([0,1],[0,1],'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{phase.capitalize()} ROC Curve')
            plt.grid(True)
            plt.legend(loc='lower right')
            roc_curve_path = os.path.join(args.output_dir, f"{phase}_roc_curve.png")
            plt.savefig(roc_curve_path)
            plt.close()
            wandb.log({f"{phase}/roc_curve": wandb.Image(roc_curve_path)})
            '''

            # 데이터 개수
            num_normal = np.sum(all_labels == 1)
            num_fraud  = np.sum(all_labels == 0)
            wandb.log({
                f"{phase}/num_normal": int(num_normal),
                f"{phase}/num_fraud": int(num_fraud)
            })


    total_steps = len(train_loader) * args.epochs
    pbar = tqdm(total=total_steps, desc="Training Steps", disable=not accelerator.is_main_process)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(train_loader, epoch, pbar)

        # 여기서는 val_loss를 triplet 관점에서 계산하는 예시(선택)
        # 굳이 normal-only valid set이 있다면 아래처럼 가능 (생략 가능)
        val_loss = train_loss  # 임시

        if (epoch % args.eval_epoch) == 0:
            # Normal+Fraud 통합 평가
            evaluate_balanced(val_loader, phase="val")
            # 원한다면 test도 epoch마다 돌릴 수 있음
            # evaluate_balanced(test_loader, phase="test")

            if accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
        else:
            if accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

        # Checkpoint
        if (epoch % args.save_epoch) == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'anchor_model_state': anchor_model.state_dict(),
                'posneg_model_state': posneg_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    if accelerator.is_main_process:
        pbar.close()
        wandb.finish()


if __name__ == "__main__":
    main()
