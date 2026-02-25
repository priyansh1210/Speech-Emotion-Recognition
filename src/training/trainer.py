"""
Training and evaluation loop for the multimodal SER model.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config: dict,
        device: torch.device,
        fold: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.fold = fold

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.grad_clip = train_cfg["grad_clip"]
        self.checkpoint_dir = train_cfg["checkpoint_dir"]
        self.num_classes = len(config["data"]["emotions"])

        self.scaler = GradScaler()
        self.best_f1 = 0.0
        self.history = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in pbar:
            audio = batch["audio"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attn_mask"].to(self.device)
            emotion_targets = batch["emotion"].to(self.device)
            vad_targets = batch["vad"].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(audio, input_ids, attn_mask)
                loss_dict = self.criterion(
                    outputs["emotion_logits"],
                    outputs["vad_scores"],
                    emotion_targets,
                    vad_targets,
                )
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = outputs["emotion_logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(emotion_targets.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = compute_metrics(all_labels, all_preds, self.num_classes)
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in tqdm(self.val_loader, desc="Eval ", leave=False):
            audio = batch["audio"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attn_mask"].to(self.device)
            emotion_targets = batch["emotion"].to(self.device)
            vad_targets = batch["vad"].to(self.device)

            with autocast():
                outputs = self.model(audio, input_ids, attn_mask)
                loss_dict = self.criterion(
                    outputs["emotion_logits"],
                    outputs["vad_scores"],
                    emotion_targets,
                    vad_targets,
                )

            total_loss += loss_dict["total"].item()
            preds = outputs["emotion_logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(emotion_targets.cpu().tolist())

        metrics = compute_metrics(all_labels, all_preds, self.num_classes)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        path = os.path.join(self.checkpoint_dir, f"fold{self.fold}_best.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        print(f"  Checkpoint saved → {path}")

    def fit(self) -> list:
        print(f"\n{'='*60}")
        print(f" Fold {self.fold} | Training for {self.epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            self.scheduler.step()

            elapsed = time.time() - t0
            val_f1 = val_metrics["weighted_f1"]

            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val WF1: {val_f1:.4f} | "
                f"Val UA: {val_metrics['ua']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.save_checkpoint(epoch, val_metrics)

            self.history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

        print(f"\nBest Val WF1: {self.best_f1:.4f}")
        return self.history


def build_optimizer(model: nn.Module, config: dict):
    """Build AdamW with differential learning rates (lower for RoBERTa)."""
    train_cfg = config["training"]
    lr_bert = train_cfg["lr_bert"]
    lr_other = train_cfg["lr_other"]
    wd = train_cfg["weight_decay"]

    bert_params = list(model.text_model.roberta.parameters())
    bert_param_ids = {id(p) for p in bert_params}
    other_params = [p for p in model.parameters() if id(p) not in bert_param_ids]

    param_groups = [
        {"params": bert_params, "lr": lr_bert},
        {"params": other_params, "lr": lr_other},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=wd)


def build_scheduler(optimizer, config: dict):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
