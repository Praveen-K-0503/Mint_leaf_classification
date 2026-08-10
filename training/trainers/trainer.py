import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from pathlib import Path

from training.callbacks.callbacks import EarlyStopping, ModelCheckpoint, HistoryLogger
from models.architectures.factory import build_model
from training.losses.focal_loss import get_loss_function

class PyTorchTrainer:
    """
    Reusable PyTorch Trainer engine supporting GPU detection, AMP, early stopping,
    checkpointing, and training history logging.
    """
    def __init__(self, config, class_counts=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Training Hardware Target: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

        # Build Model
        self.model = build_model(
            model_name=config.get("model_name", "resnet18"),
            num_classes=config.get("num_classes", 6),
            pretrained=config.get("pretrained", True)
        ).to(self.device)

        # Loss Function
        self.criterion = get_loss_function(
            loss_name=config.get("loss", "cross_entropy"),
            class_counts=class_counts,
            device=self.device,
            gamma=config.get("focal_gamma", 2.0)
        )

        # Optimizer
        opt_name = config.get("optimizer", "adam").lower()
        lr = config.get("learning_rate", 0.001)
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Learning Rate Scheduler
        sched_name = config.get("scheduler", "cosine").lower()
        epochs = config.get("epochs", 5)
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif sched_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        else:
            self.scheduler = None

        # AMP Scaler
        self.use_amp = self.device.type == "cuda" and config.get("use_amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Callbacks
        ckpt_path = config.get("checkpoint_path", "models/checkpoints/best_model.pt")
        self.checkpoint = ModelCheckpoint(ckpt_path, monitor="val_macro_f1", mode="max")
        self.early_stopping = EarlyStopping(patience=config.get("patience", 5))
        self.history_logger = HistoryLogger(config.get("history_path", "outputs/reports/training_dataset/training_history.json"))

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for images, targets, _ in dataloader:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')
        return epoch_loss, epoch_acc, epoch_f1

    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _ in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_loss / len(dataloader.dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        val_bal_acc = balanced_accuracy_score(all_targets, all_preds)
        val_macro_f1 = f1_score(all_targets, all_preds, average='macro')
        return val_loss, val_acc, val_bal_acc, val_macro_f1

    def fit(self, train_loader, val_loader):
        epochs = self.config.get("epochs", 5)
        print(f"\n🚀 Starting Training ({epochs} Epochs Target)...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_bal_acc, val_macro_f1 = self.validate_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - t0
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "train_macro_f1": round(train_f1, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_balanced_acc": round(val_bal_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
                "epoch_time_sec": round(elapsed, 2)
            }

            self.history_logger.log_epoch(metrics)
            print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} BalAcc: {val_bal_acc:.4f} MacroF1: {val_macro_f1:.4f}")

            # Checkpoint & Early Stopping
            self.checkpoint.check_and_save(val_macro_f1, self.model, self.optimizer, epoch, extra_info=metrics)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("🛑 Early stopping triggered!")
                break

        return self.history_logger.history
