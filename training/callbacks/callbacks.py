import os
import json
import torch
from pathlib import Path

class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    def __init__(self, patience=5, delta=0.0001, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"   [EarlyStopping Counter]: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class ModelCheckpoint:
    """
    Saves best model checkpoint based on validation metric.
    """
    def __init__(self, filepath, monitor='val_macro_f1', mode='max'):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_metric = -float('inf') if mode == 'max' else float('inf')

    def check_and_save(self, current_metric, model, optimizer, epoch, extra_info=None):
        is_best = False
        if self.mode == 'max' and current_metric > self.best_metric:
            is_best = True
        elif self.mode == 'min' and current_metric < self.best_metric:
            is_best = True

        if is_best:
            self.best_metric = current_metric
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': self.best_metric,
                'extra_info': extra_info or {}
            }
            torch.save(state, self.filepath)
            print(f"   💾 Saved Best Checkpoint ({self.monitor}: {current_metric:.4f}) to: {self.filepath}")
        return is_best

class HistoryLogger:
    """
    Logs training history across epochs to JSON.
    """
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.history = []

    def log_epoch(self, epoch_metrics):
        self.history.append(epoch_metrics)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)
