import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from models.architectures.factory import get_model_metrics

class ModelEvaluator:
    """
    Evaluation Engine computing comprehensive 6-class performance metrics:
    - Overall Accuracy & Balanced Accuracy
    - Macro F1 & Weighted F1
    - Per-class Precision, Recall, F1
    - Confusion Matrix DataFrame
    - Inference Latency (ms/image), Parameter Count, Model Size (MB)
    """
    def __init__(self, model, classes, device="cpu"):
        self.model = model
        self.classes = classes
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, test_loader, checkpoint_path=None):
        all_preds = []
        all_targets = []
        all_paths = []
        inference_times = []

        print(f"\n🔍 Evaluating model on test set ({len(test_loader.dataset)} images)...")

        with torch.no_grad():
            for images, targets, paths in test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                t0 = time.time()
                outputs = self.model(images)
                t_elapsed = (time.time() - t0) * 1000.0  # ms for batch
                
                # Per-image latency estimate
                per_image_ms = t_elapsed / images.size(0)
                inference_times.append(per_image_ms)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_paths.extend(paths)

        # Overall Metrics
        acc = accuracy_score(all_targets, all_preds)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

        # Per-Class Metrics
        p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
            all_targets, all_preds, labels=range(len(self.classes)), zero_division=0
        )

        per_class_df = pd.DataFrame({
            "class_name": self.classes,
            "precision": [round(x, 4) for x in p_class],
            "recall": [round(x, 4) for x in r_class],
            "f1_score": [round(x, 4) for x in f1_class],
            "support": support_class
        })

        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds, labels=range(len(self.classes)))
        cm_df = pd.DataFrame(cm, index=self.classes, columns=self.classes)

        # Hardware & Complexity Metrics
        model_meta = get_model_metrics(self.model, device=self.device)
        avg_latency_ms = float(np.mean(inference_times))

        ckpt_size_mb = 0.0
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt_size_mb = round(Path(checkpoint_path).stat().st_size / (1024 * 1024), 2)
        else:
            ckpt_size_mb = model_meta["model_size_mb"]

        metrics_summary = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_precision": round(float(precision_macro), 4),
            "macro_recall": round(float(recall_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(precision_weighted), 4),
            "weighted_recall": round(float(recall_weighted), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "avg_inference_latency_ms": round(avg_latency_ms, 3),
            "total_parameters": model_meta["total_params"],
            "trainable_parameters": model_meta["trainable_params"],
            "model_checkpoint_size_mb": ckpt_size_mb,
            "total_test_samples": len(all_targets)
        }

        return {
            "summary": metrics_summary,
            "per_class_df": per_class_df,
            "confusion_matrix_df": cm_df,
            "confusion_matrix_array": cm.tolist(),
            "raw_predictions": list(zip(all_paths, [int(x) for x in all_targets], [int(x) for x in all_preds]))
        }
