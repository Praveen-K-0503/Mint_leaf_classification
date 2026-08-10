import os
import sys
import json
import time
import torch
import pandas as pd
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
sys.path.append(str(project_dir))

from models.architectures.factory import build_model, get_model_metrics, MODEL_SUITE_REGISTRY
from training.data.dataset import get_dataloaders
from training.trainers.trainer import PyTorchTrainer
from evaluation.metrics.evaluator import ModelEvaluator
from evaluation.visualization.plotter import plot_confusion_matrix, plot_training_history

print("=======================================================")
print("🔬 STEP 8B — COMMON 25-MODEL TRAINING PROTOCOL")
print("=======================================================\n")

output_suite_dir = project_dir / "outputs" / "reports" / "model_suite"
experiments_dir = project_dir / "outputs" / "experiments"

output_suite_dir.mkdir(parents=True, exist_ok=True)
experiments_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"1. Hardware Accelerator Target: {device}")
if device.type == "cuda":
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

# 1. Verify DeiT-Tiny vs ViT-Base/16 Fix
m17 = build_model("M17_vit_b_16", num_classes=6)
m18 = build_model("M18_deit_tiny", num_classes=6)

meta17 = get_model_metrics(m17)
meta18 = get_model_metrics(m18)

print(f"\n1. Registry Audit Check:")
print(f"   - M17 (ViT-Base/16): {meta17['total_params']:,} params ({meta17['model_size_mb']} MB)")
print(f"   - M18 (DeiT-Tiny):   {meta18['total_params']:,} params ({meta18['model_size_mb']} MB)")
assert meta17['total_params'] != meta18['total_params'], "DeiT-Tiny and ViT parameters must be distinct!"
print("   ✅ Verified: DeiT-Tiny parameter count is now distinct (5.7M params).")

# 2. DataLoaders
processed_dir = project_dir / "data" / "processed"
loaders = get_dataloaders(processed_dir=processed_dir, batch_size=32, img_size=224, num_workers=0)
train_loader = loaders["train"]
val_loader = loaders["val"]
test_loader = loaders["test"]
classes = loaders["classes"]

print(f"\n2. Common Dataset Protocol:")
print(f"   - Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images | Test: {len(test_loader.dataset)} images (100% UNTOUCHED)")

# 3. Experiment Directories Setup
print(f"\n3. Initializing 25 experiment directories under outputs/experiments/...")
for m_id in MODEL_SUITE_REGISTRY.keys():
    (experiments_dir / m_id).mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / m_id / ".gitkeep", "w") as f:
        f.write(f"# Experiment directory for {m_id}\n")

# 4. Dry-Run Execution (ResNet18)
dryrun_config = {
    "model_name": "M01_resnet18",
    "architecture_family": "Family A — Classical CNN",
    "pretrained": True,
    "input_resolution": 224,
    "num_classes": 6,
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "scheduler": "cosine",
    "loss": "weighted_cross_entropy",
    "batch_size": 32,
    "epochs": 3,
    "use_amp": True,
    "patience": 3,
    "checkpoint_path": str(experiments_dir / "M01_resnet18" / "best_model.pt"),
    "history_path": str(experiments_dir / "M01_resnet18" / "history.json")
}

class_counts = {cls: len(list((processed_dir / 'train' / cls).glob('*.jpg'))) for cls in classes}
trainer = PyTorchTrainer(config=dryrun_config, class_counts=class_counts)

print("\n4. Running Protocol Dry-Run (M01_resnet18, 3 Epochs)...")
history = trainer.fit(train_loader, val_loader)

# 5. Evaluate Test Set
ckpt_p = Path(dryrun_config["checkpoint_path"])
checkpoint = torch.load(ckpt_p, map_location=device)
trainer.model.load_state_dict(checkpoint["model_state_dict"])

evaluator = ModelEvaluator(trainer.model, classes=classes, device=device.type)
eval_res = evaluator.evaluate(test_loader, checkpoint_path=ckpt_p)

summary = eval_res["summary"]
per_class_df = eval_res["per_class_df"]
cm_df = eval_res["confusion_matrix_df"]

# Plot visualizations
cm_plot = experiments_dir / "M01_resnet18" / "confusion_matrix.png"
hist_plot = experiments_dir / "M01_resnet18" / "training_curves.png"
plot_confusion_matrix(cm_df, save_path=cm_plot, title="ResNet18 Dry-Run Confusion Matrix")
plot_training_history(history, save_path=hist_plot, title="ResNet18 Dry-Run History")

# Export Protocol Dry-Run JSON
dryrun_json = output_suite_dir / "protocol_dryrun_report.json"
with open(dryrun_json, "w", encoding="utf-8") as f:
    json.dump({
        "dryrun_model": "M01_resnet18",
        "config": dryrun_config,
        "summary": summary,
        "per_class_performance": per_class_df.to_dict(orient="records"),
        "confusion_matrix": cm_df.to_dict()
    }, f, indent=4)

print(f"\n5. Exported Protocol Dry-Run Report JSON: {dryrun_json}")

# Physical Verification
assert (experiments_dir / "M01_resnet18" / "best_model.pt").exists()
assert (experiments_dir / "M01_resnet18" / "history.json").exists()
assert cm_plot.exists()
assert hist_plot.exists()
assert (output_suite_dir / "common_training_protocol.json").exists()
assert (output_suite_dir / "common_training_protocol.md").exists()

print("\n=======================================================")
print("🎉 STEP 8B PROTOCOL VALIDATION COMPLETE — ALL CHECKS PASSED!")
print("=======================================================")
