import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_confusion_matrix(cm_df, save_path=None, title="Model Confusion Matrix"):
    """
    Plots Seaborn heatmap for confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens', cbar=False, linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved Confusion Matrix Heatmap: {save_path}")
    plt.close()

def plot_training_history(history, save_path=None, title="Training & Validation History"):
    """
    Plots training loss and macro F1 curves over epochs.
    """
    df_hist = pd.DataFrame(history)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Loss Curve
    axes[0].plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_title('Loss Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Accuracy & Macro F1 Curve
    axes[1].plot(df_hist['epoch'], df_hist['val_acc'], label='Val Accuracy', marker='o', linewidth=2)
    axes[1].plot(df_hist['epoch'], df_hist['val_macro_f1'], label='Val Macro F1', marker='^', linewidth=2)
    axes[1].plot(df_hist['epoch'], df_hist['val_balanced_acc'], label='Val Balanced Acc', marker='d', linewidth=2)
    axes[1].set_title('Validation Performance Metrics', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Saved Training History Plot: {save_path}")
    plt.close()
