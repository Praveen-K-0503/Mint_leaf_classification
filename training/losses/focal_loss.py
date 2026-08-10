import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(loss_name="cross_entropy", class_counts=None, device="cpu", gamma=2.0):
    """
    Loss function factory.
    Supports:
      - 'cross_entropy': Standard Cross Entropy
      - 'weighted_cross_entropy': Class-Weighted Cross Entropy based on inverse class frequency
      - 'focal_loss': Focal Loss with gamma=2.0
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
        
    elif loss_name == "weighted_cross_entropy":
        if class_counts is not None:
            total_samples = sum(class_counts.values())
            # Compute inverse class frequencies
            weights = [total_samples / (len(class_counts) * class_counts[cls]) for cls in sorted(class_counts.keys())]
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            return nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            return nn.CrossEntropyLoss()
            
    elif loss_name == "focal_loss":
        alpha = None
        if class_counts is not None:
            total_samples = sum(class_counts.values())
            weights = [total_samples / (len(class_counts) * class_counts[cls]) for cls in sorted(class_counts.keys())]
            alpha = torch.tensor(weights, dtype=torch.float32).to(device)
        return FocalLoss(gamma=gamma, alpha=alpha)
        
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
