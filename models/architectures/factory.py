import os
import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_name="resnet18", num_classes=6, pretrained=True):
    """
    Model Factory supporting multiple model families.
    """
    model_name_clean = model_name.lower().replace("-", "_")
    weights = "DEFAULT" if pretrained else None

    if "resnet18" in model_name_clean:
        model = models.resnet18(weights=weights if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif "resnet50" in model_name_clean:
        model = models.resnet50(weights=weights if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif "mobilenet_v3_small" in model_name_clean:
        model = models.mobilenet_v3_small(weights=weights if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    elif "mobilenet_v3_large" in model_name_clean:
        model = models.mobilenet_v3_large(weights=weights if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    elif "efficientnet_b0" in model_name_clean:
        model = models.efficientnet_b0(weights=weights if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif "densenet121" in model_name_clean:
        model = models.densenet121(weights=weights if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif "convnext_tiny" in model_name_clean:
        model = models.convnext_tiny(weights=weights if pretrained else None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
    elif "vit_b_16" in model_name_clean:
        model = models.vit_b_16(weights=weights if pretrained else None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
    else:
        # Fallback to ResNet18
        print(f"⚠️ Model '{model_name}' not specifically mapped. Defaulting to ResNet18 backbone.")
        model = models.resnet18(weights=weights if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    return model

def get_model_metrics(model, device="cpu", input_size=(3, 224, 224)):
    """
    Computes total parameter count, trainable parameter count, and model size on disk (MB).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB (float32 = 4 bytes per param)
    size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(size_mb, 2)
    }
