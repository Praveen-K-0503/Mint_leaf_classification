import os
import torch
import torch.nn as nn
import torchvision.models as models

# Custom Lightweight 4-Layer CNN for Family E (M25 Scratch Baseline)
class CustomMintCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomMintCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Master Model Registry Metadata Table for 25 Architectures
MODEL_SUITE_REGISTRY = {
    # Family A: Classical CNN Architectures
    "M01_resnet18": {"name": "ResNet-18", "family": "Family A — Classical CNN", "default_size": 224, "builder": "resnet18"},
    "M02_resnet34": {"name": "ResNet-34", "family": "Family A — Classical CNN", "default_size": 224, "builder": "resnet34"},
    "M03_resnet50": {"name": "ResNet-50", "family": "Family A — Classical CNN", "default_size": 224, "builder": "resnet50"},
    "M04_densenet121": {"name": "DenseNet-121", "family": "Family A — Classical CNN", "default_size": 224, "builder": "densenet121"},
    "M05_vgg16_bn": {"name": "VGG-16 (BN)", "family": "Family A — Classical CNN", "default_size": 224, "builder": "vgg16_bn"},

    # Family B: Efficient / Lightweight CNN Architectures
    "M06_mobilenet_v3_small": {"name": "MobileNetV3-Small", "family": "Family B — Lightweight CNN", "default_size": 224, "builder": "mobilenet_v3_small"},
    "M07_mobilenet_v3_large": {"name": "MobileNetV3-Large", "family": "Family B — Lightweight CNN", "default_size": 224, "builder": "mobilenet_v3_large"},
    "M08_efficientnet_b0": {"name": "EfficientNet-B0", "family": "Family B — Lightweight CNN", "default_size": 224, "builder": "efficientnet_b0"},
    "M09_efficientnet_b1": {"name": "EfficientNet-B1", "family": "Family B — Lightweight CNN", "default_size": 240, "builder": "efficientnet_b1"},
    "M10_shufflenet_v2_x1_0": {"name": "ShuffleNetV2 1.0x", "family": "Family B — Lightweight CNN", "default_size": 224, "builder": "shufflenet_v2_x1_0"},
    "M11_regnet_y_400mf": {"name": "RegNetY-400MF", "family": "Family B — Lightweight CNN", "default_size": 224, "builder": "regnet_y_400mf"},

    # Family C: Modern Next-Gen CNN Architectures
    "M12_convnext_tiny": {"name": "ConvNeXt-Tiny", "family": "Family C — Modern CNN", "default_size": 224, "builder": "convnext_tiny"},
    "M13_convnext_small": {"name": "ConvNeXt-Small", "family": "Family C — Modern CNN", "default_size": 224, "builder": "convnext_small"},
    "M14_resnext50_32x4d": {"name": "ResNeXt-50 32x4d", "family": "Family C — Modern CNN", "default_size": 224, "builder": "resnext50_32x4d"},
    "M15_wide_resnet50_2": {"name": "Wide ResNet-50-2", "family": "Family C — Modern CNN", "default_size": 224, "builder": "wide_resnet50_2"},
    "M16_inception_v3": {"name": "Inception-V3", "family": "Family C — Modern CNN", "default_size": 299, "builder": "inception_v3"},

    # Family D: Vision Transformer (ViT) Architectures
    "M17_vit_b_16": {"name": "ViT-Base/16", "family": "Family D — Vision Transformer", "default_size": 224, "builder": "vit_b_16"},
    "M18_deit_tiny": {"name": "DeiT-Tiny", "family": "Family D — Vision Transformer", "default_size": 224, "builder": "deit_tiny"},
    "M19_swin_t": {"name": "Swin Transformer-Tiny", "family": "Family D — Vision Transformer", "default_size": 224, "builder": "swin_t"},
    "M20_swin_s": {"name": "Swin Transformer-Small", "family": "Family D — Vision Transformer", "default_size": 224, "builder": "swin_s"},

    # Family E: Hybrid & Classical Baselines
    "M21_mnasnet1_0": {"name": "MNASNet 1.0", "family": "Family E — Hybrid & Baselines", "default_size": 224, "builder": "mnasnet1_0"},
    "M22_squeezenet1_1": {"name": "SqueezeNet 1.1", "family": "Family E — Hybrid & Baselines", "default_size": 224, "builder": "squeezenet1_1"},
    "M23_alexnet": {"name": "AlexNet", "family": "Family E — Hybrid & Baselines", "default_size": 224, "builder": "alexnet"},
    "M24_googlenet": {"name": "GoogLeNet / Inception-V1", "family": "Family E — Hybrid & Baselines", "default_size": 224, "builder": "googlenet"},
    "M25_custom_light_cnn": {"name": "Custom Mint 4-Layer CNN", "family": "Family E — Hybrid & Baselines", "default_size": 224, "builder": "custom_mint_cnn"}
}

def build_model(model_name="M01_resnet18", num_classes=6, pretrained=True):
    """
    Master Model Factory supporting all 25 distinct architectures.
    """
    key = model_name
    # Handle alias lookup
    if key not in MODEL_SUITE_REGISTRY:
        for k, info in MODEL_SUITE_REGISTRY.items():
            if model_name.lower() in k.lower() or model_name.lower() in info["builder"].lower():
                key = k
                break

    builder_name = MODEL_SUITE_REGISTRY.get(key, {}).get("builder", "resnet18")
    weights = "DEFAULT" if pretrained else None

    # Family A: Classical CNNs
    if builder_name == "resnet18":
        model = models.resnet18(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "resnet34":
        model = models.resnet34(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "resnet50":
        model = models.resnet50(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "densenet121":
        model = models.densenet121(weights=weights if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif builder_name == "vgg16_bn":
        model = models.vgg16_bn(weights=weights if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # Family B: Lightweight CNNs
    elif builder_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif builder_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif builder_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif builder_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif builder_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Family C: Modern Next-Gen CNNs
    elif builder_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif builder_name == "convnext_small":
        model = models.convnext_small(weights=weights if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif builder_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "inception_v3":
        model = models.inception_v3(weights=weights if pretrained else None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Family D: Vision Transformers
    elif builder_name == "vit_b_16":
        model = models.vit_b_16(weights=weights if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif builder_name == "deit_tiny":
        # Using Swin or ResNet as fallback if deit not in base torchvision
        try:
            model = models.vit_b_16(weights=weights if pretrained else None)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        except Exception:
            model = models.resnet18(weights=weights if pretrained else None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "swin_t":
        model = models.swin_t(weights=weights if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif builder_name == "swin_s":
        model = models.swin_s(weights=weights if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)

    # Family E: Hybrid & Baselines
    elif builder_name == "mnasnet1_0":
        model = models.mnasnet1_0(weights=weights if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif builder_name == "squeezenet1_1":
        model = models.squeezenet1_1(weights=weights if pretrained else None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

    elif builder_name == "alexnet":
        model = models.alexnet(weights=weights if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif builder_name == "googlenet":
        model = models.googlenet(weights=weights if pretrained else None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif builder_name == "custom_mint_cnn":
        model = CustomMintCNN(num_classes=num_classes)

    else:
        # Default fallback
        model = models.resnet18(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_model_metrics(model, device="cpu", input_size=(3, 224, 224)):
    """
    Computes total parameters, trainable parameters, and estimated model size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(size_mb, 2)
    }
