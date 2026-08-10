import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd

class MintDataset(Dataset):
    """
    Custom PyTorch Dataset for Mint Leaf AI 6-class dataset.
    Loads images directly from data/processed/<split>/<class>/ or manifest.
    """
    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 6 Primary Classes in deterministic sorted order
        self.classes = sorted([
            "Blight_Rhizoctonia",
            "Healthy",
            "Leaf_Spot",
            "Mint_Rust",
            "Post_Harvest_Deteriorated",
            "Powdery_Mildew"
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_folder = self.split_dir / cls_name
            if cls_folder.exists():
                for img_path in sorted(cls_folder.glob("*.jpg")) + sorted(cls_folder.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image as RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, str(img_path)

def get_transforms(img_size=224, is_train=True):
    """
    Standard PyTorch ImageNet normalization & resizing.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_dataloaders(processed_dir, batch_size=32, img_size=224, num_workers=0):
    """
    Constructs PyTorch DataLoaders for train, validation, and test splits.
    """
    processed_dir = Path(processed_dir)
    
    train_dataset = MintDataset(processed_dir / "train", transform=get_transforms(img_size, is_train=True))
    val_dataset = MintDataset(processed_dir / "validation", transform=get_transforms(img_size, is_train=False))
    test_dataset = MintDataset(processed_dir / "test", transform=get_transforms(img_size, is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": train_dataset.idx_to_class,
        "classes": train_dataset.classes
    }
