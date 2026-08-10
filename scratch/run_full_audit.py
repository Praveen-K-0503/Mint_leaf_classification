import os
import sys
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image

data_raw_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai\data\raw")
output_reports = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai\outputs\reports")
output_vis = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai\outputs\visualizations")

output_reports.mkdir(parents=True, exist_ok=True)
output_vis.mkdir(parents=True, exist_ok=True)

# Locate dataset root containing target folders
target_names = {"Mint leaf", "Mentha (Mint)", "Fresh", "Spoiled", "Dried", "Augmented Mint Leaf"}
dataset_root = None

for root, dirs, files in os.walk(data_raw_dir):
    if target_names.intersection(set(dirs)):
        dataset_root = Path(root)
        break

if not dataset_root:
    dataset_root = data_raw_dir

supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

inventory_rows = []
folder_counts = defaultdict(int)
format_counts = defaultdict(int)
resolutions = []
hash_map = defaultdict(list)
corrupted_files = []

for root, dirs, files in os.walk(dataset_root):
    rel_root = Path(root).relative_to(dataset_root)
    parts = rel_root.parts
    if not parts:
        continue
    top_folder = parts[0]
    
    for f in files:
        ext = Path(f).suffix.lower()
        if ext in supported_extensions:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(data_raw_dir.parent)
            
            # MD5 Hash
            file_hash = None
            try:
                with open(full_path, 'rb') as fb:
                    file_bytes = fb.read()
                    file_hash = hashlib.md5(file_bytes).hexdigest()
            except Exception as e:
                corrupted_files.append((str(rel_path), f"Hash error: {e}"))
                continue
                
            # Image verification & metadata
            w, h, aspect_ratio, color_mode = None, None, None, None
            try:
                with Image.open(full_path) as img:
                    img.verify()
                with Image.open(full_path) as img:
                    w, h = img.size
                    color_mode = img.mode
                    aspect_ratio = round(w / h, 3)
                    resolutions.append((w, h))
            except Exception as e:
                corrupted_files.append((str(rel_path), f"Corrupt image: {e}"))
                continue
            
            folder_counts[top_folder] += 1
            format_counts[ext] += 1
            hash_map[file_hash].append(str(rel_path))
            
            inventory_rows.append({
                'class_name': top_folder,
                'filename': f,
                'path': str(rel_path),
                'file_extension': ext,
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio,
                'color_mode': color_mode,
                'image_hash': file_hash
            })

df_inventory = pd.DataFrame(inventory_rows)
total_images = len(df_inventory)

# Save Master Inventory CSV
inventory_csv_path = output_reports / "master_image_inventory.csv"
df_inventory.to_csv(inventory_csv_path, index=False)

# Duplicate calculations
duplicate_groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
total_duplicates = sum(len(paths) - 1 for paths in duplicate_groups.values())

# Most common resolution
most_common_res = Counter(resolutions).most_common(1)
if most_common_res:
    res_w, res_h = most_common_res[0][0]
    res_str = f"{res_w} × {res_h}"
else:
    res_str = "N/A"

# Save Concise JSON Summary
audit_summary = {
    'total_images': total_images,
    'folder_counts': dict(folder_counts),
    'format_counts': dict(format_counts),
    'corrupted_count': len(corrupted_files),
    'exact_duplicates_count': total_duplicates,
    'duplicate_groups_count': len(duplicate_groups),
    'most_common_resolution': res_str
}

json_path = output_reports / "dataset_audit_report.json"
with open(json_path, 'w') as jf:
    json.dump(audit_summary, jf, indent=4)

print("DATASET AUDIT COMPLETE\n")
print(f"Total Images: {total_images:,}\n")
print("Folders:")
for folder in ["Mint leaf", "Mentha (Mint)", "Fresh", "Spoiled", "Dried", "Augmented Mint Leaf"]:
    count = folder_counts.get(folder, 0)
    prefix = "└── " if folder == "Augmented Mint Leaf" else "├── "
    print(f"{prefix}{folder}: {count:,}")

print(f"\nCorrupted: {len(corrupted_files)}")
print(f"Exact Duplicates: {total_duplicates:,}\n")
print(f"Most Common Resolution:\n{res_str}\n")
print(f"Reports:\noutputs/reports/")
