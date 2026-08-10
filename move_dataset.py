import os
import hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image

data_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai\data\raw\main mint lead dataset")

valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

total_images = 0
folder_counts = defaultdict(int)
format_counts = defaultdict(int)
corrupted_images = []
hash_to_files = defaultdict(list)
dimensions = []

print(f"Scanning dataset at {data_dir}...")

for root, dirs, files in os.walk(data_dir):
    for file in files:
        ext = Path(file).suffix.lower()
        if ext in valid_exts:
            total_images += 1
            full_path = Path(root) / file
            rel_path = full_path.relative_to(data_dir)
            top_folder = rel_path.parts[0]
            
            folder_counts[top_folder] += 1
            format_counts[ext] += 1
            
            # MD5 Hash for duplicate detection
            try:
                with open(full_path, 'rb') as f:
                    file_bytes = f.read()
                    file_hash = hashlib.md5(file_bytes).hexdigest()
                    hash_to_files[file_hash].append(str(rel_path))
                
                # Pillow read for dimensions & corruption test
                with Image.open(full_path) as img:
                    img.verify()
                with Image.open(full_path) as img:
                    w, h = img.size
                    dimensions.append({'folder': top_folder, 'w': w, 'h': h, 'mode': img.mode})
            except Exception as e:
                corrupted_images.append((str(rel_path), str(e)))

duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
total_duplicates = sum(len(files) - 1 for files in duplicates.values())

print("\n--- AUDIT SUMMARY ---")
print(f"Total images: {total_images}")
print("\nFolder breakdown:")
for folder, count in folder_counts.items():
    print(f"  - {folder}: {count} images")

print("\nFormats:")
for fmt, count in format_counts.items():
    print(f"  - {fmt}: {count}")

print(f"\nCorrupted images: {len(corrupted_images)}")
print(f"Unique MD5 hashes: {len(hash_to_files)}")
print(f"Duplicate image files found: {total_duplicates}")
print(f"Duplicate hash groups: {len(duplicates)}")
