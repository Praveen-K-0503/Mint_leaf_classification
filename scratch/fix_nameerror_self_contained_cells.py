import json
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"

nb3_path = notebooks_dir / "03_disease_dataset_inventory.ipynb"

with open(nb3_path, "r", encoding="utf-8") as f:
    nb3_data = json.load(f)

for cell in nb3_data.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "OUTPUT_SOURCES_DIR" in src and "self-contained" not in src:
            # Prepend defensive path initializer
            defensive_prefix = """# Self-contained path & variable guard (allows running cell independently)
import os, sys, json, time
from pathlib import Path
import pandas as pd

cwd = Path(os.getcwd()).resolve()
BASE_PATH = cwd.parent if cwd.name == 'notebooks' else cwd
OUTPUT_SOURCES_DIR = BASE_PATH / 'outputs' / 'reports' / 'dataset_sources'
OUTPUT_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

"""
            cell["source"] = [defensive_prefix] + cell["source"]

with open(nb3_path, "w", encoding="utf-8") as f:
    json.dump(nb3_data, f, indent=1)

print("✅ Added self-contained path guard to 03_disease_dataset_inventory.ipynb!")
