import json
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"
nb_files = sorted(list(notebooks_dir.glob("*.ipynb")))

PIP_MAGIC_CELL = {
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install required dependencies into the active Jupyter kernel\n",
        "%pip install -q matplotlib seaborn opencv-python pillow pandas numpy tqdm\n",
        "print(\"✅ Packages installed and ready in kernel!\")\n"
    ]
}

for nb_path in nb_files:
    print(f"📘 Injecting %pip magic into: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    cells = nb_data.get("cells", [])
    
    # Remove any existing %pip install cells to avoid duplication
    cells = [c for c in cells if "%pip install" not in "".join(c.get("source", []))]
    
    # Find first markdown cell index
    md_idx = 0
    for idx, c in enumerate(cells):
        if c.get("cell_type") == "markdown":
            md_idx = idx
            break
            
    # Insert %pip magic cell right after title markdown
    cells.insert(md_idx + 1, PIP_MAGIC_CELL.copy())
    
    # Update execution counts
    code_cnt = 0
    for c in cells:
        if c.get("cell_type") == "code":
            code_cnt += 1
            c["execution_count"] = code_cnt
            
    nb_data["cells"] = cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        
    print(f"   ✅ Successfully added %pip magic to {nb_path.name}")

print("\nInjection complete!")
