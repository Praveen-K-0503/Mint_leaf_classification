import json
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"
nb_files = sorted(list(notebooks_dir.glob("*.ipynb")))

print("Cleaning up notebooks and removing pip magic calls that cause Jupyter kernel hangs...")

for nb_path in nb_files:
    print(f"📘 Optimizing notebook: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    cells = nb_data.get("cells", [])
    
    # Remove any cell that contains %pip install or subprocess pip install
    cleaned_cells = []
    for c in cells:
        src = "".join(c.get("source", []))
        if "%pip install" in src or "subprocess.check_call" in src:
            continue
        cleaned_cells.append(c)
        
    # Re-index execution counts
    code_cnt = 0
    for c in cleaned_cells:
        if c.get("cell_type") == "code":
            code_cnt += 1
            c["execution_count"] = code_cnt
            
    nb_data["cells"] = cleaned_cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        
    print(f"   ✅ Cleaned {nb_path.name} — Execution will now be instant (no pip hanging!)")

print("\nNotebook optimization complete!")
