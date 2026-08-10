import os
import sys
import json
import subprocess
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"

# Cell 1 Template: Dedicated Dependency Auto-Installer (Runs BEFORE any import)
CELL_1_AUTO_INSTALLER = {
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Cell 1: Environment & Dependency Setup (Auto-installs missing packages before import)\n",
        "import sys\n",
        "import subprocess\n",
        "\n",
        "required_modules = ['matplotlib', 'seaborn', 'cv2', 'PIL', 'pandas', 'numpy', 'tqdm']\n",
        "for mod in required_modules:\n",
        "    try:\n",
        "        __import__(mod)\n",
        "    except ImportError:\n",
        "        pkg = 'opencv-python' if mod == 'cv2' else ('pillow' if mod == 'PIL' else mod)\n",
        "        print(f\"📦 Auto-installing missing module '{pkg}' in current Jupyter kernel...\")\n",
        "        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", pkg])\n",
        "\n",
        "print(\"✅ All required dependencies (matplotlib, seaborn, cv2, PIL, pandas, numpy, tqdm) are active!\")\n"
    ]
}

nb_files = sorted(list(notebooks_dir.glob("*.ipynb")))

for nb_path in nb_files:
    print(f"📘 Restructuring notebook: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    cells = nb_data.get("cells", [])
    
    # Remove any duplicate auto-installer headers from old cell 1
    for c in cells:
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            if "Auto-install missing dependencies" in src:
                # Remove auto installer header lines from this cell
                lines = src.splitlines(keepends=True)
                clean_lines = [l for l in lines if not any(k in l for k in ["required_modules", "Auto-install missing", "__import__", "subprocess.check_call"])]
                c["source"] = clean_lines
                
    # Insert dedicated CELL_1_AUTO_INSTALLER as the very first code cell after title markdown
    # Find first markdown cell
    md_idx = 0
    for idx, c in enumerate(cells):
        if c.get("cell_type") == "markdown":
            md_idx = idx
            break
            
    # Insert Cell 1 right after initial title markdown
    cells.insert(md_idx + 1, CELL_1_AUTO_INSTALLER.copy())
    
    # Re-index execution counts
    code_cnt = 0
    for c in cells:
        if c.get("cell_type") == "code":
            code_cnt += 1
            c["execution_count"] = code_cnt
            
    nb_data["cells"] = cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        
    print(f"   ✅ Successfully updated {nb_path.name} with dedicated Cell 1 Dependency Auto-Installer!")

print("\nRestructuring complete!")
