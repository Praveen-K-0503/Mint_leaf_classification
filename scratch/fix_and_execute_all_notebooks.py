import os
import sys
import json
import subprocess
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"

# Step 1: Ensure packages are installed in the current environment
print("Step 1: Installing dependencies in current Python kernel...")
req_pkgs = ["matplotlib", "seaborn", "opencv-python", "pillow", "pandas", "numpy", "tqdm"]
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + req_pkgs + ["--quiet"])
    print("✅ Dependencies installed successfully!")
except Exception as e:
    print(f"Notice during pip install: {e}")

# Inline auto-installer cell block to inject into notebook Cell 1
AUTO_INSTALLER_CODE = """# Auto-install missing dependencies if running in a fresh local or Colab kernel
import sys
import subprocess

required_modules = ['matplotlib', 'seaborn', 'cv2', 'PIL', 'pandas', 'numpy', 'tqdm']
for mod in required_modules:
    try:
        __import__(mod)
    except ImportError:
        pkg = 'opencv-python' if mod == 'cv2' else ('pillow' if mod == 'PIL' else mod)
        print(f"📦 Module '{mod}' missing. Auto-installing '{pkg}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
"""

# Process all 4 notebooks
nb_files = sorted(list(notebooks_dir.glob("*.ipynb")))

print(f"\nStep 2: Processing & Injecting Auto-Installer into {len(nb_files)} Notebooks...")

for nb_path in nb_files:
    print(f"\n📘 Processing: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    cells = nb_data.get("cells", [])
    
    # Check first code cell
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if code_cells:
        first_code_cell = code_cells[0]
        source_text = "".join(first_code_cell.get("source", []))
        if "Auto-install missing dependencies" not in source_text:
            # Prepend auto-installer
            first_code_cell["source"] = [AUTO_INSTALLER_CODE + "\n"] + first_code_cell["source"]
            print(f"   ✅ Injected auto-installer into {nb_path.name}")
            
    # Save updated notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)

print("\nStep 3: Executing and testing code blocks cell by cell...")

# Execute notebooks programmatically
for nb_path in nb_files:
    print(f"\n▶ Executing: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    global_env = {'__name__': '__main__'}
    cell_idx = 0
    cell_errors = 0
    
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            cell_idx += 1
            src_lines = cell.get("source", [])
            code = "".join(src_lines)
            
            # Clean colab magics for local exec
            clean_code = "\n".join([
                f"# {l}" if l.strip().startswith("!") or l.strip().startswith("%") else l
                for l in code.splitlines()
            ])
            
            # Execute cell code
            try:
                exec(clean_code, global_env)
                print(f"   Cell {cell_idx}: ✅ PASSED")
            except Exception as err:
                cell_errors += 1
                print(f"   Cell {cell_idx}: ❌ FAILED - {type(err).__name__}: {err}")
                
    if cell_errors == 0:
        print(f"🎉 {nb_path.name} EXECUTED WITH ZERO ERRORS!")
    else:
        print(f"⚠️ {nb_path.name} encountered {cell_errors} errors.")

print("\nDone!")
