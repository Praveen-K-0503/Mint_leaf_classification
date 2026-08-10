import os
import sys
import json
import io
import contextlib
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"
nb_files = sorted(list(notebooks_dir.glob("*.ipynb")))

print(f"=======================================================")
print(f"🚀 RE-EXECUTING ALL {len(nb_files)} OPTIMIZED NOTEBOOKS")
print(f"=======================================================\n")

for nb_path in nb_files:
    print(f"📘 Executing notebook: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    global_env = {'__name__': '__main__'}
    
    code_cell_idx = 0
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            code_cell_idx += 1
            src_lines = cell.get("source", [])
            code = "".join(src_lines)
            
            clean_code = "\n".join([
                f"# {l}" if l.strip().startswith("!") or l.strip().startswith("%") else l
                for l in code.splitlines()
            ])
            
            f_out = io.StringIO()
            try:
                with contextlib.redirect_stdout(f_out):
                    exec(clean_code, global_env)
                output_text = f_out.getvalue().strip()
                status = "✅ INSTANT PASSED"
            except Exception as e:
                output_text = f"ERROR: {type(e).__name__} - {e}"
                status = f"❌ FAILED: {type(e).__name__} - {e}"
                
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in output_text.splitlines()]
                }
            ]
            cell["execution_count"] = code_cell_idx
            print(f"   Cell {code_cell_idx}: {status}")
            
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        
    print(f"💾 Updated {nb_path.name}!\n")

print("All optimized notebooks executed instantly!")
